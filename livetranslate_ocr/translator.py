"""翻译层：记忆库 → LLM 流式意译 → 本地词典直译，三级降级。

为什么要 LLM：要求里最核心的一条是"口语化"，不能直译。逐词对齐的机器翻译
（包括 NLLB/opus-mt 这类神经模型）本质仍在做词级对应，翻出来是翻译腔。
LLM 能同时拿到上文、术语表和风格约束，才可能输出"这不科学啊"而不是"这是不科学的"。

## 三级降级（本项目与音频版最大的差别）

| 顺序 | 来源 | `source` | 代价 |
| --- | --- | --- | --- |
| 1 | 记忆库精确命中 | `tm-exact` | 0 |
| 2 | 记忆库模糊命中 | `tm-fuzzy` | ~0 |
| 3 | **本地词典直译** | `local` | ~0，质量粗糙 |
| 4 | LLM 流式意译 | `llm` / `llm-partial` | 1~2 秒，要联网 |

第 3 档**只在 LLM 不可用时才启用**。这不是优化，是必须：正常时每句都走直译
等于把质量白降一档，而那正是这个项目最不该牺牲的东西。

"LLM 不可用"由 `OfflineCfg` 的断路器判定：连续失败 `fail_threshold` 次进入离线态，
之后每隔 `retry_interval_s` 拿**当前这一句**当探针再试一次（不额外发 ping，
省一次往返），成功就自动切回。`fail_threshold <= 0` 表示永不降级，每句都硬试。

**降级绝不能静默。** 老项目最惨的教训就是"出问题的时候屏幕上一点线索都没有"，
用户以为是程序坏了。所以每次降级都带上人能读懂的原因（`detail`），
字幕窗照 `source_label()` 显示「离线词典」而不是假装一切正常。

## 缓存纪律

`local` 的结果**绝不写进记忆库**。直译质量粗糙，一旦写进去，网络恢复后
同一句话会被这条烂译文顶掉——这正是老项目踩过的"缓存投毒"，而且它很难查：
表现是"网络明明好了，这句话还是怪怪的"。`llm-partial`（流被截断）同理不写。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable

from . import langs
from .config import Config
from .dicts import Dicts
from .llm import ChatClient, LlmError, is_transient
from .tm import TranslationMemory, variant_tag

log = logging.getLogger(__name__)

# 风格约束放在 system 里且保持稳定，让服务端的 prompt 缓存能命中前缀
SYSTEM_PROMPT = """你是直播实时字幕的翻译引擎。观众听不懂原声，只能靠你的译文理解主播说了什么，所以译文必须像目标语言的母语者随口说出来的话。

【风格要求】
- 口语优先。用日常说话的方式，不要新闻稿腔、不要书面语、不要翻译腔。
- 意译优先于直译。允许调整语序、拆分或合并短句、补全省略的主语和宾语、把俗语换成目标语言里对等的说法。禁止逐词对应。
- 保留说话人的情绪和语域：玩笑要像玩笑，吐槽要像吐槽，激动就激动，敷衍就敷衍。脏话按原意译出，不要净化。
- 语气词、口癖、笑声（umm / you know / lol / haha / I mean 之类）译成目标语言里对应的口语感觉即可，也可以按上下文省略，不要机械对应。
- 不要添加原文没有的信息、评价、解释或补充说明。

【硬性规则】
- 只输出译文本身。不要原文、不要引号、不要"译文："之类的标签、不要 Markdown、不要任何解释。
- 人名、地名、品牌、游戏术语用通行译法；下方术语表给出的必须严格照用。
- 如果原句是被截断的半句话，按最可能的语境补全成一句自然的话，不要输出"（听不清）"之类的占位符。
- 逐字转写里的口误、重复词、结巴（"I I I mean"）直接理顺，不要照搬。"""

# 这一句是给 OCR 输入路径加的，和音频版不同的地方必须写进 prompt：
# 屏幕字幕是机器生成的，偶尔会认错一两个字母，模型要按语境理解，而不是照着错字翻。
OCR_NOTE = """- 输入来自屏幕字幕的文字识别，可能有个别字母被认错（人名、枪械名的大小写或数字），按上下文理解它真正想说的词，不要把识别噪声照翻进译文。
- 输入可能没有句末标点，那是字幕的特性，不代表这是半句话，不要因此补上猜测的结尾。"""

_LABEL = re.compile(r"^\s*(译文|翻译|中文|Translation|Translated|Output|Output:)\s*[:：]\s*", re.IGNORECASE)
_QUOTES = {'"': '"', "'": "'", "「": "」", "『": "』", "《": "》", "“": "”", "‘": "’"}

# 字幕窗 meta 行显示什么。降级必须让人看得出来，不能假装正常。
_LABELS = {
    "tm-exact": "记忆库",
    "tm-fuzzy": "记忆库≈",
    "llm": "模型",
    "llm-partial": "模型·截断",
    "local": "离线词典",
    "local-raw": "离线·原文",
    "error": "无译文",
}


def source_label(source: str) -> str:
    """把 `source` 翻成人看的一小段字（给字幕窗的 meta 行）。"""
    return _LABELS.get(source, source or "未知")


def is_degraded(source: str) -> bool:
    """这条路是不是"退而求其次"——UI 据此把状态行点亮/变红。"""
    return source in ("local", "local-raw", "error")


class LlmUnavailable(RuntimeError):
    """LLM 根本用不了：没配 Key、地址写错。

    和"网络抖了一下"要分开：这类错误重试一万次也没用，所以直接长期降级，
    不要每 30 秒拿着用户的字幕去撞一次墙。
    """


@dataclass
class TranslationOutcome:
    text: str
    source: str          # tm-exact | tm-fuzzy | llm | llm-partial | local | local-raw | error
    latency_s: float
    tm_score: float = 0.0
    detail: str = ""     # 降级原因（人能读懂的一句话）
    coverage: float = 0.0  # local 时的词典覆盖率，调阈值时看它

    @property
    def label(self) -> str:
        return source_label(self.source)

    @property
    def degraded(self) -> bool:
        return is_degraded(self.source)


def sanitize(text: str) -> str:
    """清掉模型爱加的那点包装：引号、标签、Markdown、换行。"""
    out = (text or "").strip()
    out = _LABEL.sub("", out)

    # 去掉首尾成对的引号（可能嵌套一两层）
    for _ in range(2):
        if len(out) >= 2 and out[0] in _QUOTES and out[-1] == _QUOTES[out[0]]:
            out = out[1:-1].strip()

    out = out.replace("**", "").replace("__", "")
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out)
    # 字幕是单行区域，换行统一压成空格
    out = re.sub(r"\s*\n+\s*", " ", out)
    return out.strip()


class Translator:
    def __init__(self, cfg: Config, tm: TranslationMemory, dicts: Dicts):
        self.cfg = cfg
        self.tm = tm
        self.dicts = dicts
        self.variant = variant_tag(cfg.translate.dst_lang, cfg.translate.style)
        self._client = None
        self._last_emit = 0.0

        # ---- 离线断路器状态 ----
        self._fail_streak = 0
        self._offline = False
        self._permanent_offline = False
        self._retry_at = 0.0
        self._last_error = ""

    # ---------------------------------------------------------------- 客户端
    def _ensure_client(self) -> ChatClient:
        if self._client is not None:
            return self._client
        api_key = self.cfg.resolved_api_key()
        if not api_key:
            raise LlmUnavailable(
                f"没配置翻译模型的 API Key（环境变量 {self.cfg.llm.api_key_env}"
                f" 或 config.toml 的 [llm] api_key），只能用本地词典"
            )
        self._client = ChatClient(
            base_url=self.cfg.llm.base_url,
            api_key=api_key,
            timeout_s=self.cfg.llm.timeout_s,
            extra_headers=self.cfg.llm.extra_headers,
        )
        log.info("翻译模型：%s @ %s", self.cfg.llm.model, self.cfg.llm.base_url)
        return self._client

    def check_ready(self, attempts: int = 3, base_delay_s: float = 1.0) -> None:
        """启动时做一次连通性检查，配错了立刻暴露，而不是等开播才发现。

        服务端过载（503 Service is too busy）这类临时故障会退避重试——DeepSeek
        忙的时候很常见，不该因为一次抖动就把整个启动流程打断。
        配置类错误（Key 无效、余额不足、模型名错）不重试，重试也没用。
        """
        client = self._ensure_client()
        for i in range(attempts):
            try:
                client.chat(
                    self.cfg.llm.model,
                    [{"role": "user", "content": "ping"}],
                    temperature=0.0,
                    max_tokens=4,
                )
            except LlmError as exc:
                if not is_transient(exc) or i == attempts - 1:
                    raise
                delay = base_delay_s * (2**i)
                log.warning(
                    "连通性检查失败（第 %d/%d 次），%.0fs 后重试：%s",
                    i + 1, attempts, delay, exc,
                )
                time.sleep(delay)
            else:
                log.info("翻译模型连通性检查通过")
                self._note_success()
                return

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---------------------------------------------------------------- 断路器
    def _llm_ready(self, now: float) -> bool:
        """现在该不该去调 LLM。"""
        if self._permanent_offline:
            return False
        if self.cfg.offline.fail_threshold <= 0:
            return True          # 配置成"永不降级"：每句都硬试
        if self._fail_streak < self.cfg.offline.fail_threshold:
            return True
        return now >= self._retry_at

    def _note_failure(self, exc: BaseException) -> None:
        self._fail_streak += 1
        self._last_error = str(exc)
        thr = self.cfg.offline.fail_threshold
        if thr > 0 and self._fail_streak >= thr:
            if not self._offline:
                log.warning(
                    "翻译模型连续失败 %d 次，改用本地词典；%.0fs 后再试一次：%s",
                    self._fail_streak, self.cfg.offline.retry_interval_s, exc,
                )
            self._offline = True
            self._retry_at = time.monotonic() + self.cfg.offline.retry_interval_s

    def _note_success(self) -> None:
        if self._offline:
            log.info("翻译模型恢复，切回模型翻译")
        self._offline = False
        self._fail_streak = 0
        self._last_error = ""

    def health(self) -> dict:
        """给状态行/诊断用：现在是不是离线态、还要多久重试、最后一次错在哪。"""
        return {
            "offline": self._offline,
            "permanent": self._permanent_offline,
            "fail_streak": self._fail_streak,
            "retry_in_s": max(0.0, self._retry_at - time.monotonic()) if self._offline else 0.0,
            "last_error": self._last_error,
            "model": self.cfg.llm.model,
        }

    # ---------------------------------------------------------------- prompt
    def _build_messages(
        self, text: str, src_lang: str, context: list[tuple[str, str]]
    ) -> list[dict]:
        src_name = langs.display_name(src_lang) if src_lang and src_lang != "auto" else "源语言"
        dst_name = langs.display_name(self.cfg.translate.dst_lang)
        style = self.cfg.translate.style.strip()
        style_line = f"\n- 本次任务的风格偏好：{style}。" if style else ""

        system = (
            f"把{src_name}翻译成{dst_name}。\n\n" + SYSTEM_PROMPT + "\n" + OCR_NOTE + style_line
        )

        blocks: list[str] = []

        hints = self.dicts.term_hints(text, self.cfg.translate.glossary_max_terms)
        if hints:
            lines = "\n".join(f"- {h.term}：{h.desc}" for h in hints)
            blocks.append("[术语表]\n" + lines)

        if context and self.cfg.translate.context_lines > 0:
            recent = context[-self.cfg.translate.context_lines :]
            lines = "\n".join(f"原文: {s}\n译文: {d}" for s, d in recent if d)
            if lines:
                blocks.append("[前文（仅供理解语境，不要重复翻译）]\n" + lines)

        blocks.append(f"[本次待翻译]\n{text}")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(blocks)},
        ]

    def _max_tokens(self, text: str) -> int:
        raw = int(len(text) * self.cfg.llm.max_tokens_factor) + 32
        return max(32, min(raw, self.cfg.llm.max_tokens_cap))

    # ---------------------------------------------------------------- 第 3 档
    def _local(self, text: str, reason: str, t0: float,
               on_delta: Callable[[str], None] | None) -> TranslationOutcome:
        """LLM 用不了时的兜底：逐词直译；覆盖率太低就干脆显示英文原文。

        覆盖率低的时候宁可给原文——"看不懂英语原文"至少是确定的，
        而"半中半英还语序错乱"会让人以为程序坏了。
        """
        g = self.dicts.gloss(text)
        gate = self.cfg.dictionaries.min_coverage
        if g.coverage >= gate:
            out, src = g.text, "local"
        else:
            out, src = text, "local-raw"
        if on_delta:
            on_delta(out)
        log.info("降级[%s]（覆盖 %.0f%%）%r -> %r", src, g.coverage * 100, text, out)
        return TranslationOutcome(
            text=out, source=src, latency_s=time.perf_counter() - t0,
            detail=reason, coverage=g.coverage,
        )

    # ---------------------------------------------------------------- 主流程
    def translate(
        self,
        text: str,
        src_lang: str = "",
        context: list[tuple[str, str]] | None = None,
        on_delta: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> TranslationOutcome:
        """翻译一句。**永远不会抛异常，也永远不会返回"什么都没有"就完事**：
        最差也会退回英文原文，并让 `source`/`detail` 讲清楚发生了什么。
        """
        text = (text or "").strip()
        if not text:
            return TranslationOutcome(text="", source="error", latency_s=0.0,
                                      detail="空文本")
        src_lang = src_lang or self.cfg.translate.src_lang
        context = context or []
        t0 = time.perf_counter()

        # 1) 本地记忆库（不管在不在线，先查——它是零成本且质量可靠的）
        hit = self.tm.lookup(text, src_lang, self.variant)
        if hit is not None:
            if on_delta:
                on_delta(hit.target)
            kind = "tm-exact" if hit.score >= 100 else "tm-fuzzy"
            return TranslationOutcome(
                text=hit.target, source=kind,
                latency_s=time.perf_counter() - t0, tm_score=hit.score,
            )

        # 2) LLM 不可用（或断路器已跳闸）→ 直接走本地词典
        now = time.monotonic()
        if not self._llm_ready(now):
            reason = ("没配 API Key，只能用本地词典" if self._permanent_offline
                      else f"翻译模型连不上（{self._last_error or '未知错误'}），"
                           f"已切到本地词典")
            return self._local(text, reason, t0, on_delta)

        # 3) LLM 流式意译
        try:
            client = self._ensure_client()
            messages = self._build_messages(text, src_lang, context)
        except LlmUnavailable as exc:
            # 没配 Key 这类问题重试没有意义，直接钉死在离线态
            self._permanent_offline = True
            self._offline = True
            self._last_error = str(exc)
            log.warning("翻译模型不可用：%s", exc)
            return self._local(text, str(exc), t0, on_delta)
        except Exception as exc:  # 建客户端时的意外错误也不能让字幕断掉
            self._note_failure(exc)
            log.warning("翻译客户端初始化失败：%s", exc)
            return self._local(text, f"翻译模型初始化失败（{exc}）", t0, on_delta)

        deadline = t0 + self.cfg.llm.timeout_s
        buf: list[str] = []
        partial = False
        stream = None
        failure: BaseException | None = None
        try:
            stream = client.stream_chat(
                self.cfg.llm.model,
                messages,
                temperature=self.cfg.llm.temperature,
                max_tokens=self._max_tokens(text),
                **self.cfg.llm.extra_body,
            )
            for delta in stream:
                if should_stop and should_stop():
                    partial = True
                    break
                buf.append(delta)
                if on_delta:
                    now = time.perf_counter()
                    # 节流：约 25 帧/秒足够顺滑，再密就是浪费重绘
                    if now - self._last_emit >= 0.04:
                        self._last_emit = now
                        on_delta(sanitize("".join(buf)))
                if time.perf_counter() > deadline:
                    partial = True
                    log.debug("翻译超出 %.1fs 预算，使用已产出的部分", self.cfg.llm.timeout_s)
                    break
        except LlmError as exc:
            failure = exc
        except Exception as exc:
            failure = exc
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        out = sanitize("".join(buf))

        # 一个字都没流出来才算失败。已经流出了一部分就认它——
        # 断在半路的译文也比"退回英文原文"强，何况用户已经看到那半句了。
        if failure is not None and not out:
            self._note_failure(failure)
            log.warning("翻译请求失败：%s", failure)
            return self._local(text, f"翻译模型请求失败（{failure}）", t0, on_delta)

        if failure is not None:
            partial = True

        if not out:
            # 模型连通但返回空（极少见）。这不算"连不上"，不该让断路器跳闸。
            log.warning("模型返回空结果：%r", text[:40])
            return self._local(text, "翻译模型返回了空结果", t0, on_delta)

        self._note_success()
        latency = time.perf_counter() - t0

        # 4) 回填记忆库。**只有在流完整时才写**：
        #    - partial（被截断/超时/被新句打断）的译文不完整，写进去会毒化缓存；
        #    - 依赖上下文的句子由 tm.store 自己拦（句尾挂在 and/the/of 上的那种，
        #      正好就是 OCR 半句话的形态）。
        if not partial:
            if not self.tm.store(text, src_lang, self.variant, out):
                log.debug("未写入缓存（依赖上下文）：%s", text[:40])

        if on_delta:
            on_delta(out)

        return TranslationOutcome(
            text=out,
            source="llm-partial" if partial else "llm",
            latency_s=latency,
        )
