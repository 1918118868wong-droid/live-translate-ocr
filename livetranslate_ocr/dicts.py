"""两区词典：游戏术语 + 日常高频词。

两区都是能手改的 TSV（「英文 <TAB> 中文」），但**用途完全不同**，别混：

| 区 | 文件 | 用途 |
| --- | --- | --- |
| 游戏术语 | `data/terms.tsv` | ① 命中后连说明一起注入 prompt 的 [术语表]，让模型照用通行译法<br>② 离线直译时整条替换（用第三列的短词形，没有就保留英文） |
| 日常高频词 | `data/common.tsv` | **只服务离线降级**：LLM 挂掉时逐词直译，至少能看懂大意 |

**为什么日常词不进 prompt**：模型本来就认识 "the"、"and"、"watch"，
把它们当术语注进去只会占额度、稀释真正重要的术语，还会和"口语化意译"的要求打架。
所以 `term_hints()` 只查第一区，这是有意的，不是漏了。

## 术语区为什么有第三列

第二列是**给模型看的说明**，写成完整的话最有用（"残局（以少打多）"、"保留 AWP 不译"）。
但离线直译是要把词**替换进句子里**的，塞一段说明进去就成了怪话。所以：

    clutch	残局（以少打多）	残局
    awp	CS 里的狙击枪，保留 "AWP" 不译	     ← 没有第三列，离线时保留英文原词

第三列可以不给。不给就保留英文——对 `AWP`/`NiKo`/`deagle` 这类词，保留英文本来就是对的：
中文解说本来也念 "AWP"、"NiKo"。目前 173 条里 129 条有短词形，44 条有意留空
（枪械专名、战队名、选手名）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DictCfg, PROJECT_ROOT

log = logging.getLogger(__name__)

# 词边界：前面不能再挨着字母或数字。没有这个的话 `ak` 会在 `ak47`/`nak` 里命中，
# `he` 会在 `the` 里命中——这类误命中很难从译文上看不出来，是最难查的一类 bug。
#
# 前面也挡掉数字，是为了 `1v3` 这种写法：不然它会切出一个孤零零的 `v`，
# 那是个查不到、也不该查的实词，白拉低覆盖率。解说里 `1v3`、`2v4` 很常见。
_WORD_TOKENS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z']*")

# 功能词：冠词、介词、助动词、代词、连词。直译时它们留在英文里几乎不影响可读性——
# 读中文的人本来就是把这些语法成分跳过去看的。真正决定"这段话还能不能看"的，
# 是**实词**有没有译出来。
#
# 覆盖率必须只数实词，这不只是"更准"，是原来那个算法**判反了**（实测踩到）：
#   "NiKo gets the clutch"        → "NiKo gets the 残局"          功能词拖累，算成 50% → 退回英文
#   "the weather is nice today…"  → "the weather 是 nice 今天…"   实词全没译，反倒 56% → 放行
# 恰好和它要防的东西拧着。剔除功能词后前者 67% 放行、后者 25% 退回，方向才对。
#
# 这个表只用来算分母，不影响匹配本身（这些词在 common.tsv 里的照样会被替换）。
_FUNCTION_WORDS = frozenset("""
a an the of to in on at for with from by as into onto over under about
and or but so nor yet
is are was were be been being am do does did done have has had having
will would can could shall should may might must
i you he she it we they me him her us them my your his its our their
this that these those there here
not no if when while than then
""".split())


# 数词。它们**不需要译**，所以算覆盖率时一律当成"已经处理过了"——
# 中文读者看 "one versus three"、"1v3"、"two HP" 直接就能懂。而数字在解说里
# 出现得极密，把它们算成"没认出来"会让大量本来完全能用的句子卡在阈值下面：
#   "NiKo gets the clutch, one versus three."  实词 6 个，其中 one/three 不可能命中
#   → 33%，整句退回英文。而这句话里最关键的 clutch 其实是译得出来的。
# 当成已处理之后是 67%，能用了。
_NUMBER_WORDS = frozenset("""
one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
first second third fourth fifth last
half quarter double triple
""".split())


def _is_content(word: str) -> bool:
    return word.lower() not in _FUNCTION_WORDS


def _is_number(word: str) -> bool:
    return word.lower() in _NUMBER_WORDS


@dataclass(frozen=True)
class TermHint:
    """一条要注入 prompt 的术语。"""
    term: str
    desc: str


@dataclass(frozen=True)
class GlossOutcome:
    """离线直译的结果。"""
    text: str
    coverage: float          # 0~1，**实词**命中比例（功能词不计入分母，见 _FUNCTION_WORDS）
    unknown: tuple[str, ...]  # 所有没命中的词（含功能词，诊断用）
    used: bool               # 是否真的用了直译结果（覆盖率太低就返回原文）


def load_tsv(path: Path, allow_third: bool = False) -> dict[str, tuple[str, str]]:
    """读一张词典表，返回 {小写键: (说明, 短词形)}。

    容错优先：文件不在、某行没有制表符、键或值是空的，都只是少几条词，跳过就好——
    绝不能因为用户手改的一张文本表填错了一行，就让整个程序起不来。

    大小写不敏感（统一转小写当键），所以 "Thanks" 和 "thanks" 是同一个词，
    表里写一种就够了。
    """
    out: dict[str, tuple[str, str]] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("词典文件不存在：%s（这一区暂时为空，程序照常运行）", path)
        return out
    except OSError as exc:
        log.warning("读不了词典文件 %s：%s", path, exc)
        return out

    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            log.debug("%s 第 %d 行没有制表符，跳过：%r", path.name, lineno, line[:40])
            continue
        key = parts[0].strip().lower()
        desc = parts[1].strip()
        if not key or not desc:
            continue
        short = parts[2].strip() if allow_third and len(parts) > 2 else ""
        out[key] = (desc, short)
    return out


def _build_regex(keys: list[str]) -> re.Pattern[str] | None:
    """按最长优先编译一个匹配全部词条的正则。

    必须最长优先：否则 "force buy" 会先命中 "force"，
    "B site" 会被 "b" 抢先切掉。
    """
    if not keys:
        return None
    ordered = sorted(keys, key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in ordered)
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])", re.IGNORECASE)


class Dicts:
    """两区词典 + 最长匹配直译。"""

    def __init__(self, cfg: DictCfg) -> None:
        self.cfg = cfg
        terms_path = self._resolve(cfg.terms)
        common_path = self._resolve(cfg.common)

        # 第一区：允许第三列（短词形）
        self.terms = load_tsv(terms_path, allow_third=True)
        # 第二区：只有两列
        self.common = load_tsv(common_path, allow_third=False)

        # 离线直译的合并表。**第一区优先**：两区本不该重叠（自检里守着这条），
        # 万一用户手滑写重了，术语的释义比日常词的更准。
        #
        # 值可以是空串，这不是疏忽：空串表示"这个词是有意保留英文的"（AWP、NiKo、
        # Major）。它在表里留个位置，`gloss()` 才能把这个词标成"已处理"而不是
        # "没认出来"——否则一句里有三个枪械专名就会被判成覆盖率不够，整句退回英文，
        # 而"好 残局 from NiKo with the AWP"这种半中半英对看不懂英文的人反而更好用。
        self._gloss: dict[str, str] = {}
        # 第二区是两列表，「英文 <TAB> 中文」——中文就在第二列，它本身就是短词形。
        # （踩过：这里若照搬术语区的"取第三列"，日常词会一条都进不来，
        #   直译出来的覆盖率掉到 0，整句被退回英文，这一区等于白建。）
        for k, (desc, _) in self.common.items():
            if desc:
                self._gloss[k] = desc
        for k, (desc, short) in self.terms.items():
            # 有短词形就用它（拼进句子读得通）；没有就留空 = 保留英文原词。
            # **故意不拿说明去凑**——说明是给模型看的整句话，塞进直译结果里
            # 会变成"这波CS 里的狙击枪，保留 AWP 不译太秀了"。
            self._gloss[k] = short

        self._re = _build_regex(list(self._gloss))
        # 只在第一区里找要注入 prompt 的术语
        self._term_re = _build_regex(list(self.terms))
        log.info("词典已载入：术语 %d 条，日常词 %d 条（可用于离线直译 %d 条）",
                 len(self.terms), len(self.common), len(self._gloss))

    @staticmethod
    def _resolve(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    # ---------------------------------------------------------------- prompt 用
    def term_hints(self, text: str, limit: int = 6) -> list[TermHint]:
        """找出这句话里命中的游戏术语，准备注入 prompt。

        只查第一区。日常词不进 prompt（见模块开头）。
        按"长词优先、出现顺序"排，最多 limit 条——注太多会干扰模型。
        """
        if not text or self._term_re is None or limit <= 0:
            return []
        hits: list[TermHint] = []
        seen: set[str] = set()
        for m in self._term_re.finditer(text):
            k = m.group(0).lower()
            if k in seen:
                continue
            seen.add(k)
            desc = self.terms.get(k, ("", ""))[0]
            if desc:
                hits.append(TermHint(term=m.group(0), desc=desc))
            if len(hits) >= limit:
                break
        return hits

    # ---------------------------------------------------------------- 离线直译
    def gloss(self, text: str) -> GlossOutcome:
        """逐词直译。**只在 LLM 不可用时才用**（质量粗糙，但断网时能看懂大意）。

        规则：
          - 按最长优先扫描，命中的整条替换成中文；
          - 术语没有短词形的一律保留英文原词；
          - **绝不调整语序**——错乱的语序比字面语序更难读。
        """
        src = (text or "").strip()
        if not src:
            return GlossOutcome(text="", coverage=0.0, unknown=(), used=False)

        pieces: list[str] = []
        matched_spans: list[tuple[int, int]] = []
        pos = 0
        if self._re is not None:
            for m in self._re.finditer(src):
                # 先记范围再判断要不要替换：**保留英文的术语也算"处理过了"**，
                # 不然它会掉进 unknown 里压低覆盖率（见上面 _gloss 的注释）。
                matched_spans.append((m.start(), m.end()))
                zh = self._gloss.get(m.group(0).lower(), "")
                if not zh:
                    continue   # 有意保留英文原词：输出里那段不动，pos 也不动
                pieces.append(src[pos:m.start()])
                pieces.append(zh)
                pos = m.end()
        pieces.append(src[pos:])
        out = "".join(pieces)

        # 覆盖率：**实词**里有多少个落在词典的匹配范围里（为什么只数实词见 _FUNCTION_WORDS）
        total_all = total_content = handled_content = handled_all = 0
        unknown: list[str] = []
        for m in _WORD_TOKENS.finditer(src):
            s, e = m.span()
            word = m.group(0)
            if _is_number(word):
                # 数词不用译，直接记成"处理过了"（为什么见 _NUMBER_WORDS）。
                # 它不可能是"没认出来的词"，所以也不进 unknown。
                total_all += 1
                handled_all += 1
                total_content += 1
                handled_content += 1
                continue
            hit = any(s < me and ms < e for ms, me in matched_spans)
            total_all += 1
            if hit:
                handled_all += 1
            else:
                unknown.append(word)   # 诊断用：所有没命中的词都记着
            if not _is_content(word):
                continue
            total_content += 1
            if hit:
                handled_content += 1

        if total_content:
            coverage = handled_content / total_content
        elif total_all:
            # 整句都是功能词（"is it"、"and then"）——没有实词可判，退回按全部词算
            coverage = handled_all / total_all
        else:
            coverage = 1.0

        # 中英文之间补空格，中文紧邻处不加——纯粹为了看着不挤
        out = re.sub(r"(?<=[a-zA-Z0-9])(?=[一-鿿])", " ", out)
        out = re.sub(r"(?<=[一-鿿])(?=[a-zA-Z0-9])", " ", out)
        out = re.sub(r"\s{2,}", " ", out).strip()

        return GlossOutcome(text=out, coverage=coverage,
                            unknown=tuple(unknown), used=True)

    def stats(self) -> dict:
        return {"terms": len(self.terms), "common": len(self.common),
                "glossable": len(self._gloss)}
