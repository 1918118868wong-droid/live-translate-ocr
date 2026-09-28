"""离线自检：不联网、不动屏幕、不需要模型，几秒钟跑完。

状态机是**确定性**的（所有时间来自喂进去的 reading.t），所以"同一句字幕停 3 秒"
"字幕消失 2 秒"这类场景在这里是毫秒级跑完的纯数据驱动测试。这是本项目最重要的
测试资产——屏幕上的字幕千变万化，靠开直播碰运气是测不出来的。

用法：
    python tools\\selfcheck.py            # 跑全部
    python tools\\selfcheck.py 关键词      # 只跑名字里含关键词的
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from livetranslate_ocr.captions import CaptionTracker, key  # noqa: E402
from livetranslate_ocr.config import (  # noqa: E402
    PROJECT_ROOT,
    CaptionsCfg,
    Config,
    DictCfg,
    OverlayCfg,
)
from livetranslate_ocr.dicts import Dicts  # noqa: E402
from livetranslate_ocr.llm import LlmError  # noqa: E402
from livetranslate_ocr.ocr import OcrLine, Reading  # noqa: E402
from livetranslate_ocr.translator import is_degraded, source_label  # noqa: E402

DATA = PROJECT_ROOT / "data"


def real_dicts(**over) -> Dicts:
    """按项目里真实的两个词典文件建一个 Dicts（给词表那几条检查用）。"""
    cfg = DictCfg(terms=str(DATA / "terms.tsv"), common=str(DATA / "common.tsv"))
    for k, v in over.items():
        setattr(cfg, k, v)
    return Dicts(cfg)

CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------ 小工具
FRAME_S = 0.25   # 4Hz，和默认配置一致


def rd(t: float, *lines: str) -> Reading:
    """造一帧读数。lines 从上到下。"""
    objs = tuple(OcrLine(text=x, score=0.95, y=float(i * 50))
                 for i, x in enumerate(lines))
    return Reading(text="\n".join(lines), lines=objs, score=0.95, t=t)


def blank(t: float) -> Reading:
    """造一帧"什么都没读到"。"""
    return Reading(text="", lines=(), score=0.0, t=t)


def timeline(*steps) -> list[tuple[float, list[str]]]:
    """把 (重复次数, [行...]) 展开成逐帧的时间线，帧间隔 FRAME_S。

    测试里用"这条字幕停留 4 帧"来描述比手写 4 个时间戳清楚得多。
    """
    frames: list[tuple[float, list[str]]] = []
    t = 0.0
    for repeat, lines in steps:
        for _ in range(repeat):
            frames.append((t, lines))
            t += FRAME_S
    return frames


def feed(tracker: CaptionTracker, frames) -> list:
    """把时间线喂进去，收集所有发出的单元。"""
    out = []
    for t, lines in frames:
        out.extend(tracker.push(rd(t, *lines) if lines else blank(t)))
    return out


def texts(emissions) -> list[str]:
    return [e.text for e in emissions]


# ------------------------------------------------------------------ 状态机
@check("字幕机：逐词长出来的行只发一次")
def t_grow_once():
    tr = CaptionTracker(CaptionsCfg())
    # 前半句停两帧定型（但没标点，不该发），然后继续长，最后带句号收尾
    ems = feed(tr, timeline(
        (2, ["and he's"]),
        (2, ["and he's picked"]),
        (4, ["and he's picked up the AWP."]),
        (2, []),
    ))
    assert texts(ems) == ["and he's picked up the AWP."], texts(ems)
    assert ems[0].reason == "punct", ems[0].reason


@check("字幕机：单帧误读被滤掉")
def t_single_frame_misread():
    tr = CaptionTracker(CaptionsCfg())
    # 中间夹一帧认错了（niKo → n1ko），只出现一帧，接着又是正确的
    ems = feed(tr, timeline(
        (3, ["NiKo with the deagle"]),
        (1, ["n1ko with the deagle"]),
        (3, ["NiKo with the deagle."]),
        (2, []),
    ))
    joined = " ".join(texts(ems))
    assert "NiKo with the deagle." in joined, joined
    assert "n1ko" not in joined, f"单帧误读漏进去了: {joined}"
    assert len(ems) == 1, f"应只发一次，实际 {len(ems)}: {texts(ems)}"


@check("字幕机：整块消失时最后一行不能丢（上一行已先发掉）")
def t_last_line_survives_early_flush():
    """两行字幕，第一行自带句号先发掉了——第二行必须还能被收走。

    真踩过：_flush 里顺手把槽位 reset() 了，"最后一行"的内容和定型身份一起没了，
    跟着字幕整块消失，那一行就永远发不出来。**只在"上一行带句末标点先发掉"
    时才会发生**，看着就像"偶尔丢一句"，极难复现。
    """
    tr = CaptionTracker(CaptionsCfg())
    ems = feed(tr, timeline(
        (2, ["Alright chat, welcome back to the stream.",
             "we are gonna jump straight into ranked today"]),
        (7, []),
    ))
    got = texts(ems)
    assert "Alright chat, welcome back to the stream." in got, got
    assert any("jump straight into ranked today" in x for x in got), \
        f"上一行先发掉之后，第二行被丢掉了：{got}"


@check("字幕机：同一句话不会背靠背发两遍")
def t_no_back_to_back_duplicate():
    """发完之后屏幕上那几行还在，下一次读数不能把它们重新攒一遍再发。

    真踩过：发完就把槽位的"已发出"标记清掉，于是同一句话会在 250ms 后
    原样再发一遍。上下文里那句还挂着呢，屏幕上就打出了两遍同样的话。
    """
    tr = CaptionTracker(CaptionsCfg())
    ems = feed(tr, timeline(
        (3, ["the long angle", "waiting for the push."]),
        (4, []),
    ))
    got = texts(ems)
    for text in got:
        assert got.count(text) == 1, f"同一句发了两遍：{got}"
    assert len(got) == 1, f"这两行只该凑出一句，实际 {got}"


@check("字幕机：滚动时同一行不重复发")
def t_scroll_no_dup():
    tr = CaptionTracker(CaptionsCfg())
    # 典型的两行滚动：第一行满了往上滚，第二行接着长
    ems = feed(tr, timeline(
        (3, ["he's got the AWP", "and he's holding"]),
        (3, ["and he's holding", "the long angle"]),
        (3, ["the long angle", "waiting for the push."]),
        (3, []),
    ))
    got = texts(ems)
    # 每一段文字都只应出现一次
    for text in got:
        assert got.count(text) == 1, f"重复发出: {got}"
    joined = " ".join(got)
    assert "he's got the AWP" in joined, joined
    assert "and he's holding" in joined, joined


@check("字幕机：字幕消失满 idle_emit_s 才发")
def t_blank_emit():
    cfg = CaptionsCfg()
    tr = CaptionTracker(cfg)
    # 一句没有句末标点的话：只能靠"字幕消失"来收尾
    frames = timeline((3, ["no punctuation here"]))
    ems = feed(tr, frames)
    assert ems == [], f"字幕还在屏幕上时不该发: {texts(ems)}"

    # 消失但还不够久（idle_emit_s=1.0，这里只过 0.5s）→ 还不发
    t = frames[-1][0] + FRAME_S
    ems = tr.push(blank(t))
    ems += tr.push(blank(t + 0.5))
    assert ems == [], f"消失时间不够就发了: {texts(ems)}"

    # 再等一会儿，累计超过 1 秒 → 发
    ems = tr.push(blank(t + 1.25))
    assert texts(ems) == ["no punctuation here"], texts(ems)
    assert ems[0].reason == "blank", ems[0].reason


@check("字幕机：换句时把上一句发出来（哪怕没标点）")
def t_replaced_emits_previous():
    cfg = CaptionsCfg()
    tr = CaptionTracker(cfg)
    ems = feed(tr, timeline(
        (3, ["the first sentence"]),          # 定型，无标点，先不发
        (4, ["a completely different one."]),  # 换成完全不同的句子
    ))
    assert "the first sentence" in texts(ems), f"上一句被丢了: {texts(ems)}"


@check("字幕机：一直长不稳定的行也有 max_wait_s 兜底")
def t_maxwait():
    cfg = CaptionsCfg(max_wait_s=1.0)
    tr = CaptionTracker(cfg)
    # 每帧都变长一点点，永远达不到"连续两次一样"
    frames = []
    t = 0.0
    words = "one two three four five six seven eight".split()
    for i in range(1, len(words) + 1):
        for _ in range(2):   # 每帧变一次，但帧间隔要够密才达不到 stable_reads
            frames.append((t, [" ".join(words[:i])]))
            t += 0.1
    ems = feed(tr, frames)
    assert ems, "一直在长、又超了 max_wait_s，应该兜底发一次，不能一句都不发"
    assert all(e.reason in ("maxwait", "maxhold", "punct", "replaced")
               for e in ems), [e.reason for e in ems]


@check("字幕机：缓冲超长会把这一刀切下去（不等句末标点）")
def t_maxlen():
    cfg = CaptionsCfg(max_chars=40)
    tr = CaptionTracker(cfg)
    # 上面那行已经够长了、又没有句末标点：不能再攒着等标点，要切开先发
    long_line = "word " * 12   # 60 字符 > max_chars=40
    ems = feed(tr, timeline((3, [long_line.strip(), "short tail"]), (5, [])))
    assert ems, "缓冲超过 max_chars 就该切开，不能一直攒着"
    assert any(e.reason == "maxlen" for e in ems), \
        f"应该由 maxlen 触发，实际 {[e.reason for e in ems]}"
    joined = " ".join(texts(ems))
    # 那一行是 12 个 "word"。切开发出去时不能把它重复发两遍（这是最典型的翻车方式）
    assert joined.count("word") == 12, f"内容被重复发了: {joined!r}"


@check("字幕机：垃圾帧永不发出")
def t_junk_never_emits():
    tr = CaptionTracker(CaptionsCfg())
    ems = feed(tr, timeline(
        (3, ["a"]),        # 太短（min_chars=2）
        (3, ["."]),
        (3, ["12"]),       # 纯数字，短
        (5, []),
    ))
    assert ems == [], f"垃圾帧不该发出任何东西: {texts(ems)}"


@check("字幕机：两行变一行、再变两行不会串味")
def t_line_count_changes():
    tr = CaptionTracker(CaptionsCfg())
    ems = feed(tr, timeline(
        (4, ["first line here.", "second one"]),
        (3, ["second one keeps going."]),
        (3, ["a brand new sentence.", "and its second line"]),
        (4, []),
    ))
    got = " ".join(texts(ems))
    assert "first line here." in got, got
    for text in texts(ems):
        assert texts(ems).count(text) == 1, f"重复: {texts(ems)}"


@check("字幕机：同一个键不会被发两次（比分板那种静态文本）")
def t_static_hud_dedup():
    tr = CaptionTracker(CaptionsCfg())
    ems = feed(tr, timeline(
        (4, ["12 - 9"]),          # 比分板被框进来了
        (4, ["12 - 9"]),
        (4, ["12 - 9"]),
    ))
    assert len(texts(ems)) <= 1, f"静态文本重复发了: {texts(ems)}"


# ------------------------------------------------------------------ 词表
@check("词表：第一区是游戏术语且不含日常词")
def t_terms_red_line():
    from livetranslate_ocr.config import PROJECT_ROOT
    path = PROJECT_ROOT / "data" / "terms.tsv"
    assert path.is_file(), f"缺文件 {path}"
    terms = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            k, v = line.split("\t", 1)
            terms[k.strip().lower()] = v.strip()
    assert len(terms) > 100, f"术语太少了：{len(terms)}"
    # 这条红线是踩过坑的：说明会被原样注入 prompt 并要求"严格照用"，
    # 日常英语里也常见的词一旦收进来，正常句子会被翻译得莫名其妙。
    BANNED = {"big", "device", "spirit", "liquid", "heroic", "half", "default",
              "throw", "he", "long", "short", "window", "save", "hold", "series",
              "split", "fake", "support", "force", "bonus", "scope", "execute"}
    hit = sorted(BANNED & set(terms))
    assert not hit, f"第一区混进了日常词（会污染正常句子）: {hit}"
    # 反向确认：真正的术语得在
    for must in ("awp", "clutch", "eco round", "force buy", "b site"):
        assert must in terms, f"术语表缺了 {must!r}"


@check("词表：第二区覆盖日常高频词")
def t_common_dict():
    from livetranslate_ocr.config import PROJECT_ROOT
    path = PROJECT_ROOT / "data" / "common.tsv"
    assert path.is_file(), f"缺文件 {path}"
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "\t" in line:
            rows += 1
    assert rows > 200, f"第二区太少了：{rows} 行"


@check("词表：两区不重叠（同一个词不会既是术语又是日常词）")
def t_dicts_disjoint():
    from livetranslate_ocr.config import PROJECT_ROOT

    def load(name):
        out = set()
        for line in (PROJECT_ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "\t" in line:
                out.add(line.split("\t", 1)[0].strip().lower())
        return out

    overlap = sorted(load("terms.tsv") & load("common.tsv"))
    assert not overlap, f"两区职责重叠了（同一词在两处）: {overlap}"


@check("词表：离线直译真的把两区都用上了")
def t_gloss_uses_both_zones():
    """守着"第二区整区失效"这类 bug——它不会报错，只是直译悄悄退化成没词可用。

    第二区是两列表（英文<TAB>中文），中文在**第二列**；术语区是三列表，短词形在
    第三列。取错列的话日常词一条都进不来，覆盖率掉到 0、整句退回英文，
    表面上"程序正常"，实际上这一区白建了（真踩过）。
    """
    d = real_dicts()
    st = d.stats()
    assert st["glossable"] >= 450, \
        f"可直译词条只有 {st['glossable']} 条，像是少合并了一区（{st}）"
    # 整句都是日常词，也必须能直译出来
    out = d.gloss("thanks for watching")
    assert out.text == "感谢观看", f"日常短语没直译出来: {out.text!r}"
    assert out.coverage == 1.0, out.coverage


@check("词表：直译按最长优先，不把短语拆开")
def t_gloss_longest_first():
    d = real_dicts()
    # "force buy" 整条命中；被拆成 force + buy 的话会看到"力量 买"
    assert "强起" in d.gloss("they force buy this round").text
    # "B site" 不能被单字母 "b" 抢走前缀
    assert "B 包点" in d.gloss("he is going to B site alone").text
    # 多词日常短语同理："let me know" 不该被拆成 let + me + know
    assert d.gloss("let me know").text == "跟我说一声"
    # "eco round" 不能被较短的 "eco" 抢走（两条都是术语，长的应该赢）
    assert d.gloss("eco round").text == "经济局", d.gloss("eco round").text


@check("词表：没有短词形的术语保留英文，且不算「没认出来」")
def t_gloss_keeps_english():
    d = real_dicts()
    out = d.gloss("NiKo with the AWP gets the clutch")
    assert "AWP" in out.text and "NiKo" in out.text, \
        f"枪械/选手名被译掉了（本该保留英文）: {out.text!r}"
    assert "残局" in out.text, out.text
    # 关键：保留英文的术语不能掉进 unknown，否则它会拖低覆盖率把整句退回英文
    assert "awp" not in [w.lower() for w in out.unknown], \
        f"保留英文的术语被算成没认出来了: {out.unknown}"


@check("词表：术语区说明绝不进直译结果（那是给模型看的整句话）")
def t_gloss_no_prose():
    d = real_dicts()
    for text in ("he picked up the AWP", "nice eco round", "that is a force buy"):
        got = d.gloss(text).text
        for prose in ("里的", "保留", "不译", "CS 里", "（"):
            assert prose not in got, f"说明被塞进译文了: {got!r}"


@check("词表：日常词不进 prompt（术语命中里不能有第二区的词）")
def t_hints_terms_only():
    d = real_dicts()
    common = set(d.common)
    # 模型本来就认识这些词，注进 prompt 只会稀释术语权重
    for phrase in ("thanks for watching", "good game", "hello everyone",
                   "let me know", "the weather is nice today"):
        leaked = [h.term for h in d.term_hints(phrase) if h.term.lower() in common]
        assert not leaked, f"日常词漏进了 prompt 术语表: {leaked}"
    # 反向确认：真术语还是要能注入
    hints = [h.term.lower() for h in d.term_hints("he gets the clutch with the AWP")]
    assert "clutch" in hints, f"真术语没注入: {hints}"


@check("词表：覆盖率太低就退回英文原句（半中半英还不如不译）")
def t_gloss_coverage_gate():
    cfg = Config().dictionaries
    d = real_dicts()
    # 一句跟游戏/日常词表都不沾边的话，覆盖率必然很低
    out = d.gloss("quantum entanglement violates bell inequalities")
    assert out.coverage < cfg.min_coverage, \
        f"这种句子不该有 {out.coverage:.0%} 的覆盖率（阈值 {cfg.min_coverage}）"
    # 而词典能覆盖的句子要过线，否则离线时什么都译不出来
    out = d.gloss("he is on an eco round and they force buy next")
    assert out.coverage >= cfg.min_coverage, \
        f"术语密集的句子覆盖率只有 {out.coverage:.0%}，离线时会被误判成不该直译"


@check("词表：覆盖率只数实词，方向不能反")
def t_coverage_counts_content_words():
    """这条守着"判据判反了"——比阈值高低重要得多。

    真踩过：分母里混进冠词/介词/be 动词之后，
      "NiKo gets the clutch"   实词只剩 gets 没译  →  算 50%，被退回英文（错）
      "the weather is nice…"   实词全都没译        →  算 56%，放行成怪话（错）
    两句话的判定正好和该有的结果反着。
    """
    gate = Config().dictionaries.min_coverage
    d = real_dicts()
    out = d.gloss("NiKo gets the clutch")
    assert out.coverage >= gate, \
        f"实词基本都译出来了（{out.text!r}），覆盖率却只有 {out.coverage:.0%}"
    out = d.gloss("the weather is nice today and I like it")
    assert out.coverage < gate, \
        f"实词全没译出来（{out.text!r}），覆盖率却有 {out.coverage:.0%}，会吐怪话"


@check("词表：数字不算「没认出来」，不拖累覆盖率")
def t_numbers_are_neutral():
    """数词不需要译，中文读者看 "one versus three"、"1v3" 直接就能懂。

    但它们是**实词**，会进分母。不特殊处理的话，一句本来译得出来的解说词
    会因为夹了几个数字就整句退回英文——而数字在解说里出现得极密。
    实测：这句原本算 33%，整句退回英文，把译得出来的 clutch 一起丢了。
    """
    gate = Config().dictionaries.min_coverage
    d = real_dicts()
    out = d.gloss("NiKo gets the clutch, one versus three.")
    assert out.coverage >= gate, \
        f"只有数字没被译（{out.text!r}），覆盖率却只有 {out.coverage:.0%}，整句被退回英文"
    # 反过来：数字不能把一句真正没译的话顶上阈值
    out = d.gloss("one two three four five")
    assert out.coverage >= gate, \
        f"纯数字串本来就该放行（{out.text!r}），覆盖率只有 {out.coverage:.0%}"


@check("词表：1v3 这种写法不会切出孤零零的字母")
def t_digit_glued_letters():
    """`1v3` 里的 v 是个查不到、也不该查的孤字。

    切出来的话它会算进实词分母，白拉低覆盖率——而 1v3、2v4 在解说里很常见。
    """
    d = real_dicts()
    out = d.gloss("it is a 1v3 now")
    assert "v" not in out.unknown, \
        f"1v3 被切出了一个孤字（没认出来的词：{out.unknown}），会白拉低覆盖率"


# ------------------------------------------------------------------ 翻译降级
class _FakeClient:
    """替掉真实 LLM：要么按剧本吐增量，要么抛错。绝不联网。"""

    def __init__(self, deltas=("他", "拿到残局了"), error=None):
        self.deltas = deltas
        self.error = error
        self.calls = 0

    def stream_chat(self, *a, **kw):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return iter(self.deltas)

    def chat(self, *a, **kw):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "pong"

    def close(self):
        pass


@contextmanager
def make_translator(client=None, cfg=None):
    """临时记忆库 + 假 LLM 的 Translator。client 为 None 时保留真实的 _ensure_client。"""
    import tempfile as _tmp

    from livetranslate_ocr.tm import TranslationMemory
    from livetranslate_ocr.translator import Translator

    cfg = cfg or Config()
    with _tmp.TemporaryDirectory() as d:
        cfg.tm.db = str(Path(d) / "tm.sqlite3")
        tm = TranslationMemory(cfg.tm)
        tr = Translator(cfg, tm, real_dicts())
        if client is not None:
            tr._ensure_client = lambda: client
        try:
            yield tr
        finally:
            tr.close()
            tm.close()


@check("降级：LLM 挂了就落到本地词典，并在 detail 里说明原因")
def t_degrade_to_local():
    with make_translator(_FakeClient(error=LlmError("connection refused"))) as tr:
        out = tr.translate("NiKo gets the clutch")
        assert out.source == "local", out.source
        assert "残局" in out.text, out.text
        assert out.degraded and out.label == "离线词典"
        # 降级绝不能静默：老项目最惨的教训就是屏幕上一点线索都没有
        assert out.detail, "降级了却没给原因"


@check("降级：连续失败够次数后就不再每句都去撞墙")
def t_breaker_stops_calling():
    cfg = Config()
    cfg.offline.fail_threshold = 2
    client = _FakeClient(error=LlmError("timeout"))
    with make_translator(client, cfg) as tr:
        tr.translate("NiKo gets the clutch")
        assert client.calls == 1
        tr.translate("s1mple hits the flick")
        assert client.calls == 2
        # 已经跳闸：后面几句不该再发起请求，否则每句都要白等一个超时
        for i in range(4):
            out = tr.translate(f"offline round number {i} now")
            assert out.source.startswith("local"), out.source
        assert client.calls == 2, f"断路器没起作用，又多调了 {client.calls - 2} 次"
        h = tr.health()
        assert h["offline"] is True and h["retry_in_s"] > 0, h


@check("降级：网络恢复后自动切回模型，不用重启")
def t_breaker_recovers():
    cfg = Config()
    cfg.offline.fail_threshold = 1
    cfg.offline.retry_interval_s = 0.0   # 下一句就能试探重试
    client = _FakeClient(error=LlmError("boom"))
    with make_translator(client, cfg) as tr:
        assert tr.translate("NiKo gets the clutch").source == "local"
        assert tr.health()["offline"] is True
        client.error = None              # 网好了
        out = tr.translate("s1mple hits the flick")
        assert out.source == "llm", out.source
        assert tr.health()["offline"] is False
        assert tr.health()["fail_streak"] == 0


@check("降级：本地词典的结果绝不写进记忆库（防缓存投毒）")
def t_local_never_cached():
    """直译质量粗糙，一旦写进缓存，网络恢复后同一句会被这条烂译文顶掉。
    这个 bug 很难查——表现是"网明明好了，这句还是怪怪的"。"""
    with make_translator(_FakeClient(error=LlmError("boom"))) as tr:
        out = tr.translate("NiKo gets the clutch")
        assert out.source == "local", out.source
        assert tr.tm.lookup("NiKo gets the clutch", "en", tr.variant) is None, \
            "直译结果被写进记忆库了"


@check("降级：模型翻译成功才写缓存，且第二次直接命中")
def t_llm_caches():
    client = _FakeClient(deltas=("NiKo", "拿下了这个残局"))
    with make_translator(client) as tr:
        out = tr.translate("NiKo gets the clutch")
        assert out.source == "llm", out.source
        assert out.text == "NiKo拿下了这个残局", out.text
        hit = tr.tm.lookup("NiKo gets the clutch", "en", tr.variant)
        assert hit is not None, "成功的译文没写进记忆库"
        n = client.calls
        out2 = tr.translate("NiKo gets the clutch")
        assert out2.source == "tm-exact", out2.source
        assert client.calls == n, "命中记忆库了还去调模型"
        assert out2.text == out.text


@check("降级：没配 API Key 也算正常可用（直接走词典，不是崩溃）")
def t_no_api_key():
    cfg = Config()
    cfg.llm.api_key = ""
    cfg.llm.api_key_env = "LIVETRANSLATE_OCR_SELFCHECK_NO_SUCH_KEY"
    with make_translator(client=None, cfg=cfg) as tr:   # 用真实的 _ensure_client
        out = tr.translate("NiKo gets the clutch")
        assert out.source == "local", out.source
        assert "API Key" in out.detail, out.detail
        h = tr.health()
        assert h["permanent"] is True, "没配 Key 应该钉死在离线态，不该每 30 秒撞一次墙"
        # 而且不该抛异常——字幕链路不能因此断掉
        assert tr.translate("another round please").source.startswith("local")


@check("降级：词典覆盖不了就显示英文原文，不吐半中半英的怪话")
def t_low_coverage_shows_original():
    cfg = Config()
    cfg.dictionaries.min_coverage = 0.95    # 拉高门槛，逼出 local-raw
    with make_translator(_FakeClient(error=LlmError("boom")), cfg) as tr:
        text = "quantum entanglement violates bell"
        out = tr.translate(text)
        assert out.source == "local-raw", f"{out.source}（覆盖 {out.coverage:.0%}）"
        assert out.text == text, f"低覆盖时应该原样退回英文: {out.text!r}"
        assert out.label == "离线·原文" and out.degraded


@check("降级：每种来源都有中文标签，降级看得见")
def t_source_labels():
    for s in ("tm-exact", "tm-fuzzy", "llm", "llm-partial",
              "local", "local-raw", "error"):
        lab = source_label(s)
        assert lab and lab != s, f"{s} 没有中文标签，用户看不懂"
    assert not is_degraded("llm") and not is_degraded("tm-exact")
    assert not is_degraded("tm-fuzzy") and not is_degraded("llm-partial")
    for s in ("local", "local-raw", "error"):
        assert is_degraded(s), f"{s} 是降级，UI 该亮出来"


@check("翻译：术语进了 prompt（连说明一起），并说明了输入是 OCR 来的")
def t_prompt_contents():
    with make_translator(_FakeClient()) as tr:
        msgs = tr._build_messages("NiKo gets the clutch with the AWP", "en", [])
        system, user = msgs[0]["content"], msgs[1]["content"]
        assert "[术语表]" in user and "clutch" in user, user
        assert "残局" in user, f"术语说明没带上，模型不知道照用什么译法: {user}"
        assert "[本次待翻译]" in user
        # 这两条是 OCR 输入路径特有的，必须写进 prompt
        assert "文字识别" in system, "没告诉模型输入是屏幕文字识别来的"
        assert "不要因此补上猜测的结尾" in system, "没提示字幕常常没有句末标点"


# ------------------------------------------------------------------ 依赖与启动脚本
@check("依赖：装的是 headless 版 opencv（不能带 Qt）")
def t_no_qt_opencv():
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        distribution("opencv-python")
    except PackageNotFoundError:
        return   # 期望路径
    raise AssertionError(
        "装了带 Qt 的 opencv-python，会和 PySide6 抢 Qt 插件（症状是崩溃或界面错乱）。"
        "卸掉它、改用 opencv-python-headless 后重跑 setup.bat"
    )


@check("依赖：ocr/captions 能在不导入 Qt 的情况下加载")
def t_headless_import():
    # 这两个模块会被自检和 --text 模式用到，不该依赖 GUI
    import livetranslate_ocr.captions  # noqa: F401
    import livetranslate_ocr.ocr  # noqa: F401
    assert "PySide6" not in sys.modules, "导入 OCR 链路时把 Qt 也拉进来了"


@check("配置：默认值就是能跑的值（region 除外）")
def t_config_defaults():
    cfg = Config()
    assert cfg.capture.fps > 0
    assert cfg.captions.stable_reads >= 1
    assert cfg.translate.src_lang == "en", "屏幕字幕固定是英文，不该是 auto"
    assert cfg.region() is None, "默认没框选过，region() 应返回 None"
    # 框选过之后要能解析出四元组
    cfg.capture.region = [10, 20, 300, 40]
    assert cfg.region() == (10, 20, 300, 40)


@check("配置：save_region 只改 region 那一行，注释原样保留")
def t_save_region_keeps_comments():
    import tempfile

    from livetranslate_ocr.config import Config as C
    from livetranslate_ocr.config import save_region

    text = ("# 顶上的注释\n[capture]\n# region 上面还有一行注释\n"
            "region = [0, 0, 0, 0]\nfps = 4.0\n\n[llm]\nmodel = \"deepseek-chat\"\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.toml"
        p.write_text(text, encoding="utf-8")
        cfg = C()
        cfg.source_path = p
        save_region(cfg, (100, 200, 1600, 90))
        after = p.read_text(encoding="utf-8")
        assert "# 顶上的注释" in after, "注释被吃掉了"
        assert "# region 上面还有一行注释" in after, "注释被吃掉了"
        assert "region = [100, 200, 1600, 90]" in after, after
        assert 'model = "deepseek-chat"' in after
        assert "fps = 4.0" in after
        # 再解析一遍，确认真的生效
        again = C()
        again.source_path = p
        from livetranslate_ocr import config as cfgmod
        cfgmod._apply(again, __import__("tomllib").loads(after))
        assert again.region() == (100, 200, 1600, 90)


@check("启动脚本：.bat 必须是纯 ASCII + CRLF")
def t_bat_files():
    bad = []
    for bat in sorted(PROJECT_ROOT.glob("*.bat")):
        data = bat.read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError as e:
            bad.append(f"{bat.name}: 含非 ASCII 字符（{e}）—— 中文文案要放进 Python 里打印")
            continue
        # 有 LF 但不是 CRLF（也就是出现了裸 LF）就是错的
        lone_lf = data.replace(b"\r\n", b"").count(b"\n")
        if lone_lf:
            bad.append(f"{bat.name}: 有 {lone_lf} 个裸 LF —— cmd.exe 会吃掉行首字符，必须全 CRLF")
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 字幕窗位置
# 这一组要真造一个 QApplication，所以走 offscreen 后端：不弹窗、不碰用户的屏幕。
# 守的是计划里点名的那个最坏情况——**字幕窗盖住采集区域**，等于自己吃自己：
# 屏幕上摆着自己要读的那行字，OCR 读回来的全是自己的译文，输出就此卡死，
# 而用户看到的现象只是"用着用着没字幕了"，极难联想到是窗口摆错了地方。
_QT_APP: list = []


def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if not _QT_APP:
        from PySide6.QtWidgets import QApplication

        _QT_APP.append(QApplication.instance() or QApplication([]))
    return _QT_APP[0]


def overlay_for(region):
    from livetranslate_ocr.overlay import SubtitleOverlay

    ov = SubtitleOverlay(OverlayCfg())
    # 先塞一句译文：窗口高度是按内容算出来的，空窗口验不出"上面塞不下"。
    ov.set_translation("这是一句用来把字幕窗撑起来的中文译文，长度接近真实的字幕。")
    ov.set_region(region)
    return ov


@contextmanager
def catch_logs():
    """收下这段期间打出来的日志，用来断言"该吼的那一声吼了"。"""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _H(level=logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(h)
    try:
        yield records
    finally:
        root.removeHandler(h)


@check("字幕窗：采集区域在底部时，字幕窗贴它上方，不重叠")
def t_overlay_above_region():
    geo = qt_app().primaryScreen().availableGeometry()
    ry = geo.y() + geo.height() - 180          # 底部那条字幕带
    ov = overlay_for((100, ry, 800, 120))
    assert not ov.overlaps_region(), "字幕窗压在采集区域上了——等于自己吃自己"
    bottom = ov.y() + ov.height()
    assert bottom <= int(ry / ov._dpr()), (
        f"字幕窗底边 {bottom} 顶到了采集区域上边 {int(ry / ov._dpr())}")


@check("字幕窗：采集区域贴顶时，改贴到它下方（绝不盖住）")
def t_overlay_below_when_region_at_top():
    geo = qt_app().primaryScreen().availableGeometry()
    ov = overlay_for((100, geo.y(), 800, 120))
    assert not ov.overlaps_region(), "贴顶的采集区域被字幕窗盖住了"
    assert ov.y() >= int((geo.y() + 120) / ov._dpr()), "应该改贴到区域下方"


@check("字幕窗：区域几乎占满屏幕时退到贴顶，而且必须吼一声")
def t_overlay_degenerate_fullscreen_region():
    geo = qt_app().primaryScreen().availableGeometry()
    with catch_logs() as recs:
        ov = overlay_for((geo.x(), geo.y(), geo.width(), geo.height()))
    assert ov.y() == geo.y(), "上面下面都放不下时，应该退到贴屏幕顶"
    assert any("几乎占满屏幕" in r for r in recs), \
        "退化成贴顶是「可能挡住采集区域」的情况，必须留下日志，不能不出声"


@check("字幕窗：用户拖过之后，新字幕不会把它弹回锚点")
def t_overlay_user_position_sticks():
    from PySide6.QtCore import QPoint

    geo = qt_app().primaryScreen().availableGeometry()
    ov = overlay_for((100, geo.y(), 800, 120))
    ov._user_position = QPoint(77, 99)
    ov.set_translation("又来了新的一句，这回会触发一次重新布局。")
    assert (ov.x(), ov.y()) == (77, 99), f"位置被自作主张改成了 {(ov.x(), ov.y())}"


# ------------------------------------------------------------------ main
def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    passed = failed = skipped = 0
    failures = []

    for name, fn in CHECKS:
        if only and only not in name:
            continue
        try:
            fn()
        except AssertionError as e:
            failed += 1
            failures.append((name, str(e)))
            print(f"✗ {name}")
            print(f"    {e}")
        except Exception as e:  # 意料之外的错误也要报出来，别吞
            failed += 1
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"✗ {name}")
            print(f"    {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"✓ {name}")

    print(f"\n{passed} 项通过" + (f"，{failed} 项失败" if failed else "") +
          (f"，{skipped} 项跳过" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
