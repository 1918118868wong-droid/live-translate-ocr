"""字幕状态机：把 4Hz 的重复读数，变成"一句话，发一次"。

这是本项目真正新写的核心，也最容易做砸。要处理的现实：

- YouTube 自动字幕是**逐词长出来的**——最后一行不断变长，长满了才上滚。
- 同一句字幕在屏幕上停留 1~3 秒，我们每秒读 4 次 → 同一行会被读到十几次。
- 字幕会**滚动**（第一行变第二行）、会**消失**（换气/静音）、偶尔会**读错**一两个字。

天真的做法（每次读到不一样的文本就翻译一遍）会让同一句话被翻译十几次，
而且每次措辞可能不同，屏幕上一顿乱闪。所以要一个状态机。

## 三个判据

**1. 稳定判据代替多数投票。** 一行连续 `stable_reads` 次读到一样，才认定"定型"。
单帧的偶发误读不可能连续重复两次，所以这一步几乎能滤掉全部 OCR 抖动，
比攒一缓冲区再投票简单得多、也够用。

**2. 靠"前缀延伸"区分"在长"和"换句"。** 这是最容易写错的地方：
自动字幕长出新词时，新文本一定是旧文本的**前缀延伸**
（"and he's picked" → "and he's picked up the AWP"）。
而换下一句时，新文本几乎不可能恰好是旧文本的前缀。
所以判断同不同一条，主要看前缀关系，相似度只在两边一样长时用来容忍 OCR 抖动。
只靠相似度会出事——"他在B点捡起了AWP" 和 "他在B点捡起了沙鹰" 相似度极高但是两句。

**3. 发出前先在缓冲里攒。** 定型的行进缓冲，遇到下面任一情况就整块发出去翻译一次：
句末标点 / 超过 `max_chars` / 字幕整块消失 / 最后一行被换掉 / 定型后干等超过 `max_hold_s`。
这样"一句话"对应"一次翻译调用"，同一个句子的前缀不会被反复翻译。

## 时间从哪来

**所有时间只认 `reading.t`，这个模块内部绝不调用 time.monotonic()。**
这是自检能确定性地测它的前提：喂一串人造读数就能把"3 秒的同一条字幕"
在毫秒内跑完，不需要真的等 3 秒、也不需要屏幕。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from .config import CaptionsCfg
from .ocr import Reading

log = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return _fuzz.ratio(a, b)
except Exception:  # pragma: no cover - rapidfuzz 是正式依赖，这里只是兜底
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0


# 句末标点。命中这些就认为"这句话说完了"，可以立刻发出去。
# 引号/括号先剥掉再判断，因为字幕常以 ." 或 !) 结尾。
_SENTENCE_END = ".!?…。！？"
_TRAILING = "\"'”’)]}）】」』"


@dataclass(frozen=True)
class Emission:
    """一个翻译单元：要送去翻译的一段英文。"""
    text: str
    reason: str          # punct / replaced / blank / maxlen / maxhold / maxwait
    partial: bool        # True = 不是停在句末标点上（后面可能还有话）
    votes: int           # 定型时连续读到了几次

    def describe(self) -> str:
        tag = {"punct": "句末", "replaced": "换句", "blank": "消失",
               "maxlen": "超长", "maxhold": "停顿", "maxwait": "超时"}.get(
                   self.reason, self.reason)
        return f"{tag}{'·半句' if self.partial else ''}"


def display(text: str) -> str:
    """把 OCR 文本收拾成能给人看、也能送去翻译的样子。

    只做无损整理：合并空白、弯引号改直引号。**不动大小写**——
    "AWP"、"NiKo" 的大小写对翻译质量有影响，术语表也是按原样匹配的。
    """
    text = (text or "").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def key(text: str) -> str:
    """比对用的键：只留字母数字。

    标点、大小写、空格都不参与比较——OCR 把逗号读成句号、把句首大写读丢，
    都不该被当成"换了一句话"。
    """
    return "".join(ch for ch in display(text).lower() if ch.isalnum())


def _ends_sentence(text: str) -> bool:
    stripped = display(text).rstrip(_TRAILING)
    return bool(stripped) and stripped[-1] in _SENTENCE_END


def _same_caption(prev_key: str, k: str, jitter: int) -> bool:
    """两个读数是不是"同一条字幕"（而不是换了一句）。

    判据按可靠性排序：
      1. 前缀关系 —— 在长（"and he" → "and he is gone"）或读短了，都是同一条；
      2. 长度接近且高度相似 —— 同样的内容，OCR 抖了一两个字。
    长度差得远又没有前缀关系的，直接判为换句（"…AWP" vs "…沙鹰" 那种）。
    """
    if not prev_key or not k:
        return False
    if k.startswith(prev_key) or prev_key.startswith(k):
        return True
    if abs(len(k) - len(prev_key)) <= 3 and _ratio(prev_key, k) >= jitter:
        return True
    return False


@dataclass
class _Slot:
    """一个位置上的读数跟踪。槽位就是下标：0 = 最上面那行。

    上面那些行（不是最后一行）滚上去之后就不会再变了，所以它们定型得很快；
    最后一行还在长，得等它停下来。
    """
    text: str = ""
    count: int = 0            # 当前这条文本连续读到几次
    settled_at: float = 0.0   # 当前这条文本第一次达到定型的时刻（算 max_hold_s 用）
    first_seen_at: float = 0.0  # 这一簇（这句）最早出现的时刻（算 max_wait_s 用）
    last_settled: str = ""    # 这一簇里最后一次确认过的文本
    emitted: bool = False     # 这一簇的内容是否已经进过缓冲

    def reset(self) -> None:
        """整簇作废。**只在"这个位置换成别的话了"时用**（见 _update_slot）。"""
        self.text = ""
        self.count = 0
        self.settled_at = 0.0
        self.first_seen_at = 0.0
        self.last_settled = ""
        self.emitted = False

    def mark_sent(self, t: float) -> None:
        """缓冲发出去了。**只给已经发出去的槽位把 max_wait 的钟重新起算。**

        这里既不能清 emitted、更不能 reset：

          - reset()（改之前的写法）把 slot.text 一并抹掉，于是"最后一行"在
            下一次读数里被当成刚冒出来的新行重新数起，first_seen_at 归零
            还会让 max_wait_s 判据立刻误触发，它"已经定型过"的身份就此丢失。
            表现是**整行内容凭空消失**，而且只在"刚发完一句、字幕紧接着就消失"
            时出现，极难复现。实测踩到：两行字幕只发出第一行，第二行永远丢掉。
          - 清 emitted 也不行：那几行还挂在屏幕上，下一次读数会被重新攒一遍，
            拼出同一句话再发一次。实测：整句背靠背连发两遍。

        emitted 是 _track / _harvest 在**真的把这一行放进缓冲**的那一刻置位的。
        所以没轮到发的槽位（比如刚发完上一句时还挂在屏上的最后一行）会保留
        emitted=False，之后照样能被 _harvest 收走——第二行就是靠这条活下来的。
        """
        if self.emitted:
            self.first_seen_at = t


class CaptionTracker:
    """把读数流变成翻译单元。用法：`for em in tracker.push(reading): ...`"""

    def __init__(self, cfg: CaptionsCfg) -> None:
        self.cfg = cfg
        self._slots: list[_Slot] = []
        self._buffer: list[str] = []                       # 已定型、等着一起发出去的行
        self._recent_keys: deque[str] = deque(maxlen=12)   # 发过的，用来去重
        self._last_text_t: float = 0.0                     # 最后一次读到非空内容的时刻
        self._last_reading: str = ""                       # 最近读到的原文（给字幕窗显示）
        self.stats = {"frames": 0, "blank_frames": 0, "emitted": 0,
                      "dropped_dupes": 0, "max_lines": 0}

    # ---------------------------------------------------------------- 对外
    def push(self, reading: Reading) -> list[Emission]:
        """喂一帧读数，返回这一帧要发出的翻译单元（通常是 0 个或 1 个）。"""
        self.stats["frames"] += 1

        if reading.empty:
            self.stats["blank_frames"] += 1
            # 短暂的空读数（OCR 偶尔漏一帧）不算"字幕消失"，要连续消失够久才算
            if (reading.t - self._last_text_t) >= self.cfg.idle_emit_s:
                return self._harvest("blank", reading.t)
            return []

        lines = [ln for ln in (display(line.text) for line in reading.lines) if ln]
        if not lines:
            return []

        self._last_text_t = reading.t
        self._last_reading = "\n".join(lines)
        self.stats["max_lines"] = max(self.stats["max_lines"], len(lines))
        return self._track(lines, reading.t)

    def display_text(self) -> str:
        """当前屏幕上的英文（给字幕窗的原文行）。

        优先显示"刚刚读到的"，读不到就显示"还在缓冲里等着发的"——
        这样字幕刚好消失的那一瞬间，屏幕上不会突然空掉。
        """
        return self._last_reading or " ".join(self._buffer)

    def reset(self) -> None:
        """丢掉全部状态。重新框选区域、或暂停后恢复时调。"""
        self._slots.clear()
        self._buffer.clear()
        self._recent_keys.clear()

    # ---------------------------------------------------------------- 内部
    def _track(self, lines: list[str], t: float) -> list[Emission]:
        """更新各槽位的定型状态，并在该发的时候发。"""
        out: list[Emission] = []

        # 槽位增减：多出来的是新出现的一行，少掉的是滚走了
        while len(self._slots) < len(lines):
            slot = _Slot()
            slot.first_seen_at = t
            self._slots.append(slot)
        while len(self._slots) > len(lines):
            self._slots.pop()

        last_index = len(lines) - 1

        for i, text in enumerate(lines):
            out.extend(self._update_slot(i, text, last_index, t))

        # 不是最后一行的那些行：滚上去了就不会再变，定型就进缓冲
        for i in range(last_index):
            slot = self._slots[i]
            if slot.count >= self.cfg.stable_reads and not slot.emitted:
                slot.emitted = True
                if self._append(slot.last_settled or slot.text, slot.count):
                    # 这一行自己就是完整的一句（带句末标点）→ 立刻发，别拖着
                    if _ends_sentence(slot.last_settled or slot.text):
                        out.append(self._flush("punct", t))
                    elif sum(len(x) for x in self._buffer) > self.cfg.max_chars:
                        out.append(self._flush("maxlen", t))

        # 最后一行：还在长，等它停下来
        slot = self._slots[last_index]
        text = slot.last_settled or slot.text
        if slot.count >= self.cfg.stable_reads and not slot.emitted:
            if _ends_sentence(text):
                slot.emitted = True
                if self._append(text, slot.count):
                    out.append(self._flush("punct", t))
            elif slot.settled_at and (t - slot.settled_at) >= self.cfg.max_hold_s:
                # 定型后一直没变、也没标点——不能无限等下去
                # （YouTube 自动字幕经常整句都没有句末标点）
                slot.emitted = True
                if self._append(text, slot.count):
                    out.append(self._flush("maxhold", t))
            elif sum(len(x) for x in self._buffer) + len(text) > self.cfg.max_chars:
                # 加上这一句就太长了，一次要翻译的字数得有上限——先发掉
                slot.emitted = True
                if self._append(text, slot.count):
                    out.append(self._flush("maxlen", t))
        elif not slot.emitted and (t - slot.first_seen_at) >= self.cfg.max_wait_s:
            # 一直在长、始终没稳定下来。宁可发最后一次确认过的文本，
            # 也不要让这一句凭空消失——延迟有上限比"偶尔丢一句"重要。
            out.extend(self._harvest("maxwait", t))

        return out

    def _harvest(self, reason: str, t: float) -> list[Emission]:
        """把"已经定型但还在等"的内容收进缓冲，然后整块发出去。

        字幕消失时走的就是这条路。**必须是"先收再发"，不能只看缓冲里有没有东西**：
        最后一行定型后如果没有句末标点，它会一直挂在槽位里等着，
        这时字幕突然消失的话，只查 `self._buffer` 会把整整一句丢掉。
        """
        for slot in self._slots:
            if slot.emitted:
                continue
            # 定型过的优先；从没定型过但至少连续读到两次的，也认了——
            # 这一句马上就要从屏幕上消失了，丢掉比偶尔认错更糟。
            text = slot.last_settled or (slot.text if slot.count >= 2 else "")
            if text:
                slot.emitted = True
                self._append(text, slot.count)
        if not self._buffer:
            return []
        return [self._flush(reason, t)]

    def _update_slot(self, i: int, text: str, last_index: int, t: float) -> list[Emission]:
        """更新单个槽位，返回因"换句"而发出的单元（一般没有）。"""
        out: list[Emission] = []
        slot = self._slots[i]
        k = key(text)

        if not slot.text:
            slot.text, slot.count = text, 1
            slot.first_seen_at = t
        elif _same_caption(key(slot.text), k, self.cfg.scroll_similarity):
            prev_k = key(slot.text)
            if len(k) > len(prev_k):
                # 又长了一点：换用新的，并重新确认（新文本还没被连续看到过）
                slot.text = text
                slot.count = 1
                slot.settled_at = 0.0
                # 屏幕上的内容变了，之前发出去的那份盖不住它了——不清 emitted 的话
                # 这后半截永远进不了缓冲（"…jump straight into" 长成
                # "…jump straight into ranked today" 时，后半句会凭空丢掉）。
                slot.emitted = False
            elif len(k) == len(prev_k):
                # 内容完全一样，只是标点/大小写可能不同。
                # **要换成最新那份**：比对用的键剥掉了标点，所以"…deagle" 和
                # "…deagle." 的键是同一个——不换的话句末标点永远存不进来，
                # "遇到句号就发"这条判据会彻底失效（踩过）。
                slot.text = text
                slot.count += 1
            else:
                # 这次读短了：保留原来更长的那份
                slot.count += 1
        else:
            # 换了一句：上一簇如果已经定型过，它是一句完整的话，不能丢
            if not slot.emitted and slot.last_settled:
                slot.emitted = True
                if self._append(slot.last_settled, slot.count):
                    out.append(self._flush("replaced", t))
            slot.reset()
            slot.text, slot.count = text, 1
            slot.first_seen_at = t

        if slot.count >= self.cfg.stable_reads:
            if not slot.settled_at:
                slot.settled_at = t
            slot.last_settled = slot.text

        return out

    def _append(self, text: str, votes: int) -> bool:
        """把一行加进缓冲。返回是否真的加了（重复的丢掉）。"""
        line = display(text)
        k = key(line)
        if len(k) < self.cfg.min_chars:
            return False
        # 至少要有一个字母。字幕是英语句子，一定含字母；不含的几乎都是
        # 框进来的界面元素——比分板 "12 - 9"、计时器 "3:45"、观战人数。
        # 把 "12" 送去翻译只会得到一句莫名其妙的译文。
        if not any(ch.isalpha() for ch in line):
            return False
        if k in self._recent_keys:
            self.stats["dropped_dupes"] += 1
            return False

        if self._buffer:
            prev_key = key(self._buffer[-1])
            if k == prev_key:
                self.stats["dropped_dupes"] += 1
                return False
            if k.startswith(prev_key):
                # 是上一段的延伸（上一帧只读到半句）→ 替换，别拼成重复的两段
                self._buffer[-1] = line
                return True
            if prev_key.startswith(k):
                # 比已经攒下的还短 → 过时的读数，忽略
                return False

        self._buffer.append(line)
        log.debug("缓冲 <- %r（定型 %d 次）", line, votes)
        return True

    def _flush(self, reason: str, t: float) -> Emission:
        """把缓冲整体发出去。"""
        text = " ".join(self._buffer).strip()
        votes = max((s.count for s in self._slots if s.emitted), default=0)
        self._buffer.clear()
        for slot in self._slots:
            # 注意是 mark_sent 不是 reset：屏上那几行还在，它们的状态得留着
            slot.mark_sent(t)
        k = key(text)
        if k:
            self._recent_keys.append(k)
        self.stats["emitted"] += 1
        # 只有停在句末标点上的才不算"半句"
        partial = not _ends_sentence(text)
        log.info("发出[%s] %r", reason, text)
        return Emission(text=text, reason=reason, partial=partial, votes=votes)
