"""把抓屏、认字、拼句、翻译串成一条实时流水线。

线程模型（只有两段，比音频版简单得多）：

    抓屏+认字线程 → [Emission] → 翻译线程 → 回调
        4 Hz，一次 130~190ms            1~2 秒
        （字幕状态机在这一段里）           （记忆库 → LLM → 本地词典）

关于"只译最新一句"：翻译要 1~2 秒，而字幕每 1~2 秒来一句，老实按 FIFO 就会
越译越旧——解说越快字幕越老，最后满屏都是一分钟前的话。队列 size=2、满了丢最旧的、
翻译线程拿到手之后再往队尾看一眼、有更新的就改译最新的那句。直播字幕要的是
"现在在说什么"，不是完整转录。这条是音频版实测出来的经验，原样搬过来。

屏幕上"没字幕"是常态（换气、广告、暂停），所以这里**不把"读不到字"当故障**。
真正的故障是另一批信号，见 health()。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .captions import CaptionTracker
from .capture import Grabber
from .config import Config
from .dicts import Dicts
from .ocr import OcrEngine, Reading
from .tm import TranslationMemory
from .translator import Translator

log = logging.getLogger(__name__)


class Sink:
    """输出回调集合。默认全空实现，调用方按需覆盖。

    所有回调都可能从工作线程被调用，Qt 侧必须通过信号转发回主线程。
    """

    def on_source(self, text: str) -> None: ...

    def on_delta(self, text: str) -> None: ...

    def on_translation(self, source: str, translation: str, meta: str) -> None: ...

    def on_status(self, text: str) -> None: ...

    def on_error(self, text: str) -> None: ...


@dataclass
class Stats:
    frames: int = 0
    frames_with_text: int = 0
    emitted: int = 0
    dropped_mt: int = 0
    superseded_mt: int = 0
    tm_hits: int = 0
    llm_calls: int = 0
    degraded: int = 0
    last_ocr_s: float = 0.0
    last_mt_s: float = 0.0
    last_total_s: float = 0.0
    last_src: str = ""
    # 诊断用。字幕突然没了几乎总能在这几个数上看出来。
    started_at: float = field(default_factory=time.monotonic)
    last_text_at: float = 0.0      # 最后一次认出文字的时刻
    last_emit_at: float = 0.0      # 最后一次发出翻译单元的时刻
    static_s: float = 0.0          # 画面静止了多久（暂停/卡住）
    region_error: str = ""         # 抓屏报的错（区域越界、设备问题）
    engine_error: str = ""         # 识别引擎报的错


def _put_drop_oldest(q: queue.Queue, item) -> bool:
    """入队；满了就丢掉队首最旧的一项。返回是否发生了丢弃。"""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                return dropped


def _signature(frame) -> int:
    """给一帧算个便宜指纹，用来判断画面是不是还在动。

    隔 8 个像素采一个，1600×90 的条带只剩 20×11 个点，几乎不要时间。
    要的不是精确，只是"跟上一帧一不一样"——用来把"视频暂停了"和"框错地方了"
    分开（前者画面完全不动，后者画面在动但认不出字）。
    """
    try:
        return hash(frame[::8, ::8].tobytes())
    except Exception:  # pragma: no cover
        return 0


class Pipeline:
    def __init__(self, cfg: Config, sink: Sink):
        self.cfg = cfg
        self.sink = sink
        self.stats = Stats()

        self.tm = TranslationMemory(cfg.tm)
        self.dicts = Dicts(cfg.dictionaries)
        self.translator = Translator(cfg, self.tm, self.dicts)

        self._region: tuple[int, int, int, int] | None = cfg.region()
        self._engine: OcrEngine | None = None
        self._mt_q: queue.Queue = queue.Queue(maxsize=2)
        self._context: deque[tuple[str, str]] = deque(
            maxlen=max(1, cfg.translate.context_lines)
        )

        self._stopping = threading.Event()
        self._paused = threading.Event()
        # 抓屏线程和外部（重新框选、暂停恢复）不在一个线程里，拼句状态又被前者独占，
        # 所以用一个事件请它"下次循环开始时把状态清掉"，而不是直接伸手进去改。
        self._tracker_reset = threading.Event()
        self._threads: list[threading.Thread] = []
        self._shown_raw = ""
        self._last_sig = 0
        self._static_since = 0.0
        self._skip_times: deque[float] = deque()
        self.src_lang = cfg.translate.src_lang or "en"

    # ---------------------------------------------------------------- 生命周期
    def check_ready(self) -> None:
        """启动前检查翻译模型能不能通。连不上不算致命——会退到本地词典。"""
        self.translator.check_ready()

    def warmup(self) -> None:
        """建识别引擎并空跑一次。

        为什么引擎在这儿建、而不在抓屏线程里建：**第一次推理特别慢**
        （模型要load、onnxruntime 要初始化），放启动时做掉，用户框完选
        开直播时就不必多等那一秒。引擎只有抓屏线程一个使用者，不存在并发问题。

        抓屏器（mss）则相反，**必须在用它那个线程里建**，所以那个留在线程里。
        """
        if self._engine is None:
            self._engine = OcrEngine(self.cfg.ocr)
        import numpy as np

        t0 = time.perf_counter()
        h = max(8, 16)
        w = 480
        self._engine.read(np.zeros((h, w, 3), dtype=np.uint8), time.monotonic())
        log.info("识别引擎预热完成（耗时 %.2fs）", time.perf_counter() - t0)
        if self._engine.last_error:
            self._emit(self.sink.on_error, f"识别引擎异常：{self._engine.last_error}")

    def start(self, with_capture: bool = True) -> None:
        """启动。`with_capture=False` 只开翻译线程——--text 模式用，它不抓屏，
        所以"还没框选区域"在那条路上不是问题，不该报错。"""
        self._spawn("mt", self._mt_loop)
        if not with_capture:
            return
        if self._region is not None:
            self._spawn("scan", self._scan_loop)
        else:
            self._emit(self.sink.on_error, "还没框选字幕区域 —— 按 Ctrl+Alt+R 框一次")

    def _spawn(self, name: str, target) -> None:
        t = threading.Thread(target=target, name=f"ltocr-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stopping.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self.translator.close()
        self.tm.close()
        if self._engine is not None:
            self._engine.close()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
            self._drain(self._mt_q)
        else:
            # 恢复时把拼句状态清干净：暂停期间字幕早就换了好几轮，
            # 留着旧槽位会把暂停前后两段不相干的文字拼成一句话。
            self._paused.clear()
            self._reset_tracker()

    def _reset_tracker(self) -> None:
        self._tracker_reset.set()

    @property
    def region(self) -> tuple[int, int, int, int] | None:
        return self._region

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        """换字幕区域（重新框选后调用）。抓屏线程下一帧就会用新的。"""
        self._region = region
        self._reset_tracker()
        self._shown_raw = ""
        if region is None:
            self._emit(self.sink.on_error, "字幕区域为空，按 Ctrl+Alt+R 重新框选")
        else:
            log.info("字幕区域改为 %s", region)
            self._emit(self.sink.on_status, f"字幕区域已更新：{list(region)}")
            # 本来是"没框选过所以没开线程"，现在有了就补上
            if not any(t.name == "ltocr-scan" for t in self._threads):
                self._spawn("scan", self._scan_loop)

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    # ---------------------------------------------------------------- 抓屏 + 认字
    def _interval(self) -> float:
        """这一帧该隔多久再抓。读不到字时降频省电（广告、暂停、没开字幕占大多数时间）。"""
        idle_since = max(self.stats.last_text_at, self.stats.started_at)
        idle = (time.monotonic() - idle_since) >= self.cfg.capture.idle_after_s
        fps = self.cfg.capture.fps_idle if idle else self.cfg.capture.fps
        return 1.0 / max(fps, 0.1)

    def _scan_loop(self) -> None:
        tracker = CaptionTracker(self.cfg.captions)
        engine = self._engine or OcrEngine(self.cfg.ocr)
        self._engine = engine

        try:
            grabber = Grabber(self.cfg.capture.monitor)
        except Exception as exc:
            self._emit(self.sink.on_error, f"抓屏初始化失败：{exc}")
            log.exception("抓屏初始化失败")
            return
        log.info("抓屏就绪：显示器 %s，区域 %s，%.1f Hz",
                 grabber.bounds, self._region, self.cfg.capture.fps)

        next_at = time.perf_counter()
        try:
            while not self._stopping.is_set():
                if self._paused.is_set():
                    time.sleep(0.1)
                    next_at = time.perf_counter()
                    continue
                region = self._region
                if region is None:
                    time.sleep(0.2)
                    next_at = time.perf_counter()
                    continue
                if self._tracker_reset.is_set():
                    self._tracker_reset.clear()
                    tracker.reset()
                    self._shown_raw = ""

                next_at += self._interval()
                try:
                    frame = grabber.grab(region)
                except Exception as exc:
                    # 区域越界（改了分辨率、换了小屏）或抓屏设备出问题。
                    # 不能退出线程——用户重新框一下就好了，退出就没救了。
                    if self.stats.region_error != str(exc):
                        self.stats.region_error = str(exc)
                        log.error("抓屏失败：%s", exc)
                        self._emit(self.sink.on_error, f"抓屏失败：{exc}")
                    time.sleep(1.0)
                    next_at = time.perf_counter()
                    continue
                if self.stats.region_error:
                    self.stats.region_error = ""
                    log.info("抓屏恢复正常")

                self._process(frame, tracker, engine)

                sleep = next_at - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    # 一帧比设定的间隔还慢（识别拖后腿），别越积越多，对齐到当下
                    next_at = time.perf_counter()
        finally:
            grabber.close()

    def _process(self, frame, tracker: CaptionTracker, engine: OcrEngine) -> None:
        t0 = time.perf_counter()
        t = time.monotonic()

        sig = _signature(frame)
        if sig == self._last_sig:
            self.stats.static_s = time.monotonic() - (self._static_since or time.monotonic())
        else:
            self._last_sig = sig
            self._static_since = time.monotonic()
            self.stats.static_s = 0.0

        reading = engine.read(frame, t)
        self.stats.frames += 1
        self.stats.last_ocr_s = time.perf_counter() - t0
        if engine.last_error:
            self.stats.engine_error = engine.last_error

        if reading.empty:
            pass
        else:
            self.stats.frames_with_text += 1
            self.stats.last_text_at = t

        # 先把"屏幕上现在是什么"推给字幕窗，别等拼句。用户需要立刻看到
        # 识别是活的——否则盯着空白窗口等 2.5 秒会以为坏了。
        raw = tracker.display_text()
        if raw != self._shown_raw:
            self._shown_raw = raw
            self._emit(self.sink.on_source, raw)

        for em in tracker.push(reading):
            self.stats.emitted += 1
            self.stats.last_emit_at = time.monotonic()
            log.info("[拼句 %s] %s", em.describe(), em.text)
            if _put_drop_oldest(self._mt_q, em):
                self.stats.dropped_mt += 1
                self._note_skip()
                log.warning("翻译队列满，丢弃最旧的一句（累计 %d）", self.stats.dropped_mt)

    # ---------------------------------------------------------------- 翻译
    def _mt_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                em = self._mt_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._paused.is_set():
                continue

            # 拿到手之后先别急着翻：翻译一句要一两秒，这期间很可能又有新句子排进来。
            # 队里那句一定比手上这句新，而直播字幕要的是"现在在说什么"，
            # 所以直接换掉手上这句、改译队尾最新的一句。
            # （音频版实测：一次灌 8 句只译最新 1 句，滞后被压在"一句翻译时间"以内。）
            skipped = 0
            while True:
                try:
                    em = self._mt_q.get_nowait()
                except queue.Empty:
                    break
                skipped += 1
            if skipped:
                self.stats.superseded_mt += skipped
                for _ in range(skipped):
                    self._note_skip()
                log.info("翻译落后，跳过 %d 句旧的，直接译最新一句", skipped)

            self._translate_one(em)

    def _translate_one(self, em) -> None:
        t0 = time.perf_counter()
        try:
            outcome = self.translator.translate(
                em.text,
                self.src_lang,
                context=list(self._context),
                on_delta=lambda partial: self._emit(self.sink.on_delta, partial),
                should_stop=self._stopping.is_set,
            )
        except Exception as exc:
            log.exception("翻译异常")
            self._emit(self.sink.on_error, f"翻译失败：{exc}")
            return

        if outcome.source == "error" or not outcome.text:
            self._emit(self.sink.on_error, f"翻译失败：{outcome.detail or '空结果'}")
            return

        total = time.perf_counter() - t0
        self.stats.last_mt_s = outcome.latency_s
        self.stats.last_total_s = total
        if outcome.source.startswith("tm"):
            self.stats.tm_hits += 1
            tag = "缓存" + (f"~{outcome.tm_score:.0f}%" if outcome.source == "tm-fuzzy" else "")
        elif outcome.source.startswith("local"):
            self.stats.degraded += 1
            tag = outcome.label
        else:
            self.stats.llm_calls += 1
            tag = "模型" if outcome.source == "llm" else "模型(截断)"

        self._context.append((em.text, outcome.text))

        log.info("[%s %.2fs] %s → %s", tag, outcome.latency_s,
                 em.text[:50], outcome.text[:60])

        bits = [f"识别 {self.stats.last_ocr_s * 1000:.0f}ms",
                f"翻译 {outcome.latency_s:.1f}s", tag]
        if em.partial:
            bits.append("半句")
        if outcome.detail and outcome.degraded:
            # 降级必须让用户看得见：老项目最惨的教训就是出问题的时候屏幕上没线索
            bits.append(outcome.detail)
        meta = " · ".join(bits) if self.cfg.overlay.show_latency else ""

        # local-raw 的"译文"就是英文原文，和原文行一模一样，显示两遍只会让人困惑。
        # 这时候把译文行让出来，只在 meta 里说明情况。
        text = outcome.text if outcome.source != "local-raw" else ""
        self._emit(self.sink.on_translation, outcome.source, text, meta)

    # ---------------------------------------------------------------- 外部注入
    def submit_text(self, text: str) -> None:
        """跳过抓屏和识别，直接把文本送进翻译链路（--text 模式 / 调试用）。"""
        text = (text or "").strip()
        if not text:
            return
        self.stats.last_src = text
        self._emit(self.sink.on_source, text)
        self._translate_one(_FakeEmission(text))

    def read_image(self, path) -> tuple[Reading, list]:
        """认一张图，再走一遍拼句状态机，返回 (原始读数, 拼出来的句子)。

        `--image` 用：一张静止的图只有一帧读数，而状态机要求"连续两次一样才算定型"，
        所以这里就喂两次再补一帧空白——正是"字幕出现、停留、消失"的真实节奏，
        走的是和直播完全相同的代码路径。
        """
        import numpy as np

        frame = _load_image(path)
        if self._engine is None:
            self._engine = OcrEngine(self.cfg.ocr)
        t0 = time.monotonic()
        reading = self._engine.read(frame, t0)
        if reading.empty:
            return reading, []

        tracker = CaptionTracker(self.cfg.captions)
        out = []
        step = 1.0 / max(self.cfg.capture.fps, 0.1)
        for i in (0, 1):
            out.extend(tracker.push(
                Reading(text=reading.text, lines=reading.lines, score=reading.score,
                        elapse_s=reading.elapse_s, t=t0 + i * step)))
        # 补一帧空白：等价于"字幕从屏幕上消失了"，让缓冲里没有标点的那句也能发出来
        blank_at = t0 + 2 * step + self.cfg.captions.idle_emit_s
        out.extend(tracker.push(Reading(text="", lines=(), score=0.0, t=blank_at)))
        return reading, out

    # ---------------------------------------------------------------- 工具
    def _emit(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # pragma: no cover - 回调不该拖垮流水线
            log.warning("回调异常：%s", exc)

    def _note_skip(self) -> None:
        self._skip_times.append(time.monotonic())

    def health(self) -> tuple[str, str]:
        """返回 (问题类型, 提示语)，都健康时是 ("", "")。

        和音频版的形制一样，类型是稳定标识符、提示语带实时数字。**必须分开**：
        界面靠类型判断"要不要记一条日志"，拿带秒数的提示语去比会变成每秒刷一条。

        OCR 版特有的难点是**区分"没字幕"和"坏了"**——屏幕上没有字是常态
        （换气、广告、暂停、没开 CC），把它当故障会天天误报。所以这里只报
        那些"正常情况下不可能出现"的情形：

          1. 识别引擎自己报错（引擎没建起来、模型坏了）
          2. 抓屏报错（改了分辨率，区域跑到屏幕外面去了）
          3. 画面在动、却很久一个字都没认出来 —— 十有八九是框错地方了。
             这一条要和"画面静止"分开：暂停时画面完全不动，那种情况下没字是应该的。
          4. 近 20 秒反复跳句（跟不上），字幕会跳着走
        """
        s = self.stats
        if self._paused.is_set() or self._region is None:
            return ("", "")
        if s.engine_error:
            return ("engine", f"识别引擎报错：{s.engine_error}")
        if s.region_error:
            return ("region", f"抓屏失败：{s.region_error}")

        # 画面在动却读不到字 = 框的位置不对。阈值取 60 秒：广告、暂停、
        # 大段没人说话都能到几十秒，太灵敏会天天误报，反而没人信。
        idle = time.monotonic() - max(s.last_text_at, s.started_at)
        if idle >= 60 and s.static_s < idle * 0.5:
            return (
                "notext",
                f"{idle:.0f} 秒没认出任何文字，但画面一直在动 —— "
                "多半是字幕区域框错地方了，按 Ctrl+Alt+R 重框",
            )
        if idle >= 60 and s.frames > 0:
            return (
                "frozen",
                f"{idle:.0f} 秒没认出文字，而且画面一直没变 —— "
                "是视频暂停了吗？（暂停时这条提示是正常的）",
            )

        cutoff = time.monotonic() - 20.0
        while self._skip_times and self._skip_times[0] < cutoff:
            self._skip_times.popleft()
        if len(self._skip_times) >= 3:
            return (
                "behind",
                f"字幕出得比翻译快，近 20 秒跳过 {len(self._skip_times)} 句"
                "（字幕会跳）",
            )
        return ("", "")

    def health_warning(self) -> str:
        return self.health()[1]

    def status_line(self) -> str:
        s = self.stats
        if self._paused.is_set():
            return "已暂停（按 Ctrl+Alt+P 继续）"
        if self._region is None:
            return "还没框选字幕区域 —— 按 Ctrl+Alt+R 框一次"
        seen = (s.frames_with_text / s.frames * 100) if s.frames else 0.0
        h = self.translator.health()
        llm = "离线词典" if h["offline"] else (h["model"] or "模型")
        return (
            f"识别 {s.last_ocr_s * 1000:.0f}ms（{seen:.0f}% 的帧有字）· "
            f"翻译 {s.last_mt_s:.1f}s · 队列 {self._mt_q.qsize()} · "
            f"缓存 {s.tm_hits} 模型 {s.llm_calls} 词典 {s.degraded} · {llm}"
        )


@dataclass(frozen=True)
class _FakeEmission:
    """给 submit_text 用的假单元：外部注入的文本没有"拼句"这一步。"""
    text: str
    reason: str = "inject"
    partial: bool = False
    votes: int = 1


def _load_image(path):
    """读一张图，返回 RGB ndarray。用 cv2（已经因为 rapidocr 装上了）。"""
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不了图片：{path}")
    return np.ascontiguousarray(img[:, :, ::-1])   # BGR → RGB
