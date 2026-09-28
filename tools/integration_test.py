# -*- coding: utf-8 -*-
"""流水线集成测试：假抓屏器 + 真 OCR + 真状态机 + 真翻译线程。

`selfcheck.py` 测的是各个零件（状态机、两个词典、降级链、字幕窗位置），
它**测不到线程之间的配合**——抓屏线程的节奏、队列丢旧的时机、翻译线程只取最新那句。
这个文件补的就是那一块，也是计划里"过载时只译最新一句"那条验证的落点。

不碰屏幕：抓屏器换成假的，按**时间轴**喂合成出来的字幕图。
不联网：故意把 Key 的环境变量名指到一个不存在的名字，强制走离线词典。

要跑真识别（RapidOCR），所以比自检慢，大约一分钟：

    .venv\\Scripts\\python.exe tools\\integration_test.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# 空帧会被识别引擎刷一屏 "text detection result is empty"，淹掉真正的输出
logging.getLogger("RapidOCR").setLevel(logging.CRITICAL)

from livetranslate_ocr import pipeline as P  # noqa: E402
from livetranslate_ocr.config import Config  # noqa: E402

W = 1600          # 和真实框选时的宽度一致（实测数据也是按 1600 宽测的）
BLANK = np.zeros((60, W, 3), np.uint8)


def _font() -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf"):
        try:
            return ImageFont.truetype(name, 40)
        except OSError:
            continue
    raise SystemExit("找不到可用的粗体字体（arialbd.ttf / arial.ttf / segoeuib.ttf），"
                     "合成字幕图这一步做不了")


FONT = None  # 在 main() 里初始化：找不到字体时才给得出人话的错误


def make(lines: list[str]):
    """画一张字幕图。黑底白字，和 YouTube 字幕的样子接近。"""
    rowh = 58
    img = Image.new("RGB", (W, rowh * len(lines) + 16), (10, 10, 12))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        bb = d.textbbox((0, 0), ln, font=FONT)
        d.text((W // 2 - (bb[2] - bb[0]) // 2, 8 + i * rowh), ln,
               font=FONT, fill=(255, 255, 255))
    return np.ascontiguousarray(np.array(img))


# ---- 剧本：每组 (持续秒数, 屏幕内容) -------------------------------------
# 阶段 1 用的：一句带句末标点的 + 两行的（最后一行还在长）
SCRIPT_CHAIN = (
    [(2.0, make(["NiKo gets the clutch, one versus three."]))]
    + [(1.6, BLANK)]
    + [(2.4, make(["Alright chat, welcome back to the stream.",
                   "we are gonna jump straight into ranked today"]))]
    + [(2.0, BLANK)]
)

# 阶段 2 用的：连着六句短句，每句后面跟一小段空白
BURST = [
    "Nice shot from the top of the stairs.",
    "Watch out for the flank on the left side.",
    "That was a clean ace in the pistol round.",
    "Buy the armor and save for the next round.",
    "Turn the corner and check the window first.",
    "Push the site and plant the bomb quickly.",
]
SCRIPT_BURST: list[tuple[float, object]] = []
for _s in BURST:
    SCRIPT_BURST += [(1.0, make([_s])), (0.6, BLANK)]
SCRIPT_BURST += [(2.0, BLANK)]

# 时间轴。抓屏器按**墙上时间**取帧，不按调用次数——真屏幕就是这样：
# 不管采集循环这一轮快慢（空闲会降频），某个时刻屏幕上是什么就是什么。
TIMELINE: list[tuple[float, object]] = []
EPOCH = [0.0]


def load(script) -> None:
    """换一段剧本。**必须重置时间轴零点**，否则新剧本从旧剧本的起点算起
    （抓屏器是启动时造的，那时已经过去好几秒），开头几句直接过期。"""
    TIMELINE.clear()
    EPOCH[0] = time.monotonic()
    t = 0.0
    for dur, img in script:
        TIMELINE.append((t, img))
        t += dur
    TIMELINE.append((t, script[-1][1]))


class FakeGrabber:
    def __init__(self, monitor: int = 0):
        self.bounds = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        self.n = 0

    def grab(self, region):
        self.n += 1
        now = time.monotonic() - EPOCH[0]
        img = TIMELINE[0][1]
        for at, frame in TIMELINE:
            if at <= now:
                img = frame
            else:
                break
        return img

    def close(self) -> None:
        pass


class Sink:
    def __init__(self):
        self.sources: list[str] = []
        self.trans: list[tuple[str, str, str]] = []
        self.errors: list[str] = []

    def on_source(self, text):
        self.sources.append(text)

    def on_delta(self, text):
        pass

    def on_translation(self, source, text, meta):
        self.trans.append((source, text, meta))

    def on_status(self, text):
        pass

    def on_error(self, text):
        self.errors.append(text)


def make_config(tmp: str) -> Config:
    cfg = Config()
    cfg.tm.db = str(pathlib.Path(tmp) / "tm.sqlite3")
    cfg.dictionaries.terms = str(PROJECT_ROOT / "data" / "terms.tsv")
    cfg.dictionaries.common = str(PROJECT_ROOT / "data" / "common.tsv")
    # 强制离线：把 Key 的环境变量名指到一个不存在的名字，用户自己配了 Key 也不影响
    cfg.llm.api_key = ""
    cfg.llm.api_key_env = "LTOCR_INTEGRATION_TEST_NO_SUCH_KEY"
    cfg.capture.region = [0, 0, W, 200]
    cfg.capture.fps = 5.0
    cfg.capture.fps_idle = 3.0
    cfg.capture.idle_after_s = 0.5
    return cfg


def wait_for(pred, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


def main() -> int:
    global FONT
    FONT = _font()

    ok = True
    real_grabber = P.Grabber
    P.Grabber = FakeGrabber
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            sink = Sink()
            pipe = P.Pipeline(cfg, sink)
            pipe.warmup()

            # ---------------- 阶段 1：全链路 ----------------
            load(SCRIPT_CHAIN)
            pipe.start()
            wait_for(lambda: len(sink.trans) >= 2, 25)
            time.sleep(0.3)
            st = pipe.stats
            print("=== 阶段 1：抓屏 → 认字 → 拼句 → 翻译 ===")
            print(f"  抓 {st.frames} 帧（{st.frames_with_text} 帧有字）· 拼出 {st.emitted} 句")
            print(f"  识别 {st.last_ocr_s * 1000:.0f}ms · 翻译 {st.last_mt_s:.2f}s")
            for text in dict.fromkeys(sink.sources):
                print("  屏幕：", text.replace("\n", " / "))
            for source, text, meta in sink.trans:
                print(f"  译文[{source}] {text!r}")
            print("  health:", pipe.health())
            if len(sink.trans) < 2:
                ok = False
                print("  !! 两行那句应该也发出来（只发第一行是已知的坑）")

            # ---------------- 阶段 2：过载，只译最新 ----------------
            load(SCRIPT_BURST)
            slow_translate = pipe.translator.translate

            def slow(*a, **kw):
                time.sleep(2.0)          # 装成"翻译比字幕来得慢"
                return slow_translate(*a, **kw)

            pipe.translator.translate = slow
            base_emit, base_tr = pipe.stats.emitted, len(sink.trans)
            wait_for(lambda: pipe.stats.emitted - base_emit >= len(BURST), 12)
            time.sleep(2.0)              # 让最后那几句翻完
            emitted = pipe.stats.emitted - base_emit
            got = len(sink.trans) - base_tr
            print("=== 阶段 2：过载（每句故意拖 2 秒）===")
            print(f"  拼出 {emitted} 句，只译了 {got} 条")
            for source, text, meta in sink.trans[base_tr:]:
                print(f"  译文[{source}] {text[:48]!r}")
            if emitted < len(BURST):
                ok = False
                print(f"  !! 只拼出 {emitted} 句，{len(BURST)} 句里有的被整句吞了")
            if got >= emitted:
                ok = False
                print("  !! 过载时一句没丢，说明翻译线程老实按 FIFO 排队了——"
                      "屏幕上会一直在说'刚才那句话'")

            pipe.stop()

            # ---------------- 阶段 3：抓屏故障 ----------------
            cfg2 = make_config(tmp)
            sink2 = Sink()
            p2 = P.Pipeline(cfg2, sink2)
            load(SCRIPT_CHAIN)
            p2.start()

            def boom(region):
                raise ValueError("采集区域完全落在屏幕外")

            good_grab = FakeGrabber.grab
            FakeGrabber.grab = staticmethod(boom)
            wait_for(lambda: bool(sink2.errors), 8)
            print("=== 阶段 3：抓屏一直失败 ===")
            print("  health:", p2.health())
            print("  回调:", sink2.errors[-1] if sink2.errors else "（无）")
            if not p2.health()[0]:
                ok = False
                print("  !! 抓屏一直失败，health() 却没报出来——用户会以为只是没字幕")
            p2.stop()
            FakeGrabber.grab = good_grab
    finally:
        P.Grabber = real_grabber

    print("\n结论：", "全部符合预期" if ok else "有不符合预期的地方（见上面的 !!）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
