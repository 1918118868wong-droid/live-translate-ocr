"""框选字幕区域：拖一个矩形出来，存进配置，并且**当场验证框得对不对**。

为什么要有"当场验证"这一步：字幕区域框歪了、框到进度条上、框得太松把比分板框
进来，这些错误在**程序里是静默的**——照样运行、照样没输出，用户只能开着直播干等，
然后怀疑是程序坏了。所以框完立刻抓几帧、真跑一次识别、把认出来的字显示在面前。
看到字就说明框对了，看不到就重框。这比"先看十分钟直播再猜"省事得多。

    python -m livetranslate_ocr.regionpick          框选并保存
    python -m livetranslate_ocr.regionpick --show   只打印当前配置的区域
    python -m livetranslate_ocr.regionpick --region 100,900,1600,90   直接设（不用鼠标）

坐标一律以**物理像素**存进配置。Qt 给的是逻辑坐标，屏幕缩放不是 100% 时两者
差一个 devicePixelRatio，所以存之前要乘回去（见 _to_physical）。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# 验证时最多抓几帧、抓多久。多抓几帧的理由：字幕会在"长出来"和"整块消失"之间
# 循环，随手抓一帧有可能是空的，那样用户会以为框错了。抓一小把取信息最多的那帧。
VERIFY_FRAMES = 6
VERIFY_SPAN_S = 1.4


def _pick_screen(app):
    """选要框的那块屏：鼠标在哪块屏上就框哪块，最符合直觉。"""
    from PySide6.QtGui import QCursor, QGuiApplication

    screen = QGuiApplication.screenAt(QCursor.pos())
    return screen or QGuiApplication.primaryScreen()


def _grab_and_read(region: tuple[int, int, int, int], monitor: int):
    """抓几帧跑一次识别，返回 (认出来的字, 说明, 最清楚的那帧)。

    **必须先把框选窗口藏起来再抓**——它就盖在屏幕上，不藏的话抓到的全是它自己。
    抓回来的那一帧还要摆给用户看（"识别引擎看到的就是这样"），所以一并返回。
    """
    from PySide6.QtWidgets import QApplication

    from .capture import Grabber
    from .config import Config
    from .ocr import OcrEngine

    app = QApplication.instance()
    if app is not None:
        app.processEvents()

    cfg = Config()
    engine = OcrEngine(cfg.ocr)
    grabber = Grabber(monitor)
    best_text, best_score, best_frame, tries, err = "", -1.0, None, 0, ""
    deadline = time.monotonic() + VERIFY_SPAN_S
    try:
        while tries < VERIFY_FRAMES and time.monotonic() < deadline:
            try:
                frame = grabber.grab(region)
            except Exception as exc:
                err = str(exc)
                break
            reading = engine.read(frame, time.monotonic())
            tries += 1
            if reading.text and reading.score > best_score:
                best_text, best_score, best_frame = reading.text, reading.score, frame
            if reading.text and tries >= 2:
                break          # 第一帧就认到了，再抓一帧确认稳定就走
            time.sleep(0.12)
    finally:
        grabber.close()
        engine.close()

    if err:
        return "", err, None
    if not best_text:
        return "", (f"抓了 {tries} 帧，一个字都没认出来 —— "
                    "确认字幕正显示着、区域框在字幕带上面"), best_frame
    return best_text, f"抓了 {tries} 帧，上面这张是认得最清楚的一帧（置信度 {best_score:.2f}）", best_frame


def _to_physical(geo_x: int, geo_y: int, x: int, y: int, w: int, h: int,
                 dpr: float) -> tuple[int, int, int, int]:
    """Qt 逻辑坐标 → 屏幕物理像素。

    geo_x/geo_y 是这块屏在 Qt 虚拟桌面里的逻辑原点，x/y/w/h 是控件内的逻辑坐标。
    Qt 的模型是"这块屏的逻辑矩形 × dpr = 它的物理矩形"，所以整体乘一次就行。
    本机 dpr = 1.0，怎么算都一样；但用户改缩放后就靠这一行。
    """
    return (int(round((geo_x + x) * dpr)), int(round((geo_y + y) * dpr)),
            max(1, int(round(w * dpr))), max(1, int(round(h * dpr))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regionpick", description="框选屏幕上的字幕区域并保存到配置")
    parser.add_argument("--show", action="store_true", help="只打印当前区域，不开窗口")
    parser.add_argument("--region", help="直接设定，格式 x,y,w,h（物理像素）")
    parser.add_argument("--monitor", type=int, default=None, help="抓哪块屏（默认读配置）")
    parser.add_argument("--keep", action="store_true",
                        help="框完不验证，直接保存（不想等那 1 秒时用）")
    args = parser.parse_args(argv)

    from . import config as config_mod

    cfg = config_mod.load()
    if args.monitor is not None:
        cfg.capture.monitor = args.monitor

    # ---- 不用开窗口的两条路 ----
    if args.show:
        region = cfg.region()
        print(f"配置文件：{cfg.source_path or '（未找到，正在用默认值）'}")
        print(f"字幕区域：{region if region else '还没框选过'}")
        print(f"抓屏频率：{cfg.capture.fps} Hz（无字时降到 {cfg.capture.fps_idle} Hz）")
        print(f"抓哪块屏：{cfg.capture.monitor}")
        return 0 if region else 1

    if args.region:
        try:
            parts = [int(v.strip()) for v in args.region.split(",")]
            if len(parts) != 4:
                raise ValueError
        except ValueError:
            print("--region 的格式是 x,y,w,h（四个整数），例如 --region 100,900,1600,90",
                  file=sys.stderr)
            return 2
        region = (parts[0], parts[1], parts[2], parts[3])
        config_mod.save_region(cfg, region)
        print(f"已保存字幕区域：{region}")
        return 0

    # ---- 图形路径 ----
    from .capture import ensure_dpi_awareness

    ensure_dpi_awareness()   # 必须在 QApplication 之前
    from PySide6.QtWidgets import QApplication

    from .pickwindow import RegionPickWindow

    app = QApplication(sys.argv[:1])
    screen = _pick_screen(app)
    dpr = float(screen.devicePixelRatio()) or 1.0
    log.info("框选屏：%s %s dpr=%s", screen.name(), screen.geometry(), dpr)

    window = RegionPickWindow(cfg, screen, dpr, verify=not args.keep)
    window.show()
    app.exec()

    if window.result_region is None:
        print("\n已取消，配置没有改动。")
        return 1

    region = window.result_region
    config_mod.save_region(cfg, region)
    print()
    print("=" * 52)
    print(f"字幕区域已保存：{list(region)}")
    print(f"  （x={region[0]}, y={region[1]}, 宽={region[2]}, 高={region[3]}，物理像素）")
    if window.verified_text:
        print()
        print("当场识别到的文字：")
        for line in window.verified_text.splitlines():
            print(f"    {line}")
        print()
        print("看到上面的字 = 框对了。现在可以关掉这个窗口去跑 run.bat。")
    else:
        print()
        print("!! 没识别到文字。看看字幕是不是正显示着；或按 Ctrl+Alt+R 重框一次。")
    print("=" * 52)
    return 0 if window.verified_text else 3


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
