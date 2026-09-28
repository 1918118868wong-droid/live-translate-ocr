"""屏幕抓取：按配置的区域，定时抓一小块屏幕交给 OCR。

只用 mss（纯 GDI 调用，比 PIL.ImageGrab 快也稳），不依赖 Qt——
抓屏在后台线程跑，跟界面线程解耦。

关于坐标：配置里的 region 是**屏幕物理像素**。这台机器 DPR = 1.0（2560×1440，
无缩放），逻辑坐标和物理坐标一致；但用户改系统缩放或接第二块屏之后就不一致了。
框选器负责把 Qt 逻辑坐标按 devicePixelRatio 换算成物理像素再存进来，
抓屏这边只管照单抓，不做二次换算。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


def ensure_dpi_awareness() -> None:
    """让本进程按真实像素工作，而不是被 Windows 按缩放比例虚拟化。

    **必须在创建 QApplication 之前调用**，晚了就无效。不调的话，系统缩放
    不是 100% 时 Windows 会给进程一个假的、被缩放过的坐标空间：Qt 报出来的
    坐标和 mss 抓到的物理像素对不上，框选结果会整体偏移一大截。

    本机 DPR = 1.0（无缩放），调不调都一样；但这是个"在别人机器上才会犯、
    而且完全看不出原因"的错，所以照调。

    优先用 Win10 之后的 PER_MONITOR_AWARE_V2（多屏各自缩放也能正确工作），
    失败再退回旧接口。非 Windows 直接跳过。
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            log.debug("设置 DPI 感知失败，按系统默认继续")


class Grabber:
    """抓屏器。**必须在它被使用的那个线程里创建**——mss 实例不跨线程。"""

    def __init__(self, monitor: int = 0) -> None:
        import mss

        self._sct = mss.mss()
        self.monitor = monitor
        self._bounds = self._pick_monitor(monitor)

    def _pick_monitor(self, monitor: int) -> dict:
        """选出要抓的那块屏的边界。

        mss.monitors[0] 是所有屏拼起来的虚拟屏，[1] 起才是各块单屏。
        配的编号越界（比如只有一块屏却填了 2）就退回主屏，并且要吼一声——
        静默回退会让人对着黑屏查半天。
        """
        monitors = self._sct.monitors
        if 0 <= monitor < len(monitors):
            return monitors[monitor]
        log.warning(
            "配置的 capture.monitor = %d 超出范围（这台机器有 %d 块屏，编号 0~%d），已退回主屏",
            monitor, len(monitors) - 1, len(monitors) - 1,
        )
        return monitors[1] if len(monitors) > 1 else monitors[0]

    @property
    def bounds(self) -> dict:
        return dict(self._bounds)

    @property
    def size(self) -> tuple[int, int]:
        return self._bounds["width"], self._bounds["height"]

    def grab(self, region: tuple[int, int, int, int]) -> np.ndarray:
        """抓 region=[x, y, w, h]（物理像素），返回 RGB 的 ndarray。

        区域会被裁到屏幕边界内：用户框选后改了分辨率、或换了块小屏，
        原始坐标可能已经越界，裁一下总比 mss 抛异常好。
        """
        x, y, w, h = region
        mon = self._bounds

        # 与屏幕求交集
        x0 = max(x, mon["left"])
        y0 = max(y, mon["top"])
        x1 = min(x + w, mon["left"] + mon["width"])
        y1 = min(y + h, mon["top"] + mon["height"])
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            raise ValueError(
                f"采集区域 {region} 完全落在屏幕外（屏幕范围 "
                f"{mon['width']}x{mon['height']} @ ({mon['left']}, {mon['top']})），"
                f"按 {self._hotkey()} 重新框选"
            )

        shot = self._sct.grab({"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
        # shot.rgb 是 RGB 的裸字节，直接 view 成数组，省一次通道重排
        return np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)

    @staticmethod
    def _hotkey() -> str:
        try:
            from . import config
            return config.Config().hotkeys.repick_region
        except Exception:
            return "ctrl+alt+r"

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def frames(region: tuple[int, int, int, int], fps: float, monitor: int = 0,
           stop_flag=None) -> Iterator[np.ndarray]:
    """按 fps 产出抓到的帧，直到 stop_flag 被置位。

    节奏用"睡到该抓的时刻"而不是"抓完再睡固定时长"：OCR 耗时几十毫秒，
    后者会让实际帧率慢慢漂移（5Hz 漂到 4Hz 这样）。
    """
    grabber = Grabber(monitor)
    interval = 1.0 / max(fps, 0.1)
    next_at = time.perf_counter()
    try:
        while not (stop_flag is not None and stop_flag.is_set()):
            next_at += interval
            yield grabber.grab(region)
            sleep = next_at - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # 抓取比设定的间隔还慢（OCR 拖后腿），别越积越多，直接对齐到当下
                next_at = time.perf_counter()
    finally:
        grabber.close()


def main() -> int:
    """把各块屏的几何信息打出来，排查"框选坐标对不对"时用。"""
    import mss

    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors):
            label = "全部屏拼接" if i == 0 else f"第 {i} 块屏"
            print(f"[{i}] {label:10s} {mon['width']}x{mon['height']} "
                  f"@ ({mon['left']}, {mon['top']})")
    print("\n注意：这是物理像素。系统如果开了缩放，界面上的坐标要乘 devicePixelRatio。")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
