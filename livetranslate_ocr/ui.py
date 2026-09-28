"""Qt 界面层：字幕窗装配、流水线回调的跨线程转发、热键、状态行、重新框选。

单独成模块是有意的——PySide6 是个几十兆的重依赖，而 --help / --init-config /
--list-monitors / --image 这些路径根本不需要它。main.py 只在真要开窗口时才导入这里，
所以依赖没装全的时候命令行依然可用，--image 也能在没界面的情况下排查问题。
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication

from . import config as config_mod
from . import langs
from .hotkeys import HotkeyManager, help_text
from .overlay import SubtitleOverlay
from .pipeline import Pipeline
from .translator import LlmUnavailable

log = logging.getLogger("livetranslate_ocr")


def _brief(exc: BaseException, limit: int = 70) -> str:
    """把异常压成一行，方便塞进字幕窗的状态栏。

    HTTP 出错的正文经常是一整坨 JSON，原样贴上去会把字幕撑爆，
    真正有用的信息（状态码）反而被挤到看不见。
    """
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "…"


class QtSink(QObject):
    """流水线回调 → Qt 信号。

    Pipeline 从工作线程调用这些方法，直接碰控件会崩；
    emit 信号让 Qt 用队列连接自动切回主线程。
    """

    sig_source = Signal(str)
    sig_delta = Signal(str)
    sig_translation = Signal(str, str, str)
    sig_status = Signal(str)
    sig_error = Signal(str)
    sig_dispatch = Signal(object)   # 把热键回调弹回主线程

    def on_source(self, text: str) -> None:
        self.sig_source.emit(text)

    def on_delta(self, text: str) -> None:
        self.sig_delta.emit(text)

    def on_translation(self, source: str, translation: str, meta: str) -> None:
        self.sig_translation.emit(source, translation, meta)

    def on_status(self, text: str) -> None:
        self.sig_status.emit(text)

    def on_error(self, text: str) -> None:
        self.sig_error.emit(text)

    @Slot(object)
    def run_dispatched(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            log.warning("派发任务异常：%s", exc)


# ---------------------------------------------------------------- 自检
# 自检要验的东西和音频版不一样：这里没有麦克风也没有模型，但有**窗口位置**这个
# 本项目独有的坑——字幕窗贴在采集区域上方，框歪了会盖住要识别的字幕，形成
# "自己吃自己"。所以自检里顺手把它也说清楚。
SELFTEST_LINES = [
    ("Alright chat, welcome back to the stream.", "好，各位，欢迎回到直播间。"),
    ("We're gonna jump straight into ranked today.", "今天咱们直接开排位吧。"),
    ("NiKo gets the clutch, one versus three.", "NiKo 拿下残局，一打三。"),
    ("Okay okay, my bad, that was completely my fault.", "行行行，我的锅，那波纯纯是我自己的问题。"),
    ("Let's just run it back, no big deal.", "再来一把就是了，不碍事。"),
]


class SelfTest:
    """不联网、不动屏幕、不认字，只验证字幕窗本身（字体、位置、穿透、自动隐藏）。"""

    def __init__(self, overlay: SubtitleOverlay):
        self.overlay = overlay
        self._i = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        note = ""
        if self.overlay.overlaps_region():
            note = "　!! 字幕窗正盖着采集区域（anchor 设错了？）"
        self.overlay.set_status("自检模式：不抓屏、不调用模型，仅验证字幕窗口" + note)
        self._timer.start(2600)

    def _tick(self) -> None:
        if self._i >= len(SELFTEST_LINES):
            self.overlay.set_status("自检结束。Ctrl+Alt+Q 或 Ctrl+C 退出。")
            self._timer.stop()
            return
        src, dst = SELFTEST_LINES[self._i]
        self._i += 1
        self.overlay.show_exchange(src, dst, "识别 0ms · 翻译 0.0s · 自检")


# ---------------------------------------------------------------- 主程序
class App:
    NOTICE_S = 6.0               # 普通提示在状态栏停留多久
    PICK_HIDE_MS = 250           # 藏起字幕窗后等多久再让用户框（等合成器真的把它抹掉）

    def __init__(self, cfg: config_mod.Config, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        self.qt_app = QApplication(sys.argv[:1])
        self.qt_app.setQuitOnLastWindowClosed(False)
        # 窗口关闭不会退出进程，但用户 Ctrl+C 要能退——见 _exec()

        self.overlay = SubtitleOverlay(cfg.overlay)
        self.overlay.set_region(cfg.region())
        self.sink = QtSink()
        self.sink.sig_source.connect(self.overlay.set_source)
        self.sink.sig_delta.connect(self.overlay.set_translation)
        self.sink.sig_translation.connect(self._on_translation)
        self.sink.sig_status.connect(self._on_status)
        self.sink.sig_error.connect(self._on_error)
        self.sink.sig_dispatch.connect(self.sink.run_dispatched)

        self.pipeline: Pipeline | None = None
        self.hotkeys: HotkeyManager | None = None
        self._error = ""            # 粘住的错误，直到有译文成功才清
        self._notice = ""           # 普通提示，过一会儿让位给实时状态行
        self._notice_deadline = 0.0
        self._has_content = False   # 屏幕上是否已经出现过字幕
        self._paused = False
        self._last_health = ""      # 上一次的链路诊断类型，只在种类变化时记日志
        self._picking = False       # 正在框选（框选期间不抓屏，免得把框选界面认进去）
        self._pick_win = None
        self._pick_restore = (False, False)   # (字幕窗之前可见吗, 之前是暂停吗)
        self._first_scan_warned = False

    def _pick_band(self):
        """要框哪块屏：鼠标在哪块屏就在哪块框，最符合直觉。"""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        return screen, (float(screen.devicePixelRatio()) if screen else 1.0) or 1.0

    # ---- 回调 ----
    @Slot(str, str, str)
    def _on_translation(self, source: str, translation: str, meta: str) -> None:
        # 出译文了，说明链路是通的——把之前的错误和提示都清掉
        self._error = ""
        self._notice = ""
        self._has_content = True
        self.overlay.set_translation(translation)
        self.overlay.set_meta(meta)

    @Slot(str)
    def _on_status(self, text: str) -> None:
        """普通提示：显示几秒，然后把位置让回实时状态行。"""
        if text:
            log.warning("状态：%s", text)
        self._notice = text or ""
        self._notice_deadline = time.time() + self.NOTICE_S if text else 0.0
        self.overlay.set_status(self._notice)

    @Slot(str)
    def _on_error(self, text: str) -> None:
        """错误：一直挂着，直到有译文成功才清。

        不做成"显示几秒就走"——那样反复出现的错误会闪，反而看不见。
        """
        if text:
            log.warning("错误：%s", text)
        self._error = text or ""
        self.overlay.set_status(self._error)

    def _check_overlay_overlap(self) -> None:
        """字幕窗有没有盖住采集区域。盖住就是**自己吃自己**：屏幕上摆着自己要读的那行字，
        OCR 读回来的全是自己的译文，输出就此卡死——而现象只是"用着用着没字幕了"，
        极难联想到是窗口摆错了地方。所以启动时和每次重新框选之后都查一次。
        """
        if self.pipeline is None or self.pipeline.region is None:
            return
        if not self.overlay.overlaps_region():
            return
        log.warning("字幕窗正盖着采集区域 %s —— 屏幕上的译文会被当成原文认回来，"
                    "请把字幕窗拖开，或把 [overlay] anchor 改成 top",
                    self.pipeline.region)
        self._on_status("字幕窗压着采集区域了，会把译文当成原文认回来 —— "
                        "把它拖开，或把 [overlay] anchor 改成 top")

    # ---- 热键动作 ----
    def _actions(self) -> dict:
        return {
            "toggle_show": self._toggle_show,
            "toggle_clickthrough": self.overlay.toggle_clickthrough,
            "toggle_original": self.overlay.toggle_original,
            "pause": self._toggle_pause,
            "repick_region": self.repick_region,
            "font_up": lambda: self.overlay.nudge_font(2),
            "font_down": lambda: self.overlay.nudge_font(-2),
            "quit": self.quit,
        }

    def _toggle_show(self) -> None:
        if self.overlay.isVisible():
            self.overlay.hide()
        else:
            self.overlay.show()

    def _toggle_pause(self) -> None:
        if self.pipeline is None:
            return
        self._paused = not self._paused
        self.pipeline.set_paused(self._paused)
        self.overlay.set_paused(self._paused)

    # ---- 重新框选 ----
    def repick_region(self) -> None:
        """按热键重新框选字幕区域。

        三个前置动作都不能省：
          - 暂停抓屏：不然框选时那层压暗的全屏遮罩会被认成字幕，翻出一堆胡说
          - 藏起字幕窗：它置顶且半透明，不藏起来就会被抓进验证帧里
            （"框完验证时看到的是自己的界面"这个坑，pickwindow 自己也躲了一次）
          - 等 250ms 再开框：hide() 只是发了个请求，窗口管理器不一定立刻把它
            从屏幕上抹掉，立刻抓屏有可能还抓得到
        """
        if self._picking:
            return
        self._picking = True
        self._pick_restore = (self.overlay.isVisible(), self._paused)
        if self.pipeline is not None:
            self.pipeline.set_paused(True)
        self.overlay.hide()
        QTimer.singleShot(self.PICK_HIDE_MS, self._open_picker)

    def _open_picker(self) -> None:
        try:
            from .pickwindow import RegionPickWindow

            screen, dpr = self._pick_band()
            win = RegionPickWindow(self.cfg, screen, dpr, verify=True)
            win.finished.connect(self._on_picked)
            self._pick_win = win
            win.show()
            win.raise_()
            win.activateWindow()
            log.info("开始重新框选（屏 %s，缩放 %g×）", screen.name() if screen else "?", dpr)
        except Exception as exc:
            log.exception("打开框选窗口失败")
            self._finish_pick()
            self._on_error(f"打开框选窗口失败：{_brief(exc)}")

    @Slot(object, bool)
    def _on_picked(self, region, picked: bool) -> None:
        try:
            if region:
                config_mod.save_region(self.cfg, tuple(region))
                if self.pipeline is not None:
                    self.pipeline.set_region(tuple(region))
                self.overlay.set_region(tuple(region))
                self._notice_msg(f"字幕区域已更新：{list(region)}")
                log.info("字幕区域已更新并保存：%s", list(region))
                # 新框的位置很可能和字幕窗撞上（尤其是重新框到屏幕顶部时），立刻查一次
                self._check_overlay_overlap()
            elif picked:
                self._notice_msg("这次框的地方没认到字，区域没有改（字幕是不是没显示？）")
            else:
                self._notice_msg("框选已取消，仍用原来的区域")
        finally:
            self._finish_pick()

    def _finish_pick(self) -> None:
        was_visible, was_paused = self._pick_restore
        win, self._pick_win = self._pick_win, None
        if win is not None:
            win.deleteLater()
        self._picking = False
        if self.pipeline is not None:
            self.pipeline.set_paused(was_paused)
        # set_paused(False) 会把状态行清成空，先把它做掉再放提示，否则提示会被清掉
        self.overlay.set_paused(was_paused)
        if was_visible:
            self.overlay.show()
        if self._notice:
            self.overlay.set_status(self._notice)

    def _notice_msg(self, text: str) -> None:
        self._notice = text
        self._notice_deadline = time.time() + self.NOTICE_S

    def quit(self) -> None:
        log.info("退出中…")
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.qt_app.quit()

    # ---- 启动 ----
    def _status_tick(self) -> None:
        """状态行的显示优先级：错误 > 提示 > 实时状态行。

        实时状态行（识别耗时/命中率）在还没出现过字幕时一直显示——那正是
        "为什么没字幕"最需要看的东西。出过字幕之后就让位，保持画面干净。

        但"干净"只在链路健康时才是好事。"用着用着没字幕了"恰恰发生在出过字幕
        之后，那时屏幕越干净越没线索。所以这里再问一句 health()，只有它确实报出
        问题才重新占用状态行；一切正常时依然什么都不显示。
        """
        if self.pipeline is None or self._picking:
            return
        if self._error:
            self.overlay.set_status(self._error)
            return
        if self._notice and time.time() < self._notice_deadline:
            self.overlay.set_status(self._notice)
            return
        self._notice = ""
        if not self._has_content:
            self.overlay.set_status(self.pipeline.status_line())
            return
        kind, problem = self.pipeline.health()
        # 只在"问题的种类"变化时记一笔：这个工具最难受的失败方式是"屏幕上什么都没了
        # 却查不出原因"，留个时间点，事后翻日志就知道是哪一刻开始坏的、哪一刻好的。
        # 比的必须是 kind 而不是提示语本身——提示语里带秒数，每秒都在变，
        # 拿它比就成了每秒刷一条日志。
        if kind != self._last_health:
            self._last_health = kind
            if problem:
                log.warning("链路诊断：%s", problem)
        self.overlay.set_status(problem)

    def _background_startup(self) -> None:
        """连通性检查和引擎预热放后台线程，别让窗口卡着不显示。

        两件事互不依赖，所以分开 try：翻译服务连不上（DeepSeek 过载时很常见）
        不该连带把识别引擎也卡住——否则字幕窗看着像整个坏了，其实读字是好的。
        """
        assert self.pipeline is not None

        # 没配 Key / 连不上，**都不是致命错误**：还有本地词典那一档兜底。
        # 音频版这里原本是 return 2 直接退出，本版刻意改了——降级链的设计
        # 就是为了让"没有网"不是一个死局。
        llm_ok = True
        try:
            self.sink.sig_status.emit("正在检查翻译服务连通性…")
            self.pipeline.check_ready()
        except LlmUnavailable as exc:
            llm_ok = False
            log.warning("翻译服务不可用，改用离线词典：%s", exc)
            self.sink.sig_status.emit(
                "翻译服务不可用，先用离线词典直译（质量粗糙）；配好 Key 后重启即可"
            )
        except Exception as exc:
            llm_ok = False
            log.exception("翻译服务连通性检查失败")
            self.sink.sig_status.emit(
                f"翻译服务连不上：{_brief(exc)}　—— 先用离线词典，恢复后自动切回"
            )

        try:
            self.sink.sig_status.emit("正在加载识别引擎（首次要加载模型文件，几秒钟）…")
            self.pipeline.warmup()
        except Exception as exc:
            log.exception("识别引擎加载失败")
            self.sink.sig_error.emit(f"识别引擎加载失败：{_brief(exc)}")
            return

        if self.pipeline.region is None:
            self.sink.sig_error.emit(
                f"还没框选字幕区域 —— 打开直播并显示英文字幕，按 "
                f"{self.cfg.hotkeys.repick_region} 框一次"
            )
            return
        self.sink.sig_status.emit(
            "就绪，正在读字幕…" if llm_ok else "识别已就绪；翻译服务恢复后会自动继续"
        )

    def run(self) -> int:
        cfg = self.cfg

        if self.args.selftest:
            SelfTest(self.overlay).start()
            return self._exec()

        use_capture = not self.args.text
        self.pipeline = Pipeline(cfg, self.sink)
        self.pipeline.start(with_capture=use_capture)
        if use_capture:
            self._check_overlay_overlap()

        if use_capture:
            threading.Thread(target=self._background_startup, daemon=True).start()
        else:
            self.overlay.set_status("文本模式：从标准输入读取每行文字并翻译（Ctrl+C 退出）")
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.hotkeys = HotkeyManager(
            cfg.hotkeys,
            self._actions(),
            dispatch=lambda fn: self.sink.sig_dispatch.emit(fn),
        )
        self.hotkeys.start()

        timer = QTimer()
        timer.timeout.connect(self._status_tick)
        timer.start(1000)

        region = self.pipeline.region
        print("live-translate-ocr 已启动（读屏幕字幕）")
        print(f"  翻译方向：{langs.display_name(cfg.translate.src_lang)} → "
              f"{langs.display_name(cfg.translate.dst_lang)}")
        print(f"  字幕区域：{list(region) if region else '还没框选！按热键框一次'}")
        print(f"  抓屏频率：{cfg.capture.fps} Hz（无字时降到 {cfg.capture.fps_idle} Hz）")
        print(f"  翻译模型：{cfg.llm.base_url} / {cfg.llm.model}"
              f"{'' if cfg.resolved_api_key() else '（没配 Key，走离线词典）'}")
        print("  热键：")
        print(help_text(cfg.hotkeys))
        print()
        return self._exec()

    def _stdin_loop(self) -> None:
        for line in sys.stdin:
            if self.pipeline is None:
                return
            self.pipeline.submit_text(line.strip())

    def _exec(self) -> int:
        # 让 Ctrl+C 能穿过 Qt 事件循环
        signal.signal(signal.SIGINT, lambda *_: self.quit())
        keepalive = QTimer()
        keepalive.start(250)
        keepalive.timeout.connect(lambda: None)
        return self.qt_app.exec()
