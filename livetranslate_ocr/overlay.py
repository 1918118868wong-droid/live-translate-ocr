"""桌面置顶字幕浮窗（PySide6）。

要同时满足几个互相打架的要求，取舍如下：
  - 「置顶」用 WindowStaysOnTopHint + Tool（Tool 让它不出现在任务栏和 Alt+Tab）
  - 「不抢焦点」用 WindowDoesNotAcceptFocus + WA_ShowWithoutActivating，
    否则点一下字幕就把全屏游戏/视频切出去了
  - 「能拖动」和「能点穿」是互斥的：点击穿透开着的时候鼠标事件根本收不到。
    所以做成可切换的——热键临时关掉穿透挪位置，挪好再打开
  - 空闲时直接 hide() 而不是把背景画成透明：透明窗口在非穿透状态下依然会
    吃掉鼠标点击，挡住下面的播放器控件
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from .config import OverlayCfg

log = logging.getLogger(__name__)

_BASE_FLAGS = (
    Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowStaysOnTopHint
    | Qt.WindowType.Tool
    | Qt.WindowType.WindowDoesNotAcceptFocus
)

# 配置里的字体在有些系统上不存在（比如英文版 Windows 没有 Microsoft YaHei UI），
# 逐个往后找一个装了的。都没有就交回 Qt 默认字体，而不是画出一堆方块。
_FONT_FALLBACKS = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Segoe UI",
    "Arial",
)


def resolve_font_family(requested: str) -> str:
    try:
        available = set(QFontDatabase.families())
    except Exception:  # pragma: no cover
        return requested
    if not available:
        return requested          # 拿不到字体列表（如无头环境），原样交给 Qt
    if requested in available:
        return requested
    for candidate in _FONT_FALLBACKS:
        if candidate in available:
            log.warning("字体 %r 不存在，改用 %r", requested, candidate)
            return candidate
    log.warning("字体 %r 及其备选都不可用，使用系统默认字体", requested)
    return ""


class SubtitleOverlay(QWidget):
    def __init__(self, cfg: OverlayCfg):
        super().__init__(None)
        self.cfg = cfg
        self._clickthrough = cfg.start_clickthrough
        self._show_original = cfg.show_original
        self._paused = False

        self._src_text = ""
        self._dst_text = ""
        self._meta = ""
        self._status = ""
        self._font_size = float(cfg.font_size)
        self._orig_font_size = float(cfg.orig_font_size)
        self._user_position: QPoint | None = None
        self._drag_origin: QPoint | None = None
        self._content_shown = False
        self._font_family = resolve_font_family(cfg.font_family)
        # 采集区域（物理像素）。用来把字幕窗贴在它的正上方，免得盖住要识别的字幕。
        # 由 App 在启动时和每次重新框选后 set_region() 喂进来。
        self._region: tuple[int, int, int, int] | None = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("live-translate-ocr 字幕")

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._on_idle)

        self._apply_flags()
        self._relayout()
        self.show()

    # ---- 窗口标志 ----
    def _apply_flags(self) -> None:
        flags = _BASE_FLAGS
        if self._clickthrough:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)

    def set_clickthrough(self, enabled: bool) -> None:
        if enabled == self._clickthrough:
            return
        self._clickthrough = enabled
        self._apply_flags()
        # setWindowFlags 会隐藏窗口，必须重新 show
        self.show()
        self._relayout()
        log.info("点击穿透：%s", "开" if enabled else "关")

    def toggle_clickthrough(self) -> None:
        self.set_clickthrough(not self._clickthrough)

    # ---- 位置 ----
    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        """告诉字幕窗采集区域在哪（物理像素），它会乖乖待在这块区域外面。"""
        self._region = region
        if self._user_position is None:
            self._relayout()
            self.update()

    def _dpr(self) -> float:
        """屏幕缩放比。采集区域是物理像素，Qt 的坐标是逻辑像素，两者要靠它换算。

        本机实测 2560×1440、DPR=1.0（无缩放），但用户改成 125% 或者接第二块屏
        就不一样了——不换算的话字幕窗会跑到别的地方去。
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        try:
            return float(screen.devicePixelRatio()) or 1.0
        except Exception:  # pragma: no cover
            return 1.0

    def overlaps_region(self) -> bool:
        """字幕窗现在有没有盖住采集区域（盖住就等于自己吃自己）。"""
        if self._region is None:
            return False
        x, y, w, h = self._region
        scale = self._dpr()
        region = QRect(int(x / scale), int(y / scale),
                       max(1, int(w / scale)), max(1, int(h / scale)))
        return self.geometry().intersects(region)

    def _anchor_y(self, geo: QRect) -> int:
        """算出纵向位置。"""
        anchor = (self.cfg.anchor or "bottom").strip().lower()
        if anchor == "top":
            return geo.y()
        if anchor == "above_region" and self._region is not None:
            _, ry, _, rh = self._region
            scale = self._dpr()
            above = int(ry / scale) - self.cfg.anchor_gap - self.height()
            if above >= geo.y():
                return above
            # 上面塞不下（字幕带贴着屏幕顶）——那也别盖住它，改贴到它下面去。
            # 盖住采集区域是最坏的结果：字幕被自己挡住 → 读不到 → 永远没输出，
            # 而且现象是"用着用着突然没字幕了"，极难联想到是窗口位置。
            below = int((ry + rh) / scale) + self.cfg.anchor_gap
            if below + self.height() <= geo.y() + geo.height():
                return below
            log.warning("字幕窗放不下：采集区域 %s 几乎占满屏幕，只能贴屏幕顶",
                        self._region)
            return geo.y()
        return geo.y() + geo.height() - self.height() - self.cfg.bottom_margin

    # ---- 布局 ----
    def _screen_geometry(self) -> QRect:
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)

    def _text_height(self, text: str, font: QFont, width: int) -> int:
        if not text:
            return 0
        fm = QFontMetrics(font)
        return fm.boundingRect(
            QRect(0, 0, max(1, width), 10000),
            int(Qt.TextFlag.TextWordWrap),
            text,
        ).height()

    def _relayout(self) -> None:
        geo = self._screen_geometry()
        inner_w = int(geo.width() * self.cfg.width_ratio)
        pad = self.cfg.padding

        h = pad * 2
        if self._show_original and self._src_text:
            h += self._text_height(self._src_text, self._orig_font(), inner_w - pad * 2)
            h += self.cfg.line_spacing
        h += self._text_height(self._dst_text, self._main_font(), inner_w - pad * 2)
        if self._meta:
            h += self.cfg.line_spacing + self._text_height(self._meta, self._meta_font(), inner_w - pad * 2)
        if self._status:
            h += self.cfg.line_spacing + self._text_height(self._status, self._meta_font(), inner_w - pad * 2)

        self.resize(inner_w, max(h, pad * 2 + 8))
        if self._user_position is not None:
            # 用户拖过就不再自作主张——拖到哪就待在哪
            self.move(self._user_position)
        else:
            x = geo.x() + (geo.width() - inner_w) // 2
            self.move(x, self._anchor_y(geo))

    # ---- 字体 ----
    def _main_font(self) -> QFont:
        f = QFont(self._font_family, int(self._font_size))
        f.setWeight(QFont.Weight.DemiBold)
        f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return f

    def _orig_font(self) -> QFont:
        return QFont(self._font_family, int(self._orig_font_size))

    def _meta_font(self) -> QFont:
        return QFont(self._font_family, max(9, int(self._orig_font_size * 0.72)))

    # ---- 内容接口（由主线程调用）----
    def set_source(self, text: str) -> None:
        self._src_text = text or ""
        self._show_content()

    def set_translation(self, text: str) -> None:
        self._dst_text = text or ""
        self._show_content()

    def set_meta(self, text: str) -> None:
        self._meta = (text or "") if self.cfg.show_latency else ""
        self._relayout()
        self.update()

    def set_status(self, text: str) -> None:
        """状态/错误提示，显示在字幕下方的小字。"""
        self._status = text or ""
        if self._status:
            self._show_content()
        else:
            self._relayout()
            self.update()

    def show_exchange(self, source: str, translation: str, meta: str = "") -> None:
        self._src_text = source or ""
        self._dst_text = translation or ""
        self._meta = meta if self.cfg.show_latency else ""
        self._show_content()

    def clear_content(self) -> None:
        self._src_text = self._dst_text = self._meta = ""
        self._relayout()
        self.update()

    def _show_content(self) -> None:
        if self._paused:
            return
        self._content_shown = True
        self._relayout()
        if not self.isVisible():
            self.show()
        self.update()
        self._idle_timer.start(int(self.cfg.hide_after_s * 1000))

    def _on_idle(self) -> None:
        if self._status:
            return
        self._content_shown = False
        self.hide()

    # ---- 开关 ----
    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self.set_status("已暂停（热键继续）")
        else:
            self.set_status("")
        log.info("字幕%s", "已暂停" if paused else "已恢复")

    def toggle_original(self) -> None:
        self._show_original = not self._show_original
        self._relayout()
        self.update()

    def nudge_font(self, delta: int) -> None:
        self._font_size = max(12.0, min(72.0, self._font_size + delta))
        self._orig_font_size = max(9.0, min(40.0, self._orig_font_size + delta * 0.6))
        self._relayout()
        self.update()

    # ---- 绘制 ----
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if not self._content_shown and not self._status:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = self.rect()
        radius = 12.0
        alpha = int(max(0.0, min(1.0, self.cfg.opacity)) * 255)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 10, 14, alpha))
        painter.drawRoundedRect(rect, radius, radius)

        pad = self.cfg.padding
        inner_w = rect.width() - pad * 2
        y = pad
        flags = int(Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap)

        # 原文（暗一点，作为对照）
        if self._show_original and self._src_text:
            painter.setFont(self._orig_font())
            h = self._text_height(self._src_text, self._orig_font(), inner_w)
            painter.setPen(QColor(178, 186, 200, 235))
            painter.drawText(QRect(pad, y, inner_w, h), flags, self._src_text)
            y += h + self.cfg.line_spacing

        # 译文（主视觉，带一层深色描边提升在亮背景上的可读性）
        if self._dst_text:
            painter.setFont(self._main_font())
            h = self._text_height(self._dst_text, self._main_font(), inner_w)
            box = QRect(pad, y, inner_w, h)
            shadow = QColor(0, 0, 0, 200)
            painter.setPen(shadow)
            for dx, dy in ((1, 1), (1, 0), (0, 1)):
                painter.drawText(box.translated(dx, dy), flags, self._dst_text)
            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(box, flags, self._dst_text)
            y += h + self.cfg.line_spacing

        painter.setFont(self._meta_font())
        if self._meta:
            h = self._text_height(self._meta, self._meta_font(), inner_w)
            painter.setPen(QColor(150, 160, 175, 220))
            painter.drawText(
                QRect(pad, y, inner_w, h),
                int(Qt.AlignmentFlag.AlignRight | Qt.TextFlag.TextWordWrap),
                self._meta,
            )
            y += h + self.cfg.line_spacing

        if self._status:
            h = self._text_height(self._status, self._meta_font(), inner_w)
            painter.setPen(QColor(255, 196, 120, 235))
            painter.drawText(QRect(pad, y, inner_w, h), flags, self._status)

        painter.end()

    # ---- 鼠标：拖动窗口 ----
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._user_position = self.pos()
            self._drag_origin = None
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        # 双击切换穿透，方便不用热键也能解锁拖动
        self.toggle_clickthrough()
        event.accept()
