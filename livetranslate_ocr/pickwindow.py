"""框选用的 Qt 窗口（全屏半透明，拖拽画矩形）。

单独成模块的理由和 overlay.py / ui.py 一样：这里是纯 Qt 的东西，
而 regionpick.py 里"存配置、解析 --region"那些逻辑没有图形库也该能用。

窗口分两个阶段：

1. **框**：整屏压暗，拖出一个矩形。Esc 取消。
2. **验**：藏起自己，抓几帧真跑一次识别，然后把**抓到的画面**和**认出来的字**
   一起摆在屏幕中央。看到字，就知道框对了。

第 2 步是这个窗口存在的意义。框歪了在程序里是静默的——照样运行、照样没输出，
用户只能开着直播干等，然后怀疑是程序坏了。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .config import Config

log = logging.getLogger(__name__)

# 太小的框基本是误点，不算数（逻辑像素）
MIN_W, MIN_H = 24, 12

HUD_FONT = "Microsoft YaHei UI"


class RegionPickWindow(QWidget):
    # 窗口关掉了。热键重框那条路上，主界面要靠它知道"该读结果、收尾了"——
    # Qt 没有现成的 closed 信号（那是 QML 的），自己发一个。
    # 参数是 (区域, 有没有真的框过)：
    #   (region, True)  → 框了也验过了，存下来用
    #   (None,   True)  → 框了但一个字都没认出来，**别存**（存下去等于留个坏配置），
    #                     提示语要和"取消"分开，不然用户不知道是自己框错了
    #   (None,   False) → Esc / 没拖出框就关了，纯取消
    finished = Signal(object, bool)

    def __init__(self, cfg: Config, screen, dpr: float, verify: bool = True):
        super().__init__(None)
        self.cfg = cfg
        self._screen = screen
        self._dpr = dpr or 1.0
        self._verify = verify

        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._phase = "pick"          # pick | busy | verify
        self._note = ""
        self._text = ""
        self._thumb: QImage | None = None
        self._ok = False
        self._picked = False        # 有没有真的拖出过一个像样的框（区分取消）

        # 结果，供调用方读取
        self.result_region: tuple[int, int, int, int] | None = None
        self.verified_text = ""

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setWindowTitle("框选字幕区域")
        self.setGeometry(screen.geometry())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---------------------------------------------------------------- 画
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()

        if self._phase == "verify":
            p.fillRect(rect, QColor(0, 0, 0, 205))
            self._paint_result(p, rect)
            return

        p.fillRect(rect, QColor(0, 0, 0, 110))
        band = self._band()
        if band is not None:
            # 选中的地方提亮，让用户看清框到了什么
            p.fillRect(band, QColor(90, 190, 255, 60))
            p.setPen(QPen(QColor(90, 190, 255, 235), 2))
            p.drawRect(band)
        self._paint_hud(p, rect, band)

    def _paint_hud(self, p: QPainter, rect: QRect, band: QRect | None) -> None:
        p.setPen(QColor(255, 255, 255, 240))
        p.setFont(QFont(HUD_FONT, 15))
        head = "拖动鼠标框住屏幕上的英文字幕（只框字幕本身）"
        p.drawText(rect.adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignHCenter, head)

        p.setFont(QFont(HUD_FONT, 11))
        p.setPen(QColor(200, 200, 200, 220))
        p.drawText(rect.adjusted(0, 72, 0, 0), Qt.AlignmentFlag.AlignHCenter,
                   "别把进度条、标题、CC 按钮框进来　·　Esc 取消")

        if band is not None:
            phys = self._to_physical(band)
            p.setFont(QFont(HUD_FONT, 11))
            p.setPen(QColor(160, 220, 255, 240))
            label = (f"x={phys[0]}  y={phys[1]}  宽={phys[2]}  高={phys[3]}"
                     f"（物理像素，缩放 {self._dpr:g}×）")
            p.drawText(band.adjusted(0, -22, 0, 0),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, label)

    def _paint_result(self, p: QPainter, rect: QRect) -> None:
        panel_w = int(rect.width() * 0.72)
        panel = QRect(0, 0, panel_w, 10)
        lines = self._text.splitlines() or ["（没认到字）"]
        line_h = 30
        thumb_h = 0
        if self._thumb is not None:
            thumb_h = int(self._thumb.height() * min(1.0, panel_w * 0.9 / self._thumb.width()))
        panel.setHeight(150 + line_h * len(lines) + thumb_h)

        panel.moveTo(rect.center().x() - panel_w // 2,
                     rect.center().y() - panel.height() // 2)
        p.setBrush(QColor(24, 26, 30, 242))
        p.setPen(QPen(QColor(90, 190, 255, 200), 2))
        p.drawRoundedRect(panel, 10, 10)

        x = panel.x() + 24
        y = panel.y() + 30
        p.setPen(QColor(255, 255, 255, 245))
        p.setFont(QFont(HUD_FONT, 14, QFont.Weight.DemiBold))
        p.drawText(x, y, "框选结果" + ("　—— 看得到字就是框对了" if self._ok else ""))

        y += 34
        p.setFont(QFont(HUD_FONT, 10))
        p.setPen(QColor(190, 195, 200, 230))
        p.drawText(x, y, self._note)

        if self._thumb is not None:
            y += 12
            tw = panel_w - 48
            th = int(self._thumb.height() * tw / self._thumb.width())
            p.drawImage(QRect(x, y, tw, th), self._thumb)
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
            p.drawRect(QRect(x, y, tw, th))
            y += th

        y += 34
        p.setFont(QFont(HUD_FONT, 17, QFont.Weight.DemiBold))
        p.setPen(QColor(160, 240, 170) if self._ok else QColor(255, 150, 150))
        for line in lines:
            p.drawText(x, y, line)
            y += line_h

        p.setFont(QFont(HUD_FONT, 10))
        p.setPen(QColor(170, 175, 180, 220))
        p.drawText(panel.adjusted(0, 0, 0, -14),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                   "按任意键关闭" if self._ok else "按任意键关闭（换个位置再框：Ctrl+Alt+R）")

    # ---------------------------------------------------------------- 鼠标
    def _band(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def _to_physical(self, band: QRect) -> tuple[int, int, int, int]:
        geo = self._screen.geometry()
        return (int(round((geo.x() + band.x()) * self._dpr)),
                int(round((geo.y() + band.y()) * self._dpr)),
                max(1, int(round(band.width() * self._dpr))),
                max(1, int(round(band.height() * self._dpr))))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._phase != "pick":
            self.close()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._phase != "pick" or self._origin is None:
            return
        self._current = event.position().toPoint()
        band = self._band()
        if band is None or band.width() < MIN_W or band.height() < MIN_H:
            # 误点一下：不当作框选，继续等
            self._origin = self._current = None
            self.update()
            return
        region = self._to_physical(band)
        self._picked = True
        log.info("框选到区域 %s（物理像素）", region)
        if not self._verify:
            self.result_region = region
            self.close()
            return
        self._run_verify(region)

    # ---------------------------------------------------------------- 验证
    def _run_verify(self, region: tuple[int, int, int, int]) -> None:
        """藏起自己 → 抓帧识别 → 把画面和文字摆出来。"""
        self._phase = "busy"
        self._note = "正在抓一帧识别，验证框得对不对…"
        self.repaint()
        QGuiApplication.processEvents()
        self.hide()          # 关键：不藏起来就会把自己的界面抓进去
        QGuiApplication.processEvents()

        from .regionpick import _grab_and_read   # 延迟导入：避免和 regionpick 循环 import

        try:
            text, note, frame = _grab_and_read(region, self.cfg.capture.monitor)
        except Exception as exc:      # 抓屏/识别出问题也不能把窗口卡住
            log.warning("框选验证失败：%s", exc)
            text, note, frame = "", f"验证时出错：{exc}", None

        self._text = text
        self._note = note
        self._ok = bool(text)
        self._thumb = _to_qimage(frame)
        if text:
            self.verified_text = text
            self.result_region = region
        self._phase = "verify"
        self.show()
        self.raise_()
        self.activateWindow()
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.result_region = None
            self.verified_text = ""
            self.close()
            return
        if self._phase == "verify":
            self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
        # 只在"pick 阶段就关掉"时才算取消；verify 阶段已经存下 result_region 了，
        # 这时关窗口是确认，不能反过来把结果清成 None。
        if self._phase == "pick":
            self.result_region = None
            self.verified_text = ""
        self.finished.emit(self.result_region, self._picked)


def _to_qimage(frame) -> QImage | None:
    """numpy RGB → QImage。

    必须 .copy()：QImage 包着 numpy 的裸内存不复制数据，等这个数组被回收
    再画就是读野指针——症状是偶发花屏或直接崩，而且看不出跟哪一行有关。
    """
    if frame is None:
        return None
    try:
        h, w, _ = frame.shape
        img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return img.copy()
    except Exception as exc:  # pragma: no cover
        log.debug("缩略图转换失败：%s", exc)
        return None
