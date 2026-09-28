"""全局热键。

用 keyboard 库而不是 Qt 的 QShortcut：字幕窗口是 WindowDoesNotAcceptFocus 的，
永远不会获得键盘焦点，Qt 的快捷键在这种情况下收不到事件。

注意：keyboard 的回调跑在自己的线程里，绝不能直接碰 Qt 控件。
调用方必须传一个 dispatch 函数，把回调弹回 Qt 主线程执行。
"""

from __future__ import annotations

import logging
from typing import Callable

from .config import HotkeyCfg

log = logging.getLogger(__name__)


class HotkeyManager:
    def __init__(
        self,
        cfg: HotkeyCfg,
        actions: dict[str, Callable[[], None]],
        dispatch: Callable[[Callable[[], None]], None],
    ):
        self.cfg = cfg
        self.actions = actions
        self.dispatch = dispatch
        self._keyboard = None
        self._registered: list[str] = []

    def start(self) -> bool:
        if not self.cfg.enabled:
            log.info("热键已禁用")
            return False
        try:
            import keyboard  # noqa: PLC0415
        except ImportError:
            log.warning("未安装 keyboard 库，全局热键不可用（字幕仍可用鼠标拖动）")
            return False

        self._keyboard = keyboard
        for name, action in self.actions.items():
            combo = getattr(self.cfg, name, "")
            if not combo:
                continue
            try:
                keyboard.add_hotkey(combo, self._wrap(action), suppress=False)
                self._registered.append(f"{combo} → {name}")
            except Exception as exc:
                log.warning("注册热键 %s 失败：%s", combo, exc)

        if self._registered:
            log.info("热键已注册：\n  " + "\n  ".join(self._registered))
            return True
        return False

    def _wrap(self, action: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                self.dispatch(action)
            except Exception as exc:  # pragma: no cover
                log.warning("热键回调异常：%s", exc)

        return run

    def stop(self) -> None:
        if self._keyboard is None:
            return
        try:
            self._keyboard.unhook_all_hotkeys()
        except Exception:  # pragma: no cover
            pass
        self._keyboard = None


def help_text(cfg: HotkeyCfg) -> str:
    return (
        f"  {cfg.toggle_show:<16} 显示 / 隐藏字幕\n"
        f"  {cfg.toggle_clickthrough:<16} 切换鼠标点击穿透（穿透后才能点到下面的窗口）\n"
        f"  {cfg.toggle_original:<16} 显示 / 隐藏原文\n"
        f"  {cfg.pause:<16} 暂停 / 继续翻译\n"
        f"  {cfg.font_up:<16} 放大字号\n"
        f"  {cfg.font_down:<16} 缩小字号\n"
        f"  {cfg.quit:<16} 退出"
    )
