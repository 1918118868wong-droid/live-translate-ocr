"""最小 OCR 实验：确认能认出字幕、认得多准、要多久。

这是整个项目的第一道关：如果 OCR 认不准屏幕上的英文字幕，后面写的都白搭。
所以这个脚本只做一件事——拿一张图，跑 OCR，把结果和耗时打出来。

它**直接用正式代码的引擎**（livetranslate_ocr/ocr.py），所以这里测出来的耗时
就是实际使用时的耗时，不会出现"探针很快、正式程序很慢"的落差。

用法：
    python tools\\probe_ocr.py                  # 用合成的字幕图（不需要直播也能跑）
    python tools\\probe_ocr.py 某张截图.png      # 用真实截图（推荐：直播时截一张底部的字幕条）

合成图模拟 YouTube 自动字幕的样子：亮色跳动的画面上，白字 + 半透明黑底。
真实截图更能说明问题，但合成图至少能在没开直播时验证链路通不通。
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 要测的两句话。第一句是最典型的 CS 解说；第二句带弯引号——
# YouTube 自动字幕用的是 ' (U+2019)，OCR 把它认成什么会影响后面的归一化。
TEST_LINES = [
    "and he's picked up the AWP, that is not a good sign",
    "NiKo with the deagle, what a clutch! Let's go baby",
]

# 模拟字幕条。单行紧框（1600x80）是实测最省的一种（124ms），
# 也是推荐用户框成的样子，所以主要测这个尺寸。
STRIP_SIZES = [(1600, 80), (1600, 130)]   # 单行 / 两行滚动


def norm(s: str) -> str:
    """归一化：只看"认对了哪些字"，忽略大小写和标点差异。

    比对时用这个而不是直接比字符串——否则 OCR 把逗号认成句号也算错，
    但那对下游毫无影响，只会让实验结论变得悲观。
    """
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def make_caption_image(texts: list[str], width: int, height: int, path: Path) -> None:
    """画一条"字幕条"：噪点背景 + 半透明黑底 + 白字。

    故意加噪点和高频条纹，因为真实直播字幕底下是跳动的画面。
    拿纯黑底的图测会得满分，但说明不了任何问题。
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(42)
    bg = rng.integers(40, 210, size=(height, width, 3), dtype=np.uint8)
    for y in range(0, height, 7):   # 横向条纹：最容易干扰 OCR 的那种高频细节
        bg[y:y + 3, :, :] = (bg[y:y + 3, :, :] // 2 + 90).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    font_size = max(20, min(38, height // 3))
    font = ImageFont.truetype("arialbd.ttf", font_size)

    # 多行时从下往上摞，模拟 YouTube 的滚动字幕
    cursor = height // 2 - (len(texts) * font_size) // 2
    for text in texts:
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = (width - tw) // 2
        draw.rectangle([x - 14, cursor - box[1] - 8, x + tw + 14, cursor - box[1] + th + 8],
                       fill=(8, 8, 8, 200))
        draw.text((x, cursor - box[1]), text, font=font, fill=(255, 255, 255, 255))
        cursor += font_size + 8
    img.save(path)


def main() -> int:
    print("--- 依赖共存检查 ---")
    # 最关键的风险先验：opencv(RapidOCR 的依赖) 和 PySide6 都带 Qt，历史上会互踩。
    # 装的时候已换成 headless 版 opencv 绕开，但"装完立刻验证"才算数，
    # 而且这个 import 组合正是最终程序启动时要跑的。
    try:
        import cv2
        print(f"cv2 {cv2.__version__} 导入成功")
    except Exception as e:
        print(f"cv2 导入失败: {e}")
        return 1
    try:
        import PySide6
        from PySide6.QtWidgets import QApplication  # noqa: F401
        print(f"PySide6 {PySide6.__version__} 导入成功")
    except Exception as e:
        print(f"PySide6 导入失败: {e}")
        return 1
    # headless 版和完整版装的是同一个 cv2/ 目录，靠"有没有 Qt 插件"区分——
    # 看路径里有没有 "headless" 是无效的（踩过这个坑）。
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("opencv-python")
        print("!! 装了带 Qt 的 opencv-python，会和 PySide6 抢 Qt 插件；应改用 headless 版")
    except PackageNotFoundError:
        print(f"opencv-python-headless {version('opencv-python-headless')}（正确的那个）")
    print("两个库可以同进程共存。")

    from livetranslate_ocr.config import OcrCfg
    from livetranslate_ocr.ocr import TUNED_PARAMS, OcrEngine

    print("\n--- 建引擎 ---")
    for key, value in TUNED_PARAMS.items():
        print(f"  {key} = {value!r}")
    t0 = time.perf_counter()
    engine = OcrEngine(OcrCfg())
    print(f"建引擎耗时 {time.perf_counter() - t0:.1f}s（含加载 ONNX 模型）")

    # 有参数就认那张真实截图，否则用合成图
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"图片不存在: {target}")
            return 1
        return _run_image(engine, target)

    tmp = PROJECT_ROOT / "tmp"
    tmp.mkdir(exist_ok=True)
    all_ok = True
    for width, height in STRIP_SIZES:
        # 单行图上放第一句；两行图放"上一句 + 当前句"，模拟滚动
        texts = TEST_LINES[:1] if height < 120 else TEST_LINES
        source = " ".join(texts)
        path = tmp / f"probe_{width}x{height}.png"
        make_caption_image(texts, width, height, path)
        print(f"\n--- 合成字幕图 {width}x{height} ---")
        print(f"  构造原文: {source!r}")
        got = _run_image(engine, path, source)
        all_ok = all_ok and got

    print(f"\n实验完成：{'全部一致' if all_ok else '有差异，见上'}")
    return 0


def _run_image(engine, path: Path, source: str | None = None) -> bool:
    """对一张图跑 5 次取中位数（丢掉第一次，避免把 ONNX 首次分配的开销算进去）。"""
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(path).convert("RGB"))

    engine.read(arr, t=0.0)   # 热身
    times = []
    reading = None
    for _ in range(5):
        t0 = time.perf_counter()
        reading = engine.read(arr, t=t0)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = times[len(times) // 2]

    for line in reading.lines:
        print(f"  [{line.score:.3f}] y={line.y:7.1f}  {line.text!r}")
    if reading.empty:
        print("  (什么都没读到)")
    print(f"  中位数耗时 {median:.0f} ms（5 次：{[f'{t:.0f}' for t in times]}）")

    if source is None:
        return True
    ok = norm(reading.text) == norm(source)
    print(f"  比对: {'一致' if ok else '不一致'}")
    if not ok:
        print(f"    want: {norm(source)!r}")
        print(f"    got : {norm(reading.text)!r}")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
