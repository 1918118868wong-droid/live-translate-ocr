"""扫描加速参数：找出"够准 + 够快"的组合。

背景：默认配置对字幕条这种"又宽又扁"的图非常不友好——
Det.limit_type 默认是 min，意思是"把短边撑到 736"，于是 1600x220 会被
放大成 5353x736（390 万像素）再送进检测网络，稳定耗时约 1 秒。
而我们要 5Hz 轮询（每帧预算 200ms）。

这个脚本实测几种缩放策略的耗时和准确率，把选择建立在数字上而不是猜测上。

用法：python tools\\probe_sweep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINES = [
    "and he's picked up the AWP, that is not a good sign",
    "NiKo with the deagle, what a clutch! Let's go baby",
]


def make_strip(text: str, width: int, height: int) -> "object":
    """造一条字幕图，尺寸可调（用来模拟用户框得紧一点还是松一点）。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(7)
    bg = rng.integers(50, 180, size=(height, width, 3), dtype=np.uint8)
    for y in range(0, height, 9):
        bg[y:y + 3, :, :] = (bg[y:y + 3, :, :] // 2 + 70).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype("arialbd.ttf", 36)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = (width - tw) // 2, (height - th) // 2
    d.rectangle([x - 14, y - box[1] - 8, x + tw + 14, y - box[1] + th + 8],
                fill=(8, 8, 8, 200))
    d.text((x, y - box[1]), text, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", s.replace("’", "'").lower()).strip()


def bench(engine, arr, n: int = 5) -> tuple[float, list[str]]:
    """跑 n 次取中位数（第一次当热身丢掉，避免把 ONNX 首次分配的开销算进去）。"""
    engine(arr)
    times = []
    texts: list[str] = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = engine(arr)
        times.append((time.perf_counter() - t0) * 1000)
        texts = list(getattr(result, "txts", None) or [])
    times.sort()
    return times[len(times) // 2], texts


def stage_times(engine, arr) -> str:
    """把检测/识别分开计时，看清时间花在哪一段。"""
    engine(arr)  # 热身
    try:
        t0 = time.perf_counter()
        boxes, _ = engine.detect_and_crop(arr) if isinstance(
            engine.detect_and_crop(arr), tuple) else (engine.detect_and_crop(arr), None)
        det = (time.perf_counter() - t0) * 1000
        return f"检测 {det:.0f}ms"
    except Exception as e:
        return f"(分阶段计时失败: {type(e).__name__}: {e})"


def main() -> int:
    from rapidocr import RapidOCR

    # 先把要用的枚举取出来
    try:
        from rapidocr.utils.typings import LangRec
        en = LangRec.EN
    except Exception:
        en = None

    strips = {
        "1600x80 (单行紧框)": make_strip(LINES[0], 1600, 80),
        "1600x220 (框松)": make_strip(LINES[0], 1600, 220),
        "2560x140 (全宽紧框)": make_strip(LINES[0], 2560, 140),
    }

    # 待测的配置组合。每个都写成"扁平点号键"，这是 RapidOCR 3.x 的传参格式
    # （嵌套字典会报 "Global is not a valid key"）。
    cases = [
        ("默认（limit_type=min, 736）", {}),
        ("max/1536", {"Det.limit_type": "max", "Det.limit_side_len": 1536}),
        ("max/1536 + 关方向分类器",
         {"Det.limit_type": "max", "Det.limit_side_len": 1536, "Global.use_cls": False}),
        ("↑ + 关纵向补白",
         {"Det.limit_type": "max", "Det.limit_side_len": 1536, "Global.use_cls": False,
          "Global.use_vertical_padding": False}),
        ("max/1280 + 关分类器 + 关补白",
         {"Det.limit_type": "max", "Det.limit_side_len": 1280, "Global.use_cls": False,
          "Global.use_vertical_padding": False}),
        ("↑ + 英文识别（需联网下模型）",
         {"Det.limit_type": "max", "Det.limit_side_len": 1280, "Global.use_cls": False,
          "Global.use_vertical_padding": False, **({"Rec.lang_type": en} if en else {})}),
    ]

    print("每个配置对每种尺寸取 5 次的中位数耗时（ms）")
    print("准确率：认出文本归一化后与原文完全一致 = ✓\n")

    header = f"{'配置':34s}" + "".join(f"{k:>20s}" for k in strips)
    print(header)
    print("-" * len(header))

    for label, params in cases:
        try:
            engine = RapidOCR(params=params)
        except Exception as e:
            print(f"{label:34s}  建引擎失败: {type(e).__name__}: {e}")
            continue

        row = f"{label:34s}"
        for size_name, arr in strips.items():
            try:
                ms, texts = bench(engine, arr)
                ok = "✓" if norm(" ".join(texts)) == norm(LINES[0]) else "✗"
                row += f"{ms:>14.0f}ms {ok}"
            except Exception as e:
                row += f"{'失败:' + type(e).__name__:>20s}"
        print(row)

    print("\n--- 时间花在哪（用默认配置，1600x220） ---")
    e = RapidOCR()
    print(" ", stage_times(e, strips["1600x220 (框松)"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
