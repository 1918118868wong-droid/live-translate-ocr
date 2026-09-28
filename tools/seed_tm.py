"""预热本地记忆库：开箱即有秒回，不花一分钱额度、不联网。

记忆库的命中是**零延迟**的（内存索引 + rapidfuzz），而模型翻译要 1~2 秒。
直播间里反复出现的就是那么几十句话——"感谢打赏"、"最后一把"、"这波太秀了"
——先把它们灌进去，第一批字幕就不必等模型。

两个来源，都可以单独关掉：

1. `data/common.tsv`（第二区词典，320 条英中对照）
   本来就是"英文 <TAB> 中文"的两列表，和记忆库要的东西一模一样，直接灌。
   离线、不依赖任何别的东西。

2. **老项目 `live-translate` 的 `data/tm.sqlite3`**（可选，自动探测）
   里面除了同样的词典条目，还有 100 多条**模型真正翻过的句子**
   （"Chat, stop backseating, I know what I'm doing." → "弹幕别指挥了，我知道自己在干嘛。"）。
   这些是已经付过费的成果，白扔可惜。老项目本身一个字节都不会被改动——
   这里是只读它。

**必须过滤老库里的投毒条目**：老库里躺着 2 条 `'Thanks for the donation.' → '[译] Thanks for the donation.'`
这种被截断的占位符。它们是最早期那个 bug 留下的（模型流断在中途，
半成品被当成译文写进了缓存）。要是原样导过来，以后每次出现这句都会秒回那个烂译文，
而且**网络明明是好的**——这类问题极难排查。所以导入前逐条验一遍质量。

    python tools\\seed_tm.py                 导两个来源
    python tools\\seed_tm.py --no-old        只从 common.tsv 导（不碰老项目）
    python tools\\seed_tm.py --dry-run       只看会导多少条，不写
    python tools\\seed_tm.py --from-old PATH 手动指定老库位置
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from livetranslate_ocr.config import Config  # noqa: E402
from livetranslate_ocr.tm import TranslationMemory, variant_tag  # noqa: E402

# 老库通常就在隔壁（两个项目都在"工作资料文件夹"下）
DEFAULT_OLD_DB = PROJECT_ROOT.parent / "live-translate" / "data" / "tm.sqlite3"

_CJK = re.compile(r"[一-鿿]")
# 这些前缀是模型把"翻译"这件事本身写进了输出，不是译文
_PLACEHOLDER_PREFIX = ("[译]", "[翻译]", "译文：", "译文:", "Translation:")


def looks_wrong(src: str, dst: str) -> str:
    """这条译文能不能用。返回空串表示没问题，否则返回不能用的原因（中文）。"""
    d = (dst or "").strip()
    s = (src or "").strip()
    if not d:
        return "空译文"
    if d.startswith(_PLACEHOLDER_PREFIX):
        return "占位符（模型没真翻）"
    if not _CJK.search(d):
        return "一个中文字都没有"
    # 原文照抄：把标点抹掉再比，免得 "…donation." 和 "…donation" 被当成两样
    strip = lambda t: re.sub(r"[^\w]+", "", t).lower()  # noqa: E731
    if strip(d) == strip(s):
        return "把原文抄了一遍"
    return ""


def read_common_tsv(path: Path) -> list[tuple[str, str]]:
    """第二区词典就是现成的英中对照，直接当种子。"""
    out: list[tuple[str, str]] = []
    if not path.is_file():
        print(f"  [!] 找不到 {path}，跳过这个来源")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
            continue
        src, _, dst = line.partition("\t")
        src, dst = src.strip(), dst.strip()
        if src and dst:
            out.append((src, dst))
    return out


def read_old_db(path: Path) -> tuple[list[tuple[str, str]], Counter]:
    """读老项目的记忆库。**只读**，不写、不改。

    老库里 `src_lang` 有的是 `auto`（那时候源语言是自动检测的）。本项目屏幕上
    永远是英文字幕，所以一律按 `en` 收——否则这些条目会躺在一个查不到的分区里，
    白导一场。
    """
    stats: Counter = Counter()
    out: list[tuple[str, str]] = []
    try:
        # 只读方式打开：老库可能正被老项目用着（有 -wal/-shm），别去动它
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"  [!] 打不开老库 {path}：{exc}")
        return out, stats

    try:
        rows = db.execute("SELECT src_raw, dst, kind, src_lang FROM tm").fetchall()
    except sqlite3.Error as exc:
        print(f"  [!] 读不了老库（表结构对不上？）：{exc}")
        return out, stats
    finally:
        db.close()

    for src_raw, dst, kind, lang in rows:
        stats[f"读到({kind})"] += 1
        why = looks_wrong(src_raw, dst)
        if why:
            stats[f"丢弃：{why}"] += 1
            continue
        out.append((src_raw.strip(), dst.strip()))
        stats[f"收下({kind})"] += 1
    return out, stats


def seed(tm: TranslationMemory, pairs: list[tuple[str, str]], variant: str,
         kind: str, dry: bool) -> int:
    """写进记忆库。返回真的写进去的条数。

    用 force=True 绕过"依赖上下文的句子不缓存"那道理智检查——这里的每一条都是
    人工整理的固定对照（"Thanks for watching" → "感谢观看"），不存在"换个语境
    就不对了"的问题。
    """
    written = 0
    for src, dst in pairs:
        if dry:
            written += 1
            continue
        if tm.store(src, "en", variant, dst, kind=kind, force=True):
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="预热翻译记忆库（离线、不花额度）")
    parser.add_argument("--from-old", type=Path, default=DEFAULT_OLD_DB,
                        help="老项目的 tm.sqlite3 路径")
    parser.add_argument("--no-old", action="store_true", help="不读老项目")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = parser.parse_args()

    cfg = Config()
    variant = variant_tag(cfg.translate.dst_lang, cfg.translate.style)
    db_path = cfg.path(cfg.tm.db)

    print("=" * 46)
    print(" 预热本地记忆库（离线，不联网、不花额度）")
    print("=" * 46)
    print(f"记忆库：{db_path}")
    print(f"分区  ：{variant}（目标语言 + 风格指纹）")
    print()

    tm = TranslationMemory(cfg.tm)
    try:
        before = tm.stats().get("total", 0)

        # ---- 来源 1：第二区词典 ----
        print("[1/2] data/common.tsv")
        pairs = read_common_tsv(cfg.path(cfg.dictionaries.common))
        print(f"  读到 {len(pairs)} 条英中对照")
        n1 = seed(tm, pairs, variant, "seed", args.dry_run)
        print(f"  写入 {n1} 条（kind=seed）")
        print()

        # ---- 来源 2：老项目记忆库 ----
        print("[2/2] 老项目 live-translate 的记忆库")
        if args.no_old:
            print("  --no-old，跳过")
        elif not args.from_old.is_file():
            print(f"  找不到 {args.from_old}，跳过（不影响使用）")
        else:
            print(f"  读 {args.from_old}")
            old_pairs, stats = read_old_db(args.from_old)
            for k in sorted(stats):
                print(f"    {k}: {stats[k]}")
            n2 = seed(tm, old_pairs, variant, "import", args.dry_run)
            print(f"  写入 {n2} 条（kind=import）")
        print()

        after = tm.stats().get("total", 0)
        if args.dry_run:
            print("--dry-run：什么都没写。去掉这个参数才会真的导入。")
        else:
            print(f"记忆库：{before} → {after} 条")
            print()
            print("这些句子以后是**零延迟**命中，不走模型也不花额度。")
            print("直播里还没有的句子，第一次仍要调模型；翻过一次之后也会进这里。")
        return 0
    finally:
        tm.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        raise SystemExit(130)
