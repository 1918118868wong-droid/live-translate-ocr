"""命令行入口。

刻意不在这里 import PySide6 —— 下面这些路径都不需要图形库：
    --help / --init-config / --list-monitors / --pick / --image
把 Qt 的导入推后到真正要开窗口的时候（livetranslate_ocr/ui.py），
这样依赖只装了一半的时候命令行依然能用。**--image 尤其重要**：没有直播的时候
想复现"某张截图翻译得不对"，它是一条完全离线的路。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config as config_mod

log = logging.getLogger("livetranslate_ocr")


def setup_logging(cfg: config_mod.Config) -> None:
    level = getattr(logging, cfg.log.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        path = cfg.path(cfg.log.file)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    except OSError as exc:
        print(f"日志文件不可写（{exc}），只输出到终端", file=sys.stderr)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # 第三方库的 INFO 太吵
    for noisy in ("httpx", "httpcore", "urllib3", "onnxruntime"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live-translate-ocr",
        description="直播实时字幕翻译：读屏幕上的英文字幕 → 认字 → 口语化意译 → 桌面置顶字幕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  run.bat                             用 config.toml 的配置启动\n"
            "  run.bat --pick                      框选字幕区域（第一次必做）\n"
            "  run.bat --selftest                  只看字幕窗，不抓屏不联网\n"
            "  run.bat --text                      从终端输入英文，回车看翻译\n"
            "  run.bat --image 截图.png            认一张图上的字幕并翻译（离线排查用）\n"
            "  run.bat --list-monitors             列出各块屏的坐标\n"
        ),
    )
    p.add_argument("--config", type=Path, help="配置文件路径（默认 ./config.toml）")
    p.add_argument("--dst", help="目标语言，如 zh / zh-TW / en / ja / ko")
    p.add_argument("--src", help="源语言（屏幕字幕固定是英文，一般不用改）")
    p.add_argument("--style", help='翻译风格描述，直接进 prompt，如 "游戏直播、粗口不净化"')
    p.add_argument("--llm-model", help="翻译模型名")
    p.add_argument("--base-url", help="OpenAI 兼容 API 地址")
    p.add_argument("--fps", type=float, help="抓屏频率（每秒认几帧），默认读配置")
    p.add_argument("--monitor", type=int, help="抓哪块屏（mss 编号），默认读配置")
    p.add_argument("--region", help="直接设定字幕区域 x,y,w,h（物理像素），不用鼠标框")
    p.add_argument("--no-tm", action="store_true", help="关闭本地翻译记忆库")
    p.add_argument("--no-hotkeys", action="store_true", help="不注册全局热键")
    p.add_argument("--text", action="store_true", help="文本模式：从标准输入读取，不抓屏")
    p.add_argument("--image", type=Path, help="对一张图片跑 OCR → 翻译，然后退出")
    p.add_argument("--pick", action="store_true", help="打开框选窗口，框完存进配置后退出")
    p.add_argument("--selftest", action="store_true", help="自检字幕窗（不抓屏、不联网）")
    p.add_argument("--list-monitors", action="store_true", help="列出各块屏的坐标后退出")
    p.add_argument("--init-config", action="store_true", help="生成 config.toml 后退出")
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return p


def apply_overrides(cfg: config_mod.Config, args: argparse.Namespace) -> None:
    if args.dst:
        cfg.translate.dst_lang = args.dst
    if args.src:
        cfg.translate.src_lang = args.src
    if args.style:
        cfg.translate.style = args.style
    if args.llm_model:
        cfg.llm.model = args.llm_model
    if args.base_url:
        cfg.llm.base_url = args.base_url
    if args.fps is not None:
        cfg.capture.fps = max(0.5, float(args.fps))
    if args.monitor is not None:
        cfg.capture.monitor = int(args.monitor)
    if args.no_tm:
        cfg.tm.enabled = False
    if args.no_hotkeys:
        cfg.hotkeys.enabled = False
    if args.verbose:
        cfg.log.level = "DEBUG"


def _cmd_init_config() -> int:
    path = config_mod.DEFAULT_CONFIG_PATH
    if path.exists():
        print(f"已存在，未覆盖：{path}")
        return 1
    # 优先复制带注释的示例文件；找不到才用无注释的自动生成版
    example = config_mod.EXAMPLE_CONFIG_PATH
    if example.is_file():
        path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text(config_mod._to_toml(config_mod.Config()), encoding="utf-8")
    print(f"已生成配置文件：{path}")
    return 0


class _PrintSink:
    """--image 用的输出口：直接打到终端，不碰 Qt。"""

    def __init__(self) -> None:
        self.last = ""

    def on_source(self, text: str) -> None:
        pass          # 原文下面单独打，这里不用重复

    def on_delta(self, text: str) -> None:
        pass          # 流式中间态在终端里刷屏没意义

    def on_translation(self, source: str, translation: str, meta: str) -> None:
        self.last = translation
        body = translation if translation else "（不译：本地词典覆盖不了，见下方原文）"
        print(f"  译文：{body}")
        print(f"  来源：{meta}")

    def on_status(self, text: str) -> None:
        if text:
            print(f"  · {text}")

    def on_error(self, text: str) -> None:
        if text:
            print(f"  !! {text}", file=sys.stderr)


def _cmd_image(cfg: config_mod.Config, path: Path) -> int:
    """对一张图跑完整的 OCR → 拼句 → 翻译，全部离线、不碰屏幕。

    这是本项目最主要的排查手段：直播里看到一句翻得不对，截个图丢进来，
    反复跑、随便调词典和 prompt，不用开着直播等它再出现一次。
    """
    from .pipeline import Pipeline

    if not path.is_file():
        print(f"找不到图片：{path}", file=sys.stderr)
        return 2

    sink = _PrintSink()
    pipe = Pipeline(cfg, sink)
    try:
        pipe.warmup()
        reading, emissions = pipe.read_image(path)
    except Exception as exc:
        log.exception("图片处理失败")
        print(f"处理失败：{exc}", file=sys.stderr)
        pipe.stop()
        return 3

    print(f"图片：{path}")
    if reading.empty:
        print(f"  !! 一个字都没认出来（耗时 {reading.elapse_s * 1000:.0f}ms）")
        print("     看看图里是不是真的有英文；或者框选区域框到空白上了。")
        pipe.stop()
        return 4

    print(f"  OCR：{len(reading.lines)} 行，置信度 {reading.score:.3f}，"
          f"{reading.elapse_s * 1000:.0f}ms")
    for line in reading.lines:
        print(f"      {line.text}")

    if not emissions:
        print("  !! 认出了字，但拼句状态机没发出任何句子 —— 把日志级别调到 DEBUG 看原因")
        pipe.stop()
        return 5

    print(f"  拼出 {len(emissions)} 句：")
    for em in emissions:
        print(f"      [{em.describe()}] {em.text}")
    print()
    try:
        for em in emissions:
            pipe.submit_text(em.text)
    finally:
        pipe.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- 不需要 Qt 的纯命令行路径 ----
    if args.init_config:
        return _cmd_init_config()

    cfg = config_mod.load(args.config)
    apply_overrides(cfg, args)

    if args.list_monitors:
        setup_logging(cfg)
        from .capture import main as monitors_main

        return monitors_main()

    if args.region:
        # 和 --pick 一样，只写配置、不开窗口
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
        print(f"字幕区域已保存：{region}")
        if not (args.pick or args.image or args.text or args.selftest):
            return 0

    setup_logging(cfg)
    log.info("配置文件：%s", cfg.source_path or "（未找到，使用默认值）")

    if args.pick:
        from .regionpick import main as pick_main

        # 只转交 --monitor。--region 不转：它上面已经写进配置了，再转过去
        # regionpick 会走"直接设定、不开窗口"那条路，`--pick` 就静悄悄地什么都不做。
        pick_argv: list[str] = []
        if args.monitor is not None:
            pick_argv += ["--monitor", str(args.monitor)]
        return pick_main(pick_argv)

    if args.image:
        return _cmd_image(cfg, args.image)

    # 没配 API Key **不再是致命错误**：还有离线词典那一档兜底。
    # 音频版这里原本是 return 2 直接退出；本版刻意改了——三级降级的设计
    # 就是为了让"没有网、没有 Key"不是一个死局。下面两个非 GUI 路径也一样。
    if not cfg.resolved_api_key():
        print(
            f"\n提示：没有找到 LLM API Key（环境变量 {cfg.llm.api_key_env} "
            f"或 config.toml 的 [llm] api_key）。\n"
            f"程序照样能跑，但会一直用离线词典逐词直译，译文比较生硬。\n"
            f'想配上：set {cfg.llm.api_key_env}=sk-xxxxx    (cmd)\n'
            f'        $env:{cfg.llm.api_key_env}="sk-xxxxx"    (PowerShell)\n',
            file=sys.stderr,
        )

    # ---- 到这一步才需要图形库 ----
    try:
        from .ui import App
    except ImportError as exc:
        print(
            f"缺少图形界面依赖（{exc}）。\n"
            f"请先执行 setup.bat 安装依赖，或手动运行：\n"
            f"    .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 3

    return App(cfg, args).run()


if __name__ == "__main__":
    raise SystemExit(main())
