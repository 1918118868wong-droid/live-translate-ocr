"""配置：TOML 文件 → dataclass，缺失字段一律回落到默认值。

设计原则和老项目一致：不带 config.toml 也能跑起来（全默认值），
唯一的例外是 `[capture] region`——没框选过就不知道去哪抓字幕，程序会提示先框选。

和音频版的差别：去掉了 audio / vad / asr / wiki 四段（不再需要），
新增 capture / ocr / captions / dictionaries 四段。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.toml"


@dataclass
class CaptureCfg:
    # 字幕区域 [x, y, w, h]，屏幕物理像素。全 0 = 还没框选过，
    # 首次运行会提示先跑框选器（或按 ctrl+alt+r）。
    region: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # 轮询频率。字幕在屏幕上停留 1~3 秒，3Hz 已经够密（每条字幕能采 3~9 次，
    # 而判"定型"只要连续 2 次）。
    #
    # 定成 3 而不是 4，是实测出来的：一次 OCR 单行 168ms、两行 254ms（中位，
    # 1600 宽的紧框，最慢 399ms），而**两行是常态**。4Hz 的预算是 250ms，
    # 正好卡在两行的耗时上——OCR 线程会一直满载，白吃一个核，
    # 而字幕一条也没多采几次。3Hz 预算是 333ms，留得出余量。
    # 宁可少采几帧：这个工具是开着游戏/直播时用的，抢 CPU 的代价比多采两帧大。
    fps: float = 3.0
    # 连续读不到字的时长超过 idle_after_s，就把频率降到 fps_idle 省电
    # （广告、暂停、没开 CC 的时候占大多数时间）
    idle_after_s: float = 2.0
    fps_idle: float = 2.0
    monitor: int = 0                 # 多屏时抓哪块屏（mss 的编号，从 1 开始；0 = 全部拼起来的虚拟屏）


@dataclass
class OcrCfg:
    engine: str = "rapidocr"         # rapidocr | windows（备用：走 PowerShell 调系统 OCR）
    lang: str = "en"                 # 只认英文——认中文模型会拖慢速度且更容易出错
    # 低于这个置信度的行直接丢掉。字幕是白字黑底，正常都在 0.9 以上；
    # 掉到 0.6 以下的通常是背景花纹被误认成字。
    min_confidence: float = 0.6
    # 最多取几行。框选时可能多框进来一点（进度条、标题），多出来的行不要。
    max_lines: int = 3


@dataclass
class CaptionsCfg:
    # 一行文本连续读几次完全一样，才算"定型"。1 = 读到就发（抖），
    # 3 = 更稳但慢一拍。2 是实测折中。
    stable_reads: int = 2
    # 攒到这么长就强制发射，防止一句话太长（YouTube 一行能到 90 字符）迟迟不结束
    max_chars: int = 90
    # 字幕整块消失超过这么久，判定这句说完了，发射缓冲
    idle_emit_s: float = 1.0
    # 太短的读数直接丢（1 个字母、纯符号），基本都是误认
    min_chars: int = 2
    # 判定"还是同一条字幕"的相似度阈值（0-100，rapidfuzz）。
    # 只在两个读数**长度接近**时才用它容忍 OCR 抖动——判断"长了几个词"
    # 真正靠的是前缀关系（见 captions.py）。别调太低：不同句子的相似度
    # 也可能很高（"在B点捡起AWP" vs "在B点捡起沙鹰"），低了会漏发整句。
    scroll_similarity: int = 90
    # 最后一行定型后最多再等这么久就发出去。
    # YouTube 自动字幕经常整句都没有句末标点，光等标点会一直不发；
    # 这个上限保证"最坏情况"下的延迟不超过它。
    max_hold_s: float = 2.5
    # 一句话从出现到强制切分的硬上限。防的是"一直长、每帧都不一样"的情况
    # （语速极快时可能发生）：那样它永远达不到"连续两次一样"，会一直不发。
    max_wait_s: float = 3.0


@dataclass
class DictCfg:
    # 两区词典，都是能手改的 TSV：「英文 <TAB> 中文」
    terms: str = "data/terms.tsv"    # 第一区：游戏专业术语（注入 prompt + 离线直译优先）
    common: str = "data/common.tsv"  # 第二区：日常高频词（只服务离线降级）
    # 离线直译的最低词覆盖率。低于它宁可显示英文原句，也不要吐一堆中英夹杂的怪话
    # ——"看不懂英文"总比"看不懂一堆四不像"强，后者还让人以为程序坏了。
    min_coverage: float = 0.55


@dataclass
class LlmCfg:
    # 任何 OpenAI 兼容端点都可以：DeepSeek / OpenAI / Moonshot / 通义兼容模式 /
    # 本地 Ollama(http://localhost:11434/v1) / LM Studio / vLLM
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key: str = ""                # 不推荐写在这里，优先用 api_key_env
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.35        # 略高一点，口语化更自然；太高会开始自由发挥
    timeout_s: float = 12.0          # 单句翻译总预算，超时就用已流出的部分
    # max_tokens = 源文本长度 × 该系数 + 32，防止模型话痨
    max_tokens_factor: float = 2.5
    max_tokens_cap: int = 512
    extra_body: dict[str, Any] = field(default_factory=dict)
    # 需要额外请求头的网关（如 one-api 中转）可以在这里加
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class TranslateCfg:
    src_lang: str = "en"             # 屏幕字幕固定是英文，不用自动检测
    dst_lang: str = "zh"
    # 风格描述会直接进 system prompt，可以按主播类型微调（游戏/闲聊/带货）
    style: str = "口语化"
    context_lines: int = 4           # 送给 LLM 的上文轮数，越多越连贯但越慢
    glossary_max_terms: int = 6      # 每次最多注入几条术语，太多会干扰模型


@dataclass
class OfflineCfg:
    """离线降级：LLM 不可用时怎么退。"""
    # 连续失败几次判定为"离线态"。1 次太敏感（偶发超时会误判），
    # 2 次比较稳；进入离线态后不再每句都去撞墙。
    fail_threshold: int = 2
    # 离线态下每隔多久试探性重试一次 LLM，成功就自动切回
    retry_interval_s: float = 30.0


@dataclass
class TmCfg:
    enabled: bool = True
    db: str = "data/tm.sqlite3"
    # 模糊命中阈值（0-100）。低于 100 的命中会跳过缓存写入检查
    fuzzy_threshold: int = 93
    max_memory_entries: int = 50000  # 启动时预载进内存的条数上限
    cache_context_dependent: bool = False
    write_min_chars: int = 2


@dataclass
class OverlayCfg:
    font_family: str = "Microsoft YaHei UI"
    font_size: int = 26
    orig_font_size: int = 15
    show_original: bool = True
    opacity: float = 0.72            # 字幕框背景不透明度
    width_ratio: float = 0.74        # 占屏幕宽度比例
    # 贴哪：above_region = 贴在采集区域正上方（默认，避免挡住要 OCR 的字幕），
    # top / bottom = 手动贴顶/贴底
    anchor: str = "above_region"
    anchor_gap: int = 20             # above_region 时与采集区域留的空隙
    bottom_margin: int = 90          # anchor = bottom 时才用
    padding: int = 18
    line_spacing: int = 8
    hide_after_s: float = 7.0        # 静默多久后自动隐藏
    show_latency: bool = True        # 右下角显示 OCR/翻译耗时，调参用
    start_clickthrough: bool = False # 启动时是否点击穿透（穿透后拖不动，热键切回来）


@dataclass
class HotkeyCfg:
    enabled: bool = True
    toggle_show: str = "ctrl+alt+s"
    toggle_clickthrough: str = "ctrl+alt+c"
    toggle_original: str = "ctrl+alt+d"
    pause: str = "ctrl+alt+p"
    # 换窗口/改分辨率/框歪了，按这个重新框选
    repick_region: str = "ctrl+alt+r"
    font_up: str = "ctrl+alt+up"
    font_down: str = "ctrl+alt+down"
    quit: str = "ctrl+alt+q"


@dataclass
class LogCfg:
    level: str = "INFO"
    file: str = "logs/livetranslate-ocr.log"


@dataclass
class Config:
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    dictionaries: DictCfg = field(default_factory=DictCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    translate: TranslateCfg = field(default_factory=TranslateCfg)
    offline: OfflineCfg = field(default_factory=OfflineCfg)
    tm: TmCfg = field(default_factory=TmCfg)
    overlay: OverlayCfg = field(default_factory=OverlayCfg)
    hotkeys: HotkeyCfg = field(default_factory=HotkeyCfg)
    log: LogCfg = field(default_factory=LogCfg)
    source_path: Path | None = None

    # ---- 便捷访问 ----
    def path(self, relative: str | Path) -> Path:
        """把配置里的相对路径解析到项目根目录下，并确保父目录存在。"""
        p = Path(relative)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_api_key(self) -> str:
        return self.llm.api_key or os.environ.get(self.llm.api_key_env, "")

    def region(self) -> tuple[int, int, int, int] | None:
        """字幕区域，没框选过（宽或高为 0）时返回 None。"""
        x, y, w, h = (list(self.capture.region) + [0, 0, 0, 0])[:4]
        if w <= 0 or h <= 0:
            return None
        return int(x), int(y), int(w), int(h)


def _apply(obj: Any, data: dict[str, Any]) -> None:
    """把 dict 里的值填进 dataclass，遇到嵌套 dataclass 递归处理。

    未知键直接忽略（方便旧配置继续可用），类型不做强制转换——
    TOML 本身给了正确的类型，写错了就让它在用的时候报错，比静默纠正更好查。
    """
    if not isinstance(data, dict):
        return
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, value)


def load(path: str | Path | None = None) -> Config:
    """加载配置。path 为空时尝试项目根的 config.toml，不存在就用全默认值。"""
    cfg = Config()
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    if target.is_file():
        with open(target, "rb") as f:
            data = tomllib.load(f)
        _apply(cfg, data)
        cfg.source_path = target
    return cfg


def save_region(cfg: Config, region: tuple[int, int, int, int]) -> None:
    """把框选结果写回配置文件（只改 region 这一行，其余原样保留）。

    框选器每次框完就调这里，用户不用手动去改 TOML。
    配置文件不存在时先按默认值生成一份。
    """
    target = cfg.source_path or DEFAULT_CONFIG_PATH
    if not target.is_file():
        write_example(target)
    text = target.read_text(encoding="utf-8")
    new_line = f"region = [{region[0]}, {region[1]}, {region[2]}, {region[3]}]"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("region") and "=" in line:
            lines[i] = new_line
            break
    else:
        # 没有这一行就往 [capture] 段里补一行
        for i, line in enumerate(lines):
            if line.strip() == "[capture]":
                lines.insert(i + 1, new_line)
                break
        else:
            lines += ["", "[capture]", new_line]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cfg.capture.region = list(region)
    cfg.source_path = target


def write_example(path: str | Path | None = None) -> Path:
    """把当前默认值导出成 config.example.toml（用于首次生成配置文件）。"""
    target = Path(path) if path else EXAMPLE_CONFIG_PATH
    target.write_text(_to_toml(cfg=Config()), encoding="utf-8")
    return target


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, dict):
        # 只支持一层内联表，够 extra_body 用
        inner = ", ".join(f"{k} = {_scalar(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    return str(value)


def _to_toml(cfg: Config) -> str:
    """极简 TOML 序列化，只覆盖本项目用到的标量/数组/内联表。"""
    lines = [
        "# live-translate-ocr 配置。改完重启生效。",
        "# 完整说明见 README.md。",
        "",
    ]
    for f in fields(cfg):
        if f.name == "source_path":
            continue
        section = getattr(cfg, f.name)
        if not is_dataclass(section):
            continue
        lines.append(f"[{f.name}]")
        for sf in fields(section):
            value = getattr(section, sf.name)
            if isinstance(value, (list, tuple)):
                rendered = "[" + ", ".join(_scalar(v) for v in value) + "]"
            else:
                rendered = _scalar(value)
            lines.append(f"{sf.name} = {rendered}")
        lines.append("")
    return "\n".join(lines) + "\n"
