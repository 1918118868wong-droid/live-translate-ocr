"""环境安装：建虚拟环境、装依赖、验证 Qt 与 opencv 能共存、生成配置、预热记忆库。

纯标准库实现，所以用系统 Python 直接就能跑，不需要先装任何东西。

为什么要有这个文件：`.bat` 里不能放中文。cmd.exe 的批处理解析器是**按字节**读行的，
碰到 UTF-8 中文会因为当前代码页对不上而把行拆错——表现就是 `echo` 变成 `cho`、
`echo` 行里的内容被当成命令执行。所以 .bat 只留纯 ASCII 的壳，中文文案全放这里。

    python tools/bootstrap.py              正常安装 / 补齐缺的东西
    python tools/bootstrap.py --rebuild    删掉 .venv 重装（搬过目录、或装坏了）
    python tools/bootstrap.py --mirror     走清华镜像下依赖（网络慢的时候用）
    python tools/bootstrap.py --skip-deps  跳过装依赖，只生成配置、预热记忆库
    python tools/bootstrap.py --no-seed    跳过记忆库预热
    python tools/bootstrap.py --no-old     预热时不读老项目的记忆库
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
REQUIREMENTS = ROOT / "requirements.txt"
CONFIG = ROOT / "config.toml"
CONFIG_EXAMPLE = ROOT / "config.example.toml"
SEED = ROOT / "tools" / "seed_tm.py"
SELFCHECK = ROOT / "tools" / "selfcheck.py"

MIN_PY = (3, 11)
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 这个包声明的依赖里带 opencv-python（带 Qt 那版），必须单独装
SPECIAL_CASE = "rapidocr"

STEPS = 5

# 验证 Qt 和 opencv 能不能在同一个进程里一起用。这是本项目最要命的一个安装坑，
# 而且**症状完全不指向原因**：装了带 Qt 的 opencv-python 之后，表现是界面错乱、
# 或者一开字幕窗就崩，没人会想到是 opencv 干的。所以装完立刻验一次。
VERIFY_CODE = r"""
import sys
problems = []

# 1) 两个库能不能同时导入（真正的冲突是在这一步和后面构造窗口时暴露的）
try:
    import cv2
    print("  cv2", cv2.__version__, "->", cv2.__file__)
except Exception as e:
    problems.append("opencv 导入失败：%r" % (e,))
    cv2 = None
try:
    import PySide6
    print("  PySide6", PySide6.__version__)
except Exception as e:
    problems.append("PySide6 导入失败：%r" % (e,))

# 2) 装的是不是 headless 版。带 Qt 的那版和 PySide6 用的是同一套 import 名，
#    只能有一个在场，所以这里直接按"有没有装错那一个"来判。
from importlib.metadata import PackageNotFoundError, distribution
try:
    distribution("opencv-python")
    problems.append("装了带 Qt 的 opencv-python，它和 PySide6 抢 Qt 插件。"
                    "卸掉它、换 opencv-python-headless："
                    "  .venv\\Scripts\\python.exe -m pip uninstall -y opencv-python")
except PackageNotFoundError:
    pass
try:
    print("  opencv-python-headless", distribution("opencv-python-headless").version)
except PackageNotFoundError:
    problems.append("没装 opencv-python-headless")

# 3) 真造一个窗口 + 从 numpy 造一张图。前两步都过、这一步才崩的情况是有的，
#    因为冲突要到 Qt 插件真的被加载时才炸。
if cv2 is not None and not problems:
    try:
        import numpy as np
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv[:1])
        img = QImage(np.zeros((8, 8, 3), dtype=np.uint8).data, 8, 8, 24,
                     QImage.Format.Format_RGB888)
        assert not img.isNull()
        print("  两个库同进程实测：正常")
    except Exception as e:
        # 没有显示器时 QApplication 会失败（比如远程跑），那不算安装错
        print("  没能实际构造窗口（%s）—— 如果是远程/无桌面环境，可忽略" % (type(e).__name__,))

if problems:
    print()
    for p in problems:
        print("  [!!] " + p)
    sys.exit(1)
print("  依赖自检通过")
"""


# ---------------------------------------------------------------- 输出
def init_console() -> None:
    """把控制台切到 UTF-8，否则中文在 cp936 控制台上会变乱码。"""
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def step(n: int, text: str) -> None:
    print(f"\n[{n}/{STEPS}] {text}")


def ok(text: str) -> None:
    print(f"  {text}")


def warn(text: str) -> None:
    print(f"  [!] {text}")


def die(text: str) -> int:
    print(f"\n[错误] {text}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------- 环境
def check_python() -> bool:
    v = sys.version_info
    if v < MIN_PY:
        die(
            f"当前 Python 是 {v.major}.{v.minor}.{v.micro}，需要 "
            f"{MIN_PY[0]}.{MIN_PY[1]} 或更高。\n"
            f"       装新版：https://www.python.org/downloads/"
        )
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro}（{sys.executable}）")
    return True


def venv_relocated() -> bool:
    """判断 .venv 是不是被搬过家。

    搬过家的话 `Scripts\\*.exe` 里烤进去的绝对路径会失效。`python.exe` 本身
    没事（按 pyvenv.cfg 的位置算 prefix），但 `pip.exe` 之类的转发器会报错。
    我们全程用 `python -m pip`，所以不影响使用，只是提醒一下。
    """
    cfg = VENV / "pyvenv.cfg"
    if not cfg.exists():
        return False
    for line in cfg.read_text("utf-8", "replace").splitlines():
        if not line.startswith("command"):
            continue
        _, _, cmd = line.partition("=")
        tail = cmd.strip().split()[-1].rstrip("\\/")
        try:
            return Path(tail).resolve() != VENV.resolve()
        except OSError:
            return True
    return False


def make_venv(rebuild: bool) -> int:
    if rebuild and VENV.exists():
        warn(f"要删掉重建：{VENV}")
        if not os.environ.get("LIVETRANSLATE_YES"):
            try:
                if input("  确认删除？输入 y 回车：").strip().lower() not in ("y", "yes"):
                    return die("已取消。")
            except EOFError:
                return die("非交互环境下删除需先设 LIVETRANSLATE_YES=1")
        shutil.rmtree(VENV, ignore_errors=True)

    if VENV_PY.exists():
        ok(f"已存在，跳过：{VENV}")
        if venv_relocated():
            warn(
                "这个 .venv 是别处建的（搬过目录）。python 本身能用，"
                "但 Scripts\\pip.exe 之类的转发器是坏的——用 "
                "`.venv\\Scripts\\python.exe -m pip` 代替，"
                "或跑 setup.bat --rebuild 重建。"
            )
        return 0

    ok(f"创建虚拟环境：{VENV}")
    rc = subprocess.call([sys.executable, "-m", "venv", str(VENV)])
    if rc != 0 or not VENV_PY.exists():
        return die(
            "创建虚拟环境失败。\n"
            "       多半是系统 Python 装得不全（缺 venv 模块）——"
            "用官方安装包重装一遍 Python 即可。"
        )
    return 0


# ---------------------------------------------------------------- 依赖
def _pip(*args: str) -> int:
    return subprocess.call([str(VENV_PY), "-m", "pip", *args])


def read_requirements() -> tuple[list[str], list[str]]:
    """把 requirements.txt 拆成 (普通依赖, 需要单独装的)。

    为什么要拆：`rapidocr` 声明的依赖里带 `opencv_python`（带 Qt 那版），
    直接装着它跑就会把 opencv-python 也拖进来，和 PySide6 抢 Qt 插件。
    正确做法是它自己用 --no-deps 装、再单独补齐它声明的其余依赖。

    行尾注释和空行都去掉——不然 `mss>=10.2   # 抓屏` 会被当成一个包名丢给 pip。
    """
    if not REQUIREMENTS.exists():
        return [], []
    plain, special = [], []
    for raw in REQUIREMENTS.read_text("utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line
        for sep in (">=", "==", "<=", "~=", ">", "<", "["):
            name = name.split(sep, 1)[0]
        name = name.strip().lower().replace("_", "-")
        (special if name == SPECIAL_CASE else plain).append(line)
    return plain, special


def declared_deps(dist: str) -> list[str]:
    """读某个已装包自己声明的依赖（在 venv 里跑，因为那是它的环境）。

    动态读而不是把依赖名抄一份写死：rapidocr 以后加了新依赖，这里能自动跟上，
    而抄死的那份会变成"装完少一个包、报一堆看不通的 ImportError"。
    """
    code = (
        "from importlib.metadata import requires\n"
        f"print('\\n'.join(requires({dist!r}) or []))\n"
    )
    try:
        out = subprocess.run([str(VENV_PY), "-c", code],
                             capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def install_deps(mirror: bool) -> int:
    if not REQUIREMENTS.exists():
        return die(f"找不到 {REQUIREMENTS}")

    extra = ["-i", MIRROR] if mirror else []

    ok("升级 pip ...")
    _pip("install", "--upgrade", "pip", "--quiet")

    plain, special = read_requirements()
    ok(f"装依赖（{len(plain)} 项，首次约 250MB，要几分钟，进度如下）...")
    print()
    if plain and _pip("install", *plain, *extra) != 0:
        return die(
            "依赖安装失败。常见原因：\n"
            "       · 网络慢或被墙 —— 加镜像重试：setup.bat --mirror\n"
            "       · 缺 Visual C++ 运行库 —— 装 VC++ 2015-2022 x64\n"
            "       · 装到一半断了 —— 再跑一次 setup.bat 会接着装"
        )

    for pkg in special:
        ok(f"单独装 {pkg}（--no-deps，避免把带 Qt 的 opencv 拖进来）...")
        if _pip("install", "--no-deps", pkg, *extra) != 0:
            return die(f"装 {pkg} 失败。加镜像重试：setup.bat --mirror")
        deps = [d for d in declared_deps(SPECIAL_CASE)
                if not d.lower().replace("_", "-").startswith("opencv")]
        if deps:
            ok(f"补齐 {SPECIAL_CASE} 声明的其余依赖（{len(deps)} 项，跳过 opencv）...")
            if _pip("install", *deps, *extra) != 0:
                return die(f"补齐 {SPECIAL_CASE} 的依赖失败。加镜像重试：setup.bat --mirror")
    return 0


def verify_deps() -> int:
    ok("验证 Qt 与 opencv 能在同一个进程里共存 ...")
    print()
    rc = subprocess.call([str(VENV_PY), "-c", VERIFY_CODE], cwd=str(ROOT))
    if rc != 0:
        return die(
            "依赖有问题（见上面的 [!!]）。\n"
            "       最容易出的是 opencv 装成了带 Qt 的那版，照提示卸掉重装即可。"
        )
    print()
    return 0


# ---------------------------------------------------------------- 配置 / 词库
def make_config() -> None:
    if CONFIG.exists():
        ok("config.toml 已存在，保留不动")
        return
    if not CONFIG_EXAMPLE.exists():
        warn("找不到 config.example.toml，跳过")
        return
    shutil.copyfile(CONFIG_EXAMPLE, CONFIG)
    ok("已生成 config.toml")


def seed_dict(no_old: bool) -> None:
    if not SEED.exists():
        warn("找不到 tools/seed_tm.py，跳过")
        return
    ok("预热本地记忆库（离线，不联网、不花额度）...")
    args = [str(VENV_PY), str(SEED)]
    if no_old:
        args.append("--no-old")
    if subprocess.call(args, cwd=str(ROOT)) != 0:
        warn("预热失败。不影响使用，之后可以手动跑 seed.bat")


def run_selfcheck() -> None:
    if not SELFCHECK.exists():
        return
    ok("跑一遍离线自检（不联网、不抓屏）...")
    rc = subprocess.call([str(VENV_PY), str(SELFCHECK)], cwd=str(ROOT))
    if rc != 0:
        warn("自检没全过。装是装上了，但上面报红的那几条要看一眼。")


def next_steps() -> None:
    print(
        "\n"
        "============================================\n"
        " 安装完成\n"
        "============================================\n"
        "\n"
        "接下来：\n"
        "  1. 框选字幕区域（第一次必须做）\n"
        "       打开直播、调出英文字幕，然后跑：  pick.bat\n"
        "     拖一个框只框住字幕带本身，框完它会当场识别一次给你看。\n"
        "\n"
        "  2. 试一下字幕窗长什么样（不抓屏、不联网）：  run.bat --selftest\n"
        "\n"
        "  3. 离线试一句话（不碰屏幕，验证翻译链路）：  run.bat --text\n"
        "\n"
        "  4. 正式启动：  run.bat\n"
        "\n"
        "关于 API Key：**不配也能用**。没配的时候走本地词典逐词直译，\n"
        "能看懂大意但很生硬；配上之后才是通顺的口语化译文。\n"
        "  设置环境变量（然后重开一个终端）：\n"
        '      setx DEEPSEEK_API_KEY "sk-你的key"\n'
        "  或者编辑 config.toml 的 [llm] api_key。\n"
        "\n"
        "识别模型是打包在依赖里的，首次运行不需要联网下载。\n"
    )


# ---------------------------------------------------------------- 入口
def main() -> int:
    init_console()
    parser = argparse.ArgumentParser(description="live-translate-ocr 环境安装")
    parser.add_argument("--rebuild", action="store_true", help="删掉 .venv 重建")
    parser.add_argument("--mirror", action="store_true", help="用清华 PyPI 镜像")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖安装")
    parser.add_argument("--no-seed", action="store_true", help="跳过记忆库预热")
    parser.add_argument("--no-old", action="store_true", help="预热时不读老项目的记忆库")
    args = parser.parse_args()

    print("============================================")
    print("  live-translate-ocr 环境安装")
    print("  （读屏幕字幕，不需要显卡和麦克风）")
    print("============================================")
    print(f"项目目录：{ROOT}")

    step(1, "检查 Python")
    if not check_python():
        return 1

    step(2, "准备虚拟环境")
    rc = make_venv(args.rebuild)
    if rc:
        return rc

    step(3, "安装依赖")
    if args.skip_deps:
        ok("--skip-deps，跳过")
    else:
        rc = install_deps(args.mirror)
        if rc:
            return rc

    step(4, "验证依赖")
    if args.skip_deps:
        ok("--skip-deps，跳过")
    else:
        rc = verify_deps()
        if rc:
            return rc

    step(5, "生成配置 + 预热记忆库")
    make_config()
    if args.no_seed:
        ok("--no-seed，跳过记忆库预热")
    else:
        seed_dict(args.no_old)
    run_selfcheck()

    next_steps()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
