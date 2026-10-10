"""交互原型一键启动器：检查环境 -> 打印访问地址 -> 启动 Streamlit。

为什么需要它：
    直接把项目交给参与者/助教时，对方不一定知道要装依赖、要在哪个目录运行、
    以及怎么让别人用手机访问。本脚本把这几步收敛成一条命令，并在缺依赖/缺模型时
    给出**可直接复制**的补救命令，而不是抛一堆 traceback。

用法（在项目根目录下）：
    python -m scripts.run_app                 # 本机 + 局域网（同一 WiFi 的手机可访问）
    python -m scripts.run_app --port 8600     # 端口被占用时换端口
    python -m scripts.run_app --local-only    # 只绑定 127.0.0.1（更安全）
    python -m scripts.run_app --check-only    # 只做检查并打印地址，不启动界面

Windows 下也可以直接双击项目根目录的 run_app.cmd。
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 界面运行必需（缺任何一个都启动不了）
REQUIRED = ["pandas", "numpy", "sklearn", "joblib", "streamlit"]

# 只影响"重新训练/出报告"，缺失时只提示、不阻塞启动界面
OPTIONAL = ["matplotlib", "tabulate"]

# 界面启动必需的三件套（其余产物可缺省）
MODEL_FILES = ["model.pkl", "model_card.json", "history_sample.csv"]


def missing_packages(names: list[str]) -> list[str]:
    return [n for n in names if importlib.util.find_spec(n) is None]


def missing_model_files() -> list[str]:
    return [f for f in MODEL_FILES if not (ROOT / "models" / f).exists()]


def lan_ip() -> str | None:
    """取本机在局域网中的 IP（UDP connect 只选路由，不发数据，不会联网）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def check() -> bool:
    """返回环境是否就绪；打印的补救命令可以直接复制执行。"""
    ok = True

    missing = missing_packages(REQUIRED)
    if missing:
        ok = False
        print(f"[缺依赖] {', '.join(missing)}")
        print("        请执行：pip install -r requirements.txt")

    soft = missing_packages(OPTIONAL)
    if soft:
        print(f"[提示] 缺少 {', '.join(soft)}：不影响界面启动，但无法运行 python -m src.train / src.evaluate")

    files = missing_model_files()
    if files:
        ok = False
        print(f"[缺模型产物] {', '.join(files)}")
        print("        请依次执行（首次约 1~2 分钟）：")
        print("          python -m src.features")
        print("          python -m src.train")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="启动睡眠时长预测交互原型（Streamlit）")
    parser.add_argument("--port", type=int, default=8501, help="服务端口，默认 8501")
    parser.add_argument("--local-only", action="store_true", help="只允许本机访问（默认允许局域网）")
    parser.add_argument("--check-only", action="store_true", help="只检查环境并打印地址，不启动界面")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):        # 输出重定向到日志文件时不乱码
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print("=" * 54)
    print(" 睡眠时长预测 / 交互原型（课程作业，非医疗用途）")
    print("=" * 54)
    print(f"项目目录：{ROOT}")

    if not check():
        print("")
        print("环境未就绪：请先执行上面列出的命令，然后重新运行本脚本。")
        return 1

    host = "127.0.0.1" if args.local_only else "0.0.0.0"
    print("")
    print(f"本机访问：   http://localhost:{args.port}")
    if not args.local_only:
        ip = lan_ip()
        if ip:
            print(f"局域网访问： http://{ip}:{args.port}")
            print("             （参与者手机需连同一个 WiFi；首次访问请允许 Windows 防火墙）")
        else:
            print("局域网访问： 未取到本机 IP，请用 ipconfig 查看后拼成 http://<IP>:端口")
    print("停止服务：   在本窗口按 Ctrl+C")
    print("")

    if args.check_only:
        print("[check-only] 检查通过，未启动界面。")
        return 0

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py"),
        "--server.address", host,
        "--server.port", str(args.port),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
