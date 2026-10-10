"""网络连通性自检：这台机器适合用哪种方式把原型放到网上。

在项目根目录运行：

    python deploy/check_network.py

会并发探测上线所需的几个站点（每项最多等 6 秒），并给出结论：
哪条上线路径在这个网络里是通的。探测结果只反映"当前网络"，
换 WiFi / 换热点请重跑一次。
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):        # 输出重定向到日志文件时不乱码
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# (说明, 探测地址)
TARGETS = [
    ("GitHub（存代码 / 下载 cloudflared）", "https://github.com/"),
    ("Cloudflare 快速隧道（临时公网网址）", "https://api.trycloudflare.com/"),
    ("Streamlit Community Cloud（持久网址）", "https://share.streamlit.io/"),
    ("Hugging Face Spaces（备选持久网址）", "https://huggingface.co/"),
    ("PyPI 官方源（云端装依赖）", "https://pypi.org/simple/"),
    ("清华 PyPI 镜像（国内装依赖）", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
]

TIMEOUT = 6.0
UA = {"User-Agent": "sleep-predictor-network-check/1.0"}


def probe(item: tuple[str, str]) -> tuple[str, bool, str, float]:
    name, url = item
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=TIMEOUT) as r:
            return name, True, f"HTTP {r.status}", round(time.time() - t0, 1)
    except urllib.error.HTTPError as e:          # 有响应就说明 TLS + HTTP 通了
        return name, True, f"HTTP {e.code}", round(time.time() - t0, 1)
    except Exception as e:                       # noqa: BLE001 - 探测工具，任何异常都算不可达
        return name, False, type(e).__name__, round(time.time() - t0, 1)


def main() -> int:
    print("=" * 64)
    print(" 上线方式网络自检（每项最多等 6 秒）")
    print("=" * 64)
    print(f"项目目录：{ROOT}")

    with cf.ThreadPoolExecutor(max_workers=len(TARGETS)) as ex:
        results = list(ex.map(probe, TARGETS))

    for name, ok, detail, secs in results:
        print(f"{'[可达]  ' if ok else '[不可达]'} {name}  {detail}  ({secs}s)")

    state = {name: ok for name, ok, _, _ in results}
    tunnel_ok = state[TARGETS[1][0]]
    cloud_ok = state[TARGETS[2][0]] and state[TARGETS[0][0]]
    hf_ok = state[TARGETS[3][0]]

    print("")
    print("-" * 64)
    print("结论：")
    if tunnel_ok:
        print("  · 临时公网网址（Cloudflare 快速隧道）：可用")
        print("      → 双击 deploy\\share_web.cmd，约 20 秒后得到 https://xxx.trycloudflare.com")
    else:
        print("  · 临时公网网址（Cloudflare 快速隧道）：本网络探测不通")
        print("      → 隧道走 HTTPS(443) + QUIC(UDP 7844)，探测不通时不建议走这条；")
        print("        若确有需要，可先双击 share_web.cmd 试一次（探测不通不等于隧道一定不行）")
    if cloud_ok:
        print("  · 持久网址（Streamlit Community Cloud）：可用")
        print("      → 按 deploy\\README.md 方案 2 操作（GitHub + share.streamlit.io）")
    else:
        print("  · 持久网址（Streamlit Community Cloud）：需要 GitHub 与 share.streamlit.io 都能访问")
    if hf_ok:
        print("  · 备选持久网址（Hugging Face Spaces）：可用 → 按 deploy\\README.md 方案 3")
    print("  · 无论上面哪条通，局域网方式始终可用：run_app.cmd（参与者与你在同一个 WiFi）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
