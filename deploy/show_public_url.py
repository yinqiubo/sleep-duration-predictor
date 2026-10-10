"""打印当前临时公网网址，并做一次健康检查。

用法（在项目根目录）：

    python deploy/show_public_url.py            # 读 deploy/_tunnel.log，查一次就返回
    python deploy/show_public_url.py --wait 60  # 最多等 60 秒，等隧道地址出现

背景：`deploy/share_web.cmd` 会把 Cloudflare 的输出写到 `deploy/_tunnel.log`，
其中一行含 `https://xxxx.trycloudflare.com`，那就是对外的网页地址。
本脚本不必开着隧道也能运行，它只是"读日志 + 探测"。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "deploy" / "_tunnel.log"
URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
LOCAL_HEALTH = "http://127.0.0.1:8501/_stcore/health"


def find_url() -> str | None:
    if not LOG.exists():
        return None
    text = LOG.read_text(encoding="utf-8", errors="replace")
    matches = URL_RE.findall(text)
    return matches[-1] if matches else None


def probe(label: str, url: str, timeout: float = 20.0) -> tuple[str, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read(120).decode("utf-8", "replace").strip()
            return f"{label}: OK (HTTP {r.status}) {body}".rstrip(), "ok"
    except urllib.error.HTTPError as e:
        return f"{label}: 有响应但状态码 {e.code}", "warn"
    except Exception as e:                              # noqa: BLE001 - 诊断工具
        return f"{label}: 连接失败（{type(e).__name__}）", "fail"


def main() -> int:
    parser = argparse.ArgumentParser(description="打印并检查当前临时公网网址")
    parser.add_argument("--wait", type=float, default=0.0, help="最多等待多少秒（隧道刚启动时用）")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    url = find_url()
    deadline = time.time() + max(0.0, args.wait)
    while url is None and time.time() < deadline:
        time.sleep(3)
        url = find_url()

    if url is None:
        print(f"还没在 {LOG} 里找到 trycloudflare 地址。")
        print("先运行 deploy\\share_web.cmd，等它打印出 https:// 开头的那一行。")
        return 1

    print(f"公网地址：{url}")
    print(f"本机地址：http://localhost:8501")
    print("")
    for label, target in (("本机健康检查", LOCAL_HEALTH), ("公网健康检查", url + "/_stcore/health")):
        line, _ = probe(label, target)
        print(line)
    print("")
    print("停止：在 share_web.cmd 窗口按 Ctrl+C，然后运行 deploy\\stop_web.cmd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
