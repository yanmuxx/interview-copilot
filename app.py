"""桌面启动器：后台起本地服务，然后开一个 WebView 窗口（失败则退回浏览器）。

用法：python app.py
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

if sys.stdout is None:  # pythonw 启动（无控制台）时 print 会崩，重定向到空设备
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _free_port()
URL = f"http://127.0.0.1:{PORT}"


def _serve():
    import uvicorn
    from server import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def _wait_ready(timeout=15):
    for _ in range(timeout * 5):
        try:
            urllib.request.urlopen(URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def main():
    threading.Thread(target=_serve, daemon=True).start()
    if not _wait_ready():
        print("[app] 服务启动失败")
        sys.exit(1)

    try:
        import webview  # pywebview：原生窗口（WebView2），失败就退回浏览器
        webview.create_window("面试 Copilot", URL, width=1150, height=780, min_size=(900, 600), x=120, y=60)
        print(f"[app] 窗口已打开: {URL}")
        webview.start()   # 阻塞到窗口关闭
    except Exception as e:
        print(f"[app] 桌面窗口不可用({e})，改用浏览器打开")
        webbrowser.open(URL)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
