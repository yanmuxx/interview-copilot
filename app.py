"""桌面启动器：本地服务 + 主窗口 + 可选悬浮窗（Alt+Q 全局热键呼出/隐藏）。

悬浮窗是正常可见的置顶小卡片（屏幕共享时同样会被共享出去），
不含任何"对录屏隐身"的能力——那是刻意的设计边界。

用法：python app.py
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
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

# 全局热键：Alt+Q
_MOD_ALT_NOREPEAT = 0x0001 | 0x4000
_VK_Q = 0x51
_WM_HOTKEY = 0x0312

_OVERLAY: dict = {"win": None}   # 悬浮窗句柄状态（跨线程读写只做整体替换，无竞争问题）


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


def _toggle_overlay():
    import webview
    win = _OVERLAY["win"]
    if win is not None:
        try:
            win.destroy()
        except Exception:
            pass
        _OVERLAY["win"] = None
        return
    # 固定放主屏左上角：任何机器上坐标 (60,90) 都必然可见。
    # 不要用 webview.screens 做贴边计算——DPI 虚拟化下它可能报假宽度，把窗口送出屏幕外。
    _OVERLAY["win"] = webview.create_window(
        "copilot-overlay", URL + "/overlay", js_api=Api(_OVERLAY),
        width=380, height=280, x=60, y=90,
        frameless=True, on_top=True, easy_drag=False, transparent=True,
    )


def _hide_overlay():
    win = _OVERLAY["win"]
    if win is not None:
        try:
            win.hide()
        except Exception:
            pass


def _close_overlay():
    win = _OVERLAY.pop("win", None)
    if win is not None:
        try:
            win.destroy()
        except Exception:
            pass


def _on_main_closed():
    _close_overlay()   # 主窗口关闭 = 程序退出（与既有行为一致）


class Api:
    """暴露给页面 JS 的方法：window.pywebview.api.xxx()"""

    def __init__(self, state: dict):
        self._state = state

    def toggle_overlay(self):
        _toggle_overlay()

    def hide_overlay(self):
        _hide_overlay()

    def close_overlay(self):
        _close_overlay()


def _hotkey_loop():
    """Alt+Q 全局热键（ctypes 原生 RegisterHotKey，无第三方依赖）。失败则静默降级。"""
    try:
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(None, 1, _MOD_ALT_NOREPEAT, _VK_Q):
            return
        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == _WM_HOTKEY:
                _toggle_overlay()
    except Exception:
        pass


def main():
    threading.Thread(target=_serve, daemon=True).start()
    if not _wait_ready():
        print("[app] 服务启动失败")
        sys.exit(1)

    try:
        import webview  # pywebview：原生窗口（WebView2），失败就退回浏览器
    except Exception as e:
        print(f"[app] 桌面窗口不可用({e})，改用浏览器打开")
        webbrowser.open(URL)
        while True:
            time.sleep(3600)

    threading.Thread(target=_hotkey_loop, daemon=True).start()
    api = Api(_OVERLAY)
    main_win = webview.create_window(
        "面试 Copilot", URL, js_api=api,
        width=1150, height=780, min_size=(900, 600), x=120, y=60,
    )
    main_win.events.closed += _on_main_closed
    print(f"[app] 窗口已打开: {URL}")
    webview.start()   # 阻塞到所有窗口关闭


if __name__ == "__main__":
    main()
