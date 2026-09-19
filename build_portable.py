"""打包便携版：生成"解压即用"的绿色文件夹（含内嵌 Python、依赖、模型）并压缩成 zip。

产物：dist/面试Copilot/  →  dist/面试Copilot便携版.zip
用法：python build_portable.py
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
PKG = DIST / "面试Copilot"
PY_VER = "3.13.7"
EMBED_URLS = [
    f"https://registry.npmmirror.com/-/binary/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip",
    f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip",
]
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
BUNDLE_MODELS = [
    "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",  # GPU 用
    "models--Systran--faster-whisper-base",                    # 无 GPU 回退用
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log")

PERSONA_TEMPLATE = """\
# 候选人画像：在软件里上传简历点「解析简历」自动生成（可手动修改后保存），也可以直接手写

## 基本信息
- 例：X 年 XX 方向经验 / XX 专业在读

## 技术亮点
- 例：3~5 条一句话亮点

## 项目亮点
- 例：2~3 条带数字结果的项目

## 求职意向
- 目标岗位 / 方向
"""

BAT_MAIN = """@echo off
cd /d %~dp0
set HF_HOME=%~dp0hf_cache
start "" "runtime\\pythonw.exe" "app\\app.py"
"""
BAT_DEBUG = """@echo off
cd /d %~dp0
set HF_HOME=%~dp0hf_cache
"runtime\\python.exe" "app\\app.py"
pause
"""
README_TXT = """\
面试 Copilot 便携版 — 使用说明
=================================

【启动】双击「启动面试Copilot.bat」，稍等几秒会弹出软件窗口（无黑窗口）。
        窗口没出现或想看报错时，改用「调试模式.bat」。

【首次使用三步】
1. 左栏「LLM 设置」填你自己的 API Key（DeepSeek / GLM / Kimi 等 OpenAI 兼容接口均可），保存；
2. 上传简历（PDF/DOCX/TXT/MD）→ 点「解析简历」自动生成候选人画像 → 可手动修改 → 保存画像；
3. 点「开始面试」，然后播放任何能出声的面试（腾讯会议/Zoom/视频都行），
   程序抓系统声音，面试官提问和 AI 答案实时显示在右侧。

【说明】
- 模型已内置，断网也能转写；有 NVIDIA 显卡自动用高精度模型，没有则自动用轻量模型。
- 转写出现同音字错误（如"线程→现成"）时，编辑 app/config.toml 里的 hotwords，加上相关领域词。
- 仅支持 Windows 10/11；会话记录在 app/logs/ 下。
- 真面试中使用属灰色地带，建议以模拟面试陪练为主。
"""


def download_embeddable() -> Path:
    zip_path = DIST / f"python-{PY_VER}-embed-amd64.zip"
    if zip_path.exists():
        return zip_path
    DIST.mkdir(exist_ok=True)
    last = None
    for url in EMBED_URLS:
        try:
            print(f"下载内嵌 Python: {url}")
            urllib.request.urlretrieve(url, zip_path)
            return zip_path
        except Exception as e:
            last = e
    raise RuntimeError(f"内嵌 Python 下载失败: {last}")


def copytree(src, dst, **kw):
    print(f"复制 {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True, **kw)


def main():
    if PKG.exists():
        print("清理旧的打包目录 ...")
        shutil.rmtree(PKG)

    # 1) 内嵌 Python + 依赖
    runtime = PKG / "runtime"
    runtime.mkdir(parents=True)
    with zipfile.ZipFile(download_embeddable()) as z:
        z.extractall(runtime)
    copytree(ROOT / ".venv" / "Lib" / "site-packages", runtime / "Lib" / "site-packages", ignore=IGNORE)
    pth = runtime / f"python{sys.version_info.major}{sys.version_info.minor}._pth"
    pth.write_text("python313.zip\n.\nLib\\site-packages\nimport site\n", encoding="ascii")

    # 2) 应用代码
    app = PKG / "app"
    app.mkdir(parents=True)
    for name in ("app.py", "server.py", "main.py", "config.toml", "persona.md", "README.md",
                 "requirements.txt", "copilot", "web"):
        src = ROOT / name
        if src.is_dir():
            copytree(src, app / name, ignore=IGNORE)
        else:
            shutil.copy2(src, app / name)
    copytree(ROOT / "models", app / "models")  # Silero VAD

    # 安全：包里绝不能带开发者自己的 API Key（解析后整写，不怕用户手工改过格式）
    import tomllib
    import tomli_w

    cfg_path = app / "config.toml"
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("llm", {})["api_key"] = ""
    cfg_path.write_text(tomli_w.dumps(cfg), encoding="utf-8")
    # 隐私：包里不能带开发者自己的候选人画像，统一换成空白模板
    (app / "persona.md").write_text(PERSONA_TEMPLATE, encoding="utf-8")

    # 3) 预置转写模型（断网可用）
    hub = PKG / "hf_cache" / "hub"
    hub.mkdir(parents=True)
    for name in BUNDLE_MODELS:
        src = HF_CACHE / name
        if src.exists():
            copytree(src, hub / name)
        else:
            print(f"!! 本地缓存没有 {name}，收件人首次运行会联网下载")

    # 4) 启动脚本和说明
    (PKG / "启动面试Copilot.bat").write_text(BAT_MAIN, encoding="ascii")
    (PKG / "调试模式.bat").write_text(BAT_DEBUG, encoding="ascii")
    (PKG / "使用说明.txt").write_text(README_TXT, encoding="utf-8")

    # 5) 压缩
    print("压缩 zip（约几分钟）...")
    shutil.make_archive(str(DIST / "面试Copilot便携版"), "zip", root_dir=DIST, base_dir="面试Copilot")

    def size(p):
        p = Path(p)
        if p.is_file():
            return p.stat().st_size / 1e9
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    print(f"\n完成：{PKG}  ({size(PKG):.1f} GB)")
    print(f"      {DIST / '面试Copilot便携版.zip'}  ({size(DIST / '面试Copilot便携版.zip'):.1f} GB)")


if __name__ == "__main__":
    main()
