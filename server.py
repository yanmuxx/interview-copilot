"""本地服务：界面页面 + 配置/简历/画像接口 + 面试启停 + SSE 实时事件流。

由 app.py 启动（桌面窗口），也可以 `python -m uvicorn server:app` 后开浏览器。
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import tomllib
from pathlib import Path

import tomli_w
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from copilot.pipeline import InterviewPipeline
from copilot.resume import distill_persona, extract_text

app = FastAPI()
_LOCK = threading.Lock()
_STATE: dict = {"pipeline": None, "history": [], "starting": False, "stop_requested": False}

MAX_UPLOAD = 20 * 1024 * 1024  # 简历文件上限


@app.middleware("http")
async def csrf_guard(request: Request, call_next):
    # 界面永远和 API 同源；带 Origin 头的其他来源一律拒绝，
    # 防止恶意网页向 127.0.0.1 发跨站表单请求（换 Key / 清画像 / 干扰面试）
    origin = request.headers.get("origin")
    if origin and origin != f"http://{request.headers.get('host', '')}":
        return JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
    return await call_next(request)


# ---------------- 基础工具 ----------------
def _read_cfg() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def _write_cfg(cfg: dict):
    with open(ROOT / "config.toml", "wb") as f:
        tomli_w.dump(cfg, f)


def _persona_path() -> Path:
    return ROOT / "persona.md"


def _broadcast(ev: dict):
    with _LOCK:
        history = _STATE["history"]
        history.append(ev)
        if len(history) > 2000:          # 防长面试时 answer_delta 无限积累
            del history[:1000]
        for sub in _STATE.get("subs", []):
            sub.put(ev)


def _make_sub() -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _LOCK:
        _STATE.setdefault("subs", []).append(q)
    return q


def _drop_sub(q: queue.Queue):
    with _LOCK:
        subs = _STATE.get("subs", [])
        if q in subs:
            subs.remove(q)


# ---------------- 页面 ----------------
@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


# ---------------- 配置 ----------------
@app.get("/api/config")
def get_config():
    cfg = _read_cfg()
    return {
        "has_key": bool(cfg["llm"].get("api_key")),
        "base_url": cfg["llm"].get("base_url", ""),
        "model": cfg["llm"].get("model", ""),
    }


@app.post("/api/config")
def save_config(api_key: str = Form(""), base_url: str = Form(""), model: str = Form("")):
    cfg = _read_cfg()
    if api_key:
        cfg["llm"]["api_key"] = api_key.strip()
    cfg["llm"]["base_url"] = base_url.strip() or cfg["llm"].get("base_url", "")
    cfg["llm"]["model"] = model.strip() or cfg["llm"].get("model", "")
    _write_cfg(cfg)
    return {"ok": True, "has_key": bool(cfg["llm"]["api_key"])}


# ---------------- 画像 ----------------
@app.get("/api/persona")
def get_persona():
    return {"persona": _persona_path().read_text(encoding="utf-8")}


@app.post("/api/persona")
def save_persona(persona: str = Form("")):
    _persona_path().write_text(persona, encoding="utf-8")
    return {"ok": True}


@app.post("/api/resume/parse")
async def parse_resume(file: UploadFile | None = File(None), jd: str = Form("")):
    cfg = _read_cfg()
    api_key = cfg["llm"].get("api_key") or ""
    if not api_key:
        raise HTTPException(400, "请先在「LLM 设置」里保存 API Key")
    try:
        if not (file and file.filename):
            raise HTTPException(400, "请先选择简历文件")
        if (file.size and file.size > MAX_UPLOAD):
            raise HTTPException(413, "简历文件超过 20MB 上限")
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "简历文件超过 20MB 上限")
        text = extract_text(file.filename, data)
        persona = distill_persona(text, jd, api_key, cfg["llm"]["base_url"], cfg["llm"]["model"])
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"AI 解析失败（检查 Key/网络）: {e}")
    return {"persona": persona}


# ---------------- 面试控制 ----------------
@app.post("/api/interview/start")
def start_interview(mode: str = Form("brief")):
    # starting 标记在锁内占位：模型加载的几秒里，第二次"开始"和"停止"都不会产生竞态
    with _LOCK:
        if _STATE["pipeline"] is not None or _STATE["starting"]:
            raise HTTPException(409, "面试已在进行中")
        _STATE["starting"] = True
        _STATE["stop_requested"] = False
    try:
        cfg = _read_cfg()
        persona = _persona_path().read_text(encoding="utf-8")
        if not persona.strip():
            raise HTTPException(400, "候选人画像为空，请先导入简历或手写画像")

        pipe = InterviewPipeline(
            cfg, persona, mode if mode in ("brief", "full") else "brief", _broadcast
        )
        with _LOCK:
            aborted = _STATE["stop_requested"]  # 加载模型期间用户按了停止
            _STATE["pipeline"] = None if aborted else pipe
            if not aborted:
                _STATE["history"].clear()
        if aborted:
            return {"ok": True, "aborted": True, "has_llm": False}
        pipe.start()
        return {"ok": True, "has_llm": pipe.advisor is not None}
    finally:
        with _LOCK:
            _STATE["starting"] = False


@app.post("/api/interview/stop")
def stop_interview():
    with _LOCK:
        pipe = _STATE["pipeline"]
        _STATE["pipeline"] = None
        _STATE["stop_requested"] = True
    if pipe:
        pipe.stop()   # 内部会等转写/回答线程收尾，再广播 stopped，避免丢回答尾巴
    return {"ok": True}


@app.get("/api/interview/state")
def interview_state():
    return {"running": _STATE["pipeline"] is not None}


# ---------------- SSE 实时事件 ----------------
@app.get("/api/interview/events")
def events():
    sub = _make_sub()

    def gen():
        try:
            while True:
                try:
                    ev = sub.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"      # 心跳，防止连接被掐断
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev.get("type") == "status" and ev.get("state") == "stopped":
                    break
        finally:
            _drop_sub(sub)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/interview/history")
def history():
    with _LOCK:
        return {"events": list(_STATE["history"])}
