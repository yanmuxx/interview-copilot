"""可启停的面试流水线：采集 → VAD → 转写 → LLM，事件通过回调推给界面。"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from copilot.advisor import Advisor
from copilot.asr import Transcriber
from copilot.capture import LoopbackCapture
from copilot.vad import SileroVAD, SpeechSegmenter, ensure_model

ROOT = Path(__file__).resolve().parent.parent
_CACHE: dict = {}


def _get_transcriber(asr_cfg: dict) -> Transcriber:
    key = (f"asr:{asr_cfg['model']}:{asr_cfg.get('device', 'auto')}:"
           f"{asr_cfg.get('compute_type', 'int8')}:{asr_cfg.get('hotwords', '')}")
    if key not in _CACHE:
        _CACHE[key] = Transcriber(
            asr_cfg["model"],
            device=asr_cfg.get("device", "auto"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            hotwords=asr_cfg.get("hotwords", ""),
        )
    return _CACHE[key]


def _get_vad() -> SileroVAD:
    if "vad" not in _CACHE:
        _CACHE["vad"] = SileroVAD(ensure_model(ROOT / "models" / "silero_vad.onnx"))
    return _CACHE["vad"]


class InterviewPipeline:
    """同一时刻只允许一个实例在跑；事件（status/transcript/answer_delta/...）经 on_event 推送。"""

    def __init__(self, cfg: dict, persona: str, mode: str, on_event):
        self.cfg = cfg
        self.mode = mode
        self.on_event = on_event
        self._stop = threading.Event()
        self._seg_q: queue.Queue = queue.Queue(maxsize=8)
        self._threads: list[threading.Thread] = []

        api_key = cfg["llm"].get("api_key") or ""
        self.advisor = None
        if api_key:
            self.advisor = Advisor(
                api_key, cfg["llm"]["base_url"], cfg["llm"]["model"], persona, mode=mode
            )

        asr_cfg = cfg["asr"]
        self.transcriber = _get_transcriber(asr_cfg)
        self.language = asr_cfg.get("language") or None
        self.segmenter = SpeechSegmenter(
            _get_vad(),
            threshold=cfg["vad"]["threshold"],
            min_speech_ms=cfg["vad"]["min_speech_ms"],
            min_silence_ms=cfg["vad"]["min_silence_ms"],
        )
        self._log_file = ROOT / "logs" / f"session-{datetime.now():%Y%m%d-%H%M%S}.md"

    # ---------- 生命周期 ----------
    def start(self):
        for target, name in ((self._capture_loop, "capture"), (self._asr_worker, "asr")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self.on_event({"type": "status", "state": "listening",
                       "hint": "" if self.advisor else "未配置 API Key，只转写不出答案"})

    def stop(self):
        self._stop.set()
        cur = threading.current_thread()
        for t in self._threads:
            if t is not cur:             # 等转写/回答线程收尾，把最后一条回答送完
                t.join(timeout=5)
        self.on_event({"type": "status", "state": "stopped"})

    # ---------- 线程体 ----------
    def _capture_loop(self):
        try:
            with LoopbackCapture() as cap:
                while not self._stop.is_set():
                    try:
                        pcm = next(cap.frames())
                    except (OSError, StopIteration):  # 流被 stop 关闭
                        break
                    segment = self.segmenter.feed(pcm)
                    if segment is None:
                        continue
                    try:
                        self._seg_q.put_nowait(segment)
                    except queue.Full:      # 积压时丢最老的问题，保实时性
                        try:
                            self._seg_q.get_nowait()
                        except queue.Empty:
                            pass
                        self._seg_q.put_nowait(segment)
        except Exception as e:
            self.on_event({"type": "error", "message": f"音频采集失败: {e}"})
            self.stop()
    def _asr_worker(self):
        while not self._stop.is_set() or not self._seg_q.empty():
            try:
                audio = self._seg_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t0 = time.time()
                text = self.transcriber(audio, language=self.language)
                if not text:
                    continue
                self.on_event({"type": "transcript", "text": text, "secs": round(time.time() - t0, 1)})
                answer = self._answer(text)
                self._append_log(text, answer)
            except Exception as e:
                self.on_event({"type": "error", "message": str(e)})
            finally:
                self._seg_q.task_done()

    def _answer(self, question: str) -> str:
        if not self.advisor:
            return ""
        answer = []
        try:
            for delta in self.advisor.stream_answer(question):
                answer.append(delta)
                self.on_event({"type": "answer_delta", "delta": delta})
        finally:
            # 出错/被打断也要收尾，界面靠它去掉打字机光标
            self.on_event({"type": "answer_done", "answer": "".join(answer)})
        return "".join(answer)

    def _append_log(self, question: str, answer: str):
        self._log_file.parent.mkdir(exist_ok=True)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%H:%M:%S} 面试官\n{question}\n")
            if answer:
                f.write(f"\n**要点**\n{answer}\n")
