"""面试 Copilot 主程序：系统音频 → VAD 切分 → 本地转写 → LLM 要点提示。

用法：python main.py
对着电脑播放一段面试音频（或开个会议软件说话）即可看到效果。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from copilot.advisor import Advisor
from copilot.asr import Transcriber
from copilot.capture import LoopbackCapture
from copilot.vad import SAMPLE_RATE, SileroVAD, SpeechSegmenter, ensure_model

os.system("")  # 让 Windows 终端支持 ANSI 颜色

C = {"q": "\033[1;36m", "a": "\033[1;33m", "dim": "\033[2m", "err": "\033[1;31m", "end": "\033[0m"}
print_lock = threading.Lock()
log_file = ROOT / "logs" / f"session-{datetime.now():%Y%m%d-%H%M%S}.md"


def log_to_file(question: str, answer: str):
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%H:%M:%S} 面试官\n{question}\n")
        if answer:
            f.write(f"\n**要点**\n{answer}\n")


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def asr_worker(seg_q: queue.Queue, transcriber: Transcriber, advisor: Advisor | None, cfg: dict):
    """转写 + 出要点。转写耗时较长，放独立线程，VAD 继续跑不丢音频。"""
    while True:
        audio = seg_q.get()
        try:
            t0 = time.time()
            text = transcriber(audio, language=cfg["asr"].get("language") or None)
            if not text:
                continue
            dt = time.time() - t0
            with print_lock:
                print(f"\n{C['q']}🎤 面试官 ({dt:.1f}s): {text}{C['end']}")
            answer = ""
            if advisor:
                answer = answer_question(advisor, text)
            log_to_file(text, answer)
        except Exception as e:
            with print_lock:
                print(f"{C['err']}[asr] 出错: {e}{C['end']}")
        finally:
            seg_q.task_done()


def answer_question(advisor: Advisor, question: str) -> str:
    answer = []
    with print_lock:
        print(f"{C['a']}💡 要点: ", end="", flush=True)
    try:
        for delta in advisor.stream_answer(question):
            answer.append(delta)
            with print_lock:
                print(delta, end="", flush=True)
    finally:
        with print_lock:
            print(C["end"])
    return "".join(answer)


def main():
    cfg = load_config()
    api_key = cfg["llm"].get("api_key") or os.environ.get("LLM_API_KEY", "")

    print(f"""{C['dim']}┌──────────────────────────────────────────────┐
│  面试 Copilot  ·  Ctrl+C 退出                │
│  播放任意面试/会议音频即可，无需麦克风        │
└──────────────────────────────────────────────┘{C['end']}""")

    vad = SileroVAD(ensure_model(ROOT / "models" / "silero_vad.onnx"))
    segmenter = SpeechSegmenter(
        vad,
        threshold=cfg["vad"]["threshold"],
        min_speech_ms=cfg["vad"]["min_speech_ms"],
        min_silence_ms=cfg["vad"]["min_silence_ms"],
    )
    transcriber = Transcriber(
        cfg["asr"]["model"],
        device=cfg["asr"].get("device", "auto"),
        compute_type=cfg["asr"].get("compute_type", "int8"),
        hotwords=cfg["asr"].get("hotwords", ""),
    )

    advisor = None
    if api_key:
        persona = (ROOT / "persona.md").read_text(encoding="utf-8")
        advisor = Advisor(api_key, cfg["llm"]["base_url"], cfg["llm"]["model"], persona)
        print(f"[llm] 已接入 {cfg['llm']['model']}，问题会自动生成答题要点")
    else:
        print(f"{C['err']}[llm] 未配置 api_key（config.toml 或环境变量 LLM_API_KEY），只转写不出要点{C['end']}")

    seg_q: queue.Queue = queue.Queue(maxsize=8)
    threading.Thread(target=asr_worker, args=(seg_q, transcriber, advisor, cfg), daemon=True).start()

    print(f"{C['dim']}监听中 ...{C['end']}")
    with LoopbackCapture() as cap:
        for pcm in cap.frames():
            segment = segmenter.feed(pcm)
            if segment is not None:
                try:
                    seg_q.put_nowait(segment * 1.0)  # float32, 16kHz
                except queue.Full:
                    pass  # 转写积压时丢弃最老的问题不合适，这里直接丢新的


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C['dim']}已退出，本次会话记录在 {log_file}{C['end']}")
