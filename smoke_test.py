"""冒烟测试：逐个验证各模块能否工作（不依赖真实面试音频和 LLM key）。"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np

print("== 1. VAD ==")
from copilot.vad import SileroVAD, SpeechSegmenter, ensure_model
model_path = ensure_model(ROOT / "models" / "silero_vad.onnx")
vad = SileroVAD(model_path)
rng = np.random.default_rng(0)
sil = (rng.standard_normal(512 * 10) * 0.001).astype(np.float32)  # 近似静音
probs = [vad.prob(sil[i*512:(i+1)*512]) for i in range(10)]
print(f"静音窗口语音概率: {['%.3f' % p for p in probs]}")
assert all(p < 0.5 for p in probs), "静音不应被判为语音"
print("VAD OK")

print("== 2. 采集 ==")
from copilot.capture import LoopbackCapture
frames = 0
with LoopbackCapture() as cap:
    for pcm in cap.frames():
        frames += 1
        if frames >= 15:  # 15 * 32ms ≈ 0.5s
            break
print(f"采集到 {frames} 个 chunk，每个 {pcm.size} 样本 @{cap._rate}Hz->16kHz")
print("Capture OK")

print("== 3. 转写模型加载 ==")
from copilot.asr import Transcriber
t0 = time.time()
tr = Transcriber("base")
text = tr(sil, language="zh")
print(f"加载+推理耗时 {time.time()-t0:.1f}s，静音转写结果: '{text}' (应为空)")
print("ASR OK")
print("\n全部通过 ✔  （LLM 需要真实 api_key，启动 main.py 时自动检测）")
