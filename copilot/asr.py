"""faster-whisper 本地转写：优先 GPU（CUDA），失败自动回退 CPU。全程离线。"""
from __future__ import annotations

import os

# 默认走国内镜像下载模型，已经设置了 HF_ENDPOINT 的话以用户设置为准
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 新模型仓库走 Xet 存储后端，其 CDN（us.aws.cdn.hf.co）国内直连不通，
# 禁用后回落到镜像站的普通 HTTP 下载（large-v3 实测需要）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Windows 下让 ctranslate2 找到 pip 安装的 CUDA 运行库（nvidia-cublas-cu12 / nvidia-cudnn-cu12）
import site

for _sp in site.getsitepackages():
    for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        _d = os.path.join(_sp, _sub)
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="large-v3-turbo", device="auto", compute_type="int8", hotwords=""):
        print(f"[asr] 加载 whisper {model_size}（首次运行会自动下载模型）...")
        self.model, self.device = self._load(model_size, device, compute_type)
        self.hotwords = hotwords or None

    @staticmethod
    def _load(model_size, device, compute_type):
        if device == "cuda":
            return WhisperModel(model_size, device="cuda", compute_type=compute_type), "cuda"
        if device == "cpu":
            return WhisperModel(model_size, device="cpu", compute_type=compute_type), "cpu"
        try:  # auto：优先 GPU（float16），失败回退 CPU
            return WhisperModel(model_size, device="cuda", compute_type="float16"), "cuda"
        except Exception as e:
            # 大模型在 CPU 上慢到不可用，回退时自动换轻量模型
            if model_size in ("large-v3-turbo", "large-v3", "medium"):
                print(f"[asr] GPU 不可用({type(e).__name__})，CPU 模式自动改用 base")
                model_size = "base"
            else:
                print(f"[asr] GPU 不可用({type(e).__name__})，回退 CPU {compute_type}")
            return WhisperModel(model_size, device="cpu", compute_type=compute_type), "cpu"

    def __call__(self, audio: np.ndarray, language: str | None = None) -> str:
        # 中文面试用简体提示词引导输出，避免输出繁体
        prompt = "以下是普通话的句子。" if (language or "").startswith("zh") else None
        segments, _info = self.model.transcribe(
            audio,
            language=language or None,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt=prompt,
            hotwords=self.hotwords,  # 纠正领域同音字：线程/现成、分布式锁/分布失所
        )
        return "".join(s.text for s in segments).strip()
