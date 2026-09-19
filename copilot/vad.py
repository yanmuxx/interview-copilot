"""Silero VAD（ONNX，约 2MB）：判断面试官是否在说话，并切出完整的问题片段。"""
from __future__ import annotations

import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://hf-mirror.com/onnx-community/silero-vad/resolve/main/onnx/silero_vad.onnx",
]
SAMPLE_RATE = 16000
WINDOW = 512          # Silero 标准 512 样本窗口（32ms）
CONTEXT = 64          # 模型内部需要的 64 样本上下文


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in MODEL_URLS:
        try:
            print(f"[vad] 下载模型 {url} ...")
            with urllib.request.urlopen(url, timeout=30) as resp, open(path, "wb") as f:
                while chunk := resp.read(1 << 16):
                    f.write(chunk)
            return path
        except Exception as e:  # 网络失败/超时逐个换源
            last_err = e
            if path.exists():
                path.unlink()
    raise RuntimeError(f"Silero VAD 模型下载失败，可手动下载放到 {path}: {last_err}")


class SileroVAD:
    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        input_names = {i.name for i in self.session.get_inputs()}
        self._v5 = "state" in input_names   # v4 与 v5 的状态张量接口不同
        self.reset()

    def reset(self):
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        """chunk: 恰好 512 个样本，返回该窗口的语音概率。"""
        x = np.concatenate([self._context, chunk[None, :].astype(np.float32)], axis=1)
        self._context = chunk[None, -CONTEXT:].astype(np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        if self._v5:
            out, self._state = self.session.run(None, {"input": x, "state": self._state, "sr": sr})
        else:
            out, self._h, self._c = self.session.run(None, {"input": x, "h": self._h, "c": self._c, "sr": sr})
        return float(np.asarray(out).reshape(-1)[0])


class SpeechSegmenter:
    """把 16kHz 音频流切成"面试官的一句话"片段。

    feed() 接收任意长度样本，返回完整语音段（np.ndarray）或 None。
    """

    def __init__(self, vad: SileroVAD, threshold=0.5, min_speech_ms=250,
                 min_silence_ms=450, pre_pad_ms=200):
        self.vad = vad
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._win_ms = WINDOW * 1000 // SAMPLE_RATE
        # 预滚缓冲按"窗口个数"计（deque 的 maxlen 数的是元素个数，元素=窗口）
        self._pre_max = max(1, round(pre_pad_ms / self._win_ms))
        self._reset()

    def _reset(self):
        self._acc = np.zeros(0, dtype=np.float32)   # 凑满 512 的缓冲
        self._pre = deque(maxlen=self._pre_max)
        self._in_speech = False
        self._sil_ms = 0
        self._speech_ms = 0                          # 纯语音时长（不含预滚和静音尾）
        self._seg = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        self._acc = np.concatenate([self._acc, samples])
        segment = None
        while self._acc.size >= WINDOW:
            window, self._acc = self._acc[:WINDOW], self._acc[WINDOW:]
            p = self.vad.prob(window)
            if not self._in_speech:
                self._pre.append(window)
                if p >= self.threshold:
                    self._in_speech = True
                    self._sil_ms = 0
                    self._speech_ms = self._win_ms
                    self._seg = np.concatenate([*self._pre, window])
            else:
                self._seg = np.concatenate([self._seg, window])
                if p >= self.threshold:
                    self._speech_ms += self._win_ms
                    self._sil_ms = 0
                else:
                    self._sil_ms += self._win_ms
                    if self._sil_ms >= self.min_silence_ms:
                        # 是否成段只看纯语音时长，预滚缓冲不算数
                        if self._speech_ms >= self.min_speech_ms:
                            segment = self._seg
                        self._in_speech = False
                        self._pre.clear()
                        self._seg = np.zeros(0, dtype=np.float32)
                        self._speech_ms = 0
        return segment
