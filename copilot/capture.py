"""WASAPI 回环采集：直接抓取系统正在播放的声音（会议软件里的面试官语音）。

无需虚拟声卡、无需麦克风权限，输出 16kHz 单声道 float32 PCM。
"""
from __future__ import annotations

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16000


def _resample(x: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate or x.size == 0:
        return x
    n = max(1, round(x.size / src_rate * dst_rate))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


class LoopbackCapture:
    """以 chunk_ms 为单位产出重采样到 16kHz 的系统音频。"""

    def __init__(self, chunk_ms: int = 32):
        self.chunk_ms = chunk_ms
        self._pa = None
        self._stream = None

    @staticmethod
    def _pick_device(pa: pyaudio.PyAudio) -> dict:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if default_out.get("isLoopbackDevice"):
            return default_out
        # 默认输出设备本身不是回环设备时，找它对应的回环镜像
        for lb in pa.get_loopback_device_info_generator():
            if default_out["name"] in lb["name"]:
                return lb
        try:
            return next(iter(pa.get_loopback_device_info_generator()))
        except StopIteration:
            raise RuntimeError(
                "没有找到任何可用的回环录音设备：请确认系统有正常工作的播放设备（扬声器/耳机）"
            )

    def __enter__(self) -> "LoopbackCapture":
        self._pa = pyaudio.PyAudio()
        dev = self._pick_device(self._pa)
        self._rate = int(dev["defaultSampleRate"])
        self._channels = max(1, min(2, dev["maxInputChannels"]))
        self._frames = int(self._rate * self.chunk_ms / 1000)
        print(f"[capture] 采集设备: {dev['name']} @{self._rate}Hz x{self._channels}ch")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=self._frames,
        )
        return self

    def __exit__(self, *exc):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    def frames(self):
        while True:
            data = self._stream.read(self._frames, exception_on_overflow=False)
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            mono = pcm.reshape(-1, self._channels).mean(axis=1)
            yield _resample(mono, self._rate)
