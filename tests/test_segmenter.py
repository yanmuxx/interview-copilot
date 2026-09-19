"""SpeechSegmenter 切分逻辑测试：用脚本化的假 VAD 概率序列驱动，不依赖真实模型。"""
import numpy as np
import pytest

from copilot.vad import SpeechSegmenter

SR = 16000
WIN = 512  # 32ms


class FakeVAD:
    """按预设顺序吐概率；每个概率对应一个 512 样本窗口。"""

    def __init__(self, probs):
        self._probs = iter(probs)

    def prob(self, chunk):
        return next(self._probs)


def feed_windows(seg, n, value):
    return seg.feed(np.full(n * WIN, value, dtype=np.float32))


def test_silence_never_emits():
    seg = SpeechSegmenter(FakeVAD([0.05] * 50))
    assert feed_windows(seg, 50, 0.0) is None


def test_speech_then_silence_emits_segment():
    # 20 窗静音 → 12 窗语音(384ms≥min_speech) → 16 窗静音(512ms≥min_silence) → 出段
    probs = [0.05] * 20 + [0.95] * 12 + [0.05] * 16
    seg = SpeechSegmenter(FakeVAD(probs))
    result = seg.feed(np.zeros(len(probs) * WIN, dtype=np.float32))
    assert result is not None
    dur_ms = result.size * 1000 // SR
    # 语音 384ms + 静音尾 ~512ms + 预滚动 ~200ms
    assert 700 <= dur_ms <= 1300, f"切出时长异常: {dur_ms}ms"


def test_short_burst_discarded():
    # 2 窗语音(64ms < min_speech 250ms)应被丢弃
    probs = [0.05] * 10 + [0.95] * 2 + [0.05] * 16
    seg = SpeechSegmenter(FakeVAD(probs))
    result = seg.feed(np.zeros(len(probs) * WIN, dtype=np.float32))
    assert result is None


def test_continuous_speech_not_emitted_until_silence():
    # 一直说话没有静默：不应出段（还在攒）
    seg = SpeechSegmenter(FakeVAD([0.95] * 60))
    assert feed_windows(seg, 60, 0.0) is None


def test_threshold_boundary():
    # 概率恰好等于阈值视为语音
    probs = [0.05] * 5 + [0.5] * 12 + [0.05] * 16
    seg = SpeechSegmenter(FakeVAD(probs), threshold=0.5)
    assert seg.feed(np.zeros(len(probs) * WIN, dtype=np.float32)) is not None
