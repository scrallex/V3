"""CUDA-accelerated helpers for per-window bit histograms."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import cupy as cp
except ImportError:  # pragma: no cover - CUDA runtime may be unavailable
    cp = None
else:  # pragma: no branch - guard against environments without devices
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - prefer safe fallback
        cp = None

HAVE_CUDA = cp is not None

if HAVE_CUDA:
    _KERNEL_SOURCE = r"""
    extern "C" __global__
    void window_histogram(
        const unsigned char* __restrict__ data,
        const int window_bytes,
        const int total_windows,
        const int* __restrict__ lengths,
        unsigned int* __restrict__ zeros,
        unsigned int* __restrict__ ones,
        unsigned int* __restrict__ transitions
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= total_windows) {
            return;
        }
        const unsigned char* window = data + idx * window_bytes;
        const int window_len = lengths[idx];
        unsigned int zero_count = 0;
        unsigned int one_count = 0;
        unsigned int transition_count = 0;
        int prev_bit = -1;
        for (int byte_idx = 0; byte_idx < window_len; ++byte_idx) {
            unsigned char value = window[byte_idx];
            for (int shift = 7; shift >= 0; --shift) {
                int bit = (value >> shift) & 0x1;
                zero_count += (bit == 0);
                one_count += (bit == 1);
                if (prev_bit != -1 && bit != prev_bit) {
                    transition_count += 1;
                }
                prev_bit = bit;
            }
        }
        zeros[idx] = zero_count;
        ones[idx] = one_count;
        transitions[idx] = transition_count;
    }
    """
    _HIST_KERNEL = cp.RawKernel(_KERNEL_SOURCE, "window_histogram")
else:
    _HIST_KERNEL = None


def _entropy_from_counts(zeros: np.ndarray, ones: np.ndarray) -> np.ndarray:
    total = zeros + ones
    with np.errstate(divide="ignore", invalid="ignore"):
        p_zero = np.divide(zeros, total, where=total > 0)
        p_one = np.divide(ones, total, where=total > 0)
        entropy = np.zeros_like(p_zero, dtype=np.float32)
        nonzero_zero = p_zero > 0
        entropy[nonzero_zero] -= p_zero[nonzero_zero] * np.log2(p_zero[nonzero_zero])
        nonzero_one = p_one > 0
        entropy[nonzero_one] -= p_one[nonzero_one] * np.log2(p_one[nonzero_one])
    return np.clip(entropy, 0.0, 1.0)


def gpu_window_metrics(windows: Sequence[bytes], *, window_bytes: int) -> List[Dict[str, float]]:
    """Return coherence/stability/entropy metrics for ``windows`` using CUDA."""

    if not HAVE_CUDA or _HIST_KERNEL is None:
        raise RuntimeError("CUDA runtime is not available")
    if not windows:
        return []

    total_windows = len(windows)
    lengths = np.empty(total_windows, dtype=np.int32)
    host_buffer = np.empty(total_windows * window_bytes, dtype=np.uint8)
    for idx, chunk in enumerate(windows):
        if len(chunk) > window_bytes:
            raise ValueError("Window larger than configured window_bytes")
        lengths[idx] = len(chunk)
        start = idx * window_bytes
        end = start + window_bytes
        data_view = np.frombuffer(chunk, dtype=np.uint8)
        host_buffer[start : start + len(chunk)] = data_view
        if len(chunk) < window_bytes:
            host_buffer[start + len(chunk) : end] = 0

    data_gpu = cp.asarray(host_buffer)
    zeros_gpu = cp.empty(total_windows, dtype=cp.uint32)
    ones_gpu = cp.empty_like(zeros_gpu)
    transitions_gpu = cp.empty_like(zeros_gpu)
    lengths_gpu = cp.asarray(lengths)

    threads = 128
    blocks = (total_windows + threads - 1) // threads
    _HIST_KERNEL(
        (blocks,),
        (threads,),
        (
            data_gpu,
            np.int32(window_bytes),
            np.int32(total_windows),
            lengths_gpu,
            zeros_gpu,
            ones_gpu,
            transitions_gpu,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    zeros = cp.asnumpy(zeros_gpu).astype(np.float32)
    ones = cp.asnumpy(ones_gpu).astype(np.float32)
    transitions = cp.asnumpy(transitions_gpu).astype(np.float32)

    bit_count = lengths.astype(np.float32) * 8.0
    entropy = _entropy_from_counts(zeros, ones)
    coherence = 1.0 - entropy
    rupture = np.divide(
        transitions,
        np.maximum(bit_count - 1.0, 1.0),
        out=np.zeros_like(transitions, dtype=np.float32),
    )
    stability = 1.0 - rupture

    results: List[Dict[str, float]] = []
    for idx in range(total_windows):
        results.append(
            {
                "coherence": float(np.clip(coherence[idx], 0.0, 1.0)),
                "stability": float(np.clip(stability[idx], 0.0, 1.0)),
                "entropy": float(entropy[idx]),
                "rupture": float(np.clip(rupture[idx], 0.0, 1.0)),
                "lambda_hazard": float(np.clip(rupture[idx], 0.0, 1.0)),
            }
        )
    return results


__all__ = ["gpu_window_metrics", "HAVE_CUDA"]
