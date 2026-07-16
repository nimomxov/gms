"""
GMS — Compute Backend Abstraction  v1.0
=========================================
GPU-ready compute layer with automatic CPU fallback.

Architecture principle:
  All heavy numerical operations in GMS are routed through this
  module. Currently they run on CPU (NumPy/SciPy). When CuPy,
  numba.cuda, or OpenCL become available, zero application code
  changes are required — only this module needs updating.

Supported backends (in priority order):
  1. CuPy      — NVIDIA GPU (CUDA)
  2. numba.cuda — NVIDIA GPU (JIT compiled)
  3. NumPy      — CPU (always available, always correct)

Operations abstracted:
  - array creation and manipulation
  - FFT (used by wavelet baseline, matched filter)
  - 2-D convolution (smoothing, matched filter kernel)
  - Gaussian smoothing
  - Interpolation grid evaluation
  - Statistical reduction (mean, std, percentile)

Usage:
    from core.compute import xp, fft_module, compute_info
    # xp is cupy or numpy — use identically
    grid = xp.zeros((100, 100))
    spectrum = fft_module.fft2(grid)

    # Profiling:
    from core.compute import ComputeProfiler
    with ComputeProfiler("interpolation") as p:
        result = xp.interp(...)
    print(p.elapsed_ms)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("gms.compute")


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend():
    """Detect best available compute backend. Returns (module, name, info)."""

    # Try CuPy
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        mem = cp.cuda.runtime.memGetInfo()
        free_mb  = mem[0] // (1024**2)
        total_mb = mem[1] // (1024**2)
        info = {
            "backend": "cupy",
            "device": device.id,
            "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "vram_free_mb": free_mb,
            "vram_total_mb": total_mb,
        }
        logger.info(f"[Compute] CuPy backend: {info['gpu_name']} "
                    f"({free_mb}/{total_mb} MB free)")
        return cp, "cupy", info
    except Exception:
        pass

    # Try numba.cuda (JIT path — not used for array ops, but signals CUDA presence)
    try:
        from numba import cuda as numba_cuda
        if numba_cuda.is_available():
            dev = numba_cuda.get_current_device()
            info = {
                "backend": "numba_cuda",
                "device": 0,
                "gpu_name": str(dev.name),
                "note": "Array ops on CPU; CUDA JIT available for custom kernels",
            }
            logger.info(f"[Compute] numba.cuda present: {dev.name}")
            # Still use numpy for arrays, but flag CUDA available
            return np, "numba_cuda_numpy", info
    except Exception:
        pass

    # CPU only
    info = {
        "backend": "numpy",
        "device": "cpu",
        "note": "Install CuPy for GPU acceleration: pip install cupy-cuda12x",
    }
    logger.info("[Compute] CPU backend (NumPy)")
    return np, "numpy", info


# ── Module-level singletons ───────────────────────────────────────────────────

xp, _BACKEND_NAME, _BACKEND_INFO = _detect_backend()

try:
    if _BACKEND_NAME == "cupy":
        import cupy.fft as fft_module
    else:
        import numpy.fft as fft_module
except Exception:
    import numpy.fft as fft_module


def compute_info() -> dict:
    """Return current compute backend info for the status bar / diagnostics."""
    return dict(_BACKEND_INFO)


def is_gpu_available() -> bool:
    return _BACKEND_NAME == "cupy"


# ─────────────────────────────────────────────────────────────────────────────
# Type-safe array helpers
# (identical API for cupy and numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def to_device(array: np.ndarray):
    """Move a numpy array to the compute device (GPU if available)."""
    if is_gpu_available():
        return xp.asarray(array)
    return array


def to_host(array) -> np.ndarray:
    """Move a device array back to CPU numpy."""
    if is_gpu_available():
        return xp.asnumpy(array)
    return np.asarray(array)


def device_zeros(shape: tuple, dtype=np.float64):
    return xp.zeros(shape, dtype=dtype)


def device_ones(shape: tuple, dtype=np.float64):
    return xp.ones(shape, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Compute-accelerated operations
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing.
    CuPy path uses cupyx.scipy.ndimage; CPU path uses scipy.ndimage.
    """
    if sigma <= 0:
        return grid
    if is_gpu_available():
        try:
            from cupyx.scipy import ndimage as cp_ndimage
            g = to_device(grid)
            return to_host(cp_ndimage.gaussian_filter(g, sigma=sigma))
        except Exception as e:
            logger.debug(f"[Compute] CuPy gaussian_smooth failed, falling back: {e}")
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(grid.astype(np.float64), sigma=sigma)


def fft2(grid: np.ndarray) -> np.ndarray:
    """2-D FFT. Uses GPU if available."""
    if is_gpu_available():
        try:
            g = to_device(grid)
            return to_host(fft_module.fft2(g))
        except Exception:
            pass
    return np.fft.fft2(grid)


def ifft2(spectrum: np.ndarray) -> np.ndarray:
    """Inverse 2-D FFT."""
    if is_gpu_available():
        try:
            s = to_device(spectrum)
            return to_host(fft_module.ifft2(s)).real
        except Exception:
            pass
    return np.fft.ifft2(spectrum).real


def convolve2d(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D convolution via FFT.
    Used by matched filter and wavelet baseline.
    """
    if is_gpu_available():
        try:
            from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
            g = to_device(grid)
            k = to_device(kernel)
            return to_host(cp_fftconvolve(g, k, mode="same"))
        except Exception:
            pass
    from scipy.signal import fftconvolve
    return fftconvolve(grid, kernel, mode="same")


def percentile(array: np.ndarray, q: float) -> float:
    """Percentile computation — GPU-aware."""
    if is_gpu_available():
        try:
            return float(xp.percentile(to_device(array.ravel()), q))
        except Exception:
            pass
    return float(np.percentile(array, q))


# ─────────────────────────────────────────────────────────────────────────────
# ComputeProfiler — stage timing for status bar / diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputeProfiler:
    stage_name: str
    elapsed_ms: float = 0.0
    backend: str = field(default_factory=lambda: _BACKEND_NAME)
    _start: float = field(default=0.0, repr=False)

    def __enter__(self):
        if is_gpu_available():
            try:
                import cupy
                cupy.cuda.Stream.null.synchronize()
            except Exception:
                pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if is_gpu_available():
            try:
                import cupy
                cupy.cuda.Stream.null.synchronize()
            except Exception:
                pass
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        logger.debug(
            f"[Compute] {self.stage_name}: {self.elapsed_ms:.1f} ms "
            f"[{self.backend}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GPU Roadmap registry
# ─────────────────────────────────────────────────────────────────────────────

GPU_ROADMAP = {
    "interpolation": {
        "current":  "SciPy LinearNDInterpolator / RBF (CPU)",
        "gpu_path": "cupy + custom CUDA kernel or cupyx.scipy.interpolate",
        "expected_speedup": "10–30×",
        "priority": "HIGH",
        "blocker":  "CuPy RBF not yet available in stable release",
    },
    "wavelet_baseline": {
        "current":  "PyWavelets (CPU)",
        "gpu_path": "CuPy FFT-based wavelet or cuWavelets",
        "expected_speedup": "5–15×",
        "priority": "MEDIUM",
        "blocker":  "PyWavelets GPU port incomplete",
    },
    "matched_filter": {
        "current":  "scipy.signal.fftconvolve (CPU)",
        "gpu_path": "cupyx.scipy.signal.fftconvolve (already abstracted here)",
        "expected_speedup": "8–20×",
        "priority": "HIGH",
        "blocker":  "None — ready to enable when CuPy is installed",
    },
    "volumetric_rendering": {
        "current":  "pyqtgraph.opengl (OpenGL CPU geometry)",
        "gpu_path": "VTK GPU volume rendering or Vulkan compute",
        "expected_speedup": "50–200×",
        "priority": "HIGH",
        "blocker":  "Requires VTK Python bindings or vispy Vulkan backend",
    },
    "anomaly_detection": {
        "current":  "scipy.ndimage label + custom threshold (CPU)",
        "gpu_path": "cupy.ndimage.label + numba.cuda threshold kernel",
        "expected_speedup": "15–40×",
        "priority": "MEDIUM",
        "blocker":  "cupy.ndimage.label is available in CuPy ≥ 12",
    },
    "fft_baseline": {
        "current":  "numpy.fft.fft2 (CPU)",
        "gpu_path": "cupy.fft.fft2 (already abstracted in fft2() above)",
        "expected_speedup": "20–60×",
        "priority": "HIGH",
        "blocker":  "None — ready to enable when CuPy is installed",
    },
}


def roadmap_report() -> str:
    """Human-readable GPU roadmap for the Diagnostics tab."""
    lines = [
        "GMS GPU Acceleration Roadmap",
        "=" * 50,
        f"Current backend: {_BACKEND_NAME.upper()}",
        "",
    ]
    for stage, info in GPU_ROADMAP.items():
        lines.append(f"  [{info['priority']}] {stage}")
        lines.append(f"    Now:     {info['current']}")
        lines.append(f"    GPU:     {info['gpu_path']}")
        lines.append(f"    Speedup: {info['expected_speedup']}")
        if info["blocker"] != "None — ready to enable when CuPy is installed":
            lines.append(f"    Blocker: {info['blocker']}")
        lines.append("")
    return "\n".join(lines)
