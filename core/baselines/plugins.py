"""
GMS — Baseline Remover Plugins
All baseline removers implement BaselineRemoverBase.

Available:
  NoBaseline          — pass-through, no drift removal
  LineMedianBaseline  — row-by-row rolling median (default for cubic)
  HighPassBaseline    — 2D Gaussian high-pass

Lesson from v1.4: NEVER pair a post-grid baseline with RBF interpolation.
RBF globally minimizes bending energy. Adding another subtraction step
creates double-baseline which absorbs anomalies.

Rule:
  cubic interpolator  → LineMedianBaseline  (safe, tested)
  rbf interpolator    → NoBaseline          (mandatory)
  any                 → HighPassBaseline    (experimental)

Adding a new baseline:
  1. Subclass BaselineRemoverBase
  2. Implement name + remove()
  3. Register in REGISTRY
  4. Add to config: baseline.mode: "your_name"
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.stats import median_abs_deviation

from ..abstractions import BaselineRemoverBase, GriddedScan, BaselinedGrid, StageCompatibility

logger = logging.getLogger("gms.baseline")


def _recompute_stats(grid_z: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = grid_z[mask]
    nf = float(median_abs_deviation(vals)) if vals.size > 3 else 0.0
    dr = float(np.ptp(vals)) if vals.size > 0 else 0.0
    return nf, dr


def _grid_to_baselines(grid: GriddedScan) -> tuple:
    return grid.grid_z, grid.grid_x, grid.grid_y, grid.grid_mask


def _make_baselined(grid: GriddedScan, new_z: np.ndarray,
                    baseline_name: str, warnings: list) -> BaselinedGrid:
    nf, dr = _recompute_stats(new_z, grid.grid_mask)
    return BaselinedGrid(
        scan_id=grid.scan_id,
        grid_z=new_z,
        grid_x=grid.grid_x,
        grid_y=grid.grid_y,
        grid_mask=grid.grid_mask,
        noise_floor=nf,
        dynamic_range=dr,
        baseline_name=baseline_name,
        interp_name=grid.interp_name,
        warnings=list(grid.warnings) + warnings,
        meta=dict(grid.meta),
    )


# ─────────────────────────────────────────────────────────────────────────────

class NoBaseline(BaselineRemoverBase):
    """
    Pass-through — no drift removal.
    Use when:
    - RBF interpolation was used (already globally smooth)
    - Data is pre-corrected
    - Testing without baseline artifacts
    """

    @property
    def name(self) -> str:
        return "none"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name,
            preferred_baseline=[],
            notes="Mandatory when using RBF interpolation."
        )

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        logger.debug("  Baseline: none (pass-through)")
        return _make_baselined(grid, grid.grid_z.copy(), self.name, [])


class LineMedianBaseline(BaselineRemoverBase):
    """
    Row-by-row rolling median baseline subtraction.

    Physical motivation:
      Each row = one scan traverse. Sensor drift acts along traverse direction.
      Rolling median at window_fraction removes slow drift while preserving
      compact anomalies (anomaly smaller than window → not absorbed).

    Parameters:
      window_fraction: fraction of row length for rolling window
        0.75–0.90 = wide window → better anomaly preservation
        0.50       = moderate → may absorb mid-scale anomalies

    Edge guard:
      Edge cells blend toward row median to prevent boundary ringing.

    ⚠️ INCOMPATIBLE with RBF interpolation:
      RBF already removed slow spatial variation globally.
      Running this after RBF causes double-subtraction.
    """

    def __init__(self, window_fraction: float = 0.90):
        self.window_fraction = window_fraction

    @property
    def name(self) -> str:
        return "line_median"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name,
            preferred_baseline=["griddata_cubic", "griddata_linear"],
            incompatible_detectors=[],
            notes="⚠️ Do NOT use after RBF interpolation — causes double-subtraction."
        )

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        gz = grid.grid_z.copy()
        mask = grid.grid_mask
        nrows, ncols = gz.shape
        win_frac = self.window_fraction
        warnings = []

        for row_idx in range(nrows):
            row = gz[row_idx].copy()
            valid = mask[row_idx]
            if valid.sum() < 4:
                continue

            valid_cols = np.where(valid)[0]
            v = row[valid_cols]
            win = max(3, int(len(v) * win_frac) | 1)

            baseline = np.array([
                np.median(v[max(0, i - win//2): i + win//2 + 1])
                for i in range(len(v))
            ])

            # Edge guard: blend outermost win//4 cells toward row median
            edge = win // 4
            if edge > 0 and len(baseline) > 2 * edge:
                row_med = np.median(v)
                for k in range(edge):
                    alpha = k / edge
                    baseline[k]      = alpha * baseline[k]      + (1 - alpha) * row_med
                    baseline[-(k+1)] = alpha * baseline[-(k+1)] + (1 - alpha) * row_med

            corrected = np.zeros(ncols)
            corrected[valid_cols] = v - baseline
            gz[row_idx] = corrected

        gz[~mask] = 0.0
        logger.debug(f"  Baseline: line_median (window={win_frac:.0%})")
        return _make_baselined(grid, gz, self.name, warnings)


class HighPassBaseline(BaselineRemoverBase):
    """
    2D Gaussian high-pass filter.
    Subtracts a very broad Gaussian-blurred version of the grid.

    sigma_fraction: fraction of min(grid_shape) for low-pass sigma.
    Small sigma → aggressive high-pass (removes more).
    Large sigma → gentle high-pass (removes only broadest trends).

    Works with both cubic and RBF outputs.
    Less predictable than line_median for scanner data.
    """

    def __init__(self, sigma_fraction: float = 0.15):
        self.sigma_fraction = sigma_fraction

    @property
    def name(self) -> str:
        return "highpass"

    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        gz = grid.grid_z.copy()
        sigma = max(2.0, self.sigma_fraction * min(gz.shape))
        low_freq = ndimage.gaussian_filter(gz, sigma=sigma)
        gz_hp = gz - low_freq
        gz_hp[~grid.grid_mask] = 0.0
        logger.debug(f"  Baseline: highpass (sigma={sigma:.1f} cells)")
        return _make_baselined(grid, gz_hp, self.name, [])


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

from .adaptive import AdaptiveLocalBaseline
from .multiscale import MultiScaleAdaptiveBaseline, WaveletBaseline

BASELINE_REGISTRY: dict[str, type[BaselineRemoverBase]] = {
    "none":           NoBaseline,
    "line_median":    LineMedianBaseline,
    "highpass":       HighPassBaseline,
    "adaptive_local": AdaptiveLocalBaseline,
    "multiscale":     MultiScaleAdaptiveBaseline,
    "wavelet_bg":     WaveletBaseline,
}


def get_baseline(mode: str, params: dict = None) -> BaselineRemoverBase:
    params = params or {}
    if mode not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline '{mode}'. Known: {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[mode]
    # Pass only params that the constructor accepts
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in params.items() if k in sig.parameters}
    return cls(**valid)
