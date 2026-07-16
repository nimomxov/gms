"""
GMS — Interpolator Plugins (PATCHED)
FIX 4: _make_grid_axes now guards against zero-span / collinear input
       (was producing 1-point axes -> Delaunay crash).
FIX 6: RBF smoothing comment standardized to the real formula (0.5*noise_floor).
OPT 3: RBFInterpolator now uses neighbors= to cap O(n^3) fit cost on dense scans.
"""

import logging
import numpy as np
from scipy import interpolate as sp_interp
from scipy.spatial import Delaunay
from scipy.stats import median_abs_deviation

from ..abstractions import InterpolatorBase, RawScan, GriddedScan, StageCompatibility

logger = logging.getLogger("gms.interpolator")

# Cap for RBF neighbor solve. None => exact (global) solve. Set to an int to
# switch to local RBF, which bounds fit cost on large point sets.
RBF_NEIGHBORS = 48


def _build_hull_mask(xv, yv, xi, yi) -> np.ndarray:
    """Valid cell mask from Delaunay hull of scatter points."""
    try:
        hull = Delaunay(np.c_[xv, yv])
        gx, gy = np.meshgrid(xi, yi)
        inside = hull.find_simplex(np.c_[gx.ravel(), gy.ravel()]) >= 0
        return inside.reshape(gx.shape)
    except Exception:
        return np.ones((len(yi), len(xi)), dtype=bool)


def _make_grid_axes(xv, yv, resolution, max_cells=500):
    # FIX 4: guard degenerate geometry before it reaches linspace/Delaunay.
    x_span = float(xv.max() - xv.min())
    y_span = float(yv.max() - yv.min())
    if x_span < 1e-9 or y_span < 1e-9:
        raise ValueError(
            "Degenerate scan geometry: zero spatial span on one axis. "
            "This device likely has no valid x/y — route to line mode."
        )
    n_x = max(2, min(max_cells, int(x_span / resolution) + 1))
    n_y = max(2, min(max_cells, int(y_span / resolution) + 1))
    return (np.linspace(xv.min(), xv.max(), n_x),
            np.linspace(yv.min(), yv.max(), n_y))


def _scan_stats(zv) -> tuple[float, float]:
    nf = float(median_abs_deviation(zv)) if len(zv) > 3 else 1.0
    dr = float(np.ptp(zv))
    return nf, dr


class CubicGriddataInterpolator(InterpolatorBase):
    @property
    def name(self) -> str:
        return "griddata_cubic"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name, preferred_baseline=["line_median", "none"],
            incompatible_detectors=[], notes="Default stable choice. Works with LoG detector.")

    def interpolate(self, scan: RawScan, resolution: float = 0.1) -> GriddedScan:
        xv, yv, zv = scan.x, scan.y, scan.values
        xi, yi = _make_grid_axes(xv, yv, resolution)
        nf, _ = _scan_stats(zv)
        gx, gy = np.meshgrid(xi, yi)
        try:
            gz = sp_interp.griddata(np.c_[xv, yv], zv, (gx, gy), method="cubic")
        except Exception as e:
            logger.warning(f"  cubic failed ({e}) — fallback linear")
            gz = sp_interp.griddata(np.c_[xv, yv], zv, (gx, gy), method="linear")
        mask = _build_hull_mask(xv, yv, xi, yi)
        gz = np.nan_to_num(gz, nan=0.0)
        gz[~mask] = 0.0
        return GriddedScan(
            scan_id=scan.scan_id, grid_z=gz, grid_x=xi, grid_y=yi, grid_mask=mask,
            noise_floor=nf, dynamic_range=float(np.ptp(gz[mask])) if mask.any() else 0.0,
            interp_name=self.name, warnings=list(scan.warnings))


class LinearGriddataInterpolator(InterpolatorBase):
    @property
    def name(self) -> str:
        return "griddata_linear"

    def interpolate(self, scan: RawScan, resolution: float = 0.1) -> GriddedScan:
        xv, yv, zv = scan.x, scan.y, scan.values
        xi, yi = _make_grid_axes(xv, yv, resolution)
        nf, _ = _scan_stats(zv)
        gx, gy = np.meshgrid(xi, yi)
        gz = sp_interp.griddata(np.c_[xv, yv], zv, (gx, gy), method="linear")
        mask = _build_hull_mask(xv, yv, xi, yi)
        gz = np.nan_to_num(gz, nan=0.0)
        gz[~mask] = 0.0
        return GriddedScan(
            scan_id=scan.scan_id, grid_z=gz, grid_x=xi, grid_y=yi, grid_mask=mask,
            noise_floor=nf, dynamic_range=float(np.ptp(gz[mask])) if mask.any() else 0.0,
            interp_name=self.name, warnings=list(scan.warnings))


class RBFThinPlateInterpolator(InterpolatorBase):
    @property
    def name(self) -> str:
        return "rbf_thin_plate"

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(
            name=self.name, preferred_baseline=["none", "pre_grid_dc"],
            incompatible_detectors=["log_detector"],
            notes=("Incompatible with LoGDetector (curvature-based). Use AmplitudeDetector. "
                   "Incompatible with post-grid baselines. Use NoBaseline."))

    def interpolate(self, scan: RawScan, resolution: float = 0.1) -> GriddedScan:
        from scipy.interpolate import RBFInterpolator
        xv, yv, zv = scan.x, scan.y, scan.values
        xi, yi = _make_grid_axes(xv, yv, resolution)
        nf, _ = _scan_stats(zv)
        # FIX 6: smoothing = 0.5 * noise_floor. Near-exact interpolation that
        # only damps spikes on the order of the noise floor. NOT noise variance.
        smoothing = max(0.1, nf * 0.5)
        gx, gy = np.meshgrid(xi, yi)
        try:
            # OPT 3: neighbors= bounds the solve cost on dense point sets.
            n_pts = len(zv)
            neighbors = RBF_NEIGHBORS if (RBF_NEIGHBORS and n_pts > RBF_NEIGHBORS) else None
            rbf = RBFInterpolator(
                np.c_[xv, yv], zv, kernel="thin_plate_spline",
                smoothing=smoothing, degree=1, neighbors=neighbors)
            gz = rbf(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
            used = f"rbf_thin_plate(s={smoothing:.1f},nbr={neighbors})"
        except Exception as e:
            logger.warning(f"  RBF failed ({e}) — fallback cubic")
            gz = sp_interp.griddata(np.c_[xv, yv], zv, (gx, gy), method="cubic")
            gz = np.nan_to_num(gz, nan=0.0)
            used = "griddata_cubic_fallback"
        mask = _build_hull_mask(xv, yv, xi, yi)
        gz = np.nan_to_num(gz, nan=0.0)
        gz[~mask] = 0.0
        dr = float(np.ptp(gz[mask])) if mask.any() else 0.0
        return GriddedScan(
            scan_id=scan.scan_id, grid_z=gz, grid_x=xi, grid_y=yi, grid_mask=mask,
            noise_floor=nf, dynamic_range=dr, interp_name=used,
            warnings=list(scan.warnings), meta={"smoothing": smoothing})


INTERPOLATOR_REGISTRY: dict[str, type[InterpolatorBase]] = {
    "cubic": CubicGriddataInterpolator,
    "griddata_cubic": CubicGriddataInterpolator,
    "linear": LinearGriddataInterpolator,
    "griddata_linear": LinearGriddataInterpolator,
    "rbf": RBFThinPlateInterpolator,
    "rbf_thin_plate": RBFThinPlateInterpolator,
}


def get_interpolator(mode: str) -> InterpolatorBase:
    if mode not in INTERPOLATOR_REGISTRY:
        known = list(INTERPOLATOR_REGISTRY.keys())
        raise KeyError(f"Unknown interpolator '{mode}'. Known: {known}")
    return INTERPOLATOR_REGISTRY[mode]()