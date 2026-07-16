"""
GMS — VolumetricEngine  v4.1
================================
Audited and corrected implementation. All 9 bugs from the engineering
audit fixed. See audit report below.

Layout target:
  viewport3dWidget → vp3dWLay (QVBoxLayout, empty in .ui) → GLViewWidget

3D checkboxes: chk3dSignal, chk3dBase, chk3dDig, chk3dConf,
               chk3dRaw, chk3dGrid, chk3dBlobs

AUDIT FIXES APPLIED
===================
1. _draw_signed_mesh: removed dead intermediate face-build block that was
   overwritten; now uses a single clean vectorised face computation.
2. set_render_mode: was calling _draw_surface() twice — second call was
   redundant, caused double GL items and memory leak. Removed.
3. _draw_baseline: was using GLSurfacePlotItem which triggers the
   vertex-color IndexError. Replaced with GLMeshItem + face colors.
4. _draw_anomalies: surf_z computed from gz_norm × z_scale_old, but
   _draw_surface uses gz_signed × z_scale_new. Different z-systems caused
   markers to float at wrong height. Fixed: surf_z uses gz_signed z_scale.
5. _draw_raw_points: used gz_norm (non-polarity) for Z-coordinate. Fixed:
   uses gz_signed so raw points respect polarity and share same z-system.
6. _draw_blobs: same gz_norm/surf_z inconsistency as #4. Fixed.
7. _draw_surface_mesh (fallback): used gz_norm + plasma colormap, losing
   polarity. Fixed: uses _prepare_signal + _geomag_colormap.
8. _draw_confidence: computed gz array that was never used. Removed dead code.
9. set_render_mode guard: double-draw when _grid is None removed cleanly.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget, QSizePolicy, QMainWindow, QVBoxLayout

logger = logging.getLogger("gms.volumetric")


def _detect_gl():
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        logger.info("[Volumetric] Backend: pyqtgraph.opengl")
        return True, gl, pg
    except ImportError:
        logger.warning("[Volumetric] pyqtgraph not available — 3D disabled")
        return False, None, None

_HAS_GL, _GL, pg = _detect_gl()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(obj, *attrs, default=None):
    for a in attrs:
        v = getattr(obj, a, None)
        if v is not None:
            return v
        if isinstance(obj, dict) and a in obj:
            return obj[a]
    return default


def _conf_rgba(c: float) -> tuple:
    if c >= 0.70:
        return (0.1, 1.0, 0.25, 0.95)
    if c >= 0.45:
        return (1.0, 0.65, 0.0,  0.90)
    return     (1.0, 0.2,  0.2,  0.85)


def _build_mesh_faces(hs: int, ws: int) -> np.ndarray:
    """
    Build face index array for a (hs × ws) vertex grid.
    Returns shape (2*(hs-1)*(ws-1), 3) uint32.
    Single canonical implementation — used by all mesh builders.
    """
    r_idx = np.arange(hs - 1)
    c_idx = np.arange(ws - 1)
    rr, cc = np.meshgrid(r_idx, c_idx, indexing='ij')
    tl = (rr * ws + cc).ravel()
    f1 = np.column_stack([tl,     tl + 1,      tl + ws    ]).astype(np.uint32)
    f2 = np.column_stack([tl + 1, tl + ws + 1, tl + ws    ]).astype(np.uint32)
    return np.concatenate([f1, f2], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# VolumetricEngine
# ─────────────────────────────────────────────────────────────────────────────

class VolumetricEngine(QObject):
    anomaly_picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view:   Optional[object]   = None
        self._items:  Dict[str, List]    = {}
        self._vert_exag:  int            = 3
        self._survey_w:   float          = 5.0
        self._survey_l:   float          = 5.0
        self._grid                       = None
        self._anomalies:  list           = []
        self._current_anomalies: list    = []   # alias; kept in sync with _anomalies
        self._render_mode: str           = "Surface"
        self._layers: Dict[str, bool] = {
            "signal":   True,
            "baseline": False,
            "dig":      True,
            "conf":     False,
            "raw":      False,
            "grid":     True,
            "blobs":    False,
        }

    # ── Attach ─────────────────────────────────────────────────────────────

    def attach(self, window: QMainWindow) -> bool:
        if not _HAS_GL:
            return False
        container = window.findChild(QWidget, "viewport3dWidget")
        if container is None:
            logger.error("[Volumetric] viewport3dWidget not found")
            return False
        lay = container.layout()
        if lay is None:
            lay = QVBoxLayout(container)
            lay.setContentsMargins(0, 0, 0, 0)
        import pyqtgraph.opengl as gl
        view = gl.GLViewWidget()
        view.setObjectName("_gmsGL3D")
        view.setBackgroundColor((10, 12, 18, 255))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setMinimumSize(100, 100)
        view.setCameraPosition(distance=10, elevation=30, azimuth=45)
        # opts["center"] is the camera target AND the orbit pivot used by
        # mouse-drag rotation.  It defaults to Vector(0,0,0).  set_scan()
        # below sets it to the survey centroid once geometry is known —
        # this initial value is only a placeholder before any scan loads.
        lay.addWidget(view)
        self._view = view
        self._center_x: float = 0.0
        self._center_y: float = 0.0
        self._center_z: float = 0.0
        logger.info("[Volumetric] GLViewWidget added to vp3dWLay")
        return True

    # ── Data ───────────────────────────────────────────────────────────────

    def set_scan(self, grid, anomalies: list = None, geometry=None):
        if self._view is None:
            return
        self._grid               = grid
        self._anomalies          = anomalies or []
        self._current_anomalies  = self._anomalies
        self._geometry           = geometry   # store for orientation queries

        gx = np.asarray(grid.grid_x, dtype=float)
        gy = np.asarray(grid.grid_y, dtype=float)
        if len(gx) > 1 and len(gy) > 1:
            self._survey_w = float(gx.max() - gx.min())
            self._survey_l = float(gy.max() - gy.min())
        elif geometry is not None:
            self._survey_w = float(geometry.field_width_m)
            self._survey_l = float(geometry.field_length_m)

        diag = float(np.sqrt(self._survey_w**2 + self._survey_l**2))

        # ── Centre the orbit pivot / camera target on the survey bounding
        # box centroid.  This is the single fix that places the geometric
        # centre of the survey at the geometric centre of viewport3dWidget:
        # pyqtgraph orbits and pans around opts["center"], which defaults
        # to world origin (0,0,0).  Survey coordinates are NOT centred at
        # the origin (they run from grid_x.min()..grid_x.max()), so without
        # this the pivot sits at the corner of the survey, not its middle.
        center_x = (float(gx.min()) + float(gx.max())) / 2.0 if len(gx) > 1 else 0.0
        center_y = (float(gy.min()) + float(gy.max())) / 2.0 if len(gy) > 1 else 0.0
        center_z = 0.0   # signal surface is centred on z=0 by _prepare_signal
        self._center_x, self._center_y, self._center_z = center_x, center_y, center_z

        # ── Camera orientation matched to survey direction ────────────
        # The initial azimuth is chosen so the 3D view and the 2D heatmap
        # share the same visual orientation.
        #
        # Convention (pyqtgraph azimuth: 0° = looking along +Y axis,
        #             90° = looking along +X axis):
        #
        #   N→S / S→N scan: operator walked along Y axis.
        #     Scan lines run E→W (horizontal in map view).
        #     Best view: azimuth=45° (diagonal NE viewpoint, same as 2D).
        #
        #   E→W / W→E scan: operator walked along X axis.
        #     Scan lines run N→S (vertical in map view).
        #     Best view: azimuth=135° (diagonal NW viewpoint).
        #
        #   Reversed directions (S→N, W→E) flip the dominant axis but
        #   the visual orientation is the same; elevation stays at 30°.
        _azimuth = 45.0   # default for N↔S
        if geometry is not None:
            try:
                from core.geometry import SurveyDirection
                _dir = getattr(geometry, "direction",
                               SurveyDirection.NORTH_SOUTH)
                if _dir in (SurveyDirection.EAST_WEST,
                             SurveyDirection.WEST_EAST):
                    _azimuth = 135.0
            except Exception:
                pass
        self._view.setCameraPosition(
            distance=diag * 2.5, elevation=30, azimuth=_azimuth)
        self._view.opts["center"] = pg.Vector(center_x, center_y, center_z)
        self._view.update()

        self._clear_all()
        self._draw_grid()
        self._draw_surface()
        self._draw_anomalies()

        print(f"[3D] mesh survey={self._survey_w:.2f}x{self._survey_l:.2f}m "
              f"grid={np.asarray(grid.grid_z).shape} "
              f"anomalies={len(self._anomalies)}")

    # ── Scene management ───────────────────────────────────────────────────

    def _clear_all(self):
        if self._view is None:
            return
        for items in self._items.values():
            for item in items:
                try:
                    self._view.removeItem(item)
                except Exception:
                    pass
        self._items = {}

    # ── Signal preparation ─────────────────────────────────────────────────

    def _prepare_signal(self, gz: np.ndarray):
        """
        Polarity-preserving signed normalisation.
        Returns gz_signed in [-1, +1] centred on the survey median.
        Positive = above baseline, negative = below baseline.
        """
        from scipy.stats import median_abs_deviation as _mad
        flat = gz.ravel()
        med  = float(np.median(flat))
        mad  = float(_mad(flat, scale="normal")) + 1e-9

        # Adaptive clip coefficient: use 99.5th percentile of |z-score|
        # so that the strongest real anomaly maps to ±1 rather than being
        # saturated at the fixed ±3σ boundary.
        # Minimum of 2.5 prevents over-expansion on very clean datasets.
        raw_z    = (flat - med) / mad
        p995     = float(np.percentile(np.abs(raw_z), 99.5))
        clip_k   = max(p995, 2.5)   # never collapse below 2.5σ

        gz_signed = np.clip((gz - med) / (clip_k * mad), -1.0, 1.0).astype(np.float32)

        pos_cells = int((gz_signed >  0.05).sum())
        neg_cells = int((gz_signed < -0.05).sum())
        zc_rows   = int(np.diff(np.sign(gz_signed), axis=1).astype(bool).sum())

        print(f"[RENDER] signal_min={gz.min():.3f}  signal_max={gz.max():.3f}  "
              f"signal_mean={gz.mean():.3f}  signal_std={gz.std():.3f}")
        print(f"[GEOPHYSICS] positive_cells={pos_cells}  "
              f"negative_cells={neg_cells}  zero_crossings={zc_rows}")

        info = dict(signal_min=float(gz.min()), signal_max=float(gz.max()),
                    signal_mean=float(gz.mean()), signal_std=float(gz.std()),
                    median=med, mad=mad, positive_cells=pos_cells,
                    negative_cells=neg_cells, zero_crossings=zc_rows)
        return gz_signed, info

    def _z_scale(self) -> float:
        """Single canonical z-scale formula used by ALL layers."""
        return min(self._survey_w, self._survey_l) * 0.30 * self._vert_exag

    # ── Colormap ───────────────────────────────────────────────────────────

    def _geomag_colormap(self, gz_signed: np.ndarray) -> np.ndarray:
        """
        Polarity-aware geomagnetic colormap.
        Input: signed values in [-1, +1].
        Output: RGBA float32, same leading shape + (4,).
          +1 → red (ferrous)   0 → green (soil)   -1 → dark blue (cavity)
        """
        v   = np.asarray(gz_signed, dtype=np.float32)
        out = np.zeros((*v.shape, 4), dtype=np.float32)
        out[..., 3] = 0.93

        pos  = v >= 0.0
        vp   = v[pos]
        out[pos, 0] = np.clip(0.1  + vp * 1.8,  0, 1)
        out[pos, 1] = np.clip(0.70 - vp * 0.65, 0, 1)
        out[pos, 2] = np.clip(0.15 - vp * 0.15, 0, 1)

        # Background band: ±0.12 of the normalised range is statistically
        # indistinguishable from background noise in typical gradiometer surveys.
        soil = np.abs(v) < 0.12
        out[soil, 0] = 0.10
        out[soil, 1] = 0.72
        out[soil, 2] = 0.18

        neg  = v < 0.0
        vn   = -v[neg]
        out[neg, 0] = np.clip(0.10 - vn * 0.10, 0, 1)
        out[neg, 1] = np.clip(0.85 - vn * 0.75, 0, 1)
        out[neg, 2] = np.clip(0.75 + vn * 0.20, 0, 1)
        return out

    # ── Grid layer ─────────────────────────────────────────────────────────

    def _draw_grid(self):
        if not self._layers["grid"] or self._view is None or self._grid is None:
            return
        try:
            import pyqtgraph.opengl as gl
            gx = np.asarray(self._grid.grid_x)
            gy = np.asarray(self._grid.grid_y)
            g  = gl.GLGridItem()
            g.setSize(self._survey_w, self._survey_l, 1)
            g.setSpacing(max(self._survey_w / 10, 0.05),
                         max(self._survey_l / 10, 0.05), 1)
            g.setColor((160, 210, 160, 220))
            g.translate(float(gx.min()) + self._survey_w / 2,
                        float(gy.min()) + self._survey_l / 2, 0)
            self._view.addItem(g)
            self._items["grid"] = [g]
        except Exception as e:
            logger.error(f"[3D] Grid error: {e}")

    # ── Main signal dispatcher ─────────────────────────────────────────────

    def _draw_surface(self):
        """Clear signal layer then dispatch to current render mode."""
        if not self._layers["signal"] or self._view is None or self._grid is None:
            return

        # Clear only signal items — not other layers
        for item in self._items.get("signal", []):
            try:
                self._view.removeItem(item)
            except Exception:
                pass
        self._items["signal"] = []

        gz = np.nan_to_num(
            np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0)
        if gz.size == 0:
            return

        gx = np.asarray(self._grid.grid_x, dtype=np.float32)
        gy = np.asarray(self._grid.grid_y, dtype=np.float32)
        h, w = gz.shape
        xs = np.linspace(float(gx.min()), float(gx.max()), w)
        ys = np.linspace(float(gy.min()), float(gy.max()), h)

        gz_signed, _ = self._prepare_signal(gz)
        z_scale      = self._z_scale()

        mode = self._render_mode
        if mode == "Surface":
            self._draw_mode_surface(gz_signed, xs, ys, z_scale, h, w)
        elif mode == "Heightmap":
            self._draw_mode_heightmap(gz_signed, xs, ys, z_scale, h, w)
        elif mode == "Volumetric":
            self._draw_mode_volumetric(gz_signed, xs, ys, z_scale, h, w)
        elif mode == "Wireframe":
            self._draw_mode_wireframe(gz_signed, xs, ys, z_scale, h, w)
        elif mode == "Points":
            self._draw_raw_points_as_signal(gz_signed, xs, ys, z_scale, h, w)

        print(f"[3D] {mode}: {h}x{w} survey={self._survey_w:.2f}x{self._survey_l:.2f}m")

    # ── Shared mesh builder ────────────────────────────────────────────────

    def _draw_signed_mesh(self, gz_signed, xs, ys, z_scale, h, w,
                          invert: bool, alpha: float):
        """
        Build a GLMeshItem with face colors.

        invert=False → Heightmap: true signed elevation (positive up, negative down)
        invert=True  → Surface:  all anomalies below z=0 (depth model)

        Uses _build_mesh_faces() — single canonical face computation.
        Adaptive LOD: subsamples for surveys > 120 cells in either dimension.

        FIX #1: removed the dead intermediate face block that was
        overwriting the first computation.
        """
        try:
            import pyqtgraph.opengl as gl

            MAX_DIM = 120
            step = max(1, int(np.ceil(max(h, w) / MAX_DIM)))
            if step > 1:
                gz_s = gz_signed[::step, ::step]
                xs_s = xs[::step]
                ys_s = ys[::step]
                hs, ws = gz_s.shape
            else:
                gz_s, xs_s, ys_s, hs, ws = gz_signed, xs, ys, h, w

            gz_z = (-(np.abs(gz_s)) if invert else gz_s) * z_scale
            gz_z = gz_z.astype(np.float32)

            xx, yy = np.meshgrid(xs_s, ys_s)
            verts  = np.column_stack([
                xx.ravel(), yy.ravel(), gz_z.ravel()
            ]).astype(np.float32)

            faces  = _build_mesh_faces(hs, ws)

            v_sign = gz_s.ravel()
            fi     = (v_sign[faces[:, 0]] +
                      v_sign[faces[:, 1]] +
                      v_sign[faces[:, 2]]) / 3.0
            fc = self._geomag_colormap(fi.reshape(-1, 1)).reshape(-1, 4).astype(np.float32)
            fc[:, 3] = alpha

            mesh = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                                 smooth=False, drawEdges=False)
            mesh.setVisible(self._layers["signal"])
            self._view.addItem(mesh)
            self._items["signal"].append(mesh)

            if step > 1:
                print(f"[3D] LOD: {h}x{w} → {hs}x{ws} step={step} faces={len(faces)}")

        except Exception as e:
            logger.error(f"[3D] Mesh draw error: {e}", exc_info=True)

    # ── Render modes ───────────────────────────────────────────────────────

    def _draw_mode_surface(self, gz_signed, xs, ys, z_scale, h, w):
        """
        Interpretation mode: all anomalies below z=0.
        Depth = |signal| × z_scale. Polarity shown by colour only.
        """
        self._draw_signed_mesh(gz_signed, xs, ys, z_scale, h, w,
                               invert=True, alpha=0.92)

    def _draw_mode_heightmap(self, gz_signed, xs, ys, z_scale, h, w):
        """
        Scientific mode: true signed elevation.
        Positive anomalies rise; negative anomalies descend.
        Dipole = adjacent hill and valley.
        """
        self._draw_signed_mesh(gz_signed, xs, ys, z_scale, h, w,
                               invert=False, alpha=0.95)

    def _draw_mode_volumetric(self, gz_signed, xs, ys, z_scale, h, w):
        """
        Magnetic response cloud — ellipsoidal particle bodies per anomaly.
        Depth estimated via Peters half-width rule from extent_cells.
        Falls back to grid-driven cloud if no anomaly objects available.
        """
        import pyqtgraph.opengl as gl
        rng = np.random.default_rng(seed=42)

        dx = (xs[-1] - xs[0]) / max(w - 1, 1)
        dy = (ys[-1] - ys[0]) / max(h - 1, 1)

        all_pos, all_colors, all_sizes = [], [], []
        anomalies = self._current_anomalies

        try:
            if anomalies:
                for a in anomalies:
                    xm   = float(_get(a, "x", "centroid_x",
                                      default=(xs[0]+xs[-1])/2))
                    ym   = float(_get(a, "y", "centroid_y",
                                      default=(ys[0]+ys[-1])/2))
                    conf = float(_get(a, "confidence", "combined_confidence",
                                      default=0.4))
                    ext  = float(_get(a, "extent_cells",  default=4))
                    dip  = float(_get(a, "dipole_score",  default=0.0))
                    pol  = float(_get(a, "polarity_ratio", default=0.5))

                    # Peters half-width rule for a compact sphere in
                    # total-field data: depth ≈ 0.5 × half_width.
                    # For a gradiometer (gradient data), the apparent half-width
                    # is narrower; apply a sensor-separation correction factor
                    # of ~0.7 (empirical for 0.5 m sensor separation).
                    # Uncertainty: ±30% (standard Peters rule tolerance).
                    half_width_m   = ext * max(dx, dy) * 0.5
                    est_depth_m    = float(np.clip(
                        half_width_m * 0.50 * 0.70,   # 0.50 Peters × 0.70 gradiometer
                        0.05 * z_scale, 0.85 * z_scale))
                    depth_lower_m  = est_depth_m * 0.70   # −30%
                    depth_upper_m  = est_depth_m * 1.30   # +30%
                    body_r_xy = float(np.clip(
                        ext * max(dx, dy) * 0.55,
                        max(dx, dy), self._survey_w * 0.4))
                    if dip > 0.5:
                        body_r_xy *= 1.35
                    # Vertical semi-axis must at least reach the estimated depth
                    # centre so the body spans from z=0 to approximately 2× depth.
                    body_r_z = max(est_depth_m * 0.85, 0.05 * z_scale)

                    vol_proxy = body_r_xy ** 2 * body_r_z
                    n_pts     = int(np.clip(vol_proxy * 800, 80, 1200))

                    pts_xyz, attempts = [], 0
                    while len(pts_xyz) < n_pts and attempts < n_pts * 8:
                        attempts += 1
                        rx = rng.uniform(-1, 1)
                        ry = rng.uniform(-1, 1)
                        rz = rng.uniform(-1, 0)
                        if rx**2 + ry**2 + rz**2 <= 1.0:
                            pts_xyz.append([
                                xm + rx * body_r_xy,
                                ym + ry * body_r_xy,
                                rz * body_r_z - est_depth_m * 0.3,
                            ])
                    if not pts_xyz:
                        continue

                    pos    = np.array(pts_xyz, dtype=np.float32)
                    centre = np.array([[xm, ym, -est_depth_m * 0.3]],
                                      dtype=np.float32)
                    delta  = pos - centre
                    norm_dist = np.sqrt(
                        (delta[:, 0] / (body_r_xy + 1e-9)) ** 2 +
                        (delta[:, 1] / (body_r_xy + 1e-9)) ** 2 +
                        (delta[:, 2] / (body_r_z   + 1e-9)) ** 2)

                    if dip > 0.55:
                        # Dipole: positive and negative lobes are horizontally
                        # offset, not vertically stacked. The positive lobe lies
                        # to the North (+y) of the body centre, negative to South
                        # (or East/West depending on magnetisation direction).
                        # Default: offset along the y-axis (N-S for a N-S survey).
                        offset = body_r_xy * 0.7
                        signs  = np.where(pos[:, 1] >= ym, 0.75, -0.75)
                    else:
                        # Non-dipolar: uniform polarity from polarity_ratio.
                        # pol=1.0 → +1 (ferrous), pol=0.0 → -1 (cavity).
                        # For pol≈0.5 (equal-lobe dipole not caught above),
                        # use dipole_score to break the tie toward positive.
                        sign_val = pol * 2.0 - 1.0
                        if abs(sign_val) < 0.15 and dip > 0.4:
                            sign_val = 0.5   # weak dipole: positive-dominant
                        signs = np.full(len(pos), sign_val)

                    cols = self._geomag_colormap(
                        signs.reshape(-1, 1)).reshape(-1, 4).astype(np.float32)
                    cols[:, 3] = np.clip(
                        conf * 0.85 * (1.0 - norm_dist * 0.75), 0.04, 0.88
                    ).astype(np.float32)

                    all_pos.append(pos)
                    all_colors.append(cols)
                    all_sizes.append((7 + norm_dist * 14).astype(np.float32))

                    pol_str = ("DIPOLE" if dip > 0.55
                               else "POS" if pol >= 0.5 else "NEG")
                    print(f"[VOLUME] label={_get(a,'label','?')} "
                          f"x={xm:.2f}m y={ym:.2f}m "
                          f"depth≈{est_depth_m:.2f}m "
                          f"[{depth_lower_m:.2f}–{depth_upper_m:.2f}m] "
                          f"width≈{body_r_xy*2:.2f}m "
                          f"pol={pol_str} pts={len(pos)}")
            else:
                # Grid-driven fallback with adaptive subsampling
                thresh  = 0.15
                active  = np.argwhere(np.abs(gz_signed) >= thresh)
                if len(active) == 0:
                    logger.warning("[3D] Volumetric: no signal above threshold")
                    return
                if len(active) > 800:
                    idx    = rng.choice(len(active), 800, replace=False)
                    active = active[idx]
                sigma_xy = max(dx, dy) * 1.5
                for rc in active:
                    r, c    = int(rc[0]), int(rc[1])
                    val     = float(gz_signed[r, c])
                    xc, yc  = float(xs[c]), float(ys[r])
                    strength = abs(val)
                    n_pts   = max(int(strength * 22), 3)
                    max_dep = z_scale * strength * 0.75
                    px = xc + rng.normal(0, sigma_xy, n_pts).astype(np.float32)
                    py = yc + rng.normal(0, sigma_xy, n_pts).astype(np.float32)
                    pz = np.clip(
                        -(rng.exponential(max_dep * 0.38, n_pts)).astype(np.float32),
                        -max_dep, 0)
                    pos        = np.column_stack([px, py, pz])
                    base_col   = self._geomag_colormap(
                        np.array([[val]]))[0, 0].copy()
                    depth_frac = np.clip(-pz / (max_dep + 1e-9), 0, 1)
                    alphas     = np.clip(
                        strength * 0.70 * (1.0 - depth_frac * 0.65), 0.05, 0.78
                    ).astype(np.float32)
                    cols       = np.tile(base_col[:3], (n_pts, 1))
                    cols       = np.column_stack([cols, alphas]).astype(np.float32)
                    all_pos.append(pos)
                    all_colors.append(cols)
                    all_sizes.append((6 + depth_frac * 11).astype(np.float32))

            if not all_pos:
                logger.warning("[3D] Volumetric: nothing to render")
                return

            cloud = gl.GLScatterPlotItem(
                pos   = np.concatenate(all_pos,    axis=0).astype(np.float32),
                color = np.concatenate(all_colors, axis=0).astype(np.float32),
                size  = np.concatenate(all_sizes,  axis=0).astype(np.float32),
                pxMode=True,
            )
            cloud.setVisible(self._layers["signal"])
            self._view.addItem(cloud)
            self._items["signal"].append(cloud)
            mode_str = "anomaly-body" if anomalies else "grid-cloud"
            print(f"[3D] Volumetric ({mode_str}): "
                  f"{sum(len(p) for p in all_pos)} particles")

        except Exception as e:
            logger.error(f"[3D] Volumetric error: {e}", exc_info=True)
            self._draw_mode_surface(gz_signed, xs, ys, z_scale, h, w)

    def _draw_mode_wireframe(self, gz_signed, xs, ys, z_scale, h, w):
        """Wireframe: Heightmap elevation with visible mesh edges."""
        try:
            import pyqtgraph.opengl as gl
            gz_z  = (gz_signed * z_scale).astype(np.float32)
            xx, yy = np.meshgrid(xs, ys)
            verts  = np.column_stack([
                xx.ravel(), yy.ravel(), gz_z.ravel()
            ]).astype(np.float32)
            faces  = _build_mesh_faces(h, w)
            v_sign = gz_signed.ravel()
            fi     = (v_sign[faces[:, 0]] +
                      v_sign[faces[:, 1]] +
                      v_sign[faces[:, 2]]) / 3.0
            fc = self._geomag_colormap(fi.reshape(-1, 1)).reshape(-1, 4).astype(np.float32)
            fc[:, 3] = 0.85
            mesh = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                                 smooth=False, drawEdges=True,
                                 edgeColor=(0.55, 0.65, 0.55, 0.55))
            mesh.setVisible(self._layers["signal"])
            self._view.addItem(mesh)
            self._items["signal"].append(mesh)
        except Exception as e:
            logger.error(f"[3D] Wireframe error: {e}", exc_info=True)

    def _draw_raw_points_as_signal(self, gz_signed=None, xs=None, ys=None,
                                   z_scale=None, h=None, w=None):
        """
        Point cloud: scatter plot of all grid vertices.
        FIX #5: uses gz_signed (polarity-preserving) for Z, consistent
        with all other render modes.
        """
        try:
            import pyqtgraph.opengl as gl
            if gz_signed is None:
                gz = np.nan_to_num(
                    np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0)
                gx = np.asarray(self._grid.grid_x, dtype=np.float32)
                gy = np.asarray(self._grid.grid_y, dtype=np.float32)
                h, w   = gz.shape
                xs     = np.linspace(float(gx.min()), float(gx.max()), w)
                ys     = np.linspace(float(gy.min()), float(gy.max()), h)
                gz_signed, _ = self._prepare_signal(gz)
                z_scale      = self._z_scale()
            gz_z   = (gz_signed * z_scale).astype(np.float32)
            xx, yy = np.meshgrid(xs, ys)
            pos    = np.column_stack([
                xx.ravel(), yy.ravel(), gz_z.ravel()
            ]).astype(np.float32)
            colors = self._geomag_colormap(gz_signed).reshape(-1, 4).astype(np.float32)
            pts = gl.GLScatterPlotItem(pos=pos, size=4, color=colors, pxMode=True)
            pts.setVisible(self._layers["signal"])
            self._view.addItem(pts)
            self._items["signal"].append(pts)
        except Exception as e:
            logger.error(f"[3D] Points render error: {e}")

    def _draw_surface_mesh(self):
        """
        Emergency fallback. Calls _draw_mode_surface via the signed pipeline.
        FIX #7: no longer uses gz_norm + plasma. Uses signed signal + geomag colormap.
        """
        if self._grid is None:
            return
        gz = np.nan_to_num(
            np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0)
        gx = np.asarray(self._grid.grid_x, dtype=np.float32)
        gy = np.asarray(self._grid.grid_y, dtype=np.float32)
        h, w = gz.shape
        xs = np.linspace(float(gx.min()), float(gx.max()), w)
        ys = np.linspace(float(gy.min()), float(gy.max()), h)
        gz_signed, _ = self._prepare_signal(gz)
        z_scale      = self._z_scale()
        self._draw_mode_surface(gz_signed, xs, ys, z_scale, h, w)

    # ── Overlay layers ─────────────────────────────────────────────────────

    def _draw_anomalies(self):
        """
        Anomaly markers (dig layer).
        FIX #4: surf_z now uses _z_scale() consistently with _draw_surface.
        Markers are placed at z=0 (survey plane) with vertical stakes above.
        """
        if not self._layers["dig"] or self._view is None or not self._anomalies:
            return
        try:
            import pyqtgraph.opengl as gl
            gx = np.asarray(self._grid.grid_x, dtype=float)
            gy = np.asarray(self._grid.grid_y, dtype=float)

            for a in self._anomalies:
                xm   = float(_get(a, "x", "centroid_x",
                                  default=self._survey_w / 2))
                ym   = float(_get(a, "y", "centroid_y",
                                  default=self._survey_l / 2))
                conf = float(_get(a, "confidence", "combined_confidence",
                                  default=0.4))
                xm   = float(np.clip(xm, gx.min(), gx.max()))
                ym   = float(np.clip(ym, gy.min(), gy.max()))
                color = _conf_rgba(conf)

                # Marker at z=0 (survey plane)
                scatter = gl.GLScatterPlotItem(
                    pos=np.array([[xm, ym, 0.0]], dtype=np.float32),
                    size=16 + conf * 20, color=color, pxMode=True,
                )
                scatter.setVisible(self._layers["dig"])
                self._view.addItem(scatter)
                self._items.setdefault("dig", []).append(scatter)

                # Vertical stake above survey plane for DIG targets
                if conf >= 0.70:
                    stake_top = self._z_scale() * 0.25
                    line = gl.GLLinePlotItem(
                        pos=np.array([[xm, ym, 0.0],
                                      [xm, ym, stake_top]],
                                     dtype=np.float32),
                        color=(0.1, 1.0, 0.1, 0.9),
                        width=2.5, antialias=True,
                    )
                    self._view.addItem(line)
                    self._items.setdefault("dig", []).append(line)

                print(f"[ANOMALY] label={_get(a,'label','?')} "
                      f"x={xm:.3f}m y={ym:.3f}m conf={conf:.1%}")
        except Exception as e:
            logger.error(f"[3D] Anomaly error: {e}", exc_info=True)

    def _draw_baseline(self):
        """
        Baseline reference plane at z=0 (flat, blue-tinted).
        FIX #3: uses GLMeshItem with face colors instead of GLSurfacePlotItem
        which triggered the vertex-color IndexError.
        """
        if not self._layers["baseline"] or self._view is None or self._grid is None:
            return
        try:
            import pyqtgraph.opengl as gl
            gx = np.asarray(self._grid.grid_x, dtype=np.float32)
            gy = np.asarray(self._grid.grid_y, dtype=np.float32)
            h, w = np.asarray(self._grid.grid_z).shape
            xs = np.linspace(float(gx.min()), float(gx.max()), w)
            ys = np.linspace(float(gy.min()), float(gy.max()), h)

            xx, yy = np.meshgrid(xs, ys)
            verts  = np.column_stack([
                xx.ravel(), yy.ravel(),
                np.zeros(h * w, dtype=np.float32)
            ]).astype(np.float32)
            faces  = _build_mesh_faces(h, w)
            fc     = np.zeros((len(faces), 4), dtype=np.float32)
            fc[:, 2] = 0.55   # blue
            fc[:, 3] = 0.30   # low alpha

            mesh = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=fc,
                                 smooth=False, drawEdges=False)
            mesh.setVisible(self._layers["baseline"])
            self._view.addItem(mesh)
            self._items.setdefault("baseline", []).append(mesh)
        except Exception as e:
            logger.debug(f"[3D] Baseline layer error: {e}")

    def _draw_confidence(self):
        """
        Grid-aligned confidence squares at z=0.
        FIX #8: removed unused gz variable.
        """
        if not self._layers["conf"] or self._view is None or not self._anomalies:
            return
        try:
            import pyqtgraph.opengl as gl
            gx = np.asarray(self._grid.grid_x, dtype=float)
            gy = np.asarray(self._grid.grid_y, dtype=float)
            dx = float(gx[1] - gx[0]) if len(gx) > 1 else 0.1
            dy = float(gy[1] - gy[0]) if len(gy) > 1 else 0.1

            for a in self._anomalies:
                xm   = float(_get(a, "x", "centroid_x",
                                  default=self._survey_w / 2))
                ym   = float(_get(a, "y", "centroid_y",
                                  default=self._survey_l / 2))
                conf = float(_get(a, "confidence", "combined_confidence",
                                  default=0.4))
                unc  = float(_get(a, "mean_uncertainty", "uncertainty",
                                  default=0.25))
                xm   = float(np.clip(xm, gx.min(), gx.max()))
                ym   = float(np.clip(ym, gy.min(), gy.max()))

                # Positional uncertainty in metres ≈ unc × cell_size.
                # Round to nearest cell boundary so the square aligns with
                # the survey grid. Minimum = 1 cell, maximum = 3 m.
                unc_m  = unc * max(dx, dy) * 8.0   # scale to ~metres
                half_x = float(np.clip(
                    round(unc_m / dx) * dx, dx, 3.0))
                half_y = float(np.clip(
                    round(unc_m / dy) * dy, dy, 3.0))
                color  = (*_conf_rgba(conf)[:3], 1.0)

                for pts in [
                    # Square outline
                    np.array([[xm-half_x, ym-half_y, 0.],
                              [xm+half_x, ym-half_y, 0.],
                              [xm+half_x, ym+half_y, 0.],
                              [xm-half_x, ym+half_y, 0.],
                              [xm-half_x, ym-half_y, 0.]], dtype=np.float32),
                    # H crosshair
                    np.array([[xm-half_x*0.3, ym, 0.],
                              [xm+half_x*0.3, ym, 0.]], dtype=np.float32),
                    # V crosshair
                    np.array([[xm, ym-half_y*0.3, 0.],
                              [xm, ym+half_y*0.3, 0.]], dtype=np.float32),
                ]:
                    line = gl.GLLinePlotItem(pos=pts, color=color, width=3.5,
                                            antialias=True, mode="line_strip")
                    line.setVisible(self._layers["conf"])
                    self._view.addItem(line)
                    self._items.setdefault("conf", []).append(line)
        except Exception as e:
            logger.debug(f"[3D] Confidence layer error: {e}", exc_info=True)

    def _draw_raw_points(self):
        """
        Raw scan points overlay.
        FIX #5 (overlay version): uses gz_signed for Z, not gz_norm.
        """
        if not self._layers["raw"] or self._view is None or self._grid is None:
            return
        try:
            import pyqtgraph.opengl as gl
            gz = np.nan_to_num(
                np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0)
            gx = np.asarray(self._grid.grid_x, dtype=np.float32)
            gy = np.asarray(self._grid.grid_y, dtype=np.float32)
            h, w = gz.shape
            xs = np.linspace(float(gx.min()), float(gx.max()), w)
            ys = np.linspace(float(gy.min()), float(gy.max()), h)
            gz_signed, _ = self._prepare_signal(gz)
            gz_z = (gz_signed * self._z_scale()).astype(np.float32)
            xx, yy = np.meshgrid(xs, ys)
            pos    = np.column_stack([
                xx.ravel(), yy.ravel(), gz_z.ravel()
            ]).astype(np.float32)
            colors = self._geomag_colormap(gz_signed).reshape(-1, 4).astype(np.float32)
            colors[:, 3] = 0.70
            pts = gl.GLScatterPlotItem(pos=pos, size=3, color=colors, pxMode=True)
            pts.setVisible(self._layers["raw"])
            self._view.addItem(pts)
            self._items.setdefault("raw", []).append(pts)
        except Exception as e:
            logger.debug(f"[3D] Raw points error: {e}")

    def _draw_blobs(self):
        """
        Blob extent markers.
        FIX #6: surf_z uses _z_scale() consistently (was using gz_norm × old z_scale).
        Blobs are now at z=0 (survey plane level), size from extent_cells.
        """
        if not self._layers["blobs"] or self._view is None or not self._anomalies:
            return
        try:
            import pyqtgraph.opengl as gl
            gx = np.asarray(self._grid.grid_x, dtype=float)
            gy = np.asarray(self._grid.grid_y, dtype=float)
            for a in self._anomalies:
                xm   = float(_get(a, "x", "centroid_x",
                                  default=self._survey_w / 2))
                ym   = float(_get(a, "y", "centroid_y",
                                  default=self._survey_l / 2))
                conf = float(_get(a, "confidence", "combined_confidence",
                                  default=0.4))
                ext  = float(_get(a, "extent_cells", default=4))
                xm   = float(np.clip(xm, gx.min(), gx.max()))
                ym   = float(np.clip(ym, gy.min(), gy.max()))
                r    = (ext / 10.0) * self._survey_w * 0.1 + 0.15
                blob = gl.GLScatterPlotItem(
                    pos=np.array([[xm, ym, 0.0]], dtype=np.float32),
                    size=r * 60, color=(*_conf_rgba(conf)[:3], 0.25), pxMode=True,
                )
                blob.setVisible(self._layers["blobs"])
                self._view.addItem(blob)
                self._items.setdefault("blobs", []).append(blob)
        except Exception as e:
            logger.debug(f"[3D] Blobs layer error: {e}")

    # ── Layer control ──────────────────────────────────────────────────────

    def set_layer(self, name: str, visible: bool):
        self._layers[name] = visible
        if visible and not self._items.get(name) and self._grid is not None:
            dispatch = {
                "signal":   self._draw_surface,
                "baseline": self._draw_baseline,
                "conf":     self._draw_confidence,
                "raw":      self._draw_raw_points,
                "blobs":    self._draw_blobs,
                "dig":      self._draw_anomalies,
                "grid":     self._draw_grid,
            }
            if name in dispatch:
                dispatch[name]()
            return
        for item in self._items.get(name, []):
            try:
                item.setVisible(visible)
            except Exception:
                pass
        if self._view:
            self._view.update()

    def set_render_mode(self, mode: str):
        """
        FIX #2 + #9: was calling _draw_surface() twice.
        Now calls it exactly once, only when a grid is loaded.
        """
        self._render_mode = mode
        if self._grid is None:
            return
        for item in self._items.get("signal", []):
            try:
                self._view.removeItem(item)
            except Exception:
                pass
        self._items["signal"] = []
        self._draw_surface()

    # ── Camera and selection ───────────────────────────────────────────────

    def set_vertical_exag(self, val: int):
        self._vert_exag = max(1, int(val))
        if self._grid is not None:
            self.set_scan(self._grid, self._anomalies)

    def set_camera_preset(self, preset: str):
        if self._view is None:
            return
        d = max(self._survey_w, self._survey_l) * 2.5
        presets = {
            "top":         dict(elevation=89, azimuth=0,  distance=d),
            "side":        dict(elevation=0,  azimuth=0,  distance=d),
            "perspective": dict(elevation=30, azimuth=45, distance=d),
            "reset":       dict(elevation=30, azimuth=45, distance=d),
        }
        if preset in presets:
            self._view.setCameraPosition(**presets[preset])
            # Presets must not move the orbit pivot off the survey centroid
            self._view.opts["center"] = pg.Vector(
                self._center_x, self._center_y, self._center_z)
            self._view.update()

    def select_anomaly(self, anomaly_id: str):
        items = self._items.get("dig", [])
        for i, a in enumerate(self._anomalies):
            aid = _get(a, "anomaly_id", "group_id", default=f"T{i:03d}")
            if i < len(items):
                col = ((1., 1., 0., 1.) if str(aid) == str(anomaly_id)
                       else _conf_rgba(float(_get(a, "confidence", default=0.4))))
                try:
                    items[i].setData(color=col)
                except Exception:
                    pass