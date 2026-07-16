"""
GMS — VolumetricEngine  v3.5
================================
OKM Visualizer3D style: one GLViewWidget, embedded EXACTLY where
vp3dPH lives by replacing it in its parent layout (vp3dWLay).

Layout path (from UI file):
  tabExplorer3D
    └─ QWidget[viewport3dWidget]
         └─ QVBoxLayout[vp3dWLay]
              └─ QLabel[vp3dPH]   ← replaced with GLViewWidget

All geometry in real survey metres from BaselinedGrid.grid_x / grid_y.
Layer checkboxes wire directly to GL item setVisible().
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QMainWindow

logger = logging.getLogger("gms.volumetric")


def _detect_backend():
    try:
        import pyqtgraph.opengl as gl
        logger.info("[Volumetric] Backend: pyqtgraph.opengl")
        return "pyqtgraph", gl
    except ImportError:
        logger.warning("[Volumetric] pyqtgraph not found — fallback label")
        return "fallback", None

_BACKEND, _GL = _detect_backend()


class VolumetricEngine(QObject):
    anomaly_picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view: Optional[object]    = None   # GLViewWidget
        self._gl_items: Dict[str, List] = {}
        self._vert_exag: int            = 3
        self._current_grid              = None
        self._current_anomalies: list   = []
        self._selected_id: Optional[str]= None
        self._survey_w: float           = 5.0
        self._survey_l: float           = 5.0

        # Layer visibility state
        self._layers = {
            "signal":      True,
            "baseline":    False,
            "anomalies":   True,
            "dig_markers": True,
            "grid":        True,
            "confidence":  False,
            "raw_points":  False,
        }

    # ── Attach — replaces vp3dPH in-place ─────────────────────────────────

    def attach(self, window: QMainWindow):
        """
        Find QLabel[vp3dPH], get its parent layout (vp3dWLay), remove the
        label, and insert the GLViewWidget at the same index.
        This guarantees exactly one viewport with no duplicate widgets.
        """
        placeholder = window.findChild(QLabel, "vp3dPH")
        if placeholder is None:
            logger.error("[Volumetric] vp3dPH not found in window")
            return

        parent_widget = placeholder.parentWidget()
        if parent_widget is None:
            logger.error("[Volumetric] vp3dPH has no parent widget")
            return

        lay = parent_widget.layout()
        if lay is None:
            logger.error("[Volumetric] vp3dPH parent has no layout")
            return

        if _BACKEND == "pyqtgraph":
            self._attach_gl(placeholder, lay, parent_widget)
        else:
            placeholder.setText(
                "3D Viewer\n\nInstall: pip install pyqtgraph PyOpenGL\n"
                "to enable GPU rendering"
            )
            placeholder.setStyleSheet("color:#666; font-size:13px;")

    def _attach_gl(self, placeholder, lay, parent_widget):
        import pyqtgraph.opengl as gl

        # Find the placeholder's position in the layout
        idx = lay.indexOf(placeholder)

        # Build the GL widget BEFORE removing the placeholder
        view = gl.GLViewWidget()
        view.setObjectName("_gmsGLViewport")
        view.setBackgroundColor((12, 14, 18, 255))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setMinimumSize(100, 100)

        # Camera: perspective, OKM default angle
        view.setCameraPosition(distance=10, elevation=30, azimuth=45)

        # Remove placeholder, insert GL widget at the same slot
        lay.removeWidget(placeholder)
        placeholder.setVisible(False)
        placeholder.setParent(None)

        if idx >= 0:
            lay.insertWidget(idx, view)
        else:
            lay.addWidget(view)

        self._view = view

        # Add base grid axes
        self._add_grid_axis(self._survey_w, self._survey_l)

        logger.info("[Volumetric] GLViewWidget attached at vp3dPH slot")

    # ── Data loading ───────────────────────────────────────────────────────

    def set_scan(self, grid, anomalies: list = None, geometry=None):
        if self._view is None:
            logger.warning("[Volumetric] Not attached yet — call attach() first")
            return

        # Cleanup previous GPU resources
        self.cleanup_gpu()

        self._current_grid      = grid
        self._current_anomalies = anomalies or []

        # Survey extent: prefer grid coordinates, fall back to geometry
        gx = getattr(grid, "grid_x", None)
        gy = getattr(grid, "grid_y", None)
        if gx is not None and gy is not None and len(gx) > 1 and len(gy) > 1:
            self._survey_w = float(gx.max() - gx.min())
            self._survey_l = float(gy.max() - gy.min())
        elif geometry is not None:
            self._survey_w = float(geometry.field_width_m)
            self._survey_l = float(geometry.field_length_m)

        # Set camera distance to fit the whole survey
        diag = float(np.sqrt(self._survey_w**2 + self._survey_l**2))
        self._view.setCameraPosition(
            distance=max(diag * 2.2, 5.0),
            elevation=30, azimuth=45,
        )

        self._add_grid_axis(self._survey_w, self._survey_l)
        self._render_surface(grid)
        self._render_anomalies(anomalies or [])

    # ── Scene construction ─────────────────────────────────────────────────

    def _add_grid_axis(self, w: float, l: float):
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        for item in self._gl_items.get("grid", []):
            try:
                self._view.removeItem(item)
            except Exception:
                pass

        grid = gl.GLGridItem()
        grid.setSize(w, l, 1)
        grid.setSpacing(max(w / 10, 0.05), max(l / 10, 0.05), 1)
        grid.setColor((60, 65, 85, 100))
        grid.setVisible(self._layers["grid"])
        # Centre the grid over the survey
        grid.translate(w / 2, l / 2, 0)
        self._view.addItem(grid)
        self._gl_items["grid"] = [grid]

    def _render_surface(self, grid):
        """Render the signal surface mesh.
        Implements mesh simplification for large grids and depth‑based opacity.
        """
        if self._view is None or not self._layers["signal"]:
            return
        try:
            import pyqtgraph.opengl as gl
            import matplotlib.cm as cm

            gz = np.nan_to_num(
                np.asarray(grid.grid_z, dtype=np.float32), nan=0.0
            )
            if gz.size == 0:
                return

            gx = np.asarray(grid.grid_x, dtype=np.float32)
            gy = np.asarray(grid.grid_y, dtype=np.float32)

            # --- Mesh simplification ------------------------------------------------
            # If the grid exceeds 200 × 200 points we down‑sample to keep GPU memory low.
            max_res = 200
            h, w = gz.shape
            step_h = max(1, h // max_res)
            step_w = max(1, w // max_res)
            if step_h > 1 or step_w > 1:
                gz = gz[::step_h, ::step_w]
                gx = gx[::step_h]
                gy = gy[::step_w]
                h, w = gz.shape
                logger.debug(f"[Volumetric] Down‑sampled surface to {h}×{w} for GPU efficiency")

            xs = np.linspace(float(gx.min()), float(gx.max()), w)
            ys = np.linspace(float(gy.min()), float(gy.max()), h)

            gz_norm = (gz - gz.min()) / (gz.ptp() + 1e-9)
            # Z scaled to fraction of survey width × vertical exag
            z_scale = min(self._survey_w, self._survey_l) * 0.20 * self._vert_exag
            gz_z    = gz_norm * z_scale

            xx, yy = np.meshgrid(xs, ys)
            verts = np.column_stack([
                xx.ravel(), yy.ravel(), gz_z.ravel()
            ]).astype(np.float32)

            faces = []
            for r in range(h - 1):
                for c in range(w - 1):
                    tl = r * w + c
                    faces += [[tl, tl+1, tl+w], [tl+1, tl+w+1, tl+w]]
            if not faces:
                return
            faces = np.array(faces, dtype=np.uint32)

            # --- Colormap & depth‑based opacity -------------------------------------
            cmap   = cm.get_cmap("plasma")
            fz_avg = (gz_norm.ravel()[faces[:,0]] +
                      gz_norm.ravel()[faces[:,1]] +
                      gz_norm.ravel()[faces[:,2]]) / 3.0
            colors = cmap(fz_avg).astype(np.float32)
            # Alpha varies with normalized height (deeper = more transparent)
            colors[:,3] = 0.4 + 0.6 * gz_norm.ravel()[faces[:,0]]  # simple depth cue

            mesh = gl.GLMeshItem(
                vertexes=verts, faces=faces, faceColors=colors,
                smooth=True, drawEdges=False,
            )
            mesh.setVisible(self._layers["signal"])
            self._view.addItem(mesh)
            self._gl_items.setdefault("signal", []).append(mesh)

        except Exception as e:
            logger.error(f"[Volumetric] Surface render failed: {e}", exc_info=True)

    def _render_anomalies(self, anomalies: list):
        if self._view is None:
            return
        if not anomalies:
            return
        try:
            import pyqtgraph.opengl as gl

            grid = self._current_grid
            gz   = np.nan_to_num(
                np.asarray(grid.grid_z, dtype=np.float32), nan=0.0
            ) if grid else np.zeros((10, 10), dtype=np.float32)

            gz_norm = (gz - gz.min()) / (gz.ptp() + 1e-9)
            z_scale = min(self._survey_w, self._survey_l) * 0.20 * self._vert_exag
            surface_top_z = float(gz_norm.max()) * z_scale + 0.12

            gx = getattr(grid, "grid_x", None)
            gy = getattr(grid, "grid_y", None)
            x0 = float(gx.min()) if gx is not None else 0.0
            y0 = float(gy.min()) if gy is not None else 0.0

            for a in anomalies:
                xm   = float(_get(a, "x", "centroid_x", default=self._survey_w/2))
                ym   = float(_get(a, "y", "centroid_y", default=self._survey_l/2))
                conf = float(_get(a, "confidence", "combined_confidence", default=0.4))

                # Clamp to survey bounds
                xm = np.clip(xm, x0, x0 + self._survey_w)
                ym = np.clip(ym, y0, y0 + self._survey_l)

                color = _conf_rgba(conf)
                size  = 14 + conf * 18

                scatter = gl.GLScatterPlotItem(
                    pos=np.array([[xm, ym, surface_top_z]], dtype=np.float32),
                    size=size, color=color, pxMode=True,
                )
                scatter.setVisible(self._layers["anomalies"])
                self._view.addItem(scatter)
                self._gl_items.setdefault("anomalies", []).append(scatter)

                # Vertical stake for DIG-confidence targets
                if conf >= 0.70 and self._layers["dig_markers"]:
                    line = gl.GLLinePlotItem(
                        pos=np.array([
                            [xm, ym, 0.0],
                            [xm, ym, surface_top_z + 0.25]
                        ], dtype=np.float32),
                        color=(0.1, 1.0, 0.1, 0.9),
                        width=2.0, antialias=True,
                    )
                    self._view.addItem(line)
                    self._gl_items.setdefault("dig_markers", []).append(line)

        except Exception as e:
            logger.error(f"[Volumetric] Anomaly render failed: {e}", exc_info=True)

    def _clear_data_layers(self):
        if self._view is None:
            return
        for key in ("signal", "anomalies", "dig_markers", "baseline", "confidence"):
            for item in self._gl_items.get(key, []):
                try:
                    self._view.removeItem(item)
                except Exception:
                    pass
            self._gl_items[key] = []

    def cleanup_gpu(self):
        """Release GPU resources by removing all GL items."""
        self._clear_data_layers()
        # Additionally, clear the grid axis
        if "grid" in self._gl_items:
            for item in self._gl_items["grid"]:
                try:
                    self._view.removeItem(item)
                except Exception:
                    pass
            self._gl_items["grid"] = []
        # Force a garbage collection pass (optional, but helps with pyqtgraph)
        import gc
        gc.collect()

    # ── Layer control (wired from checkboxes) ──────────────────────────────

    def set_layer(self, name: str, visible: bool):
        self._layers[name] = visible
        for item in self._gl_items.get(name, []):
            try:
                item.setVisible(visible)
            except Exception:
                pass
        # Grid layer also controls the base grid
        if name == "grid":
            for item in self._gl_items.get("grid", []):
                try:
                    item.setVisible(visible)
                except Exception:
                    pass

    # ── Camera ─────────────────────────────────────────────────────────────

    def set_camera_preset(self, preset: str):
        if self._view is None:
            return
        dist = max(self._survey_w, self._survey_l) * 2.2
        presets = {
            "top":         dict(elevation=89, azimuth=0,   distance=dist),
            "side":        dict(elevation=0,  azimuth=0,   distance=dist),
            "perspective": dict(elevation=30, azimuth=45,  distance=dist),
            "reset":       dict(elevation=30, azimuth=45,  distance=dist),
        }
        if preset in presets:
            self._view.setCameraPosition(**presets[preset])

    def set_vertical_exag(self, value: int):
        self._vert_exag = max(1, int(value))
        if self._current_grid is not None:
            self.set_scan(self._current_grid, self._current_anomalies)

    def select_anomaly(self, anomaly_id: str):
        self._selected_id = anomaly_id
        items = self._gl_items.get("anomalies", [])
        for i, a in enumerate(self._current_anomalies):
            aid = _get(a, "anomaly_id", "group_id", default=f"T{i:03d}")
            if i < len(items):
                col = (1.0, 1.0, 0.0, 1.0) if str(aid) == str(anomaly_id) \
                      else _conf_rgba(float(_get(a, "confidence", default=0.4)))
                try:
                    items[i].setData(color=col)
                except Exception:
                    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(obj, *attrs, default=None):
    for attr in attrs:
        v = getattr(obj, attr, None)
        if v is not None:
            return v
        if isinstance(obj, dict) and attr in obj:
            return obj[attr]
    return default


def _conf_rgba(conf: float) -> tuple:
    if conf >= 0.70:
        return (0.10, 1.00, 0.25, 0.95)
    elif conf >= 0.45:
        return (1.00, 0.65, 0.00, 0.90)
    return (1.00, 0.20, 0.20, 0.85)
