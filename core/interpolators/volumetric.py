"""
GMS — VolumetricEngine  v3.6
================================
Target layout (from actual .ui file):
  tabExplorer3D
    exp3dLay (QHBoxLayout)
      vp3dLay (QVBoxLayout)
        viewer3dFrame (QFrame)
          vp3dInner (QVBoxLayout)
            viewport3dWidget (QWidget)
              vp3dWLay (QVBoxLayout)   ← ADD GL WIDGET HERE (currently empty)

3D checkboxes (actual names in .ui):
  chk3dSignal, chk3dBase, chk3dDig, chk3dConf, chk3dRaw, chk3dGrid, chk3dBlobs

All geometry in real survey metres from grid_x / grid_y arrays.
Uses GLSurfacePlotItem for the professional field surface.
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
        import pyqtgraph.opengl as gl
        logger.info("[Volumetric] Backend: pyqtgraph.opengl")
        return True, gl
    except ImportError:
        logger.warning("[Volumetric] pyqtgraph not available — 3D disabled")
        return False, None

_HAS_GL, _GL = _detect_gl()


class VolumetricEngine(QObject):
    anomaly_picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view      = None   # GLViewWidget
        self._items: Dict[str, List] = {}   # layer_name → [GL items]
        self._vert_exag = 3
        self._survey_w  = 5.0
        self._survey_l  = 5.0
        self._grid      = None
        self._anomalies: list = []
        self._layers = {
            "signal":    True,
            "baseline":  False,
            "dig":       True,
            "conf":      False,
            "raw":       False,
            "grid":      True,
            "blobs":     False,
        }

    # ── Attach ─────────────────────────────────────────────────────────────

    def attach(self, window: QMainWindow) -> bool:
        """
        Insert GLViewWidget into vp3dWLay (the empty QVBoxLayout inside
        viewport3dWidget). This layout is EMPTY in the .ui file so we
        simply addWidget — no removeWidget needed.
        """
        if not _HAS_GL:
            return False

        container = window.findChild(QWidget, "viewport3dWidget")
        if container is None:
            logger.error("[Volumetric] viewport3dWidget not found")
            return False

        lay = container.layout()   # vp3dWLay
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

        lay.addWidget(view)
        self._view = view

        logger.info("[Volumetric] GLViewWidget added to vp3dWLay")
        return True

    # ── Data ───────────────────────────────────────────────────────────────

    def set_scan(self, grid, anomalies: list = None, geometry=None):
        if self._view is None:
            return

        self._grid      = grid
        self._anomalies = anomalies or []

        # Survey extent from actual grid coordinates
        gx = np.asarray(grid.grid_x, dtype=float)
        gy = np.asarray(grid.grid_y, dtype=float)
        if len(gx) > 1 and len(gy) > 1:
            self._survey_w = float(gx.max() - gx.min())
            self._survey_l = float(gy.max() - gy.min())
        elif geometry is not None:
            self._survey_w = float(geometry.field_width_m)
            self._survey_l = float(geometry.field_length_m)

        # Auto-fit camera
        diag = np.sqrt(self._survey_w**2 + self._survey_l**2)
        self._view.setCameraPosition(
            distance=float(diag) * 2.5,
            elevation=30, azimuth=45,
        )

        self._clear_all()
        self._draw_grid()
        self._draw_surface()
        self._draw_anomalies()

        print(f"[3D] mesh survey={self._survey_w:.2f}x{self._survey_l:.2f}m "
              f"grid={np.asarray(grid.grid_z).shape} "
              f"anomalies={len(self._anomalies)}")

    # ── Drawing ────────────────────────────────────────────────────────────

    def _clear_all(self):
        if self._view is None:
            return
        for key, items in self._items.items():
            for item in items:
                try:
                    self._view.removeItem(item)
                except Exception:
                    pass
        self._items = {}

    def _draw_grid(self):
        if not self._layers["grid"] or self._view is None:
            return
        try:
            import pyqtgraph.opengl as gl
            g = gl.GLGridItem()
            g.setSize(self._survey_w, self._survey_l, 1)
            g.setSpacing(
                max(self._survey_w / 10, 0.05),
                max(self._survey_l / 10, 0.05),
                1,
            )
            g.setColor((50, 60, 80, 100))
            # Centre grid over survey
            gx = np.asarray(self._grid.grid_x)
            gy = np.asarray(self._grid.grid_y)
            g.translate(float(gx.min()) + self._survey_w / 2,
                        float(gy.min()) + self._survey_l / 2,
                        0)
            self._view.addItem(g)
            self._items["grid"] = [g]
        except Exception as e:
            logger.error(f"[3D] Grid error: {e}")

    def _draw_surface(self):
        if not self._layers["signal"] or self._view is None:
            return
        try:
            import pyqtgraph.opengl as gl
            import matplotlib.cm as cm

            gz = np.nan_to_num(
                np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0
            )
            if gz.size == 0:
                return

            gx = np.asarray(self._grid.grid_x, dtype=np.float32)
            gy = np.asarray(self._grid.grid_y, dtype=np.float32)

            h, w = gz.shape
            xs = np.linspace(float(gx.min()), float(gx.max()), w)
            ys = np.linspace(float(gy.min()), float(gy.max()), h)

            gz_norm = (gz - gz.min()) / (gz.ptp() + 1e-9)
            z_scale = min(self._survey_w, self._survey_l) * 0.25 * self._vert_exag
            gz_z    = (gz_norm * z_scale).astype(np.float32)

            # Build vertex grid
            xx, yy = np.meshgrid(xs, ys)
            verts = np.stack([xx, yy, gz_z], axis=-1).astype(np.float32)

            # Professional colormap: plasma/inferno gradient
            cmap   = cm.get_cmap("plasma")
            colors = cmap(gz_norm).astype(np.float32)
            colors[:, :, 3] = 0.88  # slight transparency

            surf = gl.GLSurfacePlotItem(
                x=xs, y=ys, z=gz_z,
                colors=colors,
                shader="shaded",
                smooth=True,
                drawEdges=False,
            )
            surf.setVisible(self._layers["signal"])
            self._view.addItem(surf)
            self._items["signal"] = [surf]

            print(f"[3D] Surface: {h}x{w} verts, "
                  f"z=[{gz_z.min():.3f},{gz_z.max():.3f}]m")

        except Exception as e:
            logger.error(f"[3D] Surface error: {e}", exc_info=True)
            # Fallback: mesh-based surface
            self._draw_surface_mesh()

    def _draw_surface_mesh(self):
        """Fallback mesh if GLSurfacePlotItem fails."""
        try:
            import pyqtgraph.opengl as gl
            import matplotlib.cm as cm

            gz = np.nan_to_num(
                np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0
            )
            gx = np.asarray(self._grid.grid_x, dtype=np.float32)
            gy = np.asarray(self._grid.grid_y, dtype=np.float32)

            h, w = gz.shape
            xs = np.linspace(float(gx.min()), float(gx.max()), w)
            ys = np.linspace(float(gy.min()), float(gy.max()), h)

            gz_norm = (gz - gz.min()) / (gz.ptp() + 1e-9)
            z_scale = min(self._survey_w, self._survey_l) * 0.25 * self._vert_exag
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
            faces = np.array(faces, dtype=np.uint32)

            cmap = cm.get_cmap("plasma")
            fi   = (gz_norm.ravel()[faces[:,0]] +
                    gz_norm.ravel()[faces[:,1]] +
                    gz_norm.ravel()[faces[:,2]]) / 3
            fc = cmap(fi).astype(np.float32)
            fc[:, 3] = 0.88

            mesh = gl.GLMeshItem(
                vertexes=verts, faces=faces, faceColors=fc,
                smooth=True, drawEdges=False,
            )
            mesh.setVisible(self._layers["signal"])
            self._view.addItem(mesh)
            self._items.setdefault("signal", []).append(mesh)
        except Exception as e:
            logger.error(f"[3D] Mesh fallback error: {e}")

    def _draw_anomalies(self):
        if not self._layers["dig"] or self._view is None:
            return
        if not self._anomalies:
            return
        try:
            import pyqtgraph.opengl as gl

            gz     = np.nan_to_num(
                np.asarray(self._grid.grid_z, dtype=np.float32), nan=0.0
            )
            gz_norm = (gz - gz.min()) / (gz.ptp() + 1e-9)
            z_scale = min(self._survey_w, self._survey_l) * 0.25 * self._vert_exag
            surf_z  = float(gz_norm.max()) * z_scale + 0.1

            gx = np.asarray(self._grid.grid_x, dtype=float)
            gy = np.asarray(self._grid.grid_y, dtype=float)

            for a in self._anomalies:
                xm   = float(_get(a, "x", "centroid_x", default=self._survey_w/2))
                ym   = float(_get(a, "y", "centroid_y", default=self._survey_l/2))
                conf = float(_get(a, "confidence", "combined_confidence", default=0.4))

                xm = float(np.clip(xm, gx.min(), gx.max()))
                ym = float(np.clip(ym, gy.min(), gy.max()))

                color = _conf_rgba(conf)

                # Sphere marker
                scatter = gl.GLScatterPlotItem(
                    pos=np.array([[xm, ym, surf_z]], dtype=np.float32),
                    size=16 + conf * 20,
                    color=color,
                    pxMode=True,
                )
                scatter.setVisible(self._layers["dig"])
                self._view.addItem(scatter)
                self._items.setdefault("dig", []).append(scatter)

                # Vertical stake for DIG targets
                if conf >= 0.70:
                    line = gl.GLLinePlotItem(
                        pos=np.array([[xm, ym, 0.0],
                                      [xm, ym, surf_z + 0.3]],
                                     dtype=np.float32),
                        color=(0.1, 1.0, 0.1, 0.9),
                        width=2.5, antialias=True,
                    )
                    self._view.addItem(line)
                    self._items.setdefault("dig", []).append(line)

                print(f"[ANOMALY] label={_get(a,'label','?')} "
                      f"x_m={xm:.3f} y_m={ym:.3f} conf={conf:.1%}")

        except Exception as e:
            logger.error(f"[3D] Anomaly error: {e}", exc_info=True)

    # ── Layer control ──────────────────────────────────────────────────────

    def set_layer(self, name: str, visible: bool):
        self._layers[name] = visible
        for item in self._items.get(name, []):
            try:
                item.setVisible(visible)
            except Exception:
                pass
        if self._view:
            self._view.update()

    def set_vertical_exag(self, val: int):
        self._vert_exag = max(1, int(val))
        if self._grid is not None:
            self.set_scan(self._grid, self._anomalies)

    def set_camera_preset(self, preset: str):
        if self._view is None:
            return
        d = max(self._survey_w, self._survey_l) * 2.5
        p = {
            "top":         dict(elevation=89, azimuth=0,  distance=d),
            "side":        dict(elevation=0,  azimuth=0,  distance=d),
            "perspective": dict(elevation=30, azimuth=45, distance=d),
            "reset":       dict(elevation=30, azimuth=45, distance=d),
        }
        if preset in p:
            self._view.setCameraPosition(**p[preset])

    def select_anomaly(self, anomaly_id: str):
        items = self._items.get("dig", [])
        for i, a in enumerate(self._anomalies):
            aid = _get(a, "anomaly_id", "group_id", default=f"T{i:03d}")
            if i < len(items):
                col = (1., 1., 0., 1.) if str(aid) == str(anomaly_id) \
                      else _conf_rgba(float(_get(a, "confidence", default=0.4)))
                try:
                    items[i].setData(color=col)
                except Exception:
                    pass


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
