"""
GMS — Viewport Keyboard Navigation  v1.0
==========================================
Arrow-key panning for both the 2D heatmap and the 3D GL viewport.

2D map:  matplotlib FigureCanvasQTAgg embedded in hmPlotLay.
         Pan by adjusting matplotlib xlim / ylim.
         Does NOT move data — only shifts the viewport window.

3D map:  pyqtgraph GLViewWidget embedded in vp3dWLay.
         Pan by directly translating opts["center"] (the camera
         target / orbit pivot) along world-space camera-right and
         camera-up vectors computed from azimuth/elevation. Distance,
         azimuth and elevation are read-only and never modified, so
         panning is a pure translation -- no rotation, no zoom change.
         Does NOT move geometry or survey coordinates.

Architecture:
  MapKeyFilter   — QObject event filter installed on the canvas widget.
                   Intercepts Qt.Key_Left/Right/Up/Down.
                   Calls a pan callback.
                   Focus is set automatically on mouse press.

  attach_2d_nav(canvas, ax)      — install on matplotlib FigureCanvasQTAgg
  attach_3d_nav(gl_view)         — install on pyqtgraph GLViewWidget

Both filters leave all mouse events (wheel, drag, click, context-menu)
completely untouched — they only intercept arrow key events.

Configurable:
  KEYBOARD_PAN_STEP_PX = 50    (pixels per keypress for 2D)
  KEYBOARD_PAN_STEP_3D = 0.15  (world-units fraction per keypress for 3D)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, QEvent
from PySide6.QtWidgets import QWidget

logger = logging.getLogger("gms.viewport_nav")

# ── Configurable pan speeds ───────────────────────────────────────────────────
KEYBOARD_PAN_STEP_PX  = 50     # pixels per arrow keypress — 2D map
KEYBOARD_PAN_STEP_3D  = 0.15   # fraction of survey width per keypress — 3D


# ─────────────────────────────────────────────────────────────────────────────
# Generic keyboard event filter
# ─────────────────────────────────────────────────────────────────────────────

class MapKeyFilter(QObject):
    """
    QObject event filter.
    Install on any widget to add arrow-key panning.
    Only consumes Key_Left / Key_Right / Key_Up / Key_Down.
    All other events pass through untouched.
    """

    def __init__(self,
                 target: QWidget,
                 pan_fn: Callable[[int, int], None],
                 parent=None):
        """
        Parameters
        ----------
        target  : the widget to install on (canvas or GLViewWidget)
        pan_fn  : callback(dx_px, dy_px) where:
                    dx_px > 0  → pan RIGHT  (view moves right)
                    dx_px < 0  → pan LEFT
                    dy_px > 0  → pan UP     (view moves up)
                    dy_px < 0  → pan DOWN
        """
        super().__init__(parent)
        self._target = target
        self._pan    = pan_fn

        target.setFocusPolicy(Qt.ClickFocus)
        target.installEventFilter(self)

    def eventFilter(self, obj, event: QEvent) -> bool:
        # Auto-focus on mouse press so arrow keys work immediately after click
        if event.type() == QEvent.MouseButtonPress:
            self._target.setFocus(Qt.MouseFocusReason)
            return False   # let mouse event continue normally

        if event.type() != QEvent.KeyPress:
            return False   # pass through everything that's not a key press

        key = event.key()
        step = KEYBOARD_PAN_STEP_PX

        if key == Qt.Key_Left:
            logger.debug("[Map] Pan Left")
            self._pan(-step, 0)
            return True

        if key == Qt.Key_Right:
            logger.debug("[Map] Pan Right")
            self._pan(step, 0)
            return True

        if key == Qt.Key_Up:
            logger.debug("[Map] Pan Up")
            self._pan(0, step)
            return True

        if key == Qt.Key_Down:
            logger.debug("[Map] Pan Down")
            self._pan(0, -step)
            return True

        return False   # all other keys pass through


# ─────────────────────────────────────────────────────────────────────────────
# 2D map — matplotlib FigureCanvasQTAgg
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapKeyNav:
    """
    Arrow-key panning for the matplotlib canvas.

    Pan is implemented by shifting xlim / ylim by a fraction of the
    current visible range — so it works correctly at any zoom level.
    The data arrays, survey coordinates, and anomaly positions are
    never touched.
    """

    def __init__(self, canvas, ax):
        """
        canvas : FigureCanvasQTAgg
        ax     : matplotlib Axes
        """
        self._canvas = canvas
        self._ax     = ax
        self._filter = MapKeyFilter(canvas, self._pan)

    def _pan(self, dx_px: int, dy_px: int):
        """
        Shift the matplotlib axes viewport.
        dx_px / dy_px are in screen pixels; we convert to data units
        using the current axis limits so zoom level is always respected.
        """
        ax = self._ax
        if ax is None:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Convert pixel step to data units:
        # KEYBOARD_PAN_STEP_PX pixels / figure_width_px × data_range
        try:
            fig_w_px = self._canvas.get_width_height()[0]
            fig_h_px = self._canvas.get_width_height()[1]
        except Exception:
            fig_w_px = fig_h_px = 600

        fig_w_px  = max(fig_w_px, 1)
        fig_h_px  = max(fig_h_px, 1)

        shift_x = (dx_px / fig_w_px) * x_range
        shift_y = (dy_px / fig_h_px) * y_range

        ax.set_xlim(xlim[0] + shift_x, xlim[1] + shift_x)
        ax.set_ylim(ylim[0] + shift_y, ylim[1] + shift_y)
        self._canvas.draw_idle()

    def update_ax(self, ax):
        """Call this after a new render to keep the axes reference current."""
        self._ax = ax


# ─────────────────────────────────────────────────────────────────────────────
# 3D map — pyqtgraph GLViewWidget
# ─────────────────────────────────────────────────────────────────────────────

class GLViewKeyNav:
    """
    Arrow-key panning for the pyqtgraph GLViewWidget.

    PURE CAMERA-TARGET TRANSLATION -- no rotation, no orbit, no zoom change.

    pyqtgraph's GLViewWidget.pan(dx, dy, dz, relative=True) is intentionally
    NOT used here: its "relative" mode interprets dx/dy as camera-LOCAL axes
    that are rotated through elevation/azimuth before being applied, and on
    some pyqtgraph versions this composes with the perspective projection in
    a way that reads visually as orbiting when the camera target is far from
    the rendered geometry.  To guarantee elevation/azimuth/distance/zoom are
    mathematically untouched, this implementation instead:

      1. Reads the CURRENT elevation, azimuth, and distance directly from
         self._view.opts (never modified).
      2. Computes the camera's local right-vector and up-vector in WORLD
         space from elevation/azimuth using standard spherical-to-cartesian
         basis vectors (same convention pyqtgraph itself uses internally
         for its own mouse-drag pan, but performed explicitly here so the
         result cannot vary across pyqtgraph versions).
      3. Adds a SCALAR multiple of those basis vectors directly to
         opts["center"] (the camera target = orbit pivot).
      4. Leaves opts["elevation"], opts["azimuth"], and opts["distance"]
         completely unchanged -- verified explicitly by the regression tests.

    Left/Right  -> translate target along the world-space camera-right axis
    Up/Down     -> translate target along the world-space camera-up axis
    """

    def __init__(self, gl_view):
        self._view   = gl_view
        self._filter = MapKeyFilter(gl_view, self._pan)

    def _camera_basis(self):
        """
        Compute world-space right/up unit vectors for the CURRENT camera
        orientation.  Pure read of elevation/azimuth -- never modifies them.

        Convention matches pyqtgraph's own spherical camera model:
          azimuth   -- rotation about world Z, degrees, 0 deg = +X axis
          elevation -- angle above the XY plane, degrees, 90 deg = looking
                       straight down the -Z axis
        """
        import numpy as np
        az = np.radians(float(self._view.opts.get("azimuth", 45.0)))
        el = np.radians(float(self._view.opts.get("elevation", 30.0)))

        # Forward vector (camera -> target direction, unit length)
        forward = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])
        world_up = np.array([0.0, 0.0, 1.0])

        # Right = forward x world_up (screen-right in world space)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm

        # True up = right x forward (screen-up in world space)
        up = np.cross(right, forward)
        up_norm = np.linalg.norm(up)
        if up_norm > 1e-9:
            up = up / up_norm
        else:
            up = world_up

        return right, up

    def _pan(self, dx_px: int, dy_px: int):
        """
        Translate the camera target (opts["center"]) by a world-space
        offset proportional to the on-screen pixel delta.  Camera DISTANCE
        is used only as a scale reference (so panning speed matches zoom
        level) and is itself never written to.
        """
        if self._view is None:
            return

        import numpy as np
        try:
            dist = float(self._view.opts.get("distance", 10.0))
        except Exception:
            dist = 10.0

        # World-unit step, proportional to current zoom (camera distance)
        scale  = KEYBOARD_PAN_STEP_3D * (dist / 10.0)
        step_x = (dx_px / KEYBOARD_PAN_STEP_PX) * scale
        step_y = (dy_px / KEYBOARD_PAN_STEP_PX) * scale

        right, up = self._camera_basis()
        offset = right * step_x + up * step_y

        try:
            center = self._view.opts.get("center", None)
            cx = float(center.x()) if center is not None else 0.0
            cy = float(center.y()) if center is not None else 0.0
            cz = float(center.z()) if center is not None else 0.0
        except Exception:
            cx = cy = cz = 0.0

        new_cx = cx + float(offset[0])
        new_cy = cy + float(offset[1])
        new_cz = cz + float(offset[2])

        try:
            import pyqtgraph as pg
            self._view.opts["center"] = pg.Vector(new_cx, new_cy, new_cz)
        except Exception as e:
            logger.debug(f"[3D] Pan error setting center: {e}")
            return

        # opts["elevation"], opts["azimuth"], opts["distance"] are
        # INTENTIONALLY never touched above -- only "center" changes.
        try:
            self._view.update()
        except Exception as e:
            logger.debug(f"[3D] Pan update error: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# Public attach helpers — called from HeatmapController and __init__.py
# ─────────────────────────────────────────────────────────────────────────────

def attach_2d_nav(canvas, ax) -> Optional[HeatmapKeyNav]:
    """
    Attach arrow-key navigation to a matplotlib FigureCanvasQTAgg.
    Returns the nav object (keep alive to prevent GC).
    """
    try:
        nav = HeatmapKeyNav(canvas, ax)
        logger.info("[Map] 2D keyboard navigation attached")
        return nav
    except Exception as e:
        logger.warning(f"[Map] 2D nav attach failed: {e}")
        return None


def attach_3d_nav(gl_view) -> Optional[GLViewKeyNav]:
    """
    Attach arrow-key navigation to a pyqtgraph GLViewWidget.
    Returns the nav object (keep alive to prevent GC).
    """
    try:
        nav = GLViewKeyNav(gl_view)
        logger.info("[Map] 3D keyboard navigation attached")
        return nav
    except Exception as e:
        logger.warning(f"[Map] 3D nav attach failed: {e}")
        return None