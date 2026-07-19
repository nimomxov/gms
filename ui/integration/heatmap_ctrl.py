"""
GMS — HeatmapController  v3.9
================================
All six rendering controls are live and correctly tiered.

Control → Action mapping
------------------------
cmbCmap    (Colormap)     RENDER_ONLY  → redraw with new colormap, same grid
sldBright  (Brightness)   RENDER_ONLY  → rescale vmin/vmax, same grid
sldCont    (Contrast)     RENDER_ONLY  → rescale vmin/vmax, same grid
sldSmooth  (Smoothing)    SMOOTH_ONLY  → gaussian-filter grid_z copy, redraw
cmbInterp  (Interpolator) RERUN_INTERP → rebuild pipeline interpolator stage, rerun
cmbBase    (Baseline)     RERUN_BASE   → rebuild pipeline baseline stage, rerun

RENDER_ONLY:  updates im.set_cmap() / im.set_clim() + canvas.draw_idle() only.
              The AxesImage is kept alive between renders so the update
              is a single Agg blit — no imshow() call, no layout change.

SMOOTH_ONLY:  applies scipy.ndimage.gaussian_filter(grid_z, sigma) to a copy
              of the stored BaselinedGrid.  Updates im.set_data() + draw_idle().

RERUN_INTERP/RERUN_BASE: rebuilds GMSPipeline with new interpolator/baseline key,
              calls process_scan(filepath) on the stored scan file, stores the
              new BaselinedGrid, then re-renders.  Runs on the global QThreadPool
              so the UI stays responsive.

Figure setup (v3.8 fix preserved)
----------------------------------
Uses matplotlib.figure.Figure() directly — NOT plt.subplots() — so pyplot's
figure manager never touches the canvas and FigureCanvasQTAgg is guaranteed to
be the one and only canvas attached to the figure.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import matplotlib
import matplotlib.cm as mcm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.patches as mpatches

from PySide6.QtCore  import QObject, QRunnable, QThreadPool, Signal, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QComboBox, QSlider, QCheckBox, QVBoxLayout,
    QLabel, QFrame, QSizePolicy, QMessageBox,
)

from .app_state import GMSApplicationState

logger = logging.getLogger("gms.heatmap")

_GRID_COLOR  = "#00FFCC"
_GRID_ALPHA  = 0.55
_GRID_LW     = 0.8

RENDER_ONLY  = "render_only"
SMOOTH_ONLY  = "smooth_only"
RERUN_INTERP = "interpolation"
RERUN_BASE   = "baseline"

_CTRL_TIER = {
    "cmbCmap":       RENDER_ONLY,
    "sldBright":     RENDER_ONLY,
    "sldCont":       RENDER_ONLY,
    "chkLSignal":    RENDER_ONLY,
    "chkLBaseline":  RENDER_ONLY,
    "chkLAnomalies": RENDER_ONLY,
    "chkLDigZones":  RENDER_ONLY,
    "chkLGrid":      RENDER_ONLY,
    "chkLConfidence":RENDER_ONLY,
    "chkLRawPts":    RENDER_ONLY,
    "sldSmooth":     SMOOTH_ONLY,
    "cmbInterp":     RERUN_INTERP,
    "cmbBase":       RERUN_BASE,
}

# ── Display label → registry key maps ──────────────────────────────────────
# gms_main_window.ui pre-populates cmbInterp / cmbBase / cmbCmap with
# human-friendly display text that does NOT match the internal registry
# keys (INTERPOLATOR_REGISTRY / BASELINE_REGISTRY) or matplotlib colormap
# names.  These maps translate the combo box selection to a valid key
# BEFORE it is passed to the pipeline or to im.set_cmap().
#
# Mapping rationale (no 1:1 match in the registry → closest algorithm):
#   Nearest    → linear     (no nearest-neighbour interpolator registered)
#   Gaussian   → rbf        (closest smooth-kernel interpolator)
#   Polynomial → multiscale (closest curve-fit baseline; no literal
#                            polynomial-fit baseline is registered)
#   Rolling Mean → line_median (closest moving-window baseline)
_INTERP_DISPLAY_TO_KEY = {
    "linear":  "linear",
    "nearest": "linear",
    "cubic":   "cubic",
    "gaussian":"rbf",
    "rbf":     "rbf_thin_plate",
}
_BASE_DISPLAY_TO_KEY = {
    "wavelet":      "wavelet_bg",
    "polynomial":   "multiscale",
    "rolling mean": "line_median",
    "none":         "none",
}
_CMAP_DISPLAY_TO_KEY = {
    "inferno":   "inferno",
    "turbo":     "turbo",
    "viridis":   "viridis",
    "plasma":    "plasma",
    "grayscale": "gray",
    "custom...": "plasma",   # placeholder until custom colormap UI exists
}


def _resolve_key(display_text: str, table: dict, default: str) -> str:
    """Map a combo box display label to its registry/colormap key.

    Falls back to `default` when the label is unrecognised (e.g. the
    table was extended without updating the map — fail safe, never
    raise KeyError into the pipeline).
    """
    key = table.get(display_text.strip().lower())
    if key is None:
        logger.warning(
            "[Heatmap] Unrecognised combo value %r — falling back to %r",
            display_text, default)
        return default
    return key


def _w(parent, cls, name):
    f = parent.findChild(cls, name)
    if f is None:
        logger.debug("[Heatmap] %s[%s] not found", cls.__name__, name)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Background rerun worker
# ─────────────────────────────────────────────────────────────────────────────

class _RerunWorker(QRunnable):
    """Run a partial pipeline rerun (new interpolator/baseline) on the thread pool."""

    class Signals(QObject):
        done   = Signal(object)   # BaselinedGrid
        failed = Signal(str)

    def __init__(self, filepath: str, interp_key: str, base_key: str,
                 gms_config: dict):
        super().__init__()
        self.signals   = _RerunWorker.Signals()
        self._filepath = filepath
        self._interp   = interp_key
        self._base     = base_key
        self._cfg      = gms_config

    def run(self):
        try:
            # GMSPipeline is constructed directly with a PipelineConfig —
            # build_pipeline() only accepts preset name or a raw dict
            # (pipeline_section), not a PipelineConfig instance.
            from core.pipeline import GMSPipeline, PipelineConfig
            cfg = PipelineConfig(interpolator=self._interp, baseline=self._base)
            pipeline = GMSPipeline(cfg, self._cfg)
            baselined, _ = pipeline.process_scan(self._filepath)
            self.signals.done.emit(baselined)
        except Exception as e:
            logger.exception("[Heatmap] Rerun failed")
            self.signals.failed.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# HeatmapController
# ─────────────────────────────────────────────────────────────────────────────

class HeatmapController(QObject):
    render_ready = Signal(bytes)

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()

        self._grid:    Optional[object] = None
        self._result:  dict             = {}

        self._scan_file:  str  = ""
        self._interp_key: str  = "cubic"
        self._base_key:   str  = "line_median"
        self._gms_config: dict = {}

        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._fig:    Optional[Figure]            = None
        self._ax                                  = None
        self._im                                  = None
        self._cbar                                = None
        self._nav_2d                              = None
        self._plot_lay: Optional[QVBoxLayout]     = None

        self._hover_cid = None
        self._hover_ann = None

        # Layout lock (anti-shrink): stored after first tight_layout call
        self._ax_pos    = None   # Bbox — restored before every ax.cla()

        # Scroll-zoom state
        self._scroll_cid = None

        # 3D sync: callable set by bootstrap, called after every
        # interpolator/baseline rerun so the 3D map stays in sync.
        # Signature: fn(baselined_grid, anomalies, geometry) → None
        self._vol_push_fn = None

        self._pending:  Optional[str] = None
        self._debounce  = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._flush)

        self._wire_controls()
        self._state.pipeline_completed.connect(self._on_pipeline_completed)
        self._state.dataset_cleared.connect(self._clear)

        logger.info("[Heatmap] v3.9 — backend: %s", matplotlib.get_backend())

    # ── Control wiring ────────────────────────────────────────────────────

    def _wire_controls(self):
        for name in ("cmbCmap", "cmbInterp", "cmbBase"):
            w = _w(self._w, QComboBox, name)
            if w:
                w.currentIndexChanged.connect(lambda _, n=name: self._touch(n))
        for name in ("sldBright", "sldCont", "sldSmooth"):
            w = _w(self._w, QSlider, name)
            if w:
                w.valueChanged.connect(lambda _, n=name: self._touch(n))
        for name in ("chkLSignal", "chkLBaseline", "chkLAnomalies",
                     "chkLDigZones", "chkLGrid", "chkLConfidence", "chkLRawPts"):
            w = _w(self._w, QCheckBox, name)
            if w:
                w.toggled.connect(lambda _, n=name: self._touch(n))
        self._populate_combos()

    def _populate_combos(self):
        """
        gms_main_window.ui already pre-populates cmbCmap / cmbInterp /
        cmbBase with human-friendly display labels (Inferno/Turbo/...,
        Linear/Nearest/Cubic/Gaussian/RBF, Wavelet/Polynomial/Rolling
        Mean/None).  This method therefore does NOT add registry-key
        items on top of them (that produced duplicate/mismatched entries
        in earlier versions).  It only restores the combo selection that
        matches the current self._interp_key / self._base_key by doing a
        REVERSE lookup through the same display->key maps used at read
        time, so selection and resolution always agree.
        """
        cmb = _w(self._w, QComboBox, "cmbCmap")
        if cmb and cmb.count() == 0:
            # Fallback only if the .ui combo is unexpectedly empty
            cmb.addItems(["Inferno", "Turbo", "Viridis", "Plasma",
                          "Grayscale", "Custom..."])

        def _select_by_key(combo: QComboBox, key: str, table: dict):
            if combo is None:
                return
            combo.blockSignals(True)
            for i in range(combo.count()):
                label = combo.itemText(i).strip().lower()
                if table.get(label) == key:
                    combo.setCurrentIndex(i)
                    break
            combo.blockSignals(False)

        ci = _w(self._w, QComboBox, "cmbInterp")
        if ci and ci.count() == 0:
            ci.addItems(["Linear", "Nearest", "Cubic", "Gaussian", "RBF"])
        _select_by_key(ci, self._interp_key, _INTERP_DISPLAY_TO_KEY)

        cb = _w(self._w, QComboBox, "cmbBase")
        if cb and cb.count() == 0:
            cb.addItems(["Wavelet", "Polynomial", "Rolling Mean", "None"])
        _select_by_key(cb, self._base_key, _BASE_DISPLAY_TO_KEY)

    def _touch(self, name: str):
        tier  = _CTRL_TIER.get(name, RENDER_ONLY)
        order = [RENDER_ONLY, SMOOTH_ONLY, RERUN_BASE, RERUN_INTERP]
        if self._pending is None:
            self._pending = tier
        elif order.index(tier) > order.index(self._pending):
            self._pending = tier
        self._debounce.start()

    def _flush(self):
        tier, self._pending = self._pending, None
        if self._grid is None:
            return
        if tier == RENDER_ONLY:
            self._update_display()
        elif tier == SMOOTH_ONLY:
            self._apply_smooth()
        elif tier in (RERUN_INTERP, RERUN_BASE):
            self._rerun_pipeline()

    # ── Pipeline completed ────────────────────────────────────────────────

    def _on_pipeline_completed(self, result: dict):
        grid = result.get("baselined_grid")
        if grid is None:
            logger.warning("[Heatmap] No baselined_grid in result")
            return

        self._grid       = grid
        self._result     = result
        self._scan_file  = (result.get("scan_files") or [""])[0]
        self._gms_config = result.get("gms_config") or {}

        cfg = result.get("pipeline_cfg")
        if cfg is not None:
            self._interp_key = getattr(cfg, "interpolator", self._interp_key)
            self._base_key   = getattr(cfg, "baseline",     self._base_key)

        logger.info("[Heatmap] Pipeline done — grid=%s  file=%s  interp=%s  base=%s",
                    np.asarray(grid.grid_z).shape,
                    self._scan_file, self._interp_key, self._base_key)

        self._ensure_canvas()
        self._full_render()

    # ── Canvas bootstrap (once) ───────────────────────────────────────────

    def _ensure_canvas(self) -> bool:
        if self._canvas is not None:
            return True

        self._fig = Figure(figsize=(7, 6), facecolor="#0C0D11")
        self._ax  = self._fig.add_subplot(111)

        canvas = FigureCanvasQTAgg(self._fig)
        canvas.setObjectName("_gmsHeatmapCanvas")
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumSize(200, 200)

        # Look up the placeholder ONCE and reuse the same reference
        # everywhere below.  Calling findChild() twice for the same
        # widget name can race with PySide6/shiboken deferred deletion
        # — the C++ object backing the first reference may already be
        # destroyed by the time a second findChild() call (or even a
        # second method call on the SAME reference) runs, raising
        # "RuntimeError: Internal C++ object already deleted".  Every
        # Qt call on `ph` below is additionally guarded with try/except
        # RuntimeError so a stale reference never crashes the pipeline.
        ph = _w(self._w, QLabel, "hmPlaceholder")

        lay = _w(self._w, QVBoxLayout, "hmPlotLay")
        if lay is None:
            pf = _w(self._w, QFrame, "plotFrame")
            if pf:
                lay = pf.layout()
        if lay is None and ph is not None:
            try:
                parent = ph.parentWidget()
                if parent is not None:
                    lay = parent.layout()
            except RuntimeError:
                logger.debug("[Heatmap] hmPlaceholder already deleted "
                            "while resolving fallback layout")
                ph = None
        if lay is None:
            logger.error("[Heatmap] hmPlotLay not found — canvas not embedded")
            QMessageBox.critical(self._w, "GMS — Heatmap Error",
                                 "hmPlotLay layout not found.")
            return False

        if ph is not None:
            try:
                lay.removeWidget(ph)
                ph.setVisible(False)
                ph.setParent(None)
            except RuntimeError:
                # Placeholder C++ object was already destroyed (e.g. by a
                # prior render attempt or a parent widget rebuild) —
                # nothing to remove, safe to continue embedding the canvas.
                logger.debug("[Heatmap] hmPlaceholder already deleted "
                            "before removal — continuing")

        lay.insertWidget(0, canvas)
        canvas.show()
        lay.activate()
        pw = lay.parentWidget()
        if pw:
            pw.updateGeometry()
            pw.update()

        self._canvas   = canvas
        self._plot_lay = lay
        logger.info("[Heatmap] Canvas embedded — size %dx%d visible=%s",
                    canvas.width(), canvas.height(), canvas.isVisible())

        # ── Colorbar created ONCE here with a placeholder ─────────────
        # Subsequent renders call cbar.update_normal(im) in place.
        # This prevents tight_layout from accumulating geometry drift
        # (proven: repeated remove+recreate shrinks axes by ~20% per 5 runs).
        _gz_init = np.zeros((2, 2))
        _im_init = self._ax.imshow(_gz_init, cmap="plasma", origin="lower",
                                    aspect="equal", extent=[0,1,0,1])
        self._cbar = self._fig.colorbar(_im_init, ax=self._ax,
                                         fraction=0.046, pad=0.04)
        self._cbar.ax.yaxis.set_tick_params(color="#888")
        # First and ONLY tight_layout call — locks axes geometry
        self._ax.set_aspect("equal", adjustable="box")
        self._fig.tight_layout(pad=0.4)
        self._ax_pos = self._ax.get_position()   # save the good Bbox
        self._fig.set_layout_engine("none")       # freeze: no auto-relayout
        self._ax.cla()                            # clear placeholder
        self._ax.set_facecolor("#0C0D11")

        try:
            from .viewport_nav import attach_2d_nav
            self._nav_2d = attach_2d_nav(self._canvas, self._ax)
        except Exception as e:
            logger.debug("[Heatmap] Nav attach skipped: %s", e)

        self._wire_hover()
        self._wire_scroll()
        return True

    # ── Full render (pipeline completed or rerun) ──────────────────────────

    def _full_render(self):
        try:
            self._full_render_impl()
            self._draw_possibility_markers(self._ax)
        except Exception as exc:
            logger.exception("[Heatmap] Full render failed")
            try:
                QMessageBox.critical(self._w, "GMS — Heatmap Render Error",
                                     f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    def _full_render_impl(self):
        gz, gx, gy, x0, x1, y0, y1, gz_d = self._prepare_grid()
        cmap = self._current_cmap()
        vmin, vmax = self._current_clim(gz_d)

        if self._hover_ann is not None:
            try: self._hover_ann.remove()
            except Exception: pass
            self._hover_ann = None

        # Restore the locked axes position BEFORE cla() so the geometry
        # the user sees never changes regardless of how many reruns occur.
        if self._ax_pos is not None:
            self._ax.set_position(self._ax_pos)
        self._ax.cla()
        self._ax.set_facecolor("#0C0D11")
        if self._ax_pos is not None:
            self._ax.set_position(self._ax_pos)   # cla() may reset position

        self._im = self._ax.imshow(
            gz_d, cmap=cmap, origin="lower", aspect="equal",
            extent=[x0, x1, y0, y1], interpolation="bilinear",
            vmin=vmin, vmax=vmax, zorder=1)

        # Update colorbar IN PLACE — no remove/recreate, no new axes,
        # no tight_layout.  Zero geometry drift across unlimited reruns.
        self._cbar.mappable = self._im
        self._cbar.update_normal(self._im)
        self._cbar.ax.yaxis.set_tick_params(color="#888")

        def chk(n) -> bool:
            c = _w(self._w, QCheckBox, n)
            return c.isChecked() if c else True

        if chk("chkLGrid"):
            self._draw_survey_grid(self._ax, gx, gy)
        if chk("chkLAnomalies"):
            self._draw_anomalies(self._ax, x0, x1, y0, y1)

        self._ax.set_xlabel("X (m)", color="#888", fontsize=8)
        self._ax.set_ylabel("Y (m)", color="#888", fontsize=8)
        self._ax.set_title(f"Signal Grid  ·  {self._result.get('decision','—')}",
                           color="white", fontsize=9, pad=4)
        self._ax.tick_params(colors="#666", labelsize=7)
        for sp in self._ax.spines.values():
            sp.set_edgecolor("#333")
        # set_aspect without tight_layout — layout is locked (anti-shrink fix)
        self._ax.set_aspect("equal", adjustable="box")

        if self._nav_2d is not None:
            self._nav_2d.update_ax(self._ax)
        self._wire_hover()

        self._canvas.draw()
        logger.info("[Heatmap] full_render OK — shape=%s cmap=%s vmin=%.2f vmax=%.2f",
                    gz_d.shape, cmap, vmin, vmax)

    # ── RENDER_ONLY ──────────────────────────────────────────────────────

    def _update_display(self):
        """Change colormap / vmin / vmax without rebuilding axes (single blit)."""
        if self._im is None or self._grid is None:
            return
        try:
            _, _, _, _, _, _, _, gz_d = self._prepare_grid()
            cmap       = self._current_cmap()
            vmin, vmax = self._current_clim(gz_d)

            self._im.set_data(gz_d)
            self._im.set_cmap(cmap)
            self._im.set_clim(vmin, vmax)
            self._cbar.update_normal(self._im)
            self._canvas.draw_idle()
            logger.debug("[Heatmap] display updated — cmap=%s vmin=%.2f vmax=%.2f",
                        cmap, vmin, vmax)
        except Exception:
            logger.exception("[Heatmap] _update_display failed")

    # ── SMOOTH_ONLY ──────────────────────────────────────────────────────

    def _apply_smooth(self):
        """Apply gaussian smoothing to a copy of grid_z; original grid untouched."""
        if self._im is None or self._grid is None:
            return
        try:
            from scipy.ndimage import gaussian_filter
            ss    = _w(self._w, QSlider, "sldSmooth")
            sigma = (ss.value() / 100.0 * 3.0) if ss else 0.0

            gz = np.nan_to_num(np.asarray(self._grid.grid_z, dtype=float), nan=0.0)
            if sigma > 0.05:
                gz = gaussian_filter(gz, sigma=sigma)

            sb     = _w(self._w, QSlider, "sldBright")
            sc     = _w(self._w, QSlider, "sldCont")
            bright = ((sb.value() / 100.0) - 0.5) if sb else 0.0
            contr  = (sc.value() / 100.0)          if sc else 0.5
            gz_d   = (gz + bright * float(np.nanstd(gz))) * (0.5 + contr)

            vmin, vmax = self._current_clim(gz_d)
            self._im.set_data(gz_d)
            self._im.set_clim(vmin, vmax)
            self._cbar.update_normal(self._im)
            self._canvas.draw_idle()
            logger.debug("[Heatmap] smoothing applied — sigma=%.2f", sigma)
        except Exception:
            logger.exception("[Heatmap] _apply_smooth failed")

    # ── RERUN_INTERP / RERUN_BASE ────────────────────────────────────────

    def _rerun_pipeline(self):
        if not self._scan_file:
            logger.warning("[Heatmap] No scan file stored — cannot rerun")
            return

        ci = _w(self._w, QComboBox, "cmbInterp")
        cb = _w(self._w, QComboBox, "cmbBase")
        # .ui pre-populates display labels (Linear/Nearest/Cubic/Gaussian/RBF
        # and Wavelet/Polynomial/Rolling Mean/None) that do NOT match
        # INTERPOLATOR_REGISTRY / BASELINE_REGISTRY keys. Always resolve
        # through the display map before handing the key to GMSPipeline —
        # passing the raw display text raises KeyError inside the worker.
        interp_raw = ci.currentText().strip() if ci and ci.currentText() else self._interp_key
        base_raw   = cb.currentText().strip() if cb and cb.currentText() else self._base_key
        interp_key = _resolve_key(interp_raw, _INTERP_DISPLAY_TO_KEY, self._interp_key)
        base_key   = _resolve_key(base_raw,   _BASE_DISPLAY_TO_KEY,   self._base_key)

        logger.info("[Heatmap] Rerun — interp=%s  base=%s  file=%s",
                    interp_key, base_key, self._scan_file)

        worker = _RerunWorker(
            filepath=self._scan_file, interp_key=interp_key,
            base_key=base_key, gms_config=self._gms_config)
        worker.signals.done.connect(self._on_rerun_done)
        worker.signals.failed.connect(self._on_rerun_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_rerun_done(self, new_grid):
        self._grid = new_grid
        ci = _w(self._w, QComboBox, "cmbInterp")
        cb = _w(self._w, QComboBox, "cmbBase")
        # Store the RESOLVED registry key, not the raw display text —
        # downstream code (e.g. _populate_combos selection restore)
        # expects a valid INTERPOLATOR_REGISTRY / BASELINE_REGISTRY key.
        if ci and ci.currentText():
            self._interp_key = _resolve_key(
                ci.currentText().strip(), _INTERP_DISPLAY_TO_KEY, self._interp_key)
        if cb and cb.currentText():
            self._base_key = _resolve_key(
                cb.currentText().strip(), _BASE_DISPLAY_TO_KEY, self._base_key)
        logger.info("[Heatmap] Rerun complete — new grid: %s",
                    np.asarray(new_grid.grid_z).shape)

        # ── Sync 3D map with the new interpolated/baselined grid ────
        # _vol_push_fn is set by bootstrap_integration after both
        # HeatmapController and VolumetricEngine are constructed, so
        # this avoids any circular import or architecture change.
        if self._vol_push_fn is not None:
            try:
                geo = self._state.__dict__.get("_geometry")
                anomalies = getattr(self._state, "anomaly_list", [])
                self._vol_push_fn(new_grid, anomalies, geo)
                logger.info("[Heatmap] 3D map updated with new rerun grid")
            except Exception as e:
                logger.debug("[Heatmap] 3D sync failed (non-critical): %s", e)

        self._full_render()

    def _on_rerun_failed(self, msg: str):
        logger.error("[Heatmap] Rerun failed: %s", msg)
        try:
            QMessageBox.warning(self._w, "GMS — Heatmap Rerun Failed",
                                f"Pipeline rerun failed:\n{msg}")
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────────

    def _prepare_grid(self):
        ss    = _w(self._w, QSlider, "sldSmooth")
        sigma = (ss.value() / 100.0 * 3.0) if ss else 0.0

        gz = np.nan_to_num(np.asarray(self._grid.grid_z, dtype=float), nan=0.0)
        gx = np.asarray(self._grid.grid_x, dtype=float)
        gy = np.asarray(self._grid.grid_y, dtype=float)

        if sigma > 0.05:
            from scipy.ndimage import gaussian_filter
            gz = gaussian_filter(gz, sigma=sigma)

        sb     = _w(self._w, QSlider, "sldBright")
        sc     = _w(self._w, QSlider, "sldCont")
        bright = ((sb.value() / 100.0) - 0.5) if sb else 0.0
        contr  = (sc.value() / 100.0)          if sc else 0.5
        gz_d   = (gz + bright * float(np.nanstd(gz))) * (0.5 + contr)

        x0, x1 = float(gx[0]), float(gx[-1])
        y0, y1 = float(gy[0]), float(gy[-1])
        return gz, gx, gy, x0, x1, y0, y1, gz_d

    def _current_cmap(self) -> str:
        cmb = _w(self._w, QComboBox, "cmbCmap")
        raw = cmb.currentText().strip() if cmb and cmb.currentText() else "Plasma"
        # .ui pre-populates display labels (Inferno/Turbo/Viridis/Plasma/
        # Grayscale/Custom...) that do not match matplotlib colormap names
        # 1:1 (e.g. "Grayscale" vs "gray"). Resolve via the display map first.
        key = _resolve_key(raw, _CMAP_DISPLAY_TO_KEY, "plasma")
        try:
            valid = set(mcm.colormaps)
        except AttributeError:
            valid = {"plasma","viridis","inferno","magma","hot",
                     "coolwarm","jet","seismic","turbo","gray"}
        return key if key in valid else "plasma"

    def _current_clim(self, gz_d: np.ndarray):
        sc    = _w(self._w, QSlider, "sldCont")
        contr = (sc.value() / 100.0) if sc else 0.5
        p_lo  = max(0.0,   (0.5 - contr) * 100)
        p_hi  = min(100.0, (0.5 + contr) * 100)
        vmin  = float(np.percentile(gz_d, p_lo))
        vmax  = float(np.percentile(gz_d, p_hi))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    def _draw_survey_grid(self, ax, gx, gy):
        """
        Draw the survey scan-line grid aligned to the actual scan geometry.

        Direction semantics
        -------------------
        NORTH_SOUTH / SOUTH_NORTH (vertical pattern):
            Operator walks N→S or S→N.  Each SCAN LINE is a horizontal
            strip at a fixed Y position (spaced by line_spacing_m).
            Points within each line run along X (spaced by sample_distance_m).
            Grid: horizontal lines at every Y position (scan lines)
                  vertical  lines at every X position (sample points).

        EAST_WEST / WEST_EAST (horizontal pattern):
            Operator walks E→W or W→E.  Scan lines run vertically
            at fixed X positions.  Points run along Y.
            Grid: vertical  lines at every X position (scan lines)
                  horizontal lines at every Y position (sample points).

        When no ScanGeometryConfig is available the fallback is to draw
        both axes uniformly (original behaviour).
        """
        geo = self._state.__dict__.get("_geometry", None)
        max_lines = 40

        def _subsample(arr):
            if len(arr) <= max_lines:
                return arr
            return arr[np.round(np.linspace(0, len(arr)-1, max_lines)).astype(int)]

        kw_line   = dict(color=_GRID_COLOR, alpha=_GRID_ALPHA,
                         linewidth=_GRID_LW + 0.2, zorder=3)   # scan lines slightly bolder
        kw_sample = dict(color=_GRID_COLOR, alpha=_GRID_ALPHA * 0.65,
                         linewidth=_GRID_LW * 0.7, zorder=3)   # sample spacing lighter

        if geo is None:
            # No geometry configured — draw uniform grid
            for x in _subsample(gx):
                ax.axvline(x=float(x), **kw_line)
            for y in _subsample(gy):
                ax.axhline(y=float(y), **kw_line)
            return

        try:
            from core.geometry import SurveyDirection
            direction = getattr(geo, "direction", SurveyDirection.NORTH_SOUTH)
            is_ns = direction in (SurveyDirection.NORTH_SOUTH,
                                   SurveyDirection.SOUTH_NORTH)
        except Exception:
            is_ns = True

        if is_ns:
            # Scan lines are horizontal (constant Y = line positions)
            # Sample points run along X
            for y in _subsample(gy):       # one line per scan line
                ax.axhline(y=float(y), **kw_line)
            for x in _subsample(gx):       # sample spacing along X
                ax.axvline(x=float(x), **kw_sample)
            # Axis labels reflect geometry
            ax.set_xlabel(f"X — samples  ({geo.sample_distance_m:.3f} m/pt)",
                          color="#888", fontsize=7)
            ax.set_ylabel(f"Y — scan lines  ({geo.line_spacing_m:.3f} m/line)",
                          color="#888", fontsize=7)
        else:
            # Scan lines are vertical (constant X = line positions)
            for x in _subsample(gx):       # one line per scan line
                ax.axvline(x=float(x), **kw_line)
            for y in _subsample(gy):       # sample spacing along Y
                ax.axhline(y=float(y), **kw_sample)
            ax.set_xlabel(f"X — scan lines  ({geo.line_spacing_m:.3f} m/line)",
                          color="#888", fontsize=7)
            ax.set_ylabel(f"Y — samples  ({geo.sample_distance_m:.3f} m/pt)",
                          color="#888", fontsize=7)

    def _draw_anomalies(self, ax, x0, x1, y0, y1):
        wm = (x1 - x0) or 1.0
        for a in self._result.get("confirmed_anomalies", []):
            xm   = float(a.get("x", a.get("centroid_x", (x0+x1)/2)))
            ym   = float(a.get("y", a.get("centroid_y", (y0+y1)/2)))
            conf = float(a.get("confidence", a.get("combined_confidence", 0.4)))
            if not (x0 <= xm <= x1 and y0 <= ym <= y1):
                continue
            col = ("#00FF41" if conf >= 0.70 else
                   "#FFA500" if conf >= 0.45 else "#FF4444")
            ax.plot(xm, ym, "+", color=col, markersize=20,
                    markeredgewidth=2.5, zorder=5)
            ax.add_patch(mpatches.Circle(
                (xm, ym), max(0.02 * wm, 0.01),
                fill=False, edgecolor=col, linewidth=1.3, alpha=0.65, zorder=4))

    # ── Hover tooltip ─────────────────────────────────────────────────────

    def _on_hover(self, event):
        ax = self._ax
        if ax is None or self._grid is None:
            return
        if event.inaxes != ax:
            if self._hover_ann is not None:
                self._hover_ann.set_visible(False)
                self._canvas.draw_idle()
            return
        xd, yd = event.xdata, event.ydata
        if xd is None or yd is None:
            return

        gx  = self._grid.grid_x
        gy  = self._grid.grid_y
        gz  = self._grid.grid_z
        col = int(np.clip(np.searchsorted(gx, xd, "right")-1, 0, len(gx)-1))
        row = int(np.clip(np.searchsorted(gy, yd, "right")-1, 0, len(gy)-1))
        sig = float(gz[row, col]) if gz.ndim == 2 else 0.0
        nf  = float(getattr(self._grid, "noise_floor", 0.0))
        snr = abs(sig)/nf if nf > 1e-9 else float("nan")
        snr_s = f"SNR: {snr:.2f}" if snr == snr else "SNR: —"

        geo = self._state.__dict__.get("_geometry", None)
        if geo:
            dx = getattr(geo, "sample_distance_m", None)
            dy = getattr(geo, "line_spacing_m", None)
            np_ = getattr(geo, "num_lines", None)
            pp  = getattr(geo, "points_per_line", None)
            dh  = (f"Grid {pp}×{np_} ({dx:.3f}m×{dy:.3f}m)"
                   if dx and dy else "Depth: calibration required")
        else:
            dh = "Depth: calibration required"

        txt = (f"x={xd:.3f}m  y={yd:.3f}m\n"
               f"Cell col={col} row={row}\n"
               f"Signal: {sig:+.2f}\n{snr_s}\n{dh}")

        if self._hover_ann is None:
            self._hover_ann = ax.annotate(
                txt, xy=(xd, yd), xytext=(14, 14),
                textcoords="offset points", fontsize=7.5, color="#E8E8E8",
                bbox=dict(boxstyle="round,pad=0.45", fc="#111827",
                          ec="#00FFCC", lw=0.8, alpha=0.92),
                zorder=20, visible=True)
        else:
            self._hover_ann.set_text(txt)
            self._hover_ann.xy = (xd, yd)
            self._hover_ann.set_visible(True)
        self._canvas.draw_idle()

    def _on_scroll(self, event):
        """Scroll-wheel zoom centred on the cursor position."""
        ax = self._ax
        if ax is None or event.inaxes != ax:
            return
        # Scale factor: scroll up → zoom in (factor < 1 shrinks the view range)
        factor = 0.85 if event.button == "up" else (1.0 / 0.85)
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Zoom around cursor: keep the data point under the cursor fixed
        new_xlim = [xdata + (x - xdata) * factor for x in xlim]
        new_ylim = [ydata + (y - ydata) * factor for y in ylim]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self._canvas.draw_idle()

    def _wire_scroll(self):
        if self._scroll_cid is not None:
            try:
                self._fig.canvas.mpl_disconnect(self._scroll_cid)
            except Exception:
                pass
        self._scroll_cid = self._fig.canvas.mpl_connect(
            "scroll_event", self._on_scroll)

    def _wire_hover(self):
        if self._hover_cid is not None:
            try:
                self._fig.canvas.mpl_disconnect(self._hover_cid)
            except Exception:
                pass
        self._hover_cid = self._fig.canvas.mpl_connect(
            "motion_notify_event", self._on_hover)

    # ── Clear ────────────────────────────────────────────────────────────

    def _clear(self):
        self._grid   = None
        self._result = {}
        self._im     = None
        if self._canvas is not None and self._ax is not None:
            self._ax.cla()
            self._ax.set_facecolor("#0C0D11")
            self._ax.text(0.5, 0.5, "Load a scan to begin",
                          transform=self._ax.transAxes,
                          ha="center", va="center", color="#555", fontsize=12)
            self._canvas.draw()
        # in HeatmapController.__init__:
    self._possibility_targets = []
    self._possibility_filter = "all"   # all | hide_low | hide_very_low | only_high

    def set_possibility_overlay(self, targets):
        self._possibility_targets = targets or []
        self._full_render()            # re-draw with markers

    def set_possibility_filter(self, mode: str):
        self._possibility_filter = mode
        self._full_render()

    def _draw_possibility_markers(self, ax):
        thresh = {"all": 0, "hide_very_low": 40, "hide_low": 60, "only_high": 75}
        floor = thresh.get(self._possibility_filter, 0)
        for t in self._possibility_targets:
            if t.possibility_score < floor:
                continue
            ax.scatter([t.x_m], [t.y_m], s=140, marker="o",
                       facecolors="none", edgecolors=t.color, linewidths=2.0, zorder=6)
            ax.annotate(f"{t.possibility_score:.0f}", (t.x_m, t.y_m),
                        color=t.color, fontsize=7, ha="center", va="bottom", zorder=7)
