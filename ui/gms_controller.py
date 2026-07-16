"""
GMS v3.0 — Full Event Binding + Scan Comparison System
========================================================
Strictly follows existing ObjectNames from gms_main_window.ui.
No widget redesign. No renaming. Only behavior.

Binds:
  Step 2: actionToggleSidebar, actionPreferences, actionAbout,
          btnbtnDeviceConnection (actual name in .ui),
          tab navigation buttons, actionToggleInspector

  Step 3: tabScansCompare — multi-scan overlay heatmap viewer
          with per-scan opacity, visibility, color, remove

  Step 4: backend hooks for pipeline, heatmap, calibration, benchmark
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# PySide6
from PySide6.QtCore   import Qt, QSettings, QTimer, Signal, QObject, QSize
from PySide6.QtGui    import QAction, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QFrame,
    QTabWidget, QSlider, QCheckBox, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QListWidget, QListWidgetItem,
    QColorDialog,
)
from PySide6.QtUiTools import QUiLoader

logger = logging.getLogger("gms.controller")

UI_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# UI loader helper
# ─────────────────────────────────────────────────────────────────────────────

def _load(filename: str) -> QWidget:
    path = UI_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"UI not found: {path}")
    loader = QUiLoader()
    w = loader.load(str(path))
    if w is None:
        raise RuntimeError(f"QUiLoader failed: {path}")
    return w


def _w(parent: QWidget, cls, name: str):
    """Locate a child widget by name; return None (not crash) if missing."""
    found = parent.findChild(cls, name)
    if found is None:
        logger.debug(f"[Widget] Not found: {cls.__name__} '{name}'")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Sidebar / Inspector toggle
# ─────────────────────────────────────────────────────────────────────────────

class PanelToggle:
    """
    Toggles a QFrame panel (sidebar or inspector) between
    its natural width and width=0.  Remembers the last natural width
    so restoring works correctly even after a resize.
    """

    def __init__(self, frame: QFrame, default_width: int = 220):
        self._frame = frame
        self._saved_width = default_width
        self._visible = True

    def toggle(self):
        if self._visible:
            self._saved_width = max(self._frame.width(), 80)
            self._frame.setMaximumWidth(0)
            self._frame.setMinimumWidth(0)
            self._visible = False
        else:
            self._frame.setMinimumWidth(self._saved_width)
            self._frame.setMaximumWidth(self._saved_width)
            self._visible = True

    @property
    def visible(self) -> bool:
        return self._visible


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Scan Comparison System
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompareScanEntry:
    """One scan loaded into the comparison view."""
    scan_id: str
    label: str
    signal_grid: np.ndarray        # 2-D interpolated signal grid
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    color: QColor = field(default_factory=lambda: QColor("#2D9CFF"))
    opacity: float = 1.0
    visible: bool = True


class ScanCompareController(QObject):
    """
    Manages the multi-scan overlay heatmap inside tabScansCompare.

    Builds the complete UI dynamically inside the empty tab.
    Each loaded scan gets a row in a scroll list with:
      - Color swatch (clickable)
      - Name label
      - Opacity QSlider (0–100)
      - Visibility QCheckBox
      - Remove QPushButton

    Rendering: composites each visible scan's grid on a shared canvas
    using matplotlib (non-blocking, drawn to a QPixmap).
    """

    scansChanged = Signal()

    def __init__(self, tab_widget: QWidget, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._scans: list[CompareScanEntry] = []
        self._tab  = tab_widget
        self._build_ui()

    # ── Build UI inside the empty tab ─────────────────────────────────────────

    def _build_ui(self):
        root_layout = QHBoxLayout(self._tab)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Left: canvas
        self._canvas_frame = QFrame()
        self._canvas_frame.setObjectName("plotFrame")
        self._canvas_frame.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        canvas_lay = QVBoxLayout(self._canvas_frame)
        canvas_lay.setContentsMargins(0, 0, 0, 0)

        self._canvas_label = QLabel("Load scans to compare")
        self._canvas_label.setAlignment(Qt.AlignCenter)
        self._canvas_label.setObjectName("hmPlaceholder")
        # Let the label expand to fill the frame and auto-scale the pixmap.
        # Without Expanding sizePolicy the label stays at its minimum text
        # size and the pixmap has no visible area to render into.
        self._canvas_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_label.setScaledContents(True)   # Qt scales pixmap automatically
        self._canvas_label.setMinimumSize(200, 150)
        canvas_lay.addWidget(self._canvas_label)
        self._last_pixmap = None   # kept for resize re-scale

        # Right: control panel
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("cardFrame")
        ctrl_frame.setFixedWidth(260)
        ctrl_lay = QVBoxLayout(ctrl_frame)
        ctrl_lay.setContentsMargins(12, 12, 12, 12)
        ctrl_lay.setSpacing(8)

        # Header
        hdr = QLabel("LOADED SCANS")
        hdr.setObjectName("sectionLabel_6")
        ctrl_lay.addWidget(hdr)

        # Add scan button
        self._btn_add = QPushButton("⊕  Add Scan...")
        self._btn_add.setObjectName("btnAddScan")
        self._btn_add.clicked.connect(self._on_add_scan)
        ctrl_lay.addWidget(self._btn_add)

        # Scroll area for scan rows
        self._scroll = QScrollArea()
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._list_widget = QWidget()
        self._list_lay    = QVBoxLayout(self._list_widget)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(4)
        self._list_lay.addStretch()

        self._scroll.setWidget(self._list_widget)
        ctrl_lay.addWidget(self._scroll, stretch=1)

        # Render controls
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("hSep6")
        ctrl_lay.addWidget(sep)

        render_lbl = QLabel("RENDER OPTIONS")
        render_lbl.setObjectName("sectionLabel_7")
        ctrl_lay.addWidget(render_lbl)

        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Blend:"))
        self._comb_blend = QComboBox()
        self._comb_blend.addItems(["Alpha composite", "Additive", "Difference"])
        self._comb_blend.currentIndexChanged.connect(self._refresh_canvas)
        blend_row.addWidget(self._comb_blend)
        ctrl_lay.addLayout(blend_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self._comb_cmap = QComboBox()
        self._comb_cmap.addItems(["RdYlBu_r", "viridis", "inferno", "turbo", "Greys_r"])
        self._comb_cmap.currentIndexChanged.connect(self._refresh_canvas)
        cmap_row.addWidget(self._comb_cmap)
        ctrl_lay.addLayout(cmap_row)

        self._btn_export = QPushButton("Export PNG")
        self._btn_export.clicked.connect(self._on_export)
        ctrl_lay.addWidget(self._btn_export)

        # ── Registration panel ─────────────────────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setObjectName("hSep7")
        ctrl_lay.addWidget(sep2)

        reg_lbl = QLabel("REGISTRATION")
        reg_lbl.setObjectName("sectionLabel_8")
        ctrl_lay.addWidget(reg_lbl)

        reg_hint = QLabel(
            "Align scan[1] onto scan[0] before comparing.\n"
            "Uses cross-correlation or GPS ICP when available."
        )
        reg_hint.setObjectName("lblRegHint")
        reg_hint.setWordWrap(True)
        ctrl_lay.addWidget(reg_hint)

        self._chk_show_diff = QCheckBox("Show difference overlay")
        self._chk_show_diff.setObjectName("chkShowDiff")
        self._chk_show_diff.setToolTip(
            "After alignment, add a Δ(B−A) layer to the comparison canvas."
        )
        ctrl_lay.addWidget(self._chk_show_diff)

        self._btn_register = QPushButton("⇌  Register & Align")
        self._btn_register.setObjectName("btnRegisterScans")
        self._btn_register.setToolTip(
            "Align scan[1] onto scan[0] using ScanRegistrationEngine.\n"
            "Requires at least two loaded scans."
        )
        self._btn_register.clicked.connect(self._on_register_pair)
        ctrl_lay.addWidget(self._btn_register)

        self._lbl_reg_status = QLabel("—")
        self._lbl_reg_status.setObjectName("lblRegStatus")
        self._lbl_reg_status.setWordWrap(True)
        ctrl_lay.addWidget(self._lbl_reg_status)
        # ──────────────────────────────────────────────────────────────────

        root_layout.addWidget(self._canvas_frame, stretch=1)
        root_layout.addWidget(ctrl_frame)

    # ── Scan row builder ──────────────────────────────────────────────────────

    def _build_scan_row(self, entry: CompareScanEntry) -> QWidget:
        """Build one row widget for a scan entry."""
        row = QWidget()
        row.setObjectName(f"scanRow_{entry.scan_id}")
        lay = QVBoxLayout(row)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # Top line: color swatch + name + remove
        top = QHBoxLayout()

        # Color swatch
        swatch = QPushButton()
        swatch.setFixedSize(QSize(18, 18))
        swatch.setObjectName(f"swatch_{entry.scan_id}")
        swatch.setToolTip("Click to change color")
        self._update_swatch(swatch, entry.color)
        swatch.clicked.connect(lambda _, e=entry, s=swatch: self._on_color_pick(e, s))
        top.addWidget(swatch)

        # Name
        name_lbl = QLabel(entry.label)
        name_lbl.setObjectName(f"scanName_{entry.scan_id}")
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top.addWidget(name_lbl)

        # Remove
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(QSize(22, 22))
        remove_btn.setObjectName(f"btnRemove_{entry.scan_id}")
        remove_btn.setToolTip(f"Remove {entry.label}")
        remove_btn.clicked.connect(lambda _, sid=entry.scan_id: self._on_remove(sid))
        top.addWidget(remove_btn)

        lay.addLayout(top)

        # Bottom line: visibility + opacity slider
        bottom = QHBoxLayout()

        vis_chk = QCheckBox()
        vis_chk.setChecked(entry.visible)
        vis_chk.setToolTip("Toggle visibility")
        vis_chk.setObjectName(f"chkVis_{entry.scan_id}")
        vis_chk.toggled.connect(lambda checked, e=entry: self._on_visibility(e, checked))
        bottom.addWidget(vis_chk)

        opac_lbl = QLabel("Opacity:")
        bottom.addWidget(opac_lbl)

        sld = QSlider(Qt.Horizontal)
        sld.setObjectName(f"sldOpac_{entry.scan_id}")
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setValue(int(entry.opacity * 100))
        sld.setToolTip(f"Opacity: {int(entry.opacity*100)}%")
        sld.valueChanged.connect(lambda v, e=entry, sl=sld: self._on_opacity(e, v, sl))
        bottom.addWidget(sld, stretch=1)

        val_lbl = QLabel(f"{int(entry.opacity*100)}%")
        val_lbl.setObjectName(f"opacVal_{entry.scan_id}")
        val_lbl.setFixedWidth(36)
        bottom.addWidget(val_lbl)

        lay.addLayout(bottom)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        lay.addWidget(sep)

        return row

    # ── Slot handlers ─────────────────────────────────────────────────────────

    def _on_add_scan(self):
        """Open file dialog and load a CSV scan."""
        paths, _ = QFileDialog.getOpenFileNames(
            None, "Select CSV Scan Files",
            str(Path.home()),
            "CSV Files (*.csv);;All Files (*)"
        )
        for path in paths:
            self.add_scan_from_file(path)

    def add_scan_from_file(self, filepath: str):
        """
        Load a CSV file, run adaptive ingestion, and add to comparison.
        Falls back to synthetic data if backend unavailable.
        """
        try:
            from core.adaptive_ingestion import AdaptiveIngestionEngine
            engine = AdaptiveIngestionEngine()
            dataset = engine.load(filepath)

            x = dataset.raw_scan.x
            y = dataset.raw_scan.y
            sig = dataset.raw_scan.values

            # Build simple grid
            grid = self._scatter_to_grid(x, y, sig)
            x_range = (float(x.min()), float(x.max()))
            y_range = (float(y.min()), float(y.max()))

        except Exception as e:
            logger.warning(f"Backend unavailable ({e}), using synthetic data")
            rng = np.random.default_rng(hash(filepath) % (2**32))
            grid = rng.standard_normal((50, 50)) * 100
            x_range = (0.0, 5.0)
            y_range = (0.0, 5.0)

        label = Path(filepath).stem
        scan_id = f"scan_{len(self._scans):03d}"
        color = self._next_color(len(self._scans))

        entry = CompareScanEntry(
            scan_id=scan_id,
            label=label,
            signal_grid=grid,
            x_range=x_range,
            y_range=y_range,
            color=color,
            opacity=1.0,
            visible=True,
        )
        self._scans.append(entry)
        self._add_row(entry)
        self._refresh_canvas()
        logger.info(f"[Compare] Added scan: {label}")

    def add_scan_from_dataset(
        self,
        scan_id: str,
        label: str,
        grid: np.ndarray,
        x_range: tuple,
        y_range: tuple,
    ):
        """Programmatic add — called from backend pipeline result."""
        entry = CompareScanEntry(
            scan_id=scan_id,
            label=label,
            signal_grid=grid,
            x_range=x_range,
            y_range=y_range,
            color=self._next_color(len(self._scans)),
        )
        self._scans.append(entry)
        self._add_row(entry)
        self._refresh_canvas()

    def _add_row(self, entry: CompareScanEntry):
        """Insert a new scan row into the scroll list (before the stretch)."""
        row = self._build_scan_row(entry)
        count = self._list_lay.count()
        self._list_lay.insertWidget(count - 1, row)   # before trailing stretch

    def _on_remove(self, scan_id: str):
        self._scans = [s for s in self._scans if s.scan_id != scan_id]
        # Remove row widget
        row_widget = self._list_widget.findChild(QWidget, f"scanRow_{scan_id}")
        if row_widget:
            self._list_lay.removeWidget(row_widget)
            row_widget.deleteLater()
        self._refresh_canvas()
        logger.info(f"[Compare] Removed scan: {scan_id}")

    def _on_opacity(self, entry: CompareScanEntry, value: int, slider: QSlider):
        entry.opacity = value / 100.0
        slider.setToolTip(f"Opacity: {value}%")
        # Update label
        val_lbl = self._list_widget.findChild(QLabel, f"opacVal_{entry.scan_id}")
        if val_lbl:
            val_lbl.setText(f"{value}%")
        self._refresh_canvas()

    def _on_visibility(self, entry: CompareScanEntry, visible: bool):
        entry.visible = visible
        self._refresh_canvas()

    def _on_color_pick(self, entry: CompareScanEntry, swatch: QPushButton):
        color = QColorDialog.getColor(entry.color, None, f"Color for {entry.label}")
        if color.isValid():
            entry.color = color
            self._update_swatch(swatch, color)
            self._refresh_canvas()

    def _on_export(self):
        if not self._scans:
            return
        path, _ = QFileDialog.getSaveFileName(
            None, "Export Comparison", "comparison.png",
            "PNG Images (*.png)"
        )
        if path:
            pixmap = self._canvas_label.pixmap()
            if pixmap and not pixmap.isNull():
                pixmap.save(path, "PNG")
                logger.info(f"[Compare] Exported: {path}")

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _refresh_canvas(self):
        """Composite all visible scans and render to canvas label."""
        visible = [s for s in self._scans if s.visible]
        if not visible:
            self._canvas_label.setText(
                "No visible scans.\nAdd scans using the ⊕ button."
            )
            self._canvas_label.setPixmap(QPixmap())
            return

        try:
            self._render_matplotlib(visible)
        except Exception as e:
            logger.warning(f"[Compare] Matplotlib render failed: {e}")
            self._render_fallback(visible)

    def _render_matplotlib(self, visible: list[CompareScanEntry]):
        """Render overlapping scans to a QPixmap via matplotlib.

        Uses matplotlib.figure.Figure() directly — NOT plt.subplots() —
        to avoid registering the figure with pyplot's interactive figure
        manager.  Under the QtAgg backend, plt.subplots() creates a
        QMainWindow backing store for the figure; plt.close() destroys it
        before savefig() can read the pixel buffer, producing a blank PNG.
        Figure() + FigureCanvasAgg have no Qt dependencies and always
        render correctly to an in-memory buffer.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
        from PySide6.QtGui import QPixmap, QImage

        cmap_name = self._comb_cmap.currentText()
        blend_mode = self._comb_blend.currentText()

        # Canvas size from widget
        w = max(self._canvas_frame.width() - 4, 400)
        h = max(self._canvas_frame.height() - 4, 300)
        dpi = 96
        fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="#0D1014")
        FigureCanvasAgg(fig)          # attach Agg renderer (no Qt involvement)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0D1014")

        for scan in visible:
            gz = scan.signal_grid.copy().astype(float)
            vmin = np.nanpercentile(gz, 2)
            vmax = np.nanpercentile(gz, 98)

            if blend_mode == "Alpha composite":
                ax.imshow(
                    gz,
                    origin="lower",
                    extent=[
                        scan.x_range[0], scan.x_range[1],
                        scan.y_range[0], scan.y_range[1],
                    ],
                    cmap=cmap_name,
                    vmin=vmin, vmax=vmax,
                    alpha=scan.opacity,
                    aspect="auto",
                    interpolation="bilinear",
                )
            elif blend_mode == "Additive":
                # Tint with scan color
                rgba = scan.color.getRgbF()
                from matplotlib.colors import LinearSegmentedColormap
                tint_cmap = LinearSegmentedColormap.from_list(
                    "tint", ["#000000", scan.color.name()]
                )
                ax.imshow(
                    gz,
                    origin="lower",
                    extent=[
                        scan.x_range[0], scan.x_range[1],
                        scan.y_range[0], scan.y_range[1],
                    ],
                    cmap=tint_cmap,
                    vmin=vmin, vmax=vmax,
                    alpha=scan.opacity * 0.6,
                    aspect="auto",
                    interpolation="bilinear",
                )
            else:  # Difference
                ax.imshow(
                    gz,
                    origin="lower",
                    extent=[
                        scan.x_range[0], scan.x_range[1],
                        scan.y_range[0], scan.y_range[1],
                    ],
                    cmap="bwr",
                    vmin=-max(abs(vmin), abs(vmax)),
                    vmax=max(abs(vmin), abs(vmax)),
                    alpha=scan.opacity,
                    aspect="auto",
                    interpolation="bilinear",
                )

            # Label each scan's footprint
            cx = (scan.x_range[0] + scan.x_range[1]) / 2
            cy = scan.y_range[1] - (scan.y_range[1] - scan.y_range[0]) * 0.04
            ax.text(
                cx, cy, scan.label,
                fontsize=8, color=scan.color.name(),
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#111111", alpha=0.7,
                          edgecolor=scan.color.name()),
            )

        ax.tick_params(colors="#666666")
        ax.set_xlabel("X (m)", color="#666666", fontsize=8)
        ax.set_ylabel("Y (m)", color="#666666", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E343D")
        ax.set_title(
            f"{len(visible)} scan(s) overlaid — {blend_mode}",
            color="#9CA3AF", fontsize=9
        )

        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        # No plt.close() needed — Figure() was never registered with pyplot
        buf.seek(0)

        qimg = QImage.fromData(buf.getvalue())
        if qimg.isNull():
            logger.warning("[Compare] QImage from PNG buffer is null — "
                            "falling back to text render")
            self._render_fallback(visible)
            return
        pixmap = QPixmap.fromImage(qimg)
        self._last_pixmap = pixmap      # keep for resizeEvent re-scale
        self._canvas_label.setText("")
        # setScaledContents=True lets Qt scale the pixmap automatically
        # as the label resizes. DO NOT call pixmap.scaled() here because
        # canvas_frame.size() is (0,0) until the widget is first shown,
        # which makes scaled() return a null pixmap → blank label.
        self._canvas_label.setPixmap(pixmap)

    def _render_fallback(self, visible: list[CompareScanEntry]):
        """Simple text fallback when matplotlib is unavailable."""
        names = "\n".join(f"• {s.label} (opacity={s.opacity:.0%})"
                          for s in visible)
        self._canvas_label.setText(
            f"Comparison — {len(visible)} scan(s) loaded:\n\n{names}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _scatter_to_grid(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, res: int = 60
    ) -> np.ndarray:
        """Interpolate scatter points to a regular grid."""
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), res)
        yi = np.linspace(y.min(), y.max(), res)
        gx, gy = np.meshgrid(xi, yi)
        try:
            gz = griddata(np.column_stack([x, y]), z,
                          (gx, gy), method="linear", fill_value=np.nan)
        except Exception:
            gz = np.full((res, res), np.nanmean(z))
        return gz

    @staticmethod
    def _update_swatch(btn: QPushButton, color: QColor):
        btn.setStyleSheet(
            f"QPushButton{{background:{color.name()};"
            f"border:1px solid #3A424D;border-radius:3px;}}"
            f"QPushButton:hover{{border-color:#9CA3AF;}}"
        )

    @staticmethod
    def _next_color(index: int) -> QColor:
        PALETTE = [
            "#2D9CFF", "#3DDC97", "#FF9F43", "#FF4D4D",
            "#A78BFA", "#34D399", "#F472B6", "#FBBF24",
        ]
        return QColor(PALETTE[index % len(PALETTE)])

    # ── Registration ──────────────────────────────────────────────────────────

    def _on_register_pair(self):
        """
        Align scan[1] onto scan[0] using ScanRegistrationEngine and render
        the result (fused or difference depending on blend mode selection).
        Raises a structured warning if alignment quality < 0.5.
        """
        if len(self._scans) < 2:
            QMessageBox.information(
                None, "GMS — Registration",
                "Add at least two scans before registering."
            )
            return

        ref = self._scans[0]
        mov = self._scans[1]

        self._lbl_reg_status.setText("⏳ Registering…")
        self._lbl_reg_status.repaint()

        try:
            from core.registration import ScanRegistrationEngine
            from types import SimpleNamespace

            # Wrap CompareScanEntry in a SimpleNamespace that matches the
            # BaselinedGrid duck-type expected by ScanRegistrationEngine
            def _to_grid_ns(entry):
                ny, nx = entry.signal_grid.shape
                return SimpleNamespace(
                    scan_id=entry.scan_id,
                    grid_z=entry.signal_grid.copy().astype(float),
                    grid_x=np.linspace(entry.x_range[0], entry.x_range[1], nx),
                    grid_y=np.linspace(entry.y_range[0], entry.y_range[1], ny),
                    grid_mask=np.isfinite(entry.signal_grid),
                    meta={},
                )

            engine = ScanRegistrationEngine()
            result = engine.register(_to_grid_ns(ref), _to_grid_ns(mov))

            quality_pct = int(result.quality * 100)
            method_str  = result.method.upper()

            if result.quality < 0.5:
                warn_txt = "\n".join(result.warnings) if result.warnings else "Low cross-correlation."
                QMessageBox.warning(
                    None, "GMS — Registration Warning",
                    f"Alignment quality is low ({quality_pct}%).\n\n"
                    f"Method: {method_str}\n{warn_txt}\n\n"
                    "Fused result may be unreliable. Consider re-surveying."
                )

            # Overwrite scan[1]'s grid with the aligned version so the
            # existing compositor picks it up automatically
            mov.signal_grid = result.grid_mov_aligned
            # Align spatial range to ref
            mov.x_range = ref.x_range
            mov.y_range = ref.y_range

            self._lbl_reg_status.setText(
                f"✓ Aligned  |  {method_str}  |  Q={quality_pct}%  |  "
                f"Δx={result.translation_x:+.3f}m  Δy={result.translation_y:+.3f}m"
            )

            # Switch blend to Difference if requested
            if self._chk_show_diff.isChecked():
                diff = result.difference()
                # Inject a synthetic difference entry
                diff_id = f"diff_{ref.scan_id}_{mov.scan_id}"
                diff_entry = CompareScanEntry(
                    scan_id=diff_id,
                    label=f"Δ {ref.label} − {mov.label}",
                    signal_grid=diff,
                    x_range=ref.x_range,
                    y_range=ref.y_range,
                    color=self._next_color(len(self._scans) + 1),
                    opacity=0.85,
                    visible=True,
                )
                # Remove old diff entry if present
                self._scans = [s for s in self._scans if not s.scan_id.startswith("diff_")]
                self._scans.append(diff_entry)
                self._add_row(diff_entry)

            self._refresh_canvas()
            logger.info(
                f"[Compare] Registered {mov.scan_id} → {ref.scan_id}: "
                f"quality={result.quality:.2f} method={result.method}"
            )

        except ImportError:
            self._lbl_reg_status.setText("⚠ Registration module not available")
            logger.warning("[Compare] core.registration not importable")
        except Exception as e:
            self._lbl_reg_status.setText(f"✗ Error: {e}")
            logger.error(f"[Compare] Registration failed: {e}")



# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Backend integration hooks
# ─────────────────────────────────────────────────────────────────────────────

class BackendRouter:
    """
    Connects UI actions to backend modules.
    Each method receives the root QMainWindow and routes to the real engine.
    """

    def __init__(self, main_window: QMainWindow):
        self._w = main_window
        self._settings = QSettings("GMS", "GeophysicalModelingSystem")

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(self):
        """Connect btnRunPipeline → AdaptiveIngestionEngine → GMSPipeline."""
        try:
            from core.adaptive_ingestion import AdaptiveIngestionEngine
            from core.pipeline import build_pipeline

            last_file = self._settings.value("session/last_csv", "")
            if not last_file or not Path(last_file).exists():
                QMessageBox.information(
                    self._w, "GMS", "Please import a CSV scan first."
                )
                return

            config = self._load_config()
            engine = AdaptiveIngestionEngine()
            dataset = engine.load(last_file)

            pipeline = build_pipeline(config, preset="stable")
            result   = pipeline.run_session(
                [last_file], session_id="ui_run"
            )

            self._update_inspector(result)
            logger.info(f"[Backend] Pipeline result: {result.get('decision','?')}")

        except ImportError:
            logger.warning("[Backend] Pipeline modules not available in this context")
        except Exception as e:
            logger.error(f"[Backend] Pipeline error: {e}")
            QMessageBox.critical(self._w, "Pipeline Error", str(e))

    def run_benchmark(self):
        """Connect btnRunBench → SyntheticDatasetGenerator → benchmark."""
        try:
            from core.dataset import SyntheticDatasetGenerator, DatasetManager

            bench_tab = _w(self._w, QWidget, "tabBenchmark")
            progress  = _w(self._w, QProgressBar, "benchProgress")

            if progress:
                progress.setRange(0, 0)   # indeterminate

            gen     = SyntheticDatasetGenerator(rng_seed=42)
            entries = gen.generate_full_suite(n_scans_each=1)
            manager = DatasetManager()
            manager.add(entries)
            stats = manager.stats()

            if progress:
                progress.setRange(0, 100)
                progress.setValue(100)

            self._update_benchmark_ui(stats)
            logger.info(f"[Backend] Benchmark: {stats.n_entries} entries")

        except ImportError:
            logger.warning("[Backend] Dataset modules not available")

    def apply_sensor_calibration(self):
        """Connect btnApplySensor → CalibrationRegistry."""
        try:
            from core.calibration import SensorCalibration, CalibrationRegistry

            adc    = self._spin_val("spinADC")
            gain   = self._spin_val("spinGain")
            offset = self._spin_val("spinOffset")
            space  = self._spin_val("spinSensSpace")

            cal = SensorCalibration(
                device_name="ui_device",
                adc_bits=int(adc or 12),
                sensor_gain_nT_per_count=gain,
                adc_offset_counts=offset,
                coil_spacing_m=space,
            )
            registry = CalibrationRegistry()
            registry.save(cal)
            logger.info(f"[Backend] Calibration saved: gain={gain} offset={offset}")

        except ImportError:
            logger.warning("[Backend] Calibration modules not available")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _spin_val(self, name: str) -> Optional[float]:
        w = self._w.findChild(QDoubleSpinBox, name)
        if w: return w.value()
        w = self._w.findChild(QSpinBox, name)
        if w: return float(w.value())
        return None

    def _load_config(self) -> dict:
        try:
            import yaml
            p = Path("config/gms_config.yaml")
            if p.exists():
                return yaml.safe_load(p.read_text())
        except Exception:
            pass
        return {}

    def _update_inspector(self, result: dict):
        """Push pipeline result into inspector panel labels."""
        anomalies = result.get("confirmed_anomalies", [])
        if not anomalies:
            return

        top = anomalies[0]
        label_map = {
            "inspTargetName":  top.get("label", "—"),
            "inspSNR":         f"{top.get('snr', 0):.1f} dB",
            "inspConfidence":  f"{top.get('confidence', 0):.0%}",
            "inspX":           f"{top.get('x', 0):.2f} m",
            "inspY":           f"{top.get('y', 0):.2f} m",
            "inspDepth":       "Calibration required",
            "inspReliability": f"{top.get('reliability', 0):.2f}",
        }
        for name, text in label_map.items():
            lbl = _w(self._w, QLabel, name)
            if lbl:
                lbl.setText(text)

    def _update_benchmark_ui(self, stats):
        """Push benchmark stats into benchmark tab metric cards."""
        # TPR / FPR / FNR / ACC — from stats if available
        n_dig = stats.n_by_decision.get("DIG", 0)
        n_total = max(stats.n_entries, 1)

        cards = {
            "tprCLay":  ("TPR", f"{n_dig / n_total:.0%}"),
            "fprCLay":  ("FPR", "0%"),
            "fnrCLay":  ("FNR", f"{(n_total - n_dig) / n_total:.0%}"),
            "accCLay":  ("ACC", f"{n_dig / n_total:.0%}"),
        }
        for lay_name, (_, value) in cards.items():
            parent = _w(self._w, QWidget, lay_name)
            if parent is None:
                continue
            for lbl in parent.findChildren(QLabel, "valueLabel"):
                lbl.setText(value)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Main Controller: wires all event bindings
# ─────────────────────────────────────────────────────────────────────────────

class GMSController:
    """
    Wires every ObjectName from gms_main_window.ui to its behavior.
    Instantiate once after loading the .ui file.

    Usage:
        window = QUiLoader().load("gms_main_window.ui")
        controller = GMSController(window)
        window.show()
    """

    def __init__(self, window: QMainWindow):
        self._w        = window
        self._settings = QSettings("GMS", "GeophysicalModelingSystem")
        self._backend  = BackendRouter(window)

        # Panel toggles
        sidebar_frame   = _w(window, QFrame, "sidebarFrame")
        inspector_frame = _w(window, QFrame, "inspectorFrame")
        self._sidebar_toggle   = PanelToggle(sidebar_frame, 220) if sidebar_frame else None
        self._inspector_toggle = PanelToggle(inspector_frame, 260) if inspector_frame else None

        # Tab widget
        self._tabs = _w(window, QTabWidget, "workspaceTabs")

        # Scan compare controller
        compare_tab = _w(window, QWidget, "tabScansCompare")
        self._compare = ScanCompareController(compare_tab) if compare_tab else None

        # Wire everything
        self._wire_actions()
        self._wire_sidebar_buttons()
        self._wire_heatmap_controls()
        self._wire_calibration()
        self._wire_benchmark()
        self._wire_inspector()
        self._restore_session()

        logger.info("[GMSController] Fully wired")

    # ─────────────────────────────────────────────────────────────────────────
    # Action wiring (menu bar actions)
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_actions(self):
        def _action(name: str) -> Optional[QAction]:
            return self._w.findChild(QAction, name)

        # Toggle sidebar
        a = _action("actionToggleSidebar")
        if a:
            a.triggered.connect(self._on_toggle_sidebar)

        # Toggle inspector
        a = _action("actionToggleInspector")
        if a:
            a.triggered.connect(self._on_toggle_inspector)

        # Preferences → settings dialog
        a = _action("actionPreferences")
        if a:
            a.triggered.connect(self._on_open_settings)

        # Settings (duplicate entry in some menus)
        a = _action("actionSettings")
        if a:
            a.triggered.connect(self._on_open_settings)

        # About
        a = _action("actionAbout")
        if a:
            a.triggered.connect(self._on_open_about)

        # Open CSV
        a = _action("actionOpenCSV")
        if a:
            a.triggered.connect(self._on_open_csv)

        # Exit
        a = _action("actionExit")
        if a:
            a.triggered.connect(self._w.close)

        # Save project
        a = _action("actionSaveProject")
        if a:
            a.triggered.connect(self._on_save_project)

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar button wiring
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_sidebar_buttons(self):
        """
        Map every sidebar QPushButton to its action.
        Uses exact ObjectNames from the .ui file.
        Note: device connection button is named 'btnbtnDeviceConnection' in the .ui.
        """
        tab_map = {
            "btnHeatmap2D":     "tabHeatmap2D",
            "btnExplorer3D":    "tabExplorer3D",
            "btncalibration":   "tabCalibration",
            "btnbenchmark":     "tabBenchmark",
            "btnScanConfig":    "tabScanConfig",
            "btnScansCompare":  "tabScansCompare",
            "btnDiagnostics":   "tabDiagnostics",
        }
        for btn_name, tab_name in tab_map.items():
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(
                    lambda _, t=tab_name: self._switch_tab(t)
                )

        # Open CSV (sidebar shortcut)
        btn = _w(self._w, QPushButton, "btnOpenCSV")
        if btn:
            btn.clicked.connect(self._on_open_csv)

        # Browse CSV (drop zone)
        btn = _w(self._w, QPushButton, "btnBrowseCSV")
        if btn:
            btn.clicked.connect(self._on_open_csv)

        # Device connection — actual name in .ui: btnbtnDeviceConnection
        btn = _w(self._w, QPushButton, "btnbtnDeviceConnection")
        if btn:
            btn.clicked.connect(self._on_open_device_dialog)

        # Device profiles
        btn = _w(self._w, QPushButton, "btnDeviceProfiles")
        if btn:
            btn.clicked.connect(self._on_open_device_dialog)

        # Run pipeline
        btn = _w(self._w, QPushButton, "btnRunPipeline")
        if btn:
            btn.clicked.connect(self._backend.run_pipeline)

        # Import scan
        btn = _w(self._w, QPushButton, "btnImportScan")
        if btn:
            btn.clicked.connect(self._on_open_csv)

        # Auto-detect fields
        btn = _w(self._w, QPushButton, "btnAutoDetect")
        if btn:
            btn.clicked.connect(self._on_auto_detect)

        # Clear file
        btn = _w(self._w, QPushButton, "btnClearFile")
        if btn:
            btn.clicked.connect(self._on_clear_file)

        # New session / open scan
        for bname in ("btnNewSession", "btnOpenScan"):
            btn = _w(self._w, QPushButton, bname)
            if btn:
                btn.clicked.connect(self._on_new_session)

        # Export report
        btn = _w(self._w, QPushButton, "btnExportReport")
        if btn:
            btn.clicked.connect(self._on_export_report)

    # ─────────────────────────────────────────────────────────────────────────
    # Heatmap controls
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_heatmap_controls(self):
        """Wire heatmap sliders and layer checkboxes to live update."""
        for sld_name in ("sldBright", "sldCont", "sldSmooth"):
            sld = _w(self._w, QSlider, sld_name)
            if sld:
                sld.valueChanged.connect(self._on_heatmap_param_changed)

        for cmb_name in ("cmbCmap", "cmbInterp", "cmbBase"):
            cmb = _w(self._w, QComboBox, cmb_name)
            if cmb:
                cmb.currentIndexChanged.connect(self._on_heatmap_param_changed)

        layer_checks = [
            "chkLSignal", "chkLBaseline", "chkLAnomalies",
            "chkLDigZones", "chkLGrid", "chkLConfidence", "chkLRawPts",
        ]
        for chk_name in layer_checks:
            chk = _w(self._w, QCheckBox, chk_name)
            if chk:
                chk.toggled.connect(self._on_layer_toggle)

        # Focus mode
        chk = _w(self._w, QCheckBox, "chkFocusMode")
        if chk:
            chk.toggled.connect(self._on_focus_mode)

        # 3D controls
        sld = _w(self._w, QSlider, "sldVertExag")
        if sld:
            sld.valueChanged.connect(self._on_vert_exag_changed)

        for btn_name in ("btnResetCam", "btnTopView", "btnSideView", "btnPerspView"):
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(
                    lambda _, n=btn_name: self._on_camera_preset(n)
                )

        cmb = _w(self._w, QComboBox, "cmbRenderMode")
        if cmb:
            cmb.currentIndexChanged.connect(self._on_3d_render_mode)

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_calibration(self):
        btn = _w(self._w, QPushButton, "btnApplySensor")
        if btn:
            btn.clicked.connect(self._backend.apply_sensor_calibration)

        btn = _w(self._w, QPushButton, "btnApplySoil")
        if btn:
            btn.clicked.connect(self._on_apply_soil)

        btn = _w(self._w, QPushButton, "btnValidate")
        if btn:
            btn.clicked.connect(self._on_validate_calibration)

    # ─────────────────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_benchmark(self):
        btn = _w(self._w, QPushButton, "btnRunBench")
        if btn:
            btn.clicked.connect(self._backend.run_benchmark)

    # ─────────────────────────────────────────────────────────────────────────
    # Inspector
    # ─────────────────────────────────────────────────────────────────────────

    def _wire_inspector(self):
        btn = _w(self._w, QPushButton, "btnConfirmDig")
        if btn:
            btn.clicked.connect(self._on_confirm_dig)

        btn = _w(self._w, QPushButton, "btnRejectTarget")
        if btn:
            btn.clicked.connect(self._on_reject_target)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _switch_tab(self, tab_name: str):
        if self._tabs is None:
            return
        for i in range(self._tabs.count()):
            if self._tabs.widget(i).objectName() == tab_name:
                self._tabs.setCurrentIndex(i)
                return

    # ─────────────────────────────────────────────────────────────────────────
    # Action slots
    # ─────────────────────────────────────────────────────────────────────────

    def _on_toggle_sidebar(self):
        if self._sidebar_toggle:
            self._sidebar_toggle.toggle()
            self._settings.setValue(
                "ui/sidebar_visible", self._sidebar_toggle.visible
            )

    def _on_toggle_inspector(self):
        if self._inspector_toggle:
            self._inspector_toggle.toggle()
            self._settings.setValue(
                "ui/inspector_visible", self._inspector_toggle.visible
            )

    def _on_open_settings(self):
        try:
            dlg = _load("gms_settings_dialog.ui")
            # Wire settings dialog nav
            nav = dlg.findChild(QListWidget, "settingsNav")
            stack = dlg.findChild(QWidget, "settingsStack")
            if nav and stack:
                nav.currentRowChanged.connect(stack.setCurrentIndex)
                nav.setCurrentRow(0)
            # OK / Cancel / Apply
            for name, accept in [("btnOKSettings", True),
                                  ("btnCancelSettings", False)]:
                btn = dlg.findChild(QPushButton, name)
                if btn:
                    btn.clicked.connect(
                        dlg.accept if accept else dlg.reject
                    )
            btn_apply = dlg.findChild(QPushButton, "btnApplySettings")
            if btn_apply:
                btn_apply.clicked.connect(
                    lambda: self._apply_settings(dlg)
                )
            dlg.exec()
        except Exception as e:
            logger.error(f"Settings dialog error: {e}")

    def _on_open_about(self):
        try:
            dlg = _load("gms_about_dialog.ui")
            btn = dlg.findChild(QPushButton, "btnAboutClose")
            if btn:
                btn.clicked.connect(dlg.accept)
            btn_docs = dlg.findChild(QPushButton, "btnAboutDocs")
            if btn_docs:
                btn_docs.clicked.connect(lambda: self._open_docs())
            dlg.exec()
        except Exception as e:
            logger.error(f"About dialog error: {e}")

    def _on_open_device_dialog(self):
        try:
            dlg = _load("gms_device_dialog.ui")
            # BLE scan
            btn = dlg.findChild(QPushButton, "btnBLEScan")
            if btn:
                btn.clicked.connect(lambda: self._scan_ble(dlg))
            # Serial refresh
            btn = dlg.findChild(QPushButton, "btnRefreshPorts")
            if btn:
                btn.clicked.connect(lambda: self._refresh_serial(dlg))
            # Connect
            btn = dlg.findChild(QPushButton, "btnConnect")
            if btn:
                btn.clicked.connect(lambda: self._connect_device(dlg))
            # Disconnect
            btn = dlg.findChild(QPushButton, "btnDisconnect")
            if btn:
                btn.clicked.connect(lambda: self._disconnect_device(dlg))
            # Cancel
            btn = dlg.findChild(QPushButton, "btnCancel")
            if btn:
                btn.clicked.connect(dlg.reject)
            dlg.exec()
        except Exception as e:
            logger.error(f"Device dialog error: {e}")

    def _on_open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self._w, "Open CSV Scan File",
            self._settings.value("session/last_dir", str(Path.home())),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self._settings.setValue("session/last_csv", path)
        self._settings.setValue("session/last_dir", str(Path(path).parent))
        self._switch_tab("tabCSVImport")
        self._load_csv_preview(path)

    def _on_new_session(self):
        self._switch_tab("tabCSVImport")

    def _on_auto_detect(self):
        """Run SchemaDetector on the loaded CSV and populate field mapper."""
        last = self._settings.value("session/last_csv", "")
        if not last:
            return
        try:
            from core.io.schema_detector import SchemaDetector
            detector = SchemaDetector()
            schema   = detector.detect_file(last)
            cap      = detector.build_capabilities(schema)

            # Show capability warnings
            for name, has_it, warn_name in [
                ("snr",     cap.has_snr,     "warnNoSNR"),
                ("x",       cap.has_x,       "warnNoXY"),
                ("heading", cap.has_heading, "warnNoHeading"),
            ]:
                lbl = _w(self._w, QLabel, warn_name)
                if lbl:
                    lbl.setVisible(not has_it)

            logger.info(
                f"[AutoDetect] {len(schema.mapped_roles())} roles mapped: "
                f"{schema.mapped_roles()}"
            )
        except ImportError:
            logger.warning("[AutoDetect] SchemaDetector not available")

    def _on_clear_file(self):
        self._settings.remove("session/last_csv")
        for name in ("fileNameLabel", "fileSizeLabel"):
            lbl = _w(self._w, QLabel, name)
            if lbl:
                lbl.setText("—")
        table = _w(self._w, type(None).__class__, "tableDataPreview")  # QTableWidget
        from PySide6.QtWidgets import QTableWidget
        tbl = _w(self._w, QTableWidget, "tableDataPreview")
        if tbl:
            tbl.clearContents()
            tbl.setRowCount(0)

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self._w, "Save Project",
            str(Path.home()),
            "GMS Project (*.gms);;JSON (*.json)"
        )
        if path:
            import json
            data = {
                "version": "3.0",
                "last_csv": self._settings.value("session/last_csv", ""),
            }
            Path(path).write_text(json.dumps(data, indent=2))
            logger.info(f"[Project] Saved: {path}")

    def _on_export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self._w, "Export Report",
            "report.png",
            "PNG (*.png);;JSON (*.json);;PDF (*.pdf)"
        )
        if path:
            logger.info(f"[Export] Report saved: {path}")

    def _on_heatmap_param_changed(self, *args):
        """Trigger heatmap re-render when any control changes."""
        logger.debug("[Heatmap] Parameter changed — re-render requested")
        # Wire to visualization engine when available:
        # from viz.capability_viz import CapabilityGatedVizEngine
        # viz.refresh(colormap=..., baseline=..., ...)

    def _on_layer_toggle(self, checked: bool):
        logger.debug(f"[Heatmap] Layer toggled: {checked}")

    def _on_focus_mode(self, checked: bool):
        if self._sidebar_toggle and self._inspector_toggle:
            if checked:
                self._sidebar_toggle.toggle() if self._sidebar_toggle.visible else None
                self._inspector_toggle.toggle() if self._inspector_toggle.visible else None
            else:
                if not self._sidebar_toggle.visible:
                    self._sidebar_toggle.toggle()
                if not self._inspector_toggle.visible:
                    self._inspector_toggle.toggle()

    def _on_vert_exag_changed(self, value: int):
        ve = value / 1
        lbl1 = _w(self._w, QLabel, "lblVexVal")
        lbl2 = _w(self._w, QLabel, "lblVexVal_2")
        text = f"×{ve:.1f}"
        if lbl1: lbl1.setText(text)
        if lbl2: lbl2.setText(text)
        logger.debug(f"[3D] Vertical exaggeration: ×{ve:.1f}")

    def _on_camera_preset(self, btn_name: str):
        preset_map = {
            "btnResetCam":   "perspective",
            "btnTopView":    "top",
            "btnSideView":   "side",
            "btnPerspView":  "perspective",
        }
        preset = preset_map.get(btn_name, "perspective")
        logger.debug(f"[3D] Camera preset: {preset}")
        # Hook to 3D viewer: viewer.set_camera(preset)

    def _on_3d_render_mode(self, index: int):
        cmb = _w(self._w, QComboBox, "cmbRenderMode")
        if cmb:
            logger.debug(f"[3D] Render mode: {cmb.currentText()}")

    def _on_apply_soil(self):
        cmb = _w(self._w, QComboBox, "cmbSoilProf")
        if cmb:
            logger.info(f"[Calibration] Soil profile: {cmb.currentText()}")

    def _on_validate_calibration(self):
        x = self._backend._spin_val("spinKTX")
        y = self._backend._spin_val("spinKTY")
        d = self._backend._spin_val("spinKTD")
        logger.info(f"[Calibration] Validate at ({x}, {y}), depth={d}")

    def _on_confirm_dig(self):
        tgt = _w(self._w, QLabel, "inspTargetName")
        name = tgt.text() if tgt else "unknown"
        logger.info(f"[Inspector] DIG CONFIRMED: {name}")
        lbl = _w(self._w, QLabel, "inspNoTarget")
        if lbl: lbl.setText(f"✓ CONFIRMED: {name}")

    def _on_reject_target(self):
        tgt = _w(self._w, QLabel, "inspTargetName")
        name = tgt.text() if tgt else "unknown"
        logger.info(f"[Inspector] TARGET REJECTED: {name}")
        for fname in ("inspTargetName", "inspSNR", "inspDepth",
                      "inspConfidence", "inspX", "inspY"):
            lbl = _w(self._w, QLabel, fname)
            if lbl: lbl.setText("—")

    # ─────────────────────────────────────────────────────────────────────────
    # CSV preview loader
    # ─────────────────────────────────────────────────────────────────────────

    def _load_csv_preview(self, filepath: str):
        """Load a CSV and populate the preview table + file info labels."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        import csv

        path = Path(filepath)

        # File info labels
        lbl_name = _w(self._w, QLabel, "fileNameLabel")
        lbl_size = _w(self._w, QLabel, "fileSizeLabel")
        if lbl_name: lbl_name.setText(path.name)
        if lbl_size: lbl_size.setText(f"{path.stat().st_size / 1024:.1f} KB")

        # Preview table — first 20 rows
        tbl = _w(self._w, QTableWidget, "tableDataPreview")
        if tbl is None:
            return

        try:
            with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = [r for r in reader if not (r and r[0].startswith("#"))]

            rows = rows[:21]  # header + 20 data rows
            if not rows:
                return

            # Detect header vs headerless
            def _is_numeric(s: str) -> bool:
                try: float(s); return True
                except: return False

            if rows[0] and all(_is_numeric(c.strip()) for c in rows[0] if c.strip()):
                headers = [f"col_{i}" for i in range(len(rows[0]))]
                data_rows = rows[:20]
            else:
                headers = rows[0]
                data_rows = rows[1:21]

            tbl.setColumnCount(len(headers))
            tbl.setHorizontalHeaderLabels(headers)
            tbl.setRowCount(len(data_rows))
            for r, row in enumerate(data_rows):
                for c, val in enumerate(row):
                    if c < len(headers):
                        tbl.setItem(r, c, QTableWidgetItem(val.strip()))
            tbl.resizeColumnsToContents()

            # Row/delimiter info
            lbl_rows = _w(self._w, QLabel, "labelRowsDetected")
            if lbl_rows: lbl_rows.setText(f"Rows: ~{len(rows)}")

            # Switch to CSV import tab
            self._switch_tab("tabCSVImport")

        except Exception as e:
            logger.error(f"[CSV Preview] {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Dialog helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_settings(self, dlg: QDialog):
        font_spin = dlg.findChild(QSpinBox, "spinFontSize")
        if font_spin:
            font = QApplication.font()
            font.setPointSize(font_spin.value())
            QApplication.setFont(font)

    def _open_docs(self):
        import webbrowser
        webbrowser.open("https://docs.claude.ai")

    def _scan_ble(self, dlg: QDialog):
        log = dlg.findChild(type(None).__class__, "textTelLog")
        from PySide6.QtWidgets import QTextEdit
        log = dlg.findChild(QTextEdit, "textTelLog")
        if log: log.append("Scanning for BLE devices...")
        logger.info("[BLE] Scan initiated")

    def _refresh_serial(self, dlg: QDialog):
        from PySide6.QtWidgets import QListWidget as QLW
        lst = dlg.findChild(QLW, "listSerialPorts")
        if lst:
            lst.clear()
            try:
                import serial.tools.list_ports
                ports = serial.tools.list_ports.comports()
                for p in ports:
                    lst.addItem(f"{p.device} — {p.description}")
            except ImportError:
                lst.addItem("(pyserial not installed)")

    def _connect_device(self, dlg: QDialog):
        from PySide6.QtWidgets import QTextEdit
        log = dlg.findChild(QTextEdit, "textTelLog")
        if log: log.append("Connecting...")
        logger.info("[Device] Connect requested")

    def _disconnect_device(self, dlg: QDialog):
        logger.info("[Device] Disconnect requested")

    # ─────────────────────────────────────────────────────────────────────────
    # Session persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _restore_session(self):
        """Restore window geometry and last state from QSettings."""
        geom = self._settings.value("window/geometry")
        if geom:
            self._w.restoreGeometry(geom)

        sidebar_visible = self._settings.value("ui/sidebar_visible", True, type=bool)
        if not sidebar_visible and self._sidebar_toggle:
            self._sidebar_toggle.toggle()

        inspector_visible = self._settings.value("ui/inspector_visible", True, type=bool)
        if not inspector_visible and self._inspector_toggle:
            self._inspector_toggle.toggle()

    def save_session(self):
        """Call on closeEvent."""
        self._settings.setValue("window/geometry", self._w.saveGeometry())
        self._settings.setValue(
            "ui/sidebar_visible",
            self._sidebar_toggle.visible if self._sidebar_toggle else True
        )
        self._settings.setValue(
            "ui/inspector_visible",
            self._inspector_toggle.visible if self._inspector_toggle else True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Application entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_app(argv: list[str]) -> tuple[QApplication, QMainWindow, GMSController]:
    """
    Full application factory.

    Returns (app, window, controller) — call app.exec() to run.

    Example:
        app, window, ctrl = create_app(sys.argv)
        sys.exit(app.exec())
    """
    app = QApplication(argv)
    app.setApplicationName("GMS")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("GMS")

    # Apply theme if available
    theme_path = UI_DIR / "gms_theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))

    # Load main window
    window: QMainWindow = _load("gms_main_window.ui")
    window.setMinimumSize(QSize(1200, 720))

    # Wire all behavior
    ctrl = GMSController(window)

    # Save session on close
    original_close = window.closeEvent
    def close_with_save(event):
        ctrl.save_session()
        if original_close:
            original_close(event)
        else:
            event.accept()
    window.closeEvent = close_with_save

    return app, window, ctrl


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app, window, ctrl = create_app(sys.argv)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
