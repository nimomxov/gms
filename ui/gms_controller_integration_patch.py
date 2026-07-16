"""
GMS — Controller Integration Patch  v3.1
==========================================
Replaces BackendRouter and GMSController._wire_* internals with
the new integration layer — WITHOUT changing any ObjectNames or UI structure.

Drop-in over gms_controller.py:
    from ui.gms_controller_integration_patch import IntegratedGMSController as GMSController

All original functionality is preserved; this patch adds:
  - GMSApplicationState as single source of truth
  - GMSEventBus for all cross-module communication
  - PipelineExecutionController (async, thread-safe)
  - HeatmapController with incremental recompute graph
  - InspectorPanelController (real backend data)
  - CapabilityGateController (feature gating)
  - AdaptiveImportWorkflow (step-by-step CSV import)
  - BenchmarkController (full benchmark workflow)
  - StatusBarController (live telemetry)
  - GMSFaultManager (graceful error handling)
  - PipelineStageProgressController (per-stage progress)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ── Path bootstrap ────────────────────────────────────────────────────────────
# This file lives at  <project>/ui/gms_controller_integration_patch.py
# The project root is one level up.  We insert it so that:
#   "from ui.integration import ..."   → <project>/ui/integration/
#   "from core.pipeline import ..."    → <project>/core/pipeline.py
# This works whether the script is run directly or imported from main.py.
_THIS_FILE   = Path(__file__).resolve()
_UI_DIR      = _THIS_FILE.parent                # .../gms/ui/
_PROJECT_ROOT = _UI_DIR.parent                  # .../gms/

for _p in [str(_PROJECT_ROOT), str(_UI_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

from PySide6.QtCore   import QSettings, QSize, QTimer
from PySide6.QtGui    import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QFrame,
    QTabWidget, QSlider, QCheckBox, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QListWidget, QListWidgetItem,
    QColorDialog, QTextEdit,
)
from PySide6.QtUiTools import QUiLoader

# ── Preserve original helpers ─────────────────────────────────────────────────
# Try both 'ui.gms_controller' (run from project root) and
# 'gms_controller' (run directly from the ui/ directory).
try:
    from ui.gms_controller import (
        _load, _w,
        PanelToggle,
        ScanCompareController,
        CompareScanEntry,
    )
except ModuleNotFoundError:
    from gms_controller import (       # noqa: F401  (run from ui/ directory)
        _load, _w,
        PanelToggle,
        ScanCompareController,
        CompareScanEntry,
    )

# ── Import integration layer ──────────────────────────────────────────────────
try:
    from ui.integration import bootstrap_integration, GMS_EVENTS
    from ui.integration.app_state import GMSApplicationState, PipelineStatus
    from ui.integration.event_bus import GMSEventBus
    from ui.integration.pipeline_exec import PipelineExecutionController
    from ui.integration.capability_gate import AdaptiveImportWorkflow
except ModuleNotFoundError:
    from integration import bootstrap_integration, GMS_EVENTS          # noqa
    from integration.app_state import GMSApplicationState, PipelineStatus  # noqa
    from integration.event_bus import GMSEventBus                      # noqa
    from integration.pipeline_exec import PipelineExecutionController  # noqa
    from integration.capability_gate import AdaptiveImportWorkflow     # noqa

logger = logging.getLogger("gms.controller.integrated")

UI_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Integrated controller
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedGMSController:
    """
    Full replacement for GMSController that activates the integration layer.
    Preserves every ObjectName and existing slot; adds real backend wiring.
    """

    def __init__(self, window: QMainWindow):
        self._w        = window
        self._settings = QSettings("GMS", "GeophysicalModelingSystem")

        # Panel toggles (unchanged)
        sidebar_frame   = _w(window, QFrame, "sidebarFrame")
        inspector_frame = _w(window, QFrame, "inspectorFrame")
        self._sidebar_toggle   = PanelToggle(sidebar_frame, 220)   if sidebar_frame   else None
        self._inspector_toggle = PanelToggle(inspector_frame, 260) if inspector_frame else None

        # Tab widget (unchanged)
        self._tabs = _w(window, QTabWidget, "workspaceTabs")

        # Scan compare controller (unchanged)
        compare_tab = _w(window, QWidget, "tabScansCompare")
        self._compare = ScanCompareController(compare_tab) if compare_tab else None

        # ── Bootstrap integration layer ───────────────────────────────────
        self._controllers = bootstrap_integration(window)

        # Convenience references
        self._state  = GMSApplicationState.instance()
        self._bus    = GMSEventBus.instance()
        self._exec   = PipelineExecutionController.instance()
        self._import = self._controllers["adaptive_import"]

        # ── Wire all original actions/buttons ────────────────────────────
        self._wire_actions()
        self._wire_sidebar_buttons()
        self._wire_heatmap_controls()
        self._wire_calibration()
        self._wire_inspector_buttons()

        # ── Wire pipeline state feedback to progress widgets ─────────────
        self._wire_pipeline_feedback()

        # ── Restore session ───────────────────────────────────────────────
        self._restore_session()

        logger.info("[IntegratedController] Fully wired with integration layer")

    # ── Pipeline feedback (status bar, progress bar, cancel button) ────────

    def _wire_pipeline_feedback(self):
        """Wire exec controller signals to UI progress widgets."""
        exec_c = self._exec

        # Stage started → status bar message
        exec_c.stage_started.connect(
            lambda name: self._w.statusBar().showMessage(f"  ▶ {name}…", 3000)
        )

        # Stage warning → status bar
        exec_c.stage_warning.connect(
            lambda stage, msg: self._w.statusBar().showMessage(
                f"  ⚠ {stage}: {msg}", 5000
            )
        )

        # Stage failed → fault dialog
        exec_c.stage_failed.connect(
            lambda stage, err: self._bus.emit_event(
                GMS_EVENTS.FAULT_RAISED,
                title=f"Stage Failed: {stage}",
                message=err,
                recovery="Try a different preset or check the scan file.",
            )
        )

        # Result ready → update compare controller
        exec_c.result_ready.connect(self._on_pipeline_result)

    def _on_pipeline_result(self, result: dict):
        """Push result to compare controller and status."""
        logger.info(f"[Controller] Pipeline result: {result.get('decision','?')}")

        # If a scan is loaded, offer to add it to compare view
        dataset = self._state.current_dataset
        if dataset and self._compare:
            try:
                self._compare.add_scan_from_dataset(dataset, result)
            except Exception:
                pass

    # ── Action wiring (menu bar) ────────────────────────────────────────────

    def _wire_actions(self):
        def _action(name):
            return self._w.findChild(QAction, name)

        acts = {
            "actionToggleSidebar":   self._on_toggle_sidebar,
            "actionToggleInspector": self._on_toggle_inspector,
            "actionPreferences":     self._on_open_settings,
            "actionSettings":        self._on_open_settings,
            "actionAbout":           self._on_open_about,
            "actionOpenCSV":         self._on_open_csv,
            "actionExit":            self._w.close,
            "actionSaveProject":     self._on_save_project,
            "actionExportReport":    self._on_export_report,
            "actionNewSession":      self._on_new_session,
            "actionBenchmark":       lambda: self._switch_tab("tabBenchmark"),
            "actionCalibration":     lambda: self._switch_tab("tabCalibration"),
            "actionDeviceProfiles":  self._on_open_device_dialog,
        }
        for name, slot in acts.items():
            a = _action(name)
            if a:
                a.triggered.connect(slot)

    # ── Sidebar button wiring (unchanged names) ─────────────────────────────

    def _wire_sidebar_buttons(self):
        tab_map = {
            "btnHeatmap2D":    "tabHeatmap2D",
            "btnExplorer3D":   "tabExplorer3D",
            "btncalibration":  "tabCalibration",
            "btnbenchmark":    "tabBenchmark",
            "btnScanConfig":   "tabScanConfig",
            "btnScansCompare": "tabScansCompare",
            "btnDiagnostics":  "tabDiagnostics",
        }
        for btn_name, tab_name in tab_map.items():
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(lambda _, t=tab_name: self._switch_tab(t))

        simple_map = {
            "btnOpenCSV":            self._on_open_csv,
            "btnBrowseCSV":          self._on_open_csv,
            "btnImportScan":         self._on_open_csv,
            "btnNewSession":         self._on_new_session,
            "btnOpenScan":           self._on_new_session,
            "btnExportReport":       self._on_export_report,
            "btnSaveProject":        self._on_save_project,
            "btnAutoDetect":         self._on_auto_detect,
            "btnClearFile":          self._on_clear_file,
            "btnbtnDeviceConnection":self._on_open_device_dialog,
            "btnDeviceProfiles":     self._on_open_device_dialog,
            "btnRunPipeline":        self._on_run_pipeline,
            "btnPresetSelector":     self._on_preset_selector,
        }
        for btn_name, slot in simple_map.items():
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(slot)

    # ── Heatmap controls ────────────────────────────────────────────────────

    def _wire_heatmap_controls(self):
        """Delegate to HeatmapController (already wired in bootstrap)."""
        # Vertical exaggeration slider
        sld = _w(self._w, QSlider, "sldVertExag")
        if sld:
            sld.valueChanged.connect(self._on_vert_exag_changed)

        # Camera preset buttons
        for btn_name in ("btnResetCam", "btnTopView", "btnSideView", "btnPerspView"):
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(lambda _, n=btn_name: self._on_camera_preset(n))

        cmb = _w(self._w, QComboBox, "cmbRenderMode")
        if cmb:
            cmb.currentIndexChanged.connect(self._on_3d_render_mode)

        # chkLGrid → volumetric engine grid visibility (2D heatmap re-renders via signal)
        chk = _w(self._w, QCheckBox, "chkLGrid")
        if chk:
            chk.toggled.connect(self._on_grid_toggle)

        # F11 fullscreen shortcut
        from PySide6.QtGui import QKeySequence, QShortcut
        fs_shortcut = QShortcut(QKeySequence("F11"), self._w)
        fs_shortcut.activated.connect(self._toggle_fullscreen)

        # Geometry submission — try both names (btSubmit is the real .ui name)
        for geo_btn_name in ("btSubmit", "btnSubmitGeometry"):
            btn_geo = _w(self._w, QPushButton, geo_btn_name)
            if btn_geo:
                btn_geo.clicked.connect(self._on_submit_geometry)
                break

    def _toggle_fullscreen(self):
        from PySide6.QtCore import Qt
        if self._w.isFullScreen():
            self._w.showNormal()
        else:
            self._w.showFullScreen()

    def _on_grid_toggle(self, checked: bool):
        """Propagate grid visibility to 3D volumetric engine and re-render 2D."""
        self._state.set_visualization("show_grid", checked)
        vol = self._controllers.get("volumetric")
        if vol:
            vol.set_layer("grid", checked)
        hm = self._controllers.get("heatmap")
        if hm and hm._grid is not None:
            hm._full_render()

    def _on_submit_geometry(self):
        """Read geometry widgets, reconstruct XY coords, apply to pipeline."""
        try:
            from core.geometry import ScanGeometryConfig, GeometryReconstructor
        except ImportError:
            logger.warning("[Geometry] core.geometry not available — using CSV XY")
            self._on_run_pipeline()
            return

        pts  = _w(self._w, QSpinBox, "spinPtsPerLine")
        lns  = _w(self._w, QSpinBox, "spinNumLines")
        # spinLineSpacing / spinSamplesDistance are the PRIMARY inputs.
        # spinFieldW / spinFieldL are DERIVED (auto-filled) — they cannot
        # be passed to ScanGeometryConfig() because field_width_m and
        # field_length_m are init=False fields computed by __post_init__.
        ls   = _w(self._w, QDoubleSpinBox, "spinLineSpacing")
        sd   = _w(self._w, QDoubleSpinBox, "spinSamplesDistance")
        zigz = _w(self._w, QCheckBox, "chkZigZag")

        pts_val  = pts.value()   if pts  else 10
        lns_val  = lns.value()   if lns  else 5
        ls_val   = ls.value()    if ls   else 1.0
        sd_val   = sd.value()    if sd   else 0.5
        zz_val   = zigz.isChecked() if zigz else True

        if pts_val < 2 or lns_val < 2:
            QMessageBox.warning(
                self._w, "GMS — Geometry",
                "Survey geometry incomplete.\n"
                "Minimum: 2 points per line and 2 scan lines."
            )
            return
        if ls_val <= 0 or sd_val <= 0:
            QMessageBox.warning(
                self._w, "GMS — Geometry",
                "Line spacing and sample distance must both be > 0."
            )
            return

        geo = ScanGeometryConfig(
            points_per_line=pts_val,
            num_lines=lns_val,
            line_spacing_m=ls_val,
            sample_distance_m=sd_val,
            zigzag=zz_val,
        )
        self._state.__dict__["_geometry"] = geo
        logger.info(
            f"[Geometry] Submitted: {pts_val}×{lns_val} pts, "
            f"{geo.field_width_m:.4f}×{geo.field_length_m:.4f}m, "
            f"zigzag={zz_val}"
        )
        self._on_run_pipeline()

    # ── Calibration ─────────────────────────────────────────────────────────

    def _wire_calibration(self):
        wires = {
            "btnApplySensor": self._on_apply_sensor,
            "btnApplySoil":   self._on_apply_soil,
            "btnValidate":    self._on_validate_calibration,
        }
        for btn_name, slot in wires.items():
            btn = _w(self._w, QPushButton, btn_name)
            if btn:
                btn.clicked.connect(slot)

    # ── Inspector buttons (confirm/reject wired in InspectorPanelController) ─

    def _wire_inspector_buttons(self):
        # Already wired by InspectorPanelController in bootstrap
        pass

    # ── Run pipeline (now uses async exec controller) ───────────────────────

    def _on_run_pipeline(self):
        # Prefer absolute path stored in state, then QSettings
        last_file = self._state.__dict__.get("_last_csv_path", "")
        if not last_file:
            last_file = self._settings.value("session/last_csv", "")
        if not last_file or not Path(last_file).exists():
            # Try resolving a bare filename relative to cwd
            bare = self._settings.value("session/last_csv", "")
            if bare and Path(bare).exists():
                last_file = str(Path(bare).resolve())
            else:
                QMessageBox.information(
                    self._w, "GMS",
                    "Please import a CSV scan file first.\n\n"
                    "Use File → Open CSV or the Import button."
                )
                return

        import yaml
        config = {}
        cfg_path = Path(__file__).parent.parent / "config" / "gms_config.yaml"
        if cfg_path.exists():
            try:
                config = yaml.safe_load(cfg_path.read_text())
            except Exception:
                pass

        self._exec.run(
            scan_files=[last_file],
            preset=self._state.active_preset,
            config=config,
            session_id="ui_run",
        )

        btn = _w(self._w, QPushButton, "btnRunPipeline")
        if btn:
            btn.setEnabled(False)
            btn.setText("Running…")

        def _restore(_=None):
            b = _w(self._w, QPushButton, "btnRunPipeline")
            if b:
                b.setEnabled(True)
                b.setText("Run Pipeline")

        self._exec.result_ready.connect(lambda _: _restore())
        self._exec.failed.connect(lambda _: _restore())
        self._exec.cancelled.connect(_restore)

    def _on_preset_selector(self):
        """Show preset selection dialog or combo."""
        from PySide6.QtWidgets import QInputDialog
        try:
            from core.pipeline import PRESETS
            presets = list(PRESETS.keys())
        except ImportError:
            presets = ["stable", "rbf", "sensitive"]

        preset, ok = QInputDialog.getItem(
            self._w, "Select Preset", "Pipeline Preset:",
            presets, presets.index(self._state.active_preset), False
        )
        if ok and preset:
            self._state.set_preset(preset)
            self._bus.emit_event(GMS_EVENTS.PRESET_CHANGED, preset=preset)
            logger.info(f"[Controller] Preset changed to: {preset}")

    # ── CSV / import ────────────────────────────────────────────────────────

    def _on_open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self._w, "Open CSV Scan File",
            self._settings.value("session/last_dir", str(Path.home())),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        abs_path = str(Path(path).resolve())
        self._settings.setValue("session/last_csv", abs_path)
        self._settings.setValue("session/last_dir", str(Path(path).parent))
        # Store absolute path on state so heatmap can find it without QSettings
        self._state.__dict__["_last_csv_path"] = abs_path
        self._switch_tab("tabCSVImport")

        dataset = self._import.run(abs_path)
        self._load_csv_preview(abs_path)

    def _on_auto_detect(self):
        last = self._settings.value("session/last_csv", "")
        if not last:
            QMessageBox.information(self._w, "GMS", "No file loaded.")
            return
        dataset = self._import.run(last)
        if dataset:
            logger.info(f"[Controller] Auto-detect complete: {dataset.grade.name}")

    def _on_clear_file(self):
        self._settings.remove("session/last_csv")
        self._state.clear()
        for name in ("fileNameLabel", "fileSizeLabel"):
            lbl = _w(self._w, QLabel, name)
            if lbl:
                lbl.setText("—")
        from PySide6.QtWidgets import QTableWidget
        tbl = _w(self._w, QTableWidget, "tableDataPreview")
        if tbl:
            tbl.clearContents()
            tbl.setRowCount(0)

    def _load_csv_preview(self, filepath: str):
        """Populate the preview table and file info labels."""
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
        import csv

        path = Path(filepath)
        lbl_name = _w(self._w, QLabel, "fileNameLabel")
        lbl_size = _w(self._w, QLabel, "fileSizeLabel")
        if lbl_name:
            lbl_name.setText(path.name)
        if lbl_size:
            try:
                lbl_size.setText(f"{path.stat().st_size / 1024:.1f} KB")
            except Exception:
                pass

        tbl = _w(self._w, QTableWidget, "tableDataPreview")
        if tbl is None:
            return
        try:
            with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = [r for r in reader if not (r and r[0].startswith("#"))]
            rows = rows[:21]
            if not rows:
                return

            def _is_numeric(s):
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

            lbl_rows = _w(self._w, QLabel, "labelRowsDetected")
            if lbl_rows:
                lbl_rows.setText(f"Rows: ~{len(rows)}")
        except Exception as e:
            logger.error(f"[CSV Preview] {e}")

    # ── Calibration slots ────────────────────────────────────────────────────

    def _on_apply_sensor(self):
        try:
            from core.calibration import SensorCalibration, CalibrationRegistry
            def _spin(name):
                w = self._w.findChild(QDoubleSpinBox, name)
                if w: return w.value()
                w = self._w.findChild(QSpinBox, name)
                return float(w.value()) if w else None

            cal = SensorCalibration(
                device_name="ui_device",
                adc_bits=int(_spin("spinADC") or 12),
                sensor_gain_nT_per_count=_spin("spinGain"),
                adc_offset_counts=_spin("spinOffset"),
                coil_spacing_m=_spin("spinSensSpace"),
            )
            CalibrationRegistry().save(cal)
            self._bus.emit_event(GMS_EVENTS.CALIBRATION_CHANGED, calibration=cal.__dict__)
            logger.info("[Controller] Calibration applied")
        except ImportError:
            logger.warning("[Controller] Calibration module not available")

    def _on_apply_soil(self):
        cmb = _w(self._w, QComboBox, "cmbSoilProf")
        if cmb:
            logger.info(f"[Calibration] Soil profile: {cmb.currentText()}")

    def _on_validate_calibration(self):
        def _spin(name):
            w = self._w.findChild(QDoubleSpinBox, name)
            return w.value() if w else 0.0
        x, y, d = _spin("spinKTX"), _spin("spinKTY"), _spin("spinKTD")
        logger.info(f"[Calibration] Validate at ({x:.2f}, {y:.2f}), depth={d:.2f}")

    # ── 3D viewer slots ─────────────────────────────────────────────────────

    def _on_vert_exag_changed(self, value: int):
        # Fix: only update lblVexVal — do NOT update lblVexVal_2 (duplicate binding)
        lbl = _w(self._w, QLabel, "lblVexVal")
        if lbl:
            lbl.setText(f"{value}×")
        # Propagate to volumetric engine (re-renders surface with new Z scale)
        vol = self._controllers.get("volumetric")
        if vol:
            vol.set_vertical_exag(value)
        logger.debug(f"[3D] Vertical exaggeration: {value}")

    def _on_camera_preset(self, btn_name: str):
        preset_map = {
            "btnResetCam":  "perspective",
            "btnTopView":   "top",
            "btnSideView":  "side",
            "btnPerspView": "perspective",
        }
        logger.debug(f"[3D] Camera preset: {preset_map.get(btn_name, 'perspective')}")

    def _on_3d_render_mode(self, index: int):
        cmb = _w(self._w, QComboBox, "cmbRenderMode")
        if cmb:
            logger.debug(f"[3D] Render mode: {cmb.currentText()}")

    # ── Panel toggles ────────────────────────────────────────────────────────

    def _on_toggle_sidebar(self):
        if self._sidebar_toggle:
            self._sidebar_toggle.toggle()
            self._settings.setValue("ui/sidebar_visible", self._sidebar_toggle.visible)

    def _on_toggle_inspector(self):
        if self._inspector_toggle:
            self._inspector_toggle.toggle()
            self._settings.setValue("ui/inspector_visible", self._inspector_toggle.visible)

    # ── Dialogs ──────────────────────────────────────────────────────────────

    def _on_open_settings(self):
        try:
            dlg = _load("gms_settings_dialog.ui")
            from PySide6.QtWidgets import QListWidget, QStackedWidget
            nav   = dlg.findChild(QListWidget, "settingsNav")
            stack = dlg.findChild(QWidget, "settingsStack")
            if nav and stack:
                nav.currentRowChanged.connect(stack.setCurrentIndex)
                nav.setCurrentRow(0)
            for name, accept in [("btnOKSettings", True), ("btnCancelSettings", False)]:
                btn = dlg.findChild(QPushButton, name)
                if btn:
                    btn.clicked.connect(dlg.accept if accept else dlg.reject)
            btn_apply = dlg.findChild(QPushButton, "btnApplySettings")
            if btn_apply:
                btn_apply.clicked.connect(lambda: self._apply_settings(dlg))
            dlg.exec()
        except Exception as e:
            logger.error(f"[Settings] {e}")

    def _on_open_about(self):
        try:
            dlg = _load("gms_about_dialog.ui")
            btn = dlg.findChild(QPushButton, "btnAboutClose")
            if btn:
                btn.clicked.connect(dlg.accept)
            dlg.exec()
        except Exception as e:
            logger.error(f"[About] {e}")

    def _on_open_device_dialog(self):
        try:
            dlg = _load("gms_device_dialog.ui")
            for name, slot in [
                ("btnBLEScan",      lambda: self._scan_ble(dlg)),
                ("btnRefreshPorts", lambda: self._refresh_serial(dlg)),
                ("btnConnect",      lambda: self._connect_device(dlg)),
                ("btnDisconnect",   lambda: self._disconnect_device(dlg)),
                ("btnCancel",       dlg.reject),
            ]:
                btn = dlg.findChild(QPushButton, name)
                if btn:
                    btn.clicked.connect(slot)
            dlg.exec()
        except Exception as e:
            logger.error(f"[Device] {e}")

    def _apply_settings(self, dlg):
        font_spin = dlg.findChild(QSpinBox, "spinFontSize")
        if font_spin:
            font = QApplication.font()
            font.setPointSize(font_spin.value())
            QApplication.setFont(font)

    def _scan_ble(self, dlg):
        log = dlg.findChild(QTextEdit, "textTelLog")
        if log:
            log.append("Scanning for BLE devices…")

    def _refresh_serial(self, dlg):
        lst = dlg.findChild(QListWidget, "listSerialPorts")
        if lst:
            lst.clear()
            try:
                import serial.tools.list_ports
                for p in serial.tools.list_ports.comports():
                    lst.addItem(f"{p.device} — {p.description}")
            except ImportError:
                lst.addItem("(pyserial not installed)")

    def _connect_device(self, dlg):
        log = dlg.findChild(QTextEdit, "textTelLog")
        if log:
            log.append("Connecting…")

    def _disconnect_device(self, dlg):
        logger.info("[Device] Disconnect requested")

    # ── Misc slots ──────────────────────────────────────────────────────────

    def _on_new_session(self):
        self._switch_tab("tabCSVImport")

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self._w, "Save Project",
            str(Path.home()), "GMS Project (*.gms);;JSON (*.json)"
        )
        if path:
            import json
            data = {
                "version": "3.1",
                "last_csv": self._settings.value("session/last_csv", ""),
                "preset": self._state.active_preset,
            }
            Path(path).write_text(json.dumps(data, indent=2))
            logger.info(f"[Project] Saved: {path}")

    def _on_export_report(self):
        result = self._state.last_result
        if not result:
            QMessageBox.information(self._w, "GMS", "Run the pipeline first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self._w, "Export Report", "gms_report.json",
            "JSON (*.json);;All Files (*)"
        )
        if path:
            import json
            Path(path).write_text(json.dumps(result, indent=2, default=str))
            logger.info(f"[Export] Report saved: {path}")

    def _switch_tab(self, tab_name: str):
        if self._tabs is None:
            return
        for i in range(self._tabs.count()):
            if self._tabs.widget(i).objectName() == tab_name:
                self._tabs.setCurrentIndex(i)
                return

    # ── Session persistence ──────────────────────────────────────────────────

    def _restore_session(self):
        geom = self._settings.value("window/geometry")
        if geom:
            self._w.restoreGeometry(geom)

        if not self._settings.value("ui/sidebar_visible", True, type=bool):
            if self._sidebar_toggle:
                self._sidebar_toggle.toggle()

        if not self._settings.value("ui/inspector_visible", True, type=bool):
            if self._inspector_toggle:
                self._inspector_toggle.toggle()

        # Restore absolute CSV path into state so _on_run_pipeline finds it
        last_csv = self._settings.value("session/last_csv", "")
        if last_csv and Path(last_csv).exists():
            self._state.__dict__["_last_csv_path"] = last_csv

    def save_session(self):
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
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_integrated_app(argv, ui_filename: str = None):
    """
    Full application factory with integration layer active.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSizePolicy

    app = QApplication(argv)
    app.setApplicationName("GMS")
    app.setApplicationVersion("3.4")
    app.setOrganizationName("GMS")

    theme_path = UI_DIR / "gms_theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))

    if ui_filename:
        window_file = ui_filename
    elif (UI_DIR / "gms_main_integrated.ui").exists():
        window_file = "gms_main_integrated.ui"
    else:
        window_file = "gms_main_window.ui"

    window: QMainWindow = _load(window_file)

    # ── Fix 1: Remove fixed-size restrictions ─────────────────────────────
    window.setMinimumSize(QSize(900, 600))
    window.setMaximumSize(QSize(16777215, 16777215))   # Qt QWIDGETSIZE_MAX
    window.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
    window.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
    window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Apply expanding policies to key panels
    _expand = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    for widget_name in ("centralwidget", "hmPlaceholder", "vp3dPH",
                        "vp3dInner", "inspectorFrame", "sidebarFrame"):
        w = window.findChild(QWidget, widget_name)
        if w:
            w.setMaximumSize(QSize(16777215, 16777215))
            w.setSizePolicy(_expand)

    ctrl = IntegratedGMSController(window)

    # ── Fix 2: Ensure inspector frame is visible ──────────────────────────
    from PySide6.QtWidgets import QFrame
    for frame_name in ("inspectorFrame", "inspectorPanel", "rightPanel"):
        f = window.findChild(QFrame, frame_name)
        if f:
            f.setVisible(True)
            f.setMaximumWidth(16777215)

    # ── Fix 3: Wire 3D viewport to pipeline result ────────────────────────
    vol = ctrl._controllers.get("volumetric")
    if vol:
        ctrl._state.pipeline_completed.connect(
            lambda result: _update_3d_from_result(vol, result, ctrl._state)
        )
        # Wire chkL3D layer checkboxes to volumetric engine
        for cb_name, layer_name in (
            ("chkLSignal",    "signal"),
            ("chkLBaseline",  "baseline"),
            ("chkLAnomalies", "anomalies"),
            ("chkLDigZones",  "dig_markers"),
            ("chkLGrid",      "grid"),
            ("chkLConfidence","confidence"),
            ("chkLRawPts",    "raw_points"),
        ):
            cb = window.findChild(QCheckBox, cb_name)
            if cb:
                cb.toggled.connect(
                    lambda checked, ln=layer_name: vol.set_layer(ln, visible=checked)
                )

    def close_with_save(event):
        ctrl.save_session()
        event.accept()

    window.closeEvent = close_with_save
    return app, window, ctrl


def _update_3d_from_result(vol_engine, result: dict, state):
    """Push pipeline result into volumetric engine for 3D rendering."""
    try:
        grid = result.get("baselined_grid")
        anomalies = state.anomaly_list
        if grid is not None:
            vol_engine.set_scan(grid, anomalies)
    except Exception as e:
        import logging
        logging.getLogger("gms.controller.integrated").debug(
            f"[3D update] {e}"
        )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app, window, ctrl = create_integrated_app(sys.argv)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
