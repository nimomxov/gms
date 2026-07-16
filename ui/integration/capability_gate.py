"""
GMS — Capability Gate Controller  v1.0
========================================
Reacts to DeviceCapabilities changes and shows/hides/enables/disables
UI elements according to what the loaded device actually supports.

Rules (from ARCHITECTURE.md):
  no SNR      → hide uncertainty widgets, confidence labels
  no XY       → disable heatmap tab, show line mode warning
  no heading  → disable geometry reconstruction
  no baseline → disable baseline comparison

Also drives the adaptive CSV import workflow:
  1. detect schema  2. show field confidence  3. allow manual correction
  4. show enabled/disabled features with reasons
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QWidget, QComboBox, QTabWidget,
    QCheckBox, QPushButton, QFrame,
)

from .app_state import GMSApplicationState
from .event_bus import GMSEventBus, GMS_EVENTS

logger = logging.getLogger("gms.capability_gate")


def _w(parent, cls, name):
    return parent.findChild(cls, name)


class CapabilityGateController(QObject):
    """
    Wires capability signals to UI visibility/enable rules.
    All gating is additive — widgets shown only when capability exists.
    """

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()

        self._state.capabilities_changed.connect(self._on_capabilities_changed)
        self._state.dataset_cleared.connect(self._on_dataset_cleared)

        # Start gated (no device loaded)
        self._gate_all(enabled=False)
        logger.info("[CapabilityGate] Controller attached")

    # ── Capability event ──────────────────────────────────────────────────

    def _on_capabilities_changed(self, capabilities):
        cap = capabilities

        has_pos     = getattr(cap, "has_position", False)
        has_snr     = getattr(cap, "has_snr", False)
        has_heading = getattr(cap, "has_heading", False)

        # ── XY / heatmap ────────────────────────────────────────────────────
        # Do NOT disable heatmap tab — geometry can provide XY even without CSV XY
        warn = _w(self._w, QLabel, "warnNoXY")
        if warn:
            warn.setVisible(not has_pos)
            warn.setText("⚠  No CSV position data — submit geometry to enable spatial view")

        # ── SNR / confidence — ONLY confidence overlays are gated ───────────
        # Never disable colormap, sliders, or layer checkboxes
        warn = _w(self._w, QLabel, "warnNoSNR")
        if warn:
            warn.setVisible(not has_snr)
            warn.setText("⚠  No SNR column — confidence percentage unavailable")

        # Confidence overlay only
        for name in ("chkLConfidence", "inspConfidence"):
            w = _w(self._w, QWidget, name)
            if w:
                w.setEnabled(has_snr)
                if not has_snr:
                    w.setToolTip("Requires SNR field in telemetry")

        # ── Heading — only path reconstruction ──────────────────────────────
        warn = _w(self._w, QLabel, "warnNoHeading")
        if warn:
            warn.setVisible(not has_heading)
            warn.setText("⚠  No heading data — path reconstruction unavailable")

        # ── ALL visualization controls stay enabled after dataset load ───────
        always_enabled = (
            "chkLSignal", "chkLBaseline", "chkLAnomalies",
            "chkLDigZones", "chkLGrid", "chkLRawPts",
            "cmbCmap", "cmbInterp", "cmbBase",
            "sldBright", "sldCont", "sldSmooth",
        )
        for name in always_enabled:
            w = self._w.findChild(QWidget, name)
            if w:
                w.setEnabled(True)
                w.setToolTip("")

        # ── Grade badge ──────────────────────────────────────────────────────
        grade = getattr(getattr(cap, "grade", None), "name", "UNKNOWN")
        grade_colors = {
            "BASIC":        "#E74C3C",
            "STANDARD":     "#F39C12",
            "ADVANCED":     "#3498DB",
            "PROFESSIONAL": "#27AE60",
        }
        color = grade_colors.get(grade, "#888888")
        for badge_name in ("badgeLabel_2", "badgeLabel_3",
                           "badgeLabel_4", "badgeLabel_5", "badgeLabel_6"):
            badge = _w(self._w, QLabel, badge_name)
            if badge and badge.text().upper() in ("GRADE", grade, ""):
                badge.setText(grade)
                badge.setStyleSheet(
                    f"background:{color};color:white;"
                    f"border-radius:3px;padding:2px 6px;font-weight:bold;"
                )

        # ── Enable run button ────────────────────────────────────────────────
        btn = _w(self._w, QPushButton, "btnRunPipeline")
        if btn:
            btn.setEnabled(True)
            btn.setToolTip(f"Run analysis — grade: {grade}")

        self._show_field_mapping(cap)
        self._show_disabled_stages(cap)

        logger.info(
            f"[CapabilityGate] Applied: grade={grade} "
            f"pos={has_pos} snr={has_snr} heading={has_heading}"
        )

    def _on_dataset_cleared(self):
        self._gate_all(enabled=False)

    # ── Field mapping display ─────────────────────────────────────────────

    def _show_field_mapping(self, cap):
        """Populate the CSV import field mapping combo boxes."""
        column_map = getattr(cap, "column_map", {})
        if not column_map:
            return

        role_widget_map = {
            "signal":  "cmbCol0_2",
            "x":       "cmbCol1_2",
            "y":       "cmbCol3_2",
            "snr":     "cmbCol4_2",
            "heading": "cmbCol4_3",
        }
        for role, widget_name in role_widget_map.items():
            cmb = _w(self._w, QComboBox, widget_name)
            mapped_col = column_map.get(role, "")
            if cmb and mapped_col:
                # Add detected mapping as first item if not present
                if cmb.findText(mapped_col) == -1:
                    cmb.insertItem(0, f"✓ {mapped_col} (auto)")
                cmb.setCurrentIndex(0)
                cmb.setToolTip(f"Auto-detected: {mapped_col}")

        # Update row/column count labels
        lbl = _w(self._w, QLabel, "labelRowsDetected")
        dataset = self._state.current_dataset
        if lbl and dataset:
            lbl.setText(f"Rows: {dataset.n_samples}")

    def _show_disabled_stages(self, cap):
        """Show disabled pipeline stages in the diagnostics area."""
        pipeline = getattr(self._state.current_dataset, "pipeline", None)
        if not pipeline:
            return

        disabled = getattr(pipeline, "disabled_stages", {})

        # Update warning badges
        features = getattr(cap, "summary", lambda: {})().get("features_enabled", {})
        for feature_name, enabled in features.items():
            if not enabled and feature_name in disabled:
                reason = disabled[feature_name]
                logger.info(f"[CapabilityGate] Disabled: {feature_name} — {reason}")

    # ── Generic enable/disable ────────────────────────────────────────────

    def _gate_all(self, enabled: bool):
        """
        Enable or disable ONLY pipeline-triggering controls (not viz controls).
        Visualization controls stay enabled as long as any dataset is loaded.
        """
        # Only gate the run button — never visualization controls
        pipeline_gated = ("btnRunPipeline",)
        for name in pipeline_gated:
            w = self._w.findChild(QWidget, name)
            if w:
                w.setEnabled(enabled)

    def _set_tab_enabled(self, tab_name: str, enabled: bool):
        tabs = _w(self._w, QTabWidget, "workspaceTabs")
        if tabs is None:
            return
        for i in range(tabs.count()):
            if tabs.widget(i).objectName() == tab_name:
                tabs.setTabEnabled(i, enabled)
                if not enabled:
                    tabs.setTabToolTip(
                        i, "Requires position (XY) data in telemetry"
                    )
                return


class AdaptiveImportWorkflow(QObject):
    """
    Orchestrates the step-by-step adaptive CSV import UX.

    Steps:
      1. Load file → CSVInspector
      2. Match device profile
      3. SemanticFieldMapper → show confidence per role
      4. Allow manual override via combo boxes
      5. Show enabled / disabled features
      6. Build pipeline
      7. Launch analysis
    """

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()
        self._bus   = GMSEventBus.instance()

    def run(self, filepath: str):
        """Execute full adaptive import workflow on the given CSV."""
        try:
            from core.adaptive_ingestion import AdaptiveIngestionEngine
            engine = AdaptiveIngestionEngine()

            # Step 1–5: Detect + map + build pipeline
            dataset = engine.load(filepath)

            # Step 6: Push to state (triggers capability gate)
            self._state.set_dataset(dataset)

            # Step 7: Emit loaded event
            self._bus.emit_event(GMS_EVENTS.SCAN_LOADED, dataset=dataset)

            # Show import summary
            self._show_import_summary(dataset)

            logger.info(
                f"[AdaptiveImport] Loaded {dataset.scan_id}: "
                f"grade={dataset.grade.name} samples={dataset.n_samples}"
            )
            return dataset

        except ValueError as e:
            self._bus.emit_event(
                GMS_EVENTS.FAULT_RAISED,
                title="CSV Import Failed",
                message=str(e),
                recovery="Check that the file has a valid signal column.",
            )
            return None

        except Exception as e:
            self._bus.emit_event(
                GMS_EVENTS.FAULT_RAISED,
                title="Unexpected Import Error",
                message=str(e),
                recovery="Check the file format and try again.",
            )
            return None

    def _show_import_summary(self, dataset):
        """Populate the CSV import tab with the detected schema summary."""
        cap = dataset.capabilities
        pipeline = dataset.pipeline

        # Capability warnings
        for has_it, warn_name, msg in (
            (cap.has_snr,      "warnNoSNR",     "No SNR column — confidence unavailable"),
            (cap.has_position, "warnNoXY",      "No XY position — heatmap unavailable"),
            (cap.has_heading,  "warnNoHeading", "No heading — path reconstruction unavailable"),
        ):
            lbl = _w(self._w, QLabel, warn_name)
            if lbl:
                lbl.setVisible(not has_it)
                lbl.setText(f"⚠  {msg}")

        # Pipeline stage counts
        enabled_count  = len(getattr(pipeline, "enabled_stages", []))
        disabled_count = len(getattr(pipeline, "disabled_stages", {}))

        lbl = _w(self._w, QLabel, "labelRowsDetected")
        if lbl:
            lbl.setText(
                f"Rows: {dataset.n_samples}  |  "
                f"Grade: {dataset.grade.name}  |  "
                f"Stages: {enabled_count} active, {disabled_count} disabled"
            )
