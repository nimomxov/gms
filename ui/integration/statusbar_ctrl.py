"""
GMS — Status Bar Controller  v1.0
===================================
Connects the status bar to REAL live telemetry.
Updates on a QTimer and via event bus subscriptions.

Displayed fields:
  Pipeline status | FPS | Queue | CPU | RAM |
  Coverage | Cursor XY | Interp mode | Baseline mode |
  Detector mode | Config hash
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtWidgets import QLabel, QStatusBar, QMainWindow, QProgressBar

from .app_state import (
    GMSApplicationState, PipelineStatus, BackendHealth
)

logger = logging.getLogger("gms.statusbar")


class StatusBarController(QObject):
    """
    Drives the main window status bar with live backend telemetry.
    Polls system metrics every second; pipeline state is event-driven.
    """

    _POLL_INTERVAL_MS = 1000

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w      = window
        self._state  = GMSApplicationState.instance()
        self._bar    = window.statusBar()

        # Persistent label segments
        self._lbl_pipeline  = QLabel("  ◦ IDLE  ")
        self._lbl_fps       = QLabel(" FPS: —  ")
        self._lbl_queue     = QLabel(" Queue: 0  ")
        self._lbl_cpu       = QLabel(" CPU: —  ")
        self._lbl_ram       = QLabel(" RAM: —  ")
        self._lbl_cursor    = QLabel(" XY: —  ")
        self._lbl_interp    = QLabel(" Interp: cubic  ")
        self._lbl_baseline  = QLabel(" Base: line_median  ")
        self._lbl_config    = QLabel(" Hash: ········  ")
        self._lbl_coverage  = QLabel(" Cov: —  ")

        # Inline progress bar (hidden when idle)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(120)
        self._progress.setMaximumHeight(14)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)

        # Build status bar
        for widget in (
            self._lbl_pipeline,
            self._progress,
            self._lbl_fps,
            self._lbl_queue,
            self._lbl_cpu,
            self._lbl_ram,
            self._lbl_coverage,
            self._lbl_cursor,
            self._lbl_interp,
            self._lbl_baseline,
            self._lbl_config,
        ):
            self._bar.addWidget(widget)
            if isinstance(widget, QLabel):
                widget.setAlignment(Qt.AlignVCenter)

        # Connect state signals
        self._state.pipeline_status_changed.connect(self._on_pipeline_status)
        self._state.pipeline_progress.connect(self._on_pipeline_progress)
        self._state.pipeline_stage_changed.connect(self._on_stage_changed)
        self._state.backend_health_updated.connect(self._on_health_updated)
        self._state.visualization_changed.connect(self._on_viz_changed)
        self._state.pipeline_completed.connect(self._on_pipeline_completed)
        self._state.pipeline_failed.connect(self._on_pipeline_failed)

        # System telemetry polling timer
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_system)
        self._poll_timer.start(self._POLL_INTERVAL_MS)

        # FPS tracking
        self._frame_times: list[float] = []
        self._last_frame = time.monotonic()

        logger.info("[StatusBar] Controller attached")

    # ── State handlers ─────────────────────────────────────────────────────

    def _on_pipeline_status(self, status: PipelineStatus):
        icons = {
            PipelineStatus.IDLE:      ("◦", "#888888"),
            PipelineStatus.LOADING:   ("↺", "#3498DB"),
            PipelineStatus.RUNNING:   ("▶", "#27AE60"),
            PipelineStatus.COMPLETED: ("✓", "#27AE60"),
            PipelineStatus.FAILED:    ("✗", "#E74C3C"),
            PipelineStatus.CANCELLED: ("⏹", "#E67E22"),
        }
        icon, color = icons.get(status, ("◦", "#888888"))
        self._lbl_pipeline.setText(f"  {icon} {status.name}  ")
        self._lbl_pipeline.setStyleSheet(f"color: {color};")

        show_progress = status == PipelineStatus.RUNNING
        self._progress.setVisible(show_progress)
        if not show_progress:
            self._progress.setValue(0)

    def _on_pipeline_progress(self, progress: float):
        self._progress.setVisible(True)
        self._progress.setValue(int(progress * 100))

    def _on_stage_changed(self, stage):
        if stage.status == "running":
            self._bar.showMessage(f"  Stage: {stage.name}…", 5000)

    def _on_health_updated(self, health: BackendHealth):
        self._lbl_fps.setText(f" FPS: {health.fps:.0f}  ")
        self._lbl_queue.setText(f" Queue: {health.queue_depth}  ")
        self._lbl_cpu.setText(f" CPU: {health.cpu_pct:.0f}%  ")
        self._lbl_ram.setText(f" RAM: {health.ram_pct:.0f}%  ")

    def _on_viz_changed(self, viz_state):
        self._lbl_interp.setText(f" Interp: {viz_state.interp_method}  ")

    def _on_pipeline_completed(self, result: dict):
        decision = result.get("decision", "—")
        n_conf   = len(result.get("confirmed_anomalies", []))
        self._bar.showMessage(
            f"  Analysis complete — {decision}  ({n_conf} confirmed targets)  ",
            8000
        )
        self._progress.setVisible(False)

    def _on_pipeline_failed(self, error: str):
        self._bar.showMessage(f"  ✗ Pipeline failed: {error}  ", 10000)
        self._progress.setVisible(False)

    # ── System polling ─────────────────────────────────────────────────────

    def _poll_system(self):
        """Read CPU/RAM and push a BackendHealth update."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except ImportError:
            cpu, ram = 0.0, 0.0

        # Update labels directly if health signal not being driven by a worker
        if self._state.pipeline_status not in (PipelineStatus.RUNNING,):
            self._lbl_cpu.setText(f" CPU: {cpu:.0f}%  ")
            self._lbl_ram.setText(f" RAM: {ram:.0f}%  ")

        # FPS approximation
        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        fps = len(self._frame_times)
        self._lbl_fps.setText(f" FPS: {fps}  ")

    # ── Cursor position (called by heatmap canvas) ─────────────────────────

    def update_cursor(self, x: float, y: float):
        self._lbl_cursor.setText(f" XY: {x:.2f}, {y:.2f}  ")

    # ── Config hash ────────────────────────────────────────────────────────

    def update_config_hash(self, config_hash: str):
        short = config_hash[:8] if config_hash else "········"
        self._lbl_config.setText(f" Hash: {short}  ")

    # ── Coverage ──────────────────────────────────────────────────────────

    def update_coverage(self, pct: float):
        self._lbl_coverage.setText(f" Cov: {pct:.0f}%  ")

    # ── Baseline label ────────────────────────────────────────────────────

    def update_baseline(self, method: str):
        self._lbl_baseline.setText(f" Base: {method}  ")
