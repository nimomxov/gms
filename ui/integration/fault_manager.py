"""
GMS — Fault Manager  v1.0
===========================
Catches, classifies, and presents all runtime errors to the user.
NEVER crashes silently.

Every fault gets:
  - readable title
  - human explanation
  - recovery action suggestion

Integrates with GMSEventBus so any backend module can raise a fault
without knowing about the UI.
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMainWindow, QMessageBox, QPushButton

from .app_state import GMSApplicationState
from .event_bus import GMSEventBus, GMS_EVENTS

logger = logging.getLogger("gms.fault_manager")


# ─────────────────────────────────────────────────────────────────────────────
# Fault categories with recovery suggestions
# ─────────────────────────────────────────────────────────────────────────────

FAULT_RECOVERIES = {
    "malformed_csv":        "Check that the file is a valid CSV with at least one numeric column.",
    "missing_columns":      "Use the field mapper in the CSV Import tab to assign columns manually.",
    "interpolation_failed": "Try a different interpolation method (e.g. switch from RBF to Cubic).",
    "matplotlib_failed":    "Ensure matplotlib is installed: pip install matplotlib",
    "opengl_failed":        "The 3D viewer requires OpenGL 3.3+. Try switching to 2D heatmap mode.",
    "memory_overload":      "Reduce scan resolution or load fewer scans simultaneously.",
    "worker_crash":         "Restart the pipeline. If the issue persists, check the log file.",
    "incompatible_stages":  "Change the preset in Pipeline Settings — some stage combinations are incompatible.",
    "generic":              "Check the application log for details.",
}


class GMSFaultManager(QObject):
    """
    Central error handler.
    Subscribes to FAULT_RAISED events from the bus and presents
    a structured error dialog to the user.
    """

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w   = window
        self._bus = GMSEventBus.instance()

        # Subscribe to fault events
        self._bus.subscribe(GMS_EVENTS.FAULT_RAISED, self._on_fault)

        # Subscribe to state fault signal too
        state = GMSApplicationState.instance()
        state.fault_raised.connect(self._on_fault_signal)

        # Install global exception hook for uncaught errors
        import sys
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._global_excepthook

        logger.info("[FaultManager] Attached")

    # ── Fault handlers ─────────────────────────────────────────────────────

    def _on_fault(self, payload: dict):
        title    = payload.get("title", "Error")
        message  = payload.get("message", "An unexpected error occurred.")
        recovery = payload.get("recovery", "")
        self._show_fault_dialog(title, message, recovery)

    def _on_fault_signal(self, title: str, message: str, recovery: str):
        self._show_fault_dialog(title, message, recovery)

    def _show_fault_dialog(self, title: str, message: str, recovery: str = ""):
        """Show a structured error dialog. Non-blocking via deferred execution."""
        logger.error(f"[Fault] {title}: {message}")

        # Classify to get recovery hint if not provided
        if not recovery:
            recovery = self._classify_recovery(message)

        dlg = QMessageBox(self._w)
        dlg.setWindowTitle(f"GMS — {title}")
        dlg.setIcon(QMessageBox.Warning)
        dlg.setText(f"<b>{title}</b>")

        detail = message
        if recovery:
            detail += f"\n\n<b>Recovery:</b> {recovery}"
        dlg.setInformativeText(detail)

        dlg.addButton("OK", QMessageBox.AcceptRole)
        log_btn = dlg.addButton("Show Log", QMessageBox.HelpRole)

        dlg.exec()

        if dlg.clickedButton() == log_btn:
            self._open_log()

    def _classify_recovery(self, message: str) -> str:
        msg_lower = message.lower()
        if "csv" in msg_lower or "parse" in msg_lower:
            return FAULT_RECOVERIES["malformed_csv"]
        if "column" in msg_lower or "field" in msg_lower:
            return FAULT_RECOVERIES["missing_columns"]
        if "interpolat" in msg_lower:
            return FAULT_RECOVERIES["interpolation_failed"]
        if "matplotlib" in msg_lower or "plot" in msg_lower:
            return FAULT_RECOVERIES["matplotlib_failed"]
        if "opengl" in msg_lower or "gl" in msg_lower:
            return FAULT_RECOVERIES["opengl_failed"]
        if "memory" in msg_lower or "oom" in msg_lower:
            return FAULT_RECOVERIES["memory_overload"]
        if "incompatible" in msg_lower or "stage" in msg_lower:
            return FAULT_RECOVERIES["incompatible_stages"]
        return FAULT_RECOVERIES["generic"]

    def _open_log(self):
        """Open the GMS log file in the system viewer."""
        import subprocess, sys, os
        log_path = "logs/gms.log"
        if not __import__("pathlib").Path(log_path).exists():
            return
        if sys.platform == "win32":
            os.startfile(log_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", log_path])
        else:
            subprocess.Popen(["xdg-open", log_path])

    # ── Global exception hook ──────────────────────────────────────────────

    def _global_excepthook(self, exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            self._original_excepthook(exc_type, exc_value, exc_tb)
            return

        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(f"[Uncaught] {exc_type.__name__}: {exc_value}\n{tb_str}")

        # Show non-blocking fault dialog
        try:
            self._show_fault_dialog(
                title=f"Unexpected Error: {exc_type.__name__}",
                message=str(exc_value),
                recovery="Restart the application. Check logs/gms.log for details.",
            )
        except Exception:
            pass  # Don't recurse into fault handling

        self._original_excepthook(exc_type, exc_value, exc_tb)

    # ── Convenience helpers for controllers ───────────────────────────────

    @staticmethod
    def raise_fault(title: str, message: str, recovery: str = ""):
        """Static helper so any module can raise a fault without holding a ref."""
        GMSEventBus.instance().emit_event(
            GMS_EVENTS.FAULT_RAISED,
            title=title,
            message=message,
            recovery=recovery,
        )
