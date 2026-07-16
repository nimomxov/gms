"""
GMS — Command History (Undo/Redo)  v1.0
=========================================
Undo/Redo engine for all user-driven state changes.

Every reversible action in GMS is a Command object.
The CommandHistory stack tracks executed commands and supports
unlimited undo / redo within a session.

Reversible actions:
  - Visualization parameter change (colormap, opacity, layers…)
  - Pipeline preset change
  - Threshold / slider adjustment
  - Target validation (confirm dig / reject)
  - Calibration change
  - Compare scan add / remove

Non-reversible (not tracked):
  - File open / close
  - Pipeline execution itself (too expensive to re-run as undo)
  - Export / save

Usage:
    history = CommandHistory.instance()
    history.execute(ChangeColormapCommand(state, old="plasma", new="viridis"))
    history.undo()    # reverts colormap
    history.redo()    # re-applies

UI wiring (in IntegratedGMSController):
    action_undo.triggered.connect(history.undo)
    action_redo.triggered.connect(history.redo)
    history.stack_changed.connect(lambda: _update_undo_redo_labels())
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger("gms.command_history")

MAX_HISTORY = 50


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Command
# ─────────────────────────────────────────────────────────────────────────────

class Command(ABC):
    """Base class for all reversible commands."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Short human-readable name shown in Edit menu."""
        ...

    @abstractmethod
    def execute(self):
        """Apply the command."""
        ...

    @abstractmethod
    def undo(self):
        """Reverse the command."""
        ...

    def redo(self):
        """Re-apply after undo. Default = execute again."""
        self.execute()


# ─────────────────────────────────────────────────────────────────────────────
# Concrete commands
# ─────────────────────────────────────────────────────────────────────────────

class ChangeVisualizationCommand(Command):
    """Change a single VisualizationState field."""

    def __init__(self, state, key: str, old_value, new_value):
        self._state     = state
        self._key       = key
        self._old       = old_value
        self._new       = new_value

    @property
    def label(self) -> str:
        return f"Change {self._key}"

    def execute(self):
        self._state.set_visualization(self._key, self._new)

    def undo(self):
        self._state.set_visualization(self._key, self._old)


class ChangePresetCommand(Command):
    """Switch pipeline preset."""

    def __init__(self, state, old_preset: str, new_preset: str):
        self._state  = state
        self._old    = old_preset
        self._new    = new_preset

    @property
    def label(self) -> str:
        return f"Preset → {self._new}"

    def execute(self):
        self._state.set_preset(self._new)

    def undo(self):
        self._state.set_preset(self._old)


class ValidateTargetCommand(Command):
    """User confirms DIG on a target."""

    def __init__(self, state, anomaly_id: str, action: str):
        self._state      = state
        self._anomaly_id = anomaly_id
        self._action     = action   # "confirm_dig" | "reject"
        self._prev_list  = list(state.anomaly_list)

    @property
    def label(self) -> str:
        return f"{self._action.replace('_', ' ').title()}: {self._anomaly_id}"

    def execute(self):
        if self._action == "reject":
            updated = [a for a in self._state.anomaly_list
                       if a.anomaly_id != self._anomaly_id]
            self._state.set_anomaly_list(updated)

    def undo(self):
        # Restore previous anomaly list
        self._state.set_anomaly_list(list(self._prev_list))


class ChangeCalibrationCommand(Command):
    """Calibration parameter change."""

    def __init__(self, state, old_cal: dict, new_cal: dict):
        self._state = state
        self._old   = dict(old_cal)
        self._new   = dict(new_cal)

    @property
    def label(self) -> str:
        return "Calibration change"

    def execute(self):
        self._state.set_calibration(self._new)

    def undo(self):
        self._state.set_calibration(self._old)


class AddCompareScanCommand(Command):
    """Add a scan to the compare view."""

    def __init__(self, compare_controller, scan_entry):
        self._ctrl  = compare_controller
        self._entry = scan_entry

    @property
    def label(self) -> str:
        return f"Add scan: {getattr(self._entry, 'scan_id', '?')}"

    def execute(self):
        try:
            self._ctrl.add_entry(self._entry)
        except Exception:
            pass

    def undo(self):
        try:
            self._ctrl.remove_entry(self._entry)
        except Exception:
            pass


class MacroCommand(Command):
    """Group of commands executed as a single undo unit."""

    def __init__(self, commands: list[Command], label: str = "Macro"):
        self._commands = commands
        self._label    = label

    @property
    def label(self) -> str:
        return self._label

    def execute(self):
        for cmd in self._commands:
            cmd.execute()

    def undo(self):
        for cmd in reversed(self._commands):
            cmd.undo()


# ─────────────────────────────────────────────────────────────────────────────
# CommandHistory — singleton
# ─────────────────────────────────────────────────────────────────────────────

class CommandHistory(QObject):
    """
    Undo/Redo stack.  Thread-safe via Qt signal delivery.
    """
    _instance: Optional["CommandHistory"] = None

    # Emitted whenever the stack changes (drives Edit menu label updates)
    stack_changed = Signal(str, str)   # undo_label, redo_label

    def __init__(self, parent=None):
        super().__init__(parent)
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    @classmethod
    def instance(cls) -> "CommandHistory":
        if cls._instance is None:
            cls._instance = CommandHistory()
        return cls._instance

    # ── Execute ────────────────────────────────────────────────────────────

    def execute(self, command: Command):
        """Execute and push to undo stack. Clears redo stack."""
        try:
            command.execute()
            self._undo_stack.append(command)
            if len(self._undo_stack) > MAX_HISTORY:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
            self._notify()
            logger.debug(f"[History] Executed: {command.label}")
        except Exception as e:
            logger.error(f"[History] Command failed: {command.label} — {e}")

    # ── Undo ──────────────────────────────────────────────────────────────

    def undo(self):
        if not self._undo_stack:
            return
        command = self._undo_stack.pop()
        try:
            command.undo()
            self._redo_stack.append(command)
            self._notify()
            logger.debug(f"[History] Undone: {command.label}")
        except Exception as e:
            logger.error(f"[History] Undo failed: {command.label} — {e}")

    # ── Redo ──────────────────────────────────────────────────────────────

    def redo(self):
        if not self._redo_stack:
            return
        command = self._redo_stack.pop()
        try:
            command.redo()
            self._undo_stack.append(command)
            self._notify()
            logger.debug(f"[History] Redone: {command.label}")
        except Exception as e:
            logger.error(f"[History] Redo failed: {command.label} — {e}")

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_label(self) -> str:
        if self._undo_stack:
            return f"Undo {self._undo_stack[-1].label}"
        return "Undo"

    @property
    def redo_label(self) -> str:
        if self._redo_stack:
            return f"Redo {self._redo_stack[-1].label}"
        return "Redo"

    def history_summary(self) -> list[str]:
        """Returns ordered list of executed command labels (oldest first)."""
        return [c.label for c in self._undo_stack]

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify()

    def _notify(self):
        self.stack_changed.emit(self.undo_label, self.redo_label)


# ─────────────────────────────────────────────────────────────────────────────
# UndoRedoController — wires history to main window actions
# ─────────────────────────────────────────────────────────────────────────────

class UndoRedoController(QObject):
    """
    Connects CommandHistory to the main window's Edit menu actions.
    Also wraps HeatmapController slider/combo changes to auto-record commands.
    """

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self._w       = window
        self._history = CommandHistory.instance()
        self._state   = None

        try:
            from .app_state import GMSApplicationState
            self._state = GMSApplicationState.instance()
        except ImportError:
            pass

        self._wire_actions()
        self._history.stack_changed.connect(self._update_labels)
        self._update_labels("Undo", "Redo")

    def _wire_actions(self):
        try:
            from PySide6.QtGui import QAction
        except ImportError:
            from PySide6.QtWidgets import QAction  # PySide6 < 6.0 fallback
        for name, slot in (
            ("actionUndo", self._history.undo),
            ("actionRedo", self._history.redo),
        ):
            action = self._w.findChild(QAction, name)
            if action:
                action.triggered.connect(slot)

    def _update_labels(self, undo_label: str, redo_label: str):
        try:
            from PySide6.QtGui import QAction
        except ImportError:
            from PySide6.QtWidgets import QAction
        for name, label, enabled in (
            ("actionUndo", undo_label, self._history.can_undo),
            ("actionRedo", redo_label, self._history.can_redo),
        ):
            action = self._w.findChild(QAction, name)
            if action:
                action.setText(label)
                action.setEnabled(enabled)

    def record_viz_change(self, key: str, old_val, new_val):
        if self._state is None:
            return
        cmd = ChangeVisualizationCommand(self._state, key, old_val, new_val)
        self._history.execute(cmd)

    def record_preset_change(self, old: str, new: str):
        if self._state is None:
            return
        cmd = ChangePresetCommand(self._state, old, new)
        self._history.execute(cmd)

    def record_target_validation(self, anomaly_id: str, action: str):
        if self._state is None:
            return
        cmd = ValidateTargetCommand(self._state, anomaly_id, action)
        self._history.execute(cmd)
