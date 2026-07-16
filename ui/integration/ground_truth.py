"""
GMS — Ground Truth Workflow  v1.0
===================================
After-excavation field validation system.

Problem:
  The pipeline makes DIG/RESCAN/NO_DIG decisions, but without
  ground truth feedback these decisions cannot be improved over time.
  Thresholds remain static. FPR/FNR drift. The system never learns.

Solution:
  After excavation, the field operator records what was actually found.
  This creates a ValidationDataset that feeds:
    - Benchmark calibration
    - Threshold auto-tuning
    - ML training corpus
    - FPR/FNR regression tracking

Workflow:
  1. Field team excavates at a DIG site
  2. Operator opens Ground Truth panel
  3. Selects the target from the anomaly list
  4. Records what was found (target category + metadata)
  5. ValidationRecord is saved to disk + SessionRegistry
  6. CalibrationFeedback module adjusts thresholds if enough records exist

ValidationRecord fields:
  session_id          → links to ProcessingSession provenance
  anomaly_id          → links to ConfirmedAnomaly
  predicted_decision  → what GMS said
  actual_category     → what was actually found
  operator_notes      → free text
  excavation_depth_m  → actual depth if measured
  gps_coords          → (lat, lon) if available

Ground truth categories:
  TRUE_DIG_FERROUS     → predicted DIG, found ferrous metal
  TRUE_DIG_NONFERROUS  → predicted DIG, found non-ferrous
  TRUE_DIG_CAVITY      → predicted DIG, found cavity/void
  TRUE_DIG_BONE        → predicted DIG, found bone/organic
  FALSE_DIG_SCRAP      → predicted DIG, found scrap / surface interference
  FALSE_DIG_BASALT     → predicted DIG, found basalt / geological
  FALSE_DIG_NOTHING    → predicted DIG, found nothing
  MISSED_TARGET        → predicted RESCAN/NO_DIG, target confirmed by other means
  CONFIRMED_CLEAN      → predicted NO_DIG, area confirmed empty

Usage:
    wf = GroundTruthWorkflow(window)
    wf.open_validation_panel(anomaly_id="T001", session_id="s_abc123")
    # user fills form, clicks Save
    # → ValidationRecord saved, CalibrationFeedback triggered
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTextEdit, QDoubleSpinBox, QDialogButtonBox, QPushButton,
    QGroupBox, QFormLayout, QMessageBox, QMainWindow,
)

logger = logging.getLogger("gms.ground_truth")

VALIDATION_DIR = Path("data/validations")


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth category enum
# ─────────────────────────────────────────────────────────────────────────────

class GTCategory(str, Enum):
    TRUE_DIG_FERROUS    = "TRUE_DIG_FERROUS"
    TRUE_DIG_NONFERROUS = "TRUE_DIG_NONFERROUS"
    TRUE_DIG_CAVITY     = "TRUE_DIG_CAVITY"
    TRUE_DIG_BONE       = "TRUE_DIG_BONE"
    FALSE_DIG_SCRAP     = "FALSE_DIG_SCRAP"
    FALSE_DIG_BASALT    = "FALSE_DIG_BASALT"
    FALSE_DIG_NOTHING   = "FALSE_DIG_NOTHING"
    MISSED_TARGET       = "MISSED_TARGET"
    CONFIRMED_CLEAN     = "CONFIRMED_CLEAN"

    @property
    def is_true_positive(self) -> bool:
        return self.value.startswith("TRUE_DIG")

    @property
    def is_false_positive(self) -> bool:
        return self.value.startswith("FALSE_DIG")

    @property
    def is_false_negative(self) -> bool:
        return self == GTCategory.MISSED_TARGET

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


# ─────────────────────────────────────────────────────────────────────────────
# ValidationRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationRecord:
    record_id: str
    session_id: str
    anomaly_id: str
    predicted_decision: str
    predicted_confidence: float
    actual_category: GTCategory
    operator_notes: str        = ""
    excavation_depth_m: float  = 0.0
    gps_lat: float             = 0.0
    gps_lon: float             = 0.0
    scan_files: list[str]      = field(default_factory=list)
    telemetry_grade: str       = "UNKNOWN"
    recorded_at: str           = ""
    gms_version: str           = "3.1"

    def __post_init__(self):
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actual_category"] = self.actual_category.value
        return d

    def save(self, directory: Path = VALIDATION_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.record_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"[GroundTruth] Saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "ValidationRecord":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["actual_category"] = GTCategory(data["actual_category"])
        return cls(**data)


# ─────────────────────────────────────────────────────────────────────────────
# ValidationDataset — in-memory corpus
# ─────────────────────────────────────────────────────────────────────────────

class ValidationDataset:
    """
    Aggregates all validation records and computes accuracy metrics.
    """

    def __init__(self):
        self._records: list[ValidationRecord] = []

    def add(self, record: ValidationRecord):
        self._records.append(record)

    def load_all(self, directory: Path = VALIDATION_DIR):
        if not directory.exists():
            return
        for p in directory.glob("*.json"):
            try:
                self._records.append(ValidationRecord.load(p))
            except Exception as e:
                logger.warning(f"[GroundTruth] Could not load {p}: {e}")
        logger.info(f"[GroundTruth] Loaded {len(self._records)} validation records")

    @property
    def n_records(self) -> int:
        return len(self._records)

    def confusion_matrix(self) -> dict:
        tp = sum(1 for r in self._records if r.actual_category.is_true_positive)
        fp = sum(1 for r in self._records if r.actual_category.is_false_positive)
        fn = sum(1 for r in self._records if r.actual_category.is_false_negative)
        tn = sum(1 for r in self._records
                 if r.actual_category == GTCategory.CONFIRMED_CLEAN)
        total = max(tp + fp + fn + tn, 1)
        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "TPR": tp / max(tp + fn, 1),
            "FPR": fp / max(fp + tn, 1),
            "FNR": fn / max(fn + tp, 1),
            "accuracy": (tp + tn) / total,
            "n_total": total,
        }

    def confidence_calibration_data(self) -> list[tuple[float, bool]]:
        """Returns (predicted_confidence, was_correct) pairs for calibration."""
        pairs = []
        for r in self._records:
            correct = r.actual_category.is_true_positive
            pairs.append((r.predicted_confidence, correct))
        return pairs

    def export_training_corpus(self, path: str = "data/training_corpus.json"):
        """Export all records as a labeled ML training dataset."""
        corpus = [r.to_dict() for r in self._records]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"n": len(corpus), "records": corpus}, indent=2),
            encoding="utf-8"
        )
        logger.info(f"[GroundTruth] Training corpus exported: {path} ({len(corpus)} records)")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationFeedback — threshold auto-tuning
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationFeedback:
    """
    Adjusts pipeline decision thresholds based on validation records.

    Strategy:
      - FPR too high → raise DIG min_confidence
      - FNR too high → lower DIG min_confidence (carefully, FPR protection)
      - Both bad     → flag for manual review, do not auto-adjust
    """

    FPR_TARGET  = 0.05    # target ≤ 5% false positive rate
    FNR_TARGET  = 0.15    # target ≤ 15% false negative rate (RESCAN acceptable)
    MIN_RECORDS = 10      # minimum records before auto-adjustment

    def __init__(self, pipeline_config=None):
        self._cfg = pipeline_config

    def suggest_adjustments(self, dataset: ValidationDataset) -> dict:
        """
        Returns suggested threshold adjustments.
        Does NOT apply them — caller decides.
        """
        if dataset.n_records < self.MIN_RECORDS:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Need ≥{self.MIN_RECORDS} validation records for auto-calibration "
                    f"(have {dataset.n_records})."
                ),
                "adjustments": {},
            }

        cm = dataset.confusion_matrix()
        fpr = cm["FPR"]
        fnr = cm["FNR"]
        adjustments = {}

        if fpr > self.FPR_TARGET and fnr <= self.FNR_TARGET:
            # Too many false digs → raise threshold
            delta = min((fpr - self.FPR_TARGET) * 0.5, 0.10)
            adjustments["min_confidence_dig"] = {
                "direction": "increase",
                "delta": round(delta, 3),
                "reason": f"FPR={fpr:.0%} exceeds target ({self.FPR_TARGET:.0%})",
            }

        elif fnr > self.FNR_TARGET and fpr <= self.FPR_TARGET:
            # Too many missed targets → lower threshold slightly
            delta = min((fnr - self.FNR_TARGET) * 0.3, 0.05)
            adjustments["min_confidence_dig"] = {
                "direction": "decrease",
                "delta": round(delta, 3),
                "reason": (
                    f"FNR={fnr:.0%} exceeds target ({self.FNR_TARGET:.0%}). "
                    f"Adjustment capped to protect FPR."
                ),
            }

        elif fpr > self.FPR_TARGET and fnr > self.FNR_TARGET:
            adjustments["status"] = "manual_review_required"

        return {
            "status": "ok" if adjustments else "no_adjustment_needed",
            "fpr": round(fpr, 3),
            "fnr": round(fnr, 3),
            "n_records": dataset.n_records,
            "adjustments": adjustments,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GroundTruthWorkflow — UI controller
# ─────────────────────────────────────────────────────────────────────────────

class GroundTruthWorkflow(QObject):
    """
    Manages the ground truth recording UI.
    Opens a dialog from the Inspector panel after user excavates a target.
    """

    validation_saved = Signal(object)   # ValidationRecord

    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w       = window
        self._dataset = ValidationDataset()
        self._dataset.load_all()   # load existing records from disk

        self._feedback = CalibrationFeedback()

    def open_validation_panel(
        self,
        anomaly_id: str,
        session_id: str = "",
        predicted_decision: str = "DIG",
        predicted_confidence: float = 0.0,
    ):
        """Open the ground truth recording dialog for a specific target."""
        dlg = _ValidationDialog(
            self._w,
            anomaly_id=anomaly_id,
            session_id=session_id,
            predicted_decision=predicted_decision,
            predicted_confidence=predicted_confidence,
        )

        if dlg.exec() == QDialog.Accepted:
            record = dlg.build_record()
            self._save_and_feedback(record)

    def _save_and_feedback(self, record: ValidationRecord):
        self._dataset.add(record)
        record.save()
        self.validation_saved.emit(record)

        # Suggest calibration adjustments after threshold of records
        if self._dataset.n_records >= CalibrationFeedback.MIN_RECORDS:
            suggestions = self._feedback.suggest_adjustments(self._dataset)
            if suggestions.get("adjustments"):
                self._show_calibration_suggestion(suggestions)

        logger.info(
            f"[GroundTruth] Recorded: {record.anomaly_id} → "
            f"{record.actual_category.value}"
        )

    def _show_calibration_suggestion(self, suggestions: dict):
        adj = suggestions.get("adjustments", {})
        if not adj:
            return
        lines = [
            "Based on field validation data, GMS suggests threshold adjustments:",
            "",
        ]
        for key, info in adj.items():
            if isinstance(info, dict):
                lines.append(
                    f"  • {key}: {info.get('direction','?')} "
                    f"by {info.get('delta', 0):.3f}  ({info.get('reason','')})"
                )

        lines += [
            "",
            "Apply these adjustments in Calibration → Thresholds.",
            f"Current FPR: {suggestions.get('fpr',0):.0%}  |  "
            f"FNR: {suggestions.get('fnr',0):.0%}",
        ]

        QMessageBox.information(
            self._w,
            "Calibration Feedback",
            "\n".join(lines),
        )

    def dataset(self) -> ValidationDataset:
        return self._dataset


# ─────────────────────────────────────────────────────────────────────────────
# Validation Dialog
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_DISPLAY = {
    "True DIG — Ferrous metal":     GTCategory.TRUE_DIG_FERROUS,
    "True DIG — Non-ferrous metal": GTCategory.TRUE_DIG_NONFERROUS,
    "True DIG — Cavity / void":     GTCategory.TRUE_DIG_CAVITY,
    "True DIG — Bone / organic":    GTCategory.TRUE_DIG_BONE,
    "False DIG — Scrap / surface":  GTCategory.FALSE_DIG_SCRAP,
    "False DIG — Basalt / geology": GTCategory.FALSE_DIG_BASALT,
    "False DIG — Nothing found":    GTCategory.FALSE_DIG_NOTHING,
    "Missed target (false negative)": GTCategory.MISSED_TARGET,
    "Confirmed clean area":         GTCategory.CONFIRMED_CLEAN,
}


class _ValidationDialog(QDialog):

    def __init__(
        self, parent,
        anomaly_id: str,
        session_id: str,
        predicted_decision: str,
        predicted_confidence: float,
    ):
        super().__init__(parent)
        self._anomaly_id           = anomaly_id
        self._session_id           = session_id
        self._predicted_decision   = predicted_decision
        self._predicted_confidence = predicted_confidence

        self.setWindowTitle("Ground Truth — Field Validation")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header
        hdr = QLabel(
            f"<b>Target:</b> {self._anomaly_id}  |  "
            f"<b>Predicted:</b> {self._predicted_decision}  "
            f"({self._predicted_confidence:.0%})"
        )
        layout.addWidget(hdr)

        # Form
        grp = QGroupBox("Excavation Result")
        form = QFormLayout(grp)

        self._cmb_category = QComboBox()
        for label in _CATEGORY_DISPLAY:
            self._cmb_category.addItem(label)
        form.addRow("What was found:", self._cmb_category)

        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setSuffix(" m")
        self._spin_depth.setRange(0.0, 10.0)
        self._spin_depth.setSingleStep(0.05)
        form.addRow("Actual depth:", self._spin_depth)

        self._spin_lat = QDoubleSpinBox()
        self._spin_lat.setDecimals(6)
        self._spin_lat.setRange(-90, 90)
        form.addRow("GPS Latitude:", self._spin_lat)

        self._spin_lon = QDoubleSpinBox()
        self._spin_lon.setDecimals(6)
        self._spin_lon.setRange(-180, 180)
        form.addRow("GPS Longitude:", self._spin_lon)

        self._txt_notes = QTextEdit()
        self._txt_notes.setMaximumHeight(80)
        self._txt_notes.setPlaceholderText(
            "Operator notes: soil type, object description, conditions…"
        )
        form.addRow("Notes:", self._txt_notes)

        layout.addWidget(grp)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def build_record(self) -> ValidationRecord:
        import hashlib, time
        raw_id = f"{self._anomaly_id}_{self._session_id}_{time.time()}"
        record_id = "gt_" + hashlib.sha256(raw_id.encode()).hexdigest()[:10]

        cat_label = self._cmb_category.currentText()
        category  = _CATEGORY_DISPLAY[cat_label]

        return ValidationRecord(
            record_id=record_id,
            session_id=self._session_id,
            anomaly_id=self._anomaly_id,
            predicted_decision=self._predicted_decision,
            predicted_confidence=self._predicted_confidence,
            actual_category=category,
            operator_notes=self._txt_notes.toPlainText().strip(),
            excavation_depth_m=self._spin_depth.value(),
            gps_lat=self._spin_lat.value(),
            gps_lon=self._spin_lon.value(),
        )
