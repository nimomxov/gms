"""
GMS — DeviceCapabilities + CapabilityExtractor + TelemetryFrame  v2.3

This module is the CAPABILITY KERNEL of the GMS platform.

It converts a SemanticMapping into a normalized DeviceCapabilities object
that controls every downstream decision:
  - which processing stages are enabled
  - which UI features are available
  - which confidence metrics are computed
  - which visualization overlays are rendered

DESIGN PRINCIPLE:
  If the telemetry does not supply a field, that capability is False.
  No estimation, no interpolation, no hallucination of missing fields.
  The UI displays a clear message explaining why the feature is disabled.

TELEMETRY GRADES (ordered by capability richness):
  BASIC        — signal only
  STANDARD     — signal + position (x/y)
  ADVANCED     — + SNR
  PROFESSIONAL — + heading + baseline + stability + noise_floor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from .mapper import SemanticMapping
from .inspector import FieldInventory

logger = logging.getLogger("gms.schema.capabilities")


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Grade
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryGrade(Enum):
    BASIC        = 1
    STANDARD     = 2
    ADVANCED     = 3
    PROFESSIONAL = 4

    def label(self) -> str:
        return {
            TelemetryGrade.BASIC:        "BASIC — signal only",
            TelemetryGrade.STANDARD:     "STANDARD — signal + position",
            TelemetryGrade.ADVANCED:     "ADVANCED — signal + position + SNR",
            TelemetryGrade.PROFESSIONAL: "PROFESSIONAL — full telemetry",
        }[self]

    def color(self) -> str:
        return {
            TelemetryGrade.BASIC:        "#AAAAAA",
            TelemetryGrade.STANDARD:     "#44AAFF",
            TelemetryGrade.ADVANCED:     "#44DD88",
            TelemetryGrade.PROFESSIONAL: "#FFCC00",
        }[self]


# ─────────────────────────────────────────────────────────────────────────────
# DeviceCapabilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceCapabilities:
    """
    Normalized capability descriptor derived from telemetry field inspection.

    Every bool flag corresponds to a hardware/telemetry capability.
    Downstream pipeline stages and UI components query this object
    before enabling any feature.

    Never set a flag True unless the actual telemetry field is present.
    """

    # ── Positional ─────────────────────────────────────────────────────────
    has_position: bool = False      # x AND y available
    has_x: bool = False
    has_y: bool = False
    has_altitude: bool = False

    # ── Signal quality ──────────────────────────────────────────────────────
    has_snr: bool = False
    has_noise_floor: bool = False
    has_stability: bool = False
    has_filtered_signal: bool = False

    # ── Heading / navigation ────────────────────────────────────────────────
    has_heading: bool = False

    # ── Reference / calibration ─────────────────────────────────────────────
    has_baseline: bool = False

    # ── Timing ──────────────────────────────────────────────────────────────
    has_timestamp: bool = False

    # ── Quality metadata ────────────────────────────────────────────────────
    has_quality_flag: bool = False
    has_temperature: bool = False
    has_speed: bool = False

    # ── Derived feature flags (set by CapabilityExtractor) ──────────────────
    # These drive UI gating decisions.
    can_compute_confidence: bool = False    # requires snr
    can_compute_uncertainty_radius: bool = False  # requires snr
    can_path_reconstruct: bool = False      # requires x, y, heading
    can_orientation_correct: bool = False   # requires heading
    can_interpolate_2d: bool = False        # requires x, y
    can_render_heatmap: bool = False        # requires x, y
    can_render_3d: bool = False             # requires x, y
    can_adaptive_baseline: bool = False     # requires position
    can_drift_compensate: bool = False      # requires baseline or position
    can_matched_filter: bool = False        # always True (signal only)
    can_depth_estimate: bool = False        # requires snr + position (and calibration)

    # ── Telemetry grade ──────────────────────────────────────────────────────
    grade: TelemetryGrade = TelemetryGrade.BASIC

    # ── Column name map: role → actual CSV column name ─────────────────────
    column_map: dict = field(default_factory=dict)

    # ── UI messages for disabled features ──────────────────────────────────
    disabled_reasons: dict = field(default_factory=dict)

    def gate(self, feature: str) -> tuple[bool, str]:
        """
        Check if a feature is available.
        Returns (enabled: bool, reason: str).
        The reason is shown in the UI when enabled=False.
        """
        attr = f"can_{feature}"
        enabled = getattr(self, attr, False)
        reason = "" if enabled else self.disabled_reasons.get(feature, f"Feature '{feature}' not available with current telemetry.")
        return enabled, reason

    def summary(self) -> dict:
        return {
            "grade": self.grade.name,
            "grade_label": self.grade.label(),
            "has_position": self.has_position,
            "has_snr": self.has_snr,
            "has_heading": self.has_heading,
            "has_baseline": self.has_baseline,
            "has_noise_floor": self.has_noise_floor,
            "has_stability": self.has_stability,
            "has_filtered_signal": self.has_filtered_signal,
            "has_timestamp": self.has_timestamp,
            "features_enabled": {
                "confidence": self.can_compute_confidence,
                "uncertainty_radius": self.can_compute_uncertainty_radius,
                "path_reconstruction": self.can_path_reconstruct,
                "orientation_correction": self.can_orientation_correct,
                "2d_heatmap": self.can_render_heatmap,
                "3d_explorer": self.can_render_3d,
                "adaptive_baseline": self.can_adaptive_baseline,
                "depth_estimation": self.can_depth_estimate,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# TelemetryFrame — unified internal data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryFrame:
    """
    Unified internal representation of one scan row/sample.

    All visualizers, processors, and detectors consume TelemetryFrame
    objects — never raw CSV directly.

    Optional fields use None when telemetry does not provide the value.
    Processing stages must handle None safely.
    """
    signal: float
    filtered_signal: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    heading: Optional[float] = None
    snr: Optional[float] = None
    stability: Optional[float] = None
    noise_floor: Optional[float] = None
    baseline: Optional[float] = None
    timestamp: Optional[int] = None
    quality: Optional[str] = None
    temperature: Optional[float] = None
    speed: Optional[float] = None
    altitude: Optional[float] = None

    @staticmethod
    def from_row(row: dict, mapping: SemanticMapping) -> "TelemetryFrame":
        """Build a TelemetryFrame from a CSV row dict using a SemanticMapping."""
        def _get(role: str) -> Optional[float]:
            col = mapping.get_column(role)
            if col is None:
                return None
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        def _get_str(role: str) -> Optional[str]:
            col = mapping.get_column(role)
            if col is None:
                return None
            val = row.get(col)
            return str(val) if val is not None else None

        def _get_int(role: str) -> Optional[int]:
            val = _get(role)
            return int(val) if val is not None else None

        signal_col = mapping.get_column("signal")
        if signal_col is None:
            raise ValueError("TelemetryFrame: no 'signal' role mapped — cannot build frame")
        raw_signal = row.get(signal_col)
        if raw_signal is None:
            raise ValueError("TelemetryFrame: signal value is None")

        return TelemetryFrame(
            signal=float(raw_signal),
            filtered_signal=_get("filtered_signal"),
            x=_get("x"),
            y=_get("y"),
            heading=_get("heading"),
            snr=_get("snr"),
            stability=_get("stability"),
            noise_floor=_get("noise_floor"),
            baseline=_get("baseline"),
            timestamp=_get_int("timestamp"),
            quality=_get_str("quality"),
            temperature=_get("temperature"),
            speed=_get("speed"),
            altitude=_get("altitude"),
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame, mapping: SemanticMapping) -> list["TelemetryFrame"]:
        """Convert an entire DataFrame to a list of TelemetryFrames."""
        frames = []
        for _, row in df.iterrows():
            try:
                frames.append(TelemetryFrame.from_row(row.to_dict(), mapping))
            except ValueError:
                pass
        return frames


# ─────────────────────────────────────────────────────────────────────────────
# CapabilityExtractor
# ─────────────────────────────────────────────────────────────────────────────

class CapabilityExtractor:
    """
    Converts a SemanticMapping into a DeviceCapabilities object.

    This is the single source of truth for capability gating.
    Called once per CSV/device connection.
    """

    def extract(self, mapping: SemanticMapping) -> DeviceCapabilities:
        """Derive capabilities from a resolved semantic mapping."""
        cap = DeviceCapabilities()
        disabled: dict[str, str] = {}

        # Store column_map for downstream use
        cap.column_map = {
            role: rm.field_name
            for role, rm in mapping.role_map.items()
            if rm.is_mapped
        }

        # ── Positional ──────────────────────────────────────────────────────
        cap.has_x = mapping.has_role("x")
        cap.has_y = mapping.has_role("y")
        cap.has_position = cap.has_x and cap.has_y
        cap.has_altitude = mapping.has_role("altitude")

        # ── Signal quality ──────────────────────────────────────────────────
        cap.has_snr = mapping.has_role("snr")
        cap.has_noise_floor = mapping.has_role("noise_floor")
        cap.has_stability = mapping.has_role("stability")
        cap.has_filtered_signal = mapping.has_role("filtered_signal")

        # ── Heading ─────────────────────────────────────────────────────────
        cap.has_heading = mapping.has_role("heading")

        # ── Reference ───────────────────────────────────────────────────────
        cap.has_baseline = mapping.has_role("baseline")

        # ── Timing ──────────────────────────────────────────────────────────
        cap.has_timestamp = mapping.has_role("timestamp")

        # ── Metadata ────────────────────────────────────────────────────────
        cap.has_quality_flag = mapping.has_role("quality")
        cap.has_temperature = mapping.has_role("temperature")
        cap.has_speed = mapping.has_role("speed")

        # ── Derived feature gates ────────────────────────────────────────────
        if cap.has_snr:
            cap.can_compute_confidence = True
            cap.can_compute_uncertainty_radius = True
        else:
            disabled["compute_confidence"] = (
                "Confidence analysis unavailable: device does not provide SNR telemetry."
            )
            disabled["compute_uncertainty_radius"] = (
                "Uncertainty radius unavailable: requires SNR telemetry."
            )

        if cap.has_position and cap.has_heading:
            cap.can_path_reconstruct = True
        else:
            missing = []
            if not cap.has_position:
                missing.append("position (x/y)")
            if not cap.has_heading:
                missing.append("heading")
            disabled["path_reconstruct"] = (
                f"Path reconstruction unavailable: missing {' and '.join(missing)}."
            )

        cap.can_orientation_correct = cap.has_heading
        if not cap.has_heading:
            disabled["orientation_correct"] = (
                "Orientation correction unavailable: device does not provide heading telemetry."
            )

        cap.can_interpolate_2d = cap.has_position
        cap.can_render_heatmap = cap.has_position
        cap.can_render_3d = cap.has_position
        if not cap.has_position:
            disabled["interpolate_2d"] = (
                "2D interpolation unavailable: device does not provide x/y position. "
                "Falling back to line visualization."
            )
            disabled["render_heatmap"] = (
                "2D heatmap unavailable: requires x/y position telemetry."
            )
            disabled["render_3d"] = (
                "3D visualization unavailable: requires x/y position telemetry."
            )

        cap.can_adaptive_baseline = cap.has_position
        cap.can_drift_compensate = cap.has_baseline or cap.has_position
        if not cap.can_drift_compensate:
            disabled["drift_compensate"] = (
                "Drift compensation limited: no baseline or position available."
            )

        cap.can_matched_filter = True  # always available — signal only needed

        cap.can_depth_estimate = False  # requires calibrated inversion + SNR + position
        disabled["depth_estimate"] = (
            "Depth estimation disabled: requires calibrated inversion model, "
            "SNR telemetry, and position data. Not available in current configuration."
        )

        cap.disabled_reasons = disabled

        # ── Compute telemetry grade ──────────────────────────────────────────
        cap.grade = self._compute_grade(cap)

        logger.info(
            f"[CapabilityExtractor] Grade: {cap.grade.name} | "
            f"position={cap.has_position} snr={cap.has_snr} "
            f"heading={cap.has_heading} baseline={cap.has_baseline} "
            f"stability={cap.has_stability} noise={cap.has_noise_floor}"
        )

        return cap

    def _compute_grade(self, cap: DeviceCapabilities) -> TelemetryGrade:
        if (cap.has_heading and cap.has_baseline
                and cap.has_stability and cap.has_noise_floor):
            return TelemetryGrade.PROFESSIONAL
        if cap.has_snr:
            return TelemetryGrade.ADVANCED
        if cap.has_position:
            return TelemetryGrade.STANDARD
        return TelemetryGrade.BASIC
