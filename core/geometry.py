"""
GMS — Survey Geometry  v3.0
=============================
SurveyDirection enum, ScanGeometryConfig, CalibrationConfig,
SurveyMetadata (scan_id + timestamp), GeometryReconstructor.

Formula (always, scan_pattern is metadata only):
  field_width_m  = (points_per_line - 1) × sample_distance_m
  field_length_m = (num_lines - 1)       × line_spacing_m
"""
from __future__ import annotations
import logging, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import numpy as np

logger = logging.getLogger("gms.geometry")

class SurveyDirection(Enum):
    NORTH_SOUTH = "North -->South"
    SOUTH_NORTH = "South --> North"
    EAST_WEST   = "East --> West"
    WEST_EAST   = "West --> East"

    @classmethod
    def from_combo_text(cls, text: str) -> "SurveyDirection":
        for m in cls:
            if m.value == text: return m
        raise ValueError(f"Unknown survey direction: '{text}'")

    @property
    def is_north_south_axis(self) -> bool:
        return self in (SurveyDirection.NORTH_SOUTH, SurveyDirection.SOUTH_NORTH)

    @property
    def reversed(self) -> bool:
        return self in (SurveyDirection.SOUTH_NORTH, SurveyDirection.WEST_EAST)

class ScanPattern(Enum):
    VERTICAL   = "vertical"
    HORIZONTAL = "horizontal"

class SensorOrientation(Enum):
    VERTICAL_GRADIENT   = "vertical"
    HORIZONTAL_GRADIENT = "horizontal"

_HEIGHT_MIDPOINTS = {
    "5-10 cm": 0.075, "10-15cm": 0.125, "15-20cm": 0.175,
    "20-30cm": 0.250, "30-40cm": 0.350, "40-50cm": 0.450,
}

def _parse_height(text: str) -> float:
    t = text.strip()
    if t in _HEIGHT_MIDPOINTS: return _HEIGHT_MIDPOINTS[t]
    import re
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if len(nums) >= 2: return (nums[0]+nums[1])/2/100.0
    if len(nums) == 1: return nums[0]/100.0
    return 0.125

@dataclass
class ScanGeometryConfig:
    num_lines:          int             = 5
    points_per_line:    int             = 10
    line_spacing_m:     float           = 1.0
    sample_distance_m:  float           = 0.5
    direction:          SurveyDirection = SurveyDirection.NORTH_SOUTH
    scan_pattern:       ScanPattern     = ScanPattern.VERTICAL
    zigzag:             bool            = True
    parallel:           bool            = False
    sensor_height_m:    float           = 0.125
    field_width_m:      float           = field(default=0.0, init=False)
    field_length_m:     float           = field(default=0.0, init=False)

    def __post_init__(self):
        self.field_width_m  = (self.points_per_line - 1) * self.sample_distance_m
        self.field_length_m = (self.num_lines - 1)       * self.line_spacing_m

    def validate(self):
        errors = []
        if self.num_lines < 2:       errors.append("• Number of Lines must be ≥ 2")
        if self.points_per_line < 2: errors.append("• Points per Line must be ≥ 2")
        if self.line_spacing_m <= 0: errors.append("• Line Spacing must be > 0 cm")
        if self.sample_distance_m <= 0: errors.append("• Sample Distance must be > 0 cm")
        if self.field_width_m <= 0 or self.field_length_m <= 0:
            errors.append(f"• Computed field invalid: "
                          f"{self.field_width_m:.3f}m × {self.field_length_m:.3f}m")
        if errors:
            raise ValueError("Survey geometry incomplete:\n" + "\n".join(errors) +
                             "\n\nFill all geometry fields before submitting.")

    @property
    def dx(self) -> float: return self.sample_distance_m
    @property
    def dy(self) -> float: return self.line_spacing_m
    @property
    def total_points(self) -> int: return self.num_lines * self.points_per_line

    def to_dict(self) -> dict:
        return dict(num_lines=self.num_lines, points_per_line=self.points_per_line,
                    line_spacing_m=round(self.line_spacing_m,4),
                    sample_distance_m=round(self.sample_distance_m,4),
                    field_width_m=round(self.field_width_m,4),
                    field_length_m=round(self.field_length_m,4),
                    direction=self.direction.value, scan_pattern=self.scan_pattern.value,
                    zigzag=self.zigzag, parallel=self.parallel,
                    sensor_height_m=round(self.sensor_height_m,4))

@dataclass
class SensorCalibration:
    adc_scaling_factor: float             = 1.0
    gain:               float             = 1.0
    offset:             float             = 0.0
    sensor_spacing_m:   float             = 0.5
    orientation:        SensorOrientation = SensorOrientation.VERTICAL_GRADIENT
    submitted:          bool              = False

    def validate(self):
        errors = []
        if self.adc_scaling_factor <= 0: errors.append("• ADC Scaling Factor must be > 0")
        if self.gain <= 0:               errors.append("• Gain must be > 0")
        if not (0.05 <= self.sensor_spacing_m <= 2.0):
            errors.append("• Sensor Spacing must be 0.05–2.0 m")
        if errors: raise ValueError("Sensor calibration invalid:\n" + "\n".join(errors))

    def to_dict(self) -> dict:
        return dict(adc_scaling_factor=self.adc_scaling_factor, gain=self.gain,
                    offset=self.offset, sensor_spacing_m=self.sensor_spacing_m,
                    orientation=self.orientation.value)

@dataclass
class SoilCalibration:
    soil_profile:              str   = "General Loam"
    mineralization_correction: float = 0.0
    basalt_compensation:       float = 0.0
    submitted:                 bool  = False

    def validate(self):
        errors = []
        if not (0 <= self.mineralization_correction <= 100):
            errors.append("• Mineralization must be 0–100")
        if not (0 <= self.basalt_compensation <= 100):
            errors.append("• Basalt Compensation must be 0–100")
        if errors: raise ValueError("Soil calibration invalid:\n" + "\n".join(errors))

    def to_dict(self) -> dict:
        return dict(soil_profile=self.soil_profile,
                    mineralization_correction=self.mineralization_correction,
                    basalt_compensation=self.basalt_compensation)

@dataclass
class CalibrationConfig:
    sensor: SensorCalibration = field(default_factory=SensorCalibration)
    soil:   SoilCalibration   = field(default_factory=SoilCalibration)

    @property
    def is_complete(self) -> bool:
        return self.sensor.submitted and self.soil.submitted

    def to_dict(self) -> dict:
        return {"sensor": self.sensor.to_dict(), "soil": self.soil.to_dict()}

@dataclass
class SurveyMetadata:
    scan_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    geometry:     Optional[ScanGeometryConfig] = None
    calibration:  CalibrationConfig            = field(default_factory=CalibrationConfig)
    phase1_complete: bool = False
    phase2_complete: bool = False

    def mark_phase1(self, geo: ScanGeometryConfig):
        geo.validate()
        self.geometry = geo
        self.phase1_complete = True
        logger.info(f"[Survey] Phase1 scan_id={self.scan_id} "
                    f"{geo.field_width_m:.2f}×{geo.field_length_m:.2f}m")

    def mark_sensor(self, cal: SensorCalibration):
        cal.validate(); cal.submitted = True
        self.calibration.sensor = cal
        self._check2()

    def mark_soil(self, cal: SoilCalibration):
        cal.validate(); cal.submitted = True
        self.calibration.soil = cal
        self._check2()

    def _check2(self):
        if self.calibration.is_complete:
            self.phase2_complete = True
            logger.info(f"[Survey] Phase2 complete scan_id={self.scan_id}")

    def validate_pipeline_ready(self):
        if not self.phase1_complete or self.geometry is None:
            raise ValueError(
                "Survey geometry has not been submitted.\n\n"
                "Fill all geometry fields and click "
                "\"Submit Survey Geometry\" before running the pipeline.")
        if not self.phase2_complete:
            missing = []
            if not self.calibration.sensor.submitted:
                missing.append("Sensor Calibration (click Apply Sensor)")
            if not self.calibration.soil.submitted:
                missing.append("Soil Calibration (click Apply Soil)")
            raise ValueError(
                "Calibration incomplete:\n\n" +
                "\n".join(f"• {m}" for m in missing) +
                "\n\nComplete all calibration steps before running the pipeline.")

    def to_dict(self) -> dict:
        return dict(scan_id=self.scan_id, timestamp=self.timestamp,
                    geometry=self.geometry.to_dict() if self.geometry else None,
                    calibration=self.calibration.to_dict(),
                    phase1_complete=self.phase1_complete,
                    phase2_complete=self.phase2_complete)

class GeometryReconstructor:
    @staticmethod
    def build_xy(geo: ScanGeometryConfig, n_samples: int = None):
        n       = n_samples if n_samples is not None else geo.total_points
        xs_line = np.linspace(0.0, geo.field_width_m,  geo.points_per_line)
        ys_line = np.linspace(0.0, geo.field_length_m, geo.num_lines)
        x_list, y_list = [], []
        for li, y_val in enumerate(ys_line):
            line_x = xs_line[::-1] if (geo.zigzag and li % 2 == 1) else xs_line
            x_list.append(line_x)
            y_list.append(np.full(geo.points_per_line, y_val))
        x_all = np.concatenate(x_list)
        y_all = np.concatenate(y_list)
        if geo.direction.reversed: y_all = geo.field_length_m - y_all
        total = len(x_all)
        if n == total:   return x_all, y_all
        if n < total:    return x_all[:n], y_all[:n]
        reps = int(np.ceil(n / total))
        return np.tile(x_all, reps)[:n], np.tile(y_all, reps)[:n]

    @staticmethod
    def apply_to_scan(scan_dataset, geo: ScanGeometryConfig):
        n = len(scan_dataset.values)
        x, y = GeometryReconstructor.build_xy(geo, n_samples=n)
        scan_dataset.x = x; scan_dataset.y = y
        cap = scan_dataset.capabilities
        cap.has_x = cap.has_y = cap.has_position = True
        logger.info(f"[Geometry] Applied to {scan_dataset.scan_id}: "
                    f"{n} pts x∈[{x.min():.2f},{x.max():.2f}] "
                    f"y∈[{y.min():.2f},{y.max():.2f}]")
        return scan_dataset