"""
GMS — Pipeline Abstractions  v2.0
ABC base classes for every swappable pipeline stage.

Design principle: each stage is completely independent.
No stage knows about the others. The pipeline assembles them.

Stages:
  1. Interpolator  — scatter points → regular grid
  2. BaselineRemover — remove slow drift from grid
  3. AnomalyDetector — find blobs in clean grid

Lesson from v1.3→v1.4 experiments:
  Tight coupling between interpolation + baseline + detection caused:
  - anomaly absorption when both RBF and row-baseline ran together
  - LoG failing silently on ultra-smooth RBF surfaces
  - parameters tuned for one combo breaking another

Solution: each stage declares its COMPATIBILITY metadata so the
pipeline can warn about known bad combinations before running.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawScan:
    """Output of ingestion. Scatter point cloud."""
    scan_id: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class GriddedScan:
    """Output of interpolation stage. Regular 2D grid."""
    scan_id: str
    grid_z: np.ndarray       # 2D signal values
    grid_x: np.ndarray       # 1D x-axis coordinates
    grid_y: np.ndarray       # 1D y-axis coordinates
    grid_mask: np.ndarray    # bool — True = valid cell
    noise_floor: float       # MAD-estimated noise (pre-interpolation)
    dynamic_range: float     # peak-to-peak signal
    interp_name: str         # which interpolator was used
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class BaselinedGrid:
    """Output of baseline removal stage."""
    scan_id: str
    grid_z: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_mask: np.ndarray
    noise_floor: float        # re-estimated after baseline removal
    dynamic_range: float
    baseline_name: str
    interp_name: str
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class RawAnomaly:
    """Single detected anomaly — output of detector stage."""
    anomaly_id: str
    cx: float               # blob centroid x (grid index)
    cy: float               # blob centroid y (grid index)
    marker_cx: float        # true marker x (dipole midpoint for metal)
    marker_cy: float        # true marker y
    extent_cells: int
    peak_amplitude: float
    snr_robust: float
    smoothness_score: float
    dipole_score: float
    polarity_ratio: float
    spatial_coherence: float
    final_score: float
    uncertainty: float
    raw_label: str
    confidence: float
    bbox: tuple
    detector_name: str


@dataclass
class DetectionResult:
    """All anomalies from one scan."""
    scan_id: str
    anomalies: list[RawAnomaly] = field(default_factory=list)
    scan_quality_score: float = 0.0
    noise_floor: float = 0.0
    detector_name: str = ""
    warnings: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageCompatibility:
    """
    Each stage declares what it pairs well with.
    The pipeline checks these before running and logs warnings.
    """
    name: str
    preferred_baseline: list[str] = field(default_factory=list)
    incompatible_detectors: list[str] = field(default_factory=list)
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base classes
# ─────────────────────────────────────────────────────────────────────────────

class InterpolatorBase(ABC):
    """
    Stage 1: Scatter points → regular 2D grid.
    Must NOT do any drift removal. That is the baseline stage's job.
    Must NOT make any classification decisions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this interpolator."""

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(name=self.name)

    @abstractmethod
    def interpolate(self, scan: RawScan, resolution: float) -> GriddedScan:
        """
        Interpolate scatter data onto a regular grid.
        Args:
            scan: raw scatter data (x, y, values already drift-removed if needed)
            resolution: target grid cell size (same units as x, y)
        Returns:
            GriddedScan with regular 2D grid
        """


class BaselineRemoverBase(ABC):
    """
    Stage 2: Remove slow spatial drift from gridded data.
    Must NOT modify the grid structure (x, y axes, mask).
    Must NOT run anomaly detection.
    Input: GriddedScan. Output: BaselinedGrid.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier."""

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(name=self.name)

    @abstractmethod
    def remove(self, grid: GriddedScan) -> BaselinedGrid:
        """Apply baseline removal. Return new grid with drift subtracted."""


class AnomalyDetectorBase(ABC):
    """
    Stage 3: Find and classify anomalies in a clean grid.
    Must NOT modify the grid.
    Must NOT do drift removal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier."""

    @property
    def compatibility(self) -> StageCompatibility:
        return StageCompatibility(name=self.name)

    @abstractmethod
    def detect(self, grid: BaselinedGrid, config: dict) -> DetectionResult:
        """Detect anomalies. Return DetectionResult."""
