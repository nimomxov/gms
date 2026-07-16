"""
GMS — Orthogonal Scan Fusion  v2.3

Physical basis:
  A true buried target produces a consistent anomaly signature regardless
  of survey direction. If two scan grids (X-direction and Y-direction)
  BOTH show an anomaly at the same position, confidence is greatly boosted.

  A noise artifact or scan-line artifact will typically appear in ONE
  direction only — it will NOT appear at the same position in the
  orthogonal scan.

How to use:
  Survey two overlapping grids perpendicular to each other:
    - Primary scan:   traverse direction = X (lines parallel to Y-axis)
    - Secondary scan: traverse direction = Y (lines parallel to X-axis)

  Same target → appears in BOTH at (x₀, y₀)
  Scan artifact → appears in primary at some (x, y), absent in secondary

  fusion_boost = spatial_overlap_score × direction_consistency_score

  Final confidence = detection_confidence × (1 + fusion_boost × MAX_BOOST)

Integration:
  OrthogonalFusionEngine.fuse(results_x, results_y) → FusedResult
  FusedResult contains boosted anomalies and a fusion quality report.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("gms.orthogonal_fusion")

MAX_BOOST = 0.30        # maximum confidence boost from orthogonal confirmation
PROXIMITY_CELLS = 10.0  # max distance to consider the same target


@dataclass
class FusionPair:
    """Two anomalies from perpendicular scans that match the same target."""
    anomaly_id_x: str
    anomaly_id_y: str
    cx_x: float
    cy_x: float
    cx_y: float
    cy_y: float
    spatial_distance: float
    direction_consistency: float   # how similar are polarity + label
    fusion_boost: float            # confidence boost applied
    fused_cx: float                # weighted centroid
    fused_cy: float


@dataclass
class FusionResult:
    """Output of orthogonal scan fusion."""
    n_pairs_found: int
    n_unconfirmed_x: int        # anomalies in X scan not confirmed in Y
    n_unconfirmed_y: int
    pairs: list[FusionPair]
    quality_score: float        # 0=no orthogonal data, 1=perfect fusion
    message: str = ""


class OrthogonalFusionEngine:
    """
    Fuses anomaly detections from two perpendicular scan directions.

    Matching algorithm:
      For each anomaly in scan_x, find the nearest anomaly in scan_y
      within proximity_cells. If found and labels are compatible:
        → Mark as orthogonally confirmed
        → Apply fusion_boost to both anomalies
        → Compute fused position (weighted by confidence)

    Confidence boost formula:
      fusion_boost = MAX_BOOST × spatial_score × direction_score
      where:
        spatial_score = exp(-distance / proximity_cells)
        direction_score = label_match * polarity_similarity

    The fused centroid is placed between the two positions,
    weighted by their individual confidences.
    """

    def __init__(self,
                 proximity_cells: float = PROXIMITY_CELLS,
                 max_boost: float = MAX_BOOST,
                 require_label_match: bool = False):
        self.proximity = proximity_cells
        self.max_boost = max_boost
        self.require_label_match = require_label_match

    def fuse(self, anomalies_x: list, anomalies_y: list) -> FusionResult:
        """
        Fuse anomalies from X-scan with anomalies from Y-scan.
        Modifies anomaly confidence IN PLACE.
        """
        if not anomalies_x or not anomalies_y:
            return FusionResult(
                n_pairs_found=0,
                n_unconfirmed_x=len(anomalies_x),
                n_unconfirmed_y=len(anomalies_y),
                pairs=[], quality_score=0.0,
                message="No orthogonal data available — fusion skipped",
            )

        pairs = []
        matched_y = set()

        for ax in anomalies_x:
            if ax.raw_label == "NOISE":
                continue

            best_dist = self.proximity + 1
            best_ay   = None

            for j, ay in enumerate(anomalies_y):
                if j in matched_y or ay.raw_label == "NOISE":
                    continue
                dist = np.sqrt((ax.cx - ay.cx)**2 + (ax.cy - ay.cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_ay   = (j, ay)

            if best_ay is None:
                continue

            j, ay = best_ay

            # Label compatibility check
            if self.require_label_match and ax.raw_label != ay.raw_label:
                continue

            # Direction consistency: similar polarity and label → higher boost
            polarity_sim = float(1.0 - abs(ax.polarity_ratio - ay.polarity_ratio) / 2.0)
            label_match  = 1.0 if ax.raw_label == ay.raw_label else 0.6

            spatial_score    = float(np.exp(-best_dist / self.proximity))
            direction_score  = polarity_sim * label_match
            fusion_boost     = round(self.max_boost * spatial_score * direction_score, 4)

            # Apply boost to BOTH anomalies
            for a in [ax, ay]:
                old_conf = a.confidence
                a.__dict__["confidence"] = round(
                    float(np.clip(a.confidence + fusion_boost, 0, 1)), 3
                )
                logger.debug(
                    f"  OrthoFusion: {a.anomaly_id} conf {old_conf:.3f}→{a.confidence:.3f} "
                    f"(boost={fusion_boost:.3f}, dist={best_dist:.1f})"
                )

            # Fused position (confidence-weighted)
            w_x = ax.confidence
            w_y = ay.confidence
            wt  = w_x + w_y + 1e-8
            fused_cx = (w_x * ax.marker_cx + w_y * ay.marker_cx) / wt
            fused_cy = (w_x * ax.marker_cy + w_y * ay.marker_cy) / wt

            pairs.append(FusionPair(
                anomaly_id_x=ax.anomaly_id,
                anomaly_id_y=ay.anomaly_id,
                cx_x=ax.cx, cy_x=ax.cy,
                cx_y=ay.cx, cy_y=ay.cy,
                spatial_distance=round(best_dist, 2),
                direction_consistency=round(direction_score, 3),
                fusion_boost=fusion_boost,
                fused_cx=round(fused_cx, 2),
                fused_cy=round(fused_cy, 2),
            ))
            matched_y.add(j)

        n_unconf_x = sum(1 for a in anomalies_x
                         if a.raw_label != "NOISE" and
                         not any(p.anomaly_id_x == a.anomaly_id for p in pairs))
        n_unconf_y = len(anomalies_y) - len(matched_y)

        n_total_x = sum(1 for a in anomalies_x if a.raw_label != "NOISE")
        quality = len(pairs) / max(n_total_x, 1)

        msg = (
            f"{len(pairs)} orthogonal pairs confirmed, "
            f"{n_unconf_x} X-only, {n_unconf_y} Y-only "
            f"(quality={quality:.0%})"
        )
        logger.info(f"  OrthogonalFusion: {msg}")

        return FusionResult(
            n_pairs_found=len(pairs),
            n_unconfirmed_x=n_unconf_x,
            n_unconfirmed_y=n_unconf_y,
            pairs=pairs,
            quality_score=round(quality, 3),
            message=msg,
        )

    def tag_scans(self, scan_files: list[str]) -> tuple[list[str], list[str]]:
        """
        Heuristic: separate scan files into X-direction and Y-direction.
        Convention:
          Files with '_x' or '_X' in name → X-direction
          Files with '_y' or '_Y' in name → Y-direction
          Otherwise: odd index → X, even index → Y
        """
        x_scans, y_scans = [], []
        for f in scan_files:
            name = str(f).lower()
            if '_x' in name or 'scan_x' in name:
                x_scans.append(f)
            elif '_y' in name or 'scan_y' in name:
                y_scans.append(f)
        if not x_scans and not y_scans:
            # Fall back: split by index
            x_scans = scan_files[::2]
            y_scans = scan_files[1::2]
        return x_scans, y_scans
