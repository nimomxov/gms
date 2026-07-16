"""
GMS — Multi-Scan Fusion Engine  v1.0
======================================
Moves GMS from single-survey interpretation to evidence-based
multi-survey decision making.

Problem statement
-----------------
CrossScanValidator (decision_engine.py) already confirms anomalies across
scans, but it operates only at grid-index level (proximity_cells) and
discards all per-scan geometry, depth, timestamp, and quality metadata.
It cannot answer:

  "This anomaly was found in 4 independent surveys.  After weighting by
   scan quality, accounting for positional spread, and penalising
   disagreement about depth — what is the consolidated confidence?"

Solution
--------
MultiScanFusionEngine operates entirely on **anomaly evidence** — never
on raw grids.  It:

  1. Converts all anomaly coordinates to real-world metres using each
     scan's ScanGeometryConfig (sample_distance × grid_index).
  2. Clusters anomalies across scans using a 2-D spatial index with
     configurable XY and depth tolerances (no arbitrary grid-index units).
  3. Applies a 5-component confidence formula to each cluster:
       - mean confidence (base signal)
       - uncertainty penalty  (high spread → discount)
       - repeatability bonus  (n_scans → reward)
       - label agreement      (disagreement → discount)
       - scan quality weight  (poor scans count less)
  4. Produces FusedTarget objects with full provenance.
  5. Emits FusionDiagnostics for every target — enabling the UI to show
     exactly why a target gained or lost confidence through fusion.

Architecture constraints (preserved from overall system)
---------------------------------------------------------
- No raw grid access.  All inputs are DetectionResult / RawAnomaly objects.
- No fabrication: if depth is unavailable, depth fields are None — not 0.
- Fusion does not modify any source anomaly object (immutable inputs).
- Scan quality weighting uses ScanReliability.penalty_factor when provided;
  falls back to DetectionResult.scan_quality_score otherwise.
- The engine is stateless — call fuse() multiple times safely.

Confidence tier rules
---------------------
  Detected in 1 scan  → LOW    (fusion_tier = "LOW")
  Detected in 2 scans → MEDIUM (fusion_tier = "MEDIUM")
  Detected in ≥3 scans → HIGH  (fusion_tier = "HIGH")

Usage
-----
    from core.fusion_engine import MultiScanFusionEngine, FusionInput

    inputs = [
        FusionInput(
            detection_result=det,           # DetectionResult from pipeline
            geometry=geo,                   # ScanGeometryConfig — for metre coords
            timestamp="2026-06-01T09:15Z",  # ISO-8601 acquisition time
            reliability=rel,                # ScanReliability (optional)
        )
        for det, geo, rel in zip(detections, geometries, reliabilities)
    ]

    engine = MultiScanFusionEngine(xy_tolerance_m=0.30, depth_tolerance_m=0.15)
    result = engine.fuse(inputs)

    for target in result.targets:
        print(target.fused_id, target.confidence, target.fusion_tier)
        print(engine.explain(target))
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .abstractions import DetectionResult, RawAnomaly

logger = logging.getLogger("gms.fusion")


# ─────────────────────────────────────────────────────────────────────────────
# Public data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionInput:
    """
    One scan's evidence package — everything needed for fusion.

    Parameters
    ----------
    detection_result : DetectionResult
        Anomalies detected in this scan (from GMSPipeline.process_scan).
    geometry : object (ScanGeometryConfig duck-type)
        Must expose .sample_distance_m and .line_spacing_m.
        Used to convert grid-index centroids to real-world metres.
        Pass None only if anomalies already carry real-world coordinates
        in cx / cy (set geometry_is_metres=True in that case).
    timestamp : str
        ISO-8601 acquisition time. Used for ordering and diagnostics.
    reliability : object (ScanReliability duck-type), optional
        Must expose .penalty_factor [0, 1] and .reliability_score [0, 1].
        When provided, used as scan quality weight in confidence fusion.
        Falls back to detection_result.scan_quality_score if absent.
    geometry_is_metres : bool
        Set True when anomaly cx/cy are already in metres (e.g. after
        registration). Default False (cx/cy are grid indices).
    """
    detection_result:   DetectionResult
    geometry:           object                  # ScanGeometryConfig duck-type
    timestamp:          str    = ""
    reliability:        object = None           # ScanReliability duck-type (optional)
    geometry_is_metres: bool   = False


@dataclass
class FusionDiagnostics:
    """
    Full audit trail explaining how a FusedTarget's confidence was computed.
    Enables the inspector panel to show exactly why a target gained or lost
    confidence relative to any individual scan.
    """
    fused_id:            str
    n_contributing:      int
    scan_ids:            list[str]
    timestamps:          list[str]

    # Per-scan raw values (before weighting)
    raw_confidences:     list[float]
    raw_uncertainties:   list[float]
    scan_quality_weights: list[float]

    # Intermediate fusion components
    mean_confidence_raw:   float     # unweighted mean
    weighted_confidence:   float     # quality-weighted mean
    uncertainty_penalty:   float     # [0, 1] — multiplicative discount
    repeatability_bonus:   float     # [0, 1] — additive reward factor
    label_agreement:       float     # [0, 1] — fraction agreeing on label
    label_agreement_weight: float    # [0, 1] — multiplicative factor

    # Final
    final_confidence:    float
    confidence_gain:     float       # final_confidence - best_single_confidence
    fusion_tier:         str         # LOW | MEDIUM | HIGH

    # Spatial diagnostics
    position_spread_m:   float       # std of XY positions in metres
    depth_spread_m:      Optional[float]   # std of depth estimates (if available)
    has_depth:           bool

    # Labels across scans
    label_votes:         dict[str, int]  # label → count

    # Any warnings generated during fusion
    warnings:            list[str]

    # Registration quality per scan (scan_id → quality score [0,1]).
    # Populated only when ScanRegistrationEngine ran before fusion.
    # Key: scan_id of the moved scan; value: quality score [0, 1].
    registration_quality: dict = None


@dataclass
class FusedTarget:
    """
    A single target consolidated from evidence across multiple scans.

    Coordinates are always in real-world metres (converted from grid
    indices using ScanGeometryConfig.sample_distance_m / line_spacing_m).

    Depth is None when the depth inversion stub is not yet calibrated —
    never fabricated.
    """
    fused_id:          str
    x:                 float              # metres
    y:                 float              # metres
    depth_m:           Optional[float]    # metres — None if uncalibrated
    depth_uncertainty_m: Optional[float]  # metres — None if depth is None

    confidence:        float              # [0, 1] fused confidence
    uncertainty:       float              # [0, 1] fused uncertainty
    fusion_tier:       str               # LOW | MEDIUM | HIGH

    supporting_scans:  list[str]         # scan_ids of contributing scans
    n_scans:           int               # len(supporting_scans)
    repeatability_score: float           # [0, 1] how consistently detected
    label:             str               # consensus anomaly label
    label_agreement:   float             # [0, 1] fraction agreeing on label

    # Provenance
    timestamps:        list[str]         # ISO-8601 of each contributing scan

    # Default fields (must come after all required fields)
    decision_tier:     str = "WEAK"      # CONFIRMED | PROBABLE | WEAK | REJECTED
    diagnostics:       FusionDiagnostics = field(repr=False, default=None)

    def to_dict(self) -> dict:
        return {
            "fused_id":            self.fused_id,
            "x_m":                 round(self.x, 4),
            "y_m":                 round(self.y, 4),
            "depth_m":             round(self.depth_m, 4) if self.depth_m is not None else None,
            "depth_uncertainty_m": round(self.depth_uncertainty_m, 4)
                                   if self.depth_uncertainty_m is not None else None,
            "confidence":          round(self.confidence, 4),
            "uncertainty":         round(self.uncertainty, 4),
            "fusion_tier":         self.fusion_tier,
            "decision_tier":       self.decision_tier,
            "supporting_scans":    self.supporting_scans,
            "n_scans":             self.n_scans,
            "repeatability_score": round(self.repeatability_score, 4),
            "label":               self.label,
            "label_agreement":     round(self.label_agreement, 4),
            "timestamps":          self.timestamps,
        }


@dataclass
class FusionResult:
    """
    Complete output of MultiScanFusionEngine.fuse().
    """
    targets:          list[FusedTarget]
    n_scans_fused:    int
    n_anomalies_in:   int              # total raw anomalies across all scans
    n_clusters:       int              # spatial clusters formed
    n_fused:          int              # clusters with ≥2 scans → FusedTarget
    n_singletons:     int              # clusters seen in only 1 scan
    fusion_timestamp: str              # ISO-8601 when fuse() was called
    warnings:         list[str]

    @property
    def high_confidence_targets(self) -> list[FusedTarget]:
        return [t for t in self.targets if t.fusion_tier == "HIGH"]

    @property
    def medium_confidence_targets(self) -> list[FusedTarget]:
        return [t for t in self.targets if t.fusion_tier == "MEDIUM"]

    @property
    def low_confidence_targets(self) -> list[FusedTarget]:
        return [t for t in self.targets if t.fusion_tier == "LOW"]

    @property
    def confirmed_targets(self) -> list["FusedTarget"]:
        return [t for t in self.targets if t.decision_tier == "CONFIRMED"]

    @property
    def probable_targets(self) -> list["FusedTarget"]:
        return [t for t in self.targets if t.decision_tier == "PROBABLE"]

    @property
    def weak_targets(self) -> list["FusedTarget"]:
        return [t for t in self.targets if t.decision_tier == "WEAK"]

    @property
    def rejected_targets(self) -> list["FusedTarget"]:
        return [t for t in self.targets if t.decision_tier == "REJECTED"]

    def summary(self) -> dict:
        return {
            "n_scans_fused":   self.n_scans_fused,
            "n_anomalies_in":  self.n_anomalies_in,
            "n_clusters":      self.n_clusters,
            "n_fused_targets": self.n_fused,
            "n_singletons":    self.n_singletons,
            "tier_counts": {
                "HIGH":   len(self.high_confidence_targets),
                "MEDIUM": len(self.medium_confidence_targets),
                "LOW":    len(self.low_confidence_targets),
            },
            "decision_tier_counts": {
                "CONFIRMED": len(self.confirmed_targets),
                "PROBABLE":  len(self.probable_targets),
                "WEAK":      len(self.weak_targets),
                "REJECTED":  len(self.rejected_targets),
            },
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _AnomalyEvidence:
    """Internal: anomaly in real-world metre coordinates, with scan context."""
    anomaly_id:   str
    scan_id:      str
    timestamp:    str
    x_m:          float
    y_m:          float
    depth_m:      Optional[float]
    confidence:   float
    uncertainty:  float
    label:        str
    snr:          float
    quality_weight: float    # [0, 1] scan-level quality weight


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _most_common(values: list) -> tuple:
    """Return (most_common_value, count, fraction) from a list."""
    if not values:
        return None, 0, 0.0
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts[best], counts[best] / len(values)


def _repeatability_bonus(n_scans: int, max_scans: int = 10) -> float:
    """
    Bonus factor [0, 1] that rewards consistent detection across many scans.

    Formula: sigmoid-like curve so that:
      n=1  → 0.00  (no bonus for a singleton)
      n=2  → 0.25
      n=3  → 0.50
      n=5  → 0.75
      n≥8  → ~1.00

    Uses: bonus = 1 - exp(-k*(n-1)) where k is chosen so n=3→0.5
    """
    if n_scans <= 1:
        return 0.0
    k = math.log(2)   # ln(2) ≈ 0.693 → n=2 gives 1 - exp(-k) ≈ 0.5 for n-1=1
    raw = 1.0 - math.exp(-k * (n_scans - 1))
    return float(np.clip(raw, 0.0, 1.0))


def _uncertainty_penalty(mean_uncertainty: float, position_spread_m: float,
                          xy_tolerance_m: float) -> float:
    """
    Multiplicative discount [0, 1] applied to weighted confidence.

    Two sources of penalty are combined:
      1. Mean anomaly uncertainty   (from detector per-anomaly field)
      2. Positional spread          (std of XY across supporting scans,
                                     normalised by XY tolerance)

    Both are bounded [0, 1] and their geometric mean forms the penalty.
    """
    # 1. Uncertainty from detector (already [0, 1])
    unc_component = float(np.clip(1.0 - mean_uncertainty, 0.0, 1.0))

    # 2. Spatial consistency: low spread → high component
    if xy_tolerance_m > 0:
        spread_norm = float(np.clip(position_spread_m / xy_tolerance_m, 0.0, 1.0))
    else:
        spread_norm = 0.0
    spread_component = 1.0 - spread_norm * 0.5   # partial penalty; spread is expected

    # Geometric mean
    return float(np.clip(math.sqrt(unc_component * spread_component), 0.0, 1.0))


def _quality_weighted_mean(confidences: list[float],
                            weights: list[float]) -> float:
    """Weighted mean confidence. Falls back to simple mean if weights all zero."""
    w = np.array(weights, dtype=float)
    c = np.array(confidences, dtype=float)
    total_w = w.sum()
    if total_w < 1e-9:
        return float(c.mean())
    return float(np.clip(np.dot(w, c) / total_w, 0.0, 1.0))


def _fusion_tier(n_scans: int) -> str:
    if n_scans >= 3:
        return "HIGH"
    if n_scans == 2:
        return "MEDIUM"
    return "LOW"


def _decision_tier(fusion_tier: str, confidence: float) -> str:
    """
    Map (fusion_tier, confidence) to a 4-level decision tier.

    Thresholds mirror decision_engine.py:
      DIG threshold:    min_confidence >= 0.70
      RESCAN threshold: min_confidence >= 0.45

    Mapping:
      CONFIRMED: HIGH tier AND confidence >= 0.70
      PROBABLE:  MEDIUM tier  OR  (HIGH tier AND confidence < 0.70)
      WEAK:      LOW tier AND confidence >= 0.45
      REJECTED:  LOW tier AND confidence < 0.45
    """
    if fusion_tier == "HIGH" and confidence >= 0.70:
        return "CONFIRMED"
    if fusion_tier == "MEDIUM" or (fusion_tier == "HIGH" and confidence < 0.70):
        return "PROBABLE"
    if fusion_tier == "LOW" and confidence >= 0.45:
        return "WEAK"
    return "REJECTED"


# ─────────────────────────────────────────────────────────────────────────────
# Spatial clustering (greedy single-linkage in metre space)
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialClusterer:
    """
    Groups _AnomalyEvidence objects that fall within xy_tolerance_m of each
    other (and, optionally, depth_tolerance_m) into clusters.

    Algorithm: greedy single-linkage.
      - O(n²) — acceptable for typical n < 200 anomalies per session.
      - Each evidence point is assigned to the first cluster whose centroid
        is within tolerance, or starts a new cluster.
      - Centroids are recomputed incrementally after each assignment.

    This is intentionally simpler than DBSCAN:
      - No epsilon / min_samples confusion for operators.
      - Predictable: adding one more anomaly cannot split an existing cluster.
      - Transparent: every assignment decision uses a single distance threshold.

    Note on depth: depth is optional. When both a candidate and the cluster
    centroid have depth estimates, depth tolerance is checked as a gate BEFORE
    XY distance. When either is None, depth tolerance is skipped.
    """

    def __init__(self, xy_tolerance_m: float, depth_tolerance_m: float):
        self.xy_tol    = xy_tolerance_m
        self.depth_tol = depth_tolerance_m

    def cluster(self, evidence: list[_AnomalyEvidence]) -> list[list[_AnomalyEvidence]]:
        if not evidence:
            return []

        clusters: list[list[_AnomalyEvidence]] = []
        centroids: list[dict] = []   # {"x": float, "y": float, "depth": float|None}

        for ev in evidence:
            assigned = False
            for idx, centroid in enumerate(centroids):
                # ── depth gate (only when both have depth) ────────────────
                if (ev.depth_m is not None and centroid["depth"] is not None
                        and self.depth_tol > 0):
                    if abs(ev.depth_m - centroid["depth"]) > self.depth_tol:
                        continue

                # ── XY distance ───────────────────────────────────────────
                dx = ev.x_m - centroid["x"]
                dy = ev.y_m - centroid["y"]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= self.xy_tol:
                    clusters[idx].append(ev)
                    # Update centroid (running mean)
                    n = len(clusters[idx])
                    centroids[idx]["x"]     = (centroid["x"]     * (n - 1) + ev.x_m) / n
                    centroids[idx]["y"]     = (centroid["y"]     * (n - 1) + ev.y_m) / n
                    # Update depth centroid only when available
                    if ev.depth_m is not None and centroid["depth"] is not None:
                        centroids[idx]["depth"] = (centroid["depth"] * (n - 1) + ev.depth_m) / n
                    elif ev.depth_m is not None and centroid["depth"] is None:
                        centroids[idx]["depth"] = ev.depth_m
                    assigned = True
                    break

            if not assigned:
                clusters.append([ev])
                centroids.append({
                    "x":     ev.x_m,
                    "y":     ev.y_m,
                    "depth": ev.depth_m,
                })

        return clusters


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class MultiScanFusionEngine:
    """
    Evidence-based fusion of anomalies detected across multiple independent scans.

    Parameters
    ----------
    xy_tolerance_m : float
        Maximum XY distance (metres) for two anomalies to be considered the
        same physical target. Typical value: 0.25–0.50 m. Default 0.30 m.
    depth_tolerance_m : float
        Maximum depth difference for two anomalies to merge into the same
        cluster. Only applied when both anomalies have depth estimates.
        Default 0.15 m.
    min_scans_for_medium : int
        Minimum contributing scans for MEDIUM tier. Default 2.
    min_scans_for_high : int
        Minimum contributing scans for HIGH tier. Default 3.
    repeatability_scale : float
        Scaling factor applied to the repeatability bonus before blending
        into the final confidence formula. Range [0, 1]. Default 0.20
        (bonus contributes up to 20% on top of base confidence).
    noise_labels : set[str]
        Anomaly labels that are filtered out before fusion. Targets labelled
        as noise in ALL contributing scans are still excluded.
        Default {"NOISE", "NOISE_ONLY"}.
    """

    def __init__(
        self,
        xy_tolerance_m:        float     = 0.30,
        depth_tolerance_m:     float     = 0.15,
        min_scans_for_medium:  int       = 2,
        min_scans_for_high:    int       = 3,
        repeatability_scale:   float     = 0.20,
        noise_labels:          set[str]  = None,
    ):
        if xy_tolerance_m <= 0:
            raise ValueError("xy_tolerance_m must be > 0")
        if repeatability_scale < 0 or repeatability_scale > 1:
            raise ValueError("repeatability_scale must be in [0, 1]")

        self.xy_tol            = xy_tolerance_m
        self.depth_tol         = depth_tolerance_m
        self.min_medium        = min_scans_for_medium
        self.min_high          = min_scans_for_high
        self.rep_scale         = repeatability_scale
        self.noise_labels      = noise_labels or {"NOISE", "NOISE_ONLY"}
        self._clusterer        = _SpatialClusterer(xy_tolerance_m, depth_tolerance_m)

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(self, inputs: list[FusionInput]) -> FusionResult:
        """
        Fuse anomaly evidence from multiple scan inputs.

        Parameters
        ----------
        inputs : list[FusionInput]
            One FusionInput per scan. Order does not matter. Minimum 1.

        Returns
        -------
        FusionResult
            All fused targets plus session-level diagnostics.
        """
        if not inputs:
            return FusionResult(
                targets=[], n_scans_fused=0, n_anomalies_in=0,
                n_clusters=0, n_fused=0, n_singletons=0,
                fusion_timestamp=_utcnow_iso(), warnings=["No inputs provided"],
            )

        warnings: list[str] = []

        # ── Step 1: Convert all anomalies to metre-space evidence ─────────────
        all_evidence: list[_AnomalyEvidence] = []
        n_raw = 0
        for inp in inputs:
            evs, w = self._extract_evidence(inp, warnings)
            all_evidence.extend(evs)
            n_raw += len(inp.detection_result.anomalies)

        if not all_evidence:
            warnings.append(
                "All anomalies were filtered (noise labels or empty scans). "
                "Nothing to fuse."
            )
            return FusionResult(
                targets=[], n_scans_fused=len(inputs), n_anomalies_in=n_raw,
                n_clusters=0, n_fused=0, n_singletons=0,
                fusion_timestamp=_utcnow_iso(), warnings=warnings,
            )

        # ── Step 2: Spatial clustering ────────────────────────────────────────
        clusters = self._clusterer.cluster(all_evidence)
        logger.info(f"[Fusion] {len(all_evidence)} evidence points → "
                    f"{len(clusters)} clusters from {len(inputs)} scans")

        # ── Step 3: Fuse each cluster → FusedTarget ───────────────────────────
        targets: list[FusedTarget] = []
        n_singletons = 0
        n_fused = 0

        for c_idx, cluster in enumerate(clusters):
            target, cluster_warnings = self._fuse_cluster(c_idx, cluster)
            warnings.extend(cluster_warnings)
            targets.append(target)

            unique_scans = len({ev.scan_id for ev in cluster})
            if unique_scans < 2:
                n_singletons += 1
            else:
                n_fused += 1

        # ── Step 4: Sort — high confidence first ─────────────────────────────
        targets.sort(key=lambda t: t.confidence, reverse=True)

        return FusionResult(
            targets=targets,
            n_scans_fused=len(inputs),
            n_anomalies_in=n_raw,
            n_clusters=len(clusters),
            n_fused=n_fused,
            n_singletons=n_singletons,
            fusion_timestamp=_utcnow_iso(),
            warnings=warnings,
        )

    def explain(self, target: FusedTarget) -> str:
        """
        Render a human-readable explanation of how a FusedTarget's confidence
        was computed.  Suitable for the Inspector panel "Why this confidence?"
        tooltip.
        """
        if target.diagnostics is None:
            return f"[{target.fused_id}] No diagnostics available."

        d = target.diagnostics
        lines = [
            f"  Target {target.fused_id}  —  {target.fusion_tier} confidence tier",
            f"  {'─' * 52}",
            f"  Contributing scans : {d.n_contributing}",
            f"  Scan IDs           : {', '.join(d.scan_ids)}",
            f"",
            f"  ┌─ Confidence components ─────────────────────────┐",
            f"  │  Mean confidence (raw)     : {d.mean_confidence_raw:.3f}",
            f"  │  Weighted mean confidence  : {d.weighted_confidence:.3f}",
            f"  │  Uncertainty penalty       : ×{d.uncertainty_penalty:.3f}",
            f"  │  Repeatability bonus       : +{d.repeatability_bonus * self.rep_scale:.3f}",
            f"  │  Label agreement weight    : ×{d.label_agreement_weight:.3f}",
            f"  │  ──────────────────────────────────────────",
            f"  │  Final fused confidence    : {d.final_confidence:.3f}",
            f"  │  Confidence gain vs best   : {d.confidence_gain:+.3f}",
            f"  └─────────────────────────────────────────────────┘",
            f"",
            f"  ┌─ Spatial ───────────────────────────────────────┐",
            f"  │  Position spread (XY std)  : {d.position_spread_m:.4f} m",
        ]
        if d.has_depth and d.depth_spread_m is not None:
            lines.append(f"  │  Depth spread (std)        : {d.depth_spread_m:.4f} m")
        else:
            lines.append(f"  │  Depth                     : unavailable (calibration required)")
        lines += [
            f"  └─────────────────────────────────────────────────┘",
            f"",
        ]
        if d.label_votes:
            lines.append(f"  Label votes:")
            for lbl, cnt in sorted(d.label_votes.items(), key=lambda x: -x[1]):
                lines.append(f"    {lbl:<22} : {cnt} scan(s)")
        if d.warnings:
            lines.append(f"")
            lines.append(f"  Warnings:")
            for w in d.warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)

    # ── Evidence extraction ───────────────────────────────────────────────────

    def _extract_evidence(
        self, inp: FusionInput, warnings: list[str]
    ) -> tuple[list[_AnomalyEvidence], list[str]]:
        """
        Convert every non-noise anomaly in a DetectionResult to
        _AnomalyEvidence in real-world metres.

        Coordinate conversion (grid index → metres):
          x_m = anomaly.cx * geometry.sample_distance_m
          y_m = anomaly.cy * geometry.line_spacing_m

        When geometry_is_metres=True, cx/cy are already in metres.

        Returns (evidence_list, local_warnings).
        """
        det  = inp.detection_result
        geo  = inp.geometry
        evs: list[_AnomalyEvidence] = []
        local_warnings: list[str] = []

        # Determine scan quality weight
        q_weight = self._quality_weight(inp)

        for a in det.anomalies:
            # Filter noise
            if a.raw_label in self.noise_labels:
                continue

            # Coordinate conversion
            if inp.geometry_is_metres:
                x_m = float(a.cx)
                y_m = float(a.cy)
            elif geo is not None:
                dx = getattr(geo, "sample_distance_m", None) or getattr(geo, "dx", lambda: None)()
                dy = getattr(geo, "line_spacing_m",   None) or getattr(geo, "dy", lambda: None)()
                if dx is None or dy is None:
                    local_warnings.append(
                        f"[{det.scan_id}] geometry missing sample_distance_m / "
                        f"line_spacing_m — using cx/cy as metres"
                    )
                    x_m = float(a.cx)
                    y_m = float(a.cy)
                else:
                    x_m = float(a.cx) * float(dx)
                    y_m = float(a.cy) * float(dy)
            else:
                # No geometry — treat indices as metres with a warning
                if not getattr(self, "_geo_warned_" + det.scan_id, False):
                    local_warnings.append(
                        f"[{det.scan_id}] No geometry provided — "
                        f"cx/cy treated as metres (spatial accuracy unverified)"
                    )
                    setattr(self, "_geo_warned_" + det.scan_id, True)
                x_m = float(a.cx)
                y_m = float(a.cy)

            evs.append(_AnomalyEvidence(
                anomaly_id=a.anomaly_id,
                scan_id=det.scan_id,
                timestamp=inp.timestamp,
                x_m=x_m,
                y_m=y_m,
                depth_m=None,         # DepthInversionPlugin not yet calibrated
                confidence=float(np.clip(a.confidence, 0.0, 1.0)),
                uncertainty=float(np.clip(a.uncertainty, 0.0, 1.0)),
                label=a.raw_label,
                snr=float(a.snr_robust),
                quality_weight=q_weight,
            ))

        return evs, local_warnings

    def _quality_weight(self, inp: FusionInput) -> float:
        """
        Derive a [0, 1] scan quality weight.

        Priority:
          1. ScanReliability.penalty_factor   (most informed)
          2. DetectionResult.scan_quality_score
          3. 1.0 (unknown — do not penalise)
        """
        if inp.reliability is not None:
            pf = getattr(inp.reliability, "penalty_factor", None)
            if pf is not None:
                return float(np.clip(pf, 0.0, 1.0))
        qs = getattr(inp.detection_result, "scan_quality_score", None)
        if qs is not None and qs > 0:
            return float(np.clip(qs, 0.0, 1.0))
        return 1.0

    # ── Cluster fusion ────────────────────────────────────────────────────────

    def _fuse_cluster(
        self, c_idx: int, cluster: list[_AnomalyEvidence]
    ) -> tuple[FusedTarget, list[str]]:
        """
        Apply the 5-component confidence formula to one cluster and produce
        a FusedTarget with full FusionDiagnostics.

        Formula
        -------
        1. quality_weighted_confidence = Σ(w_i × conf_i) / Σ(w_i)
        2. uncertainty_penalty         = f(mean_uncertainty, position_spread)
        3. repeatability_bonus         = g(n_unique_scans)
        4. label_agreement_weight      = label_agreement_fraction^0.5
        5. final_confidence = clip(
             weighted_conf
             × uncertainty_penalty
             × label_agreement_weight
             + repeatability_bonus × rep_scale
           , 0, 1)
        """
        warnings: list[str] = []

        # ── Unique scan count (cross-scan confirmation) ────────────────────
        unique_scan_ids = sorted({ev.scan_id for ev in cluster})
        n_unique = len(unique_scan_ids)

        # ── Positional centroid (quality-weighted mean) ────────────────────
        weights  = np.array([ev.quality_weight for ev in cluster], dtype=float)
        xs       = np.array([ev.x_m           for ev in cluster], dtype=float)
        ys       = np.array([ev.y_m           for ev in cluster], dtype=float)
        w_sum    = weights.sum()
        if w_sum < 1e-9:
            w_norm = np.ones(len(cluster)) / len(cluster)
        else:
            w_norm = weights / w_sum

        cx_m = float(np.dot(w_norm, xs))
        cy_m = float(np.dot(w_norm, ys))

        # ── Depth (None if all uncalibrated) ──────────────────────────────
        depths = [ev.depth_m for ev in cluster if ev.depth_m is not None]
        has_depth   = len(depths) > 0
        depth_m     = float(np.mean(depths))   if has_depth else None
        depth_std   = float(np.std(depths))    if has_depth else None
        depth_unc_m = depth_std                if has_depth else None

        # ── Position spread ────────────────────────────────────────────────
        xs_unique = np.array([ev.x_m for ev in cluster], dtype=float)
        ys_unique = np.array([ev.y_m for ev in cluster], dtype=float)
        pos_spread = float(np.sqrt(np.var(xs_unique) + np.var(ys_unique)))

        # ── Raw and weighted confidence ────────────────────────────────────
        confs   = [ev.confidence    for ev in cluster]
        uncs    = [ev.uncertainty   for ev in cluster]
        qws     = [ev.quality_weight for ev in cluster]

        mean_conf_raw     = float(np.mean(confs))
        weighted_conf     = _quality_weighted_mean(confs, qws)
        mean_unc          = float(np.mean(uncs))
        best_single_conf  = float(max(confs))

        # ── Confidence components ──────────────────────────────────────────
        unc_penalty  = _uncertainty_penalty(mean_unc, pos_spread, self.xy_tol)
        rep_bonus    = _repeatability_bonus(n_unique)
        best_label, _, label_agree = _most_common([ev.label for ev in cluster])
        # Square-root softening: 0.7 agreement → 0.84 weight (not full 30% cut)
        label_weight = float(np.sqrt(max(label_agree, 0.0)))

        # ── Final confidence ───────────────────────────────────────────────
        base = weighted_conf * unc_penalty * label_weight
        final_conf = float(np.clip(base + rep_bonus * self.rep_scale, 0.0, 1.0))
        conf_gain  = final_conf - best_single_conf

        # ── Fusion tier ────────────────────────────────────────────────────
        tier = _fusion_tier(n_unique)
        # Decision tier is computed after final_conf is known (below) and
        # stored on the FusedTarget. We compute it here post-formula for clarity.
        # (assignment happens at FusedTarget construction)

        # ── Repeatability score ────────────────────────────────────────────
        # Combines detection consistency with positional consistency.
        # detection_rate: fraction of all input scans where this target appeared
        # spatial_rate:   1 - normalised position spread
        detection_rate  = float(np.clip(n_unique / max(n_unique, 1), 0.0, 1.0))
        tol_norm_spread = float(np.clip(pos_spread / max(self.xy_tol, 1e-9), 0.0, 1.0))
        repeatability   = float(np.clip(
            0.6 * detection_rate + 0.4 * (1.0 - tol_norm_spread),
            0.0, 1.0
        ))

        # ── Fused uncertainty ──────────────────────────────────────────────
        # Decreases with more scans (averaging effect) and spatial consistency
        fused_unc = float(np.clip(
            mean_unc * (1.0 - 0.2 * min(n_unique - 1, 4) / 4.0),
            0.0, 1.0
        ))

        # ── Warnings ──────────────────────────────────────────────────────
        if n_unique == 1:
            warnings.append(
                f"Cluster C{c_idx:03d} has only 1 contributing scan. "
                f"Cannot confirm — tier=LOW."
            )
        if label_agree < 0.6 and n_unique >= 2:
            warnings.append(
                f"Cluster C{c_idx:03d}: low label agreement ({label_agree:.0%}). "
                f"Scans disagree on anomaly type — inspect manually."
            )
        if pos_spread > self.xy_tol * 0.8 and n_unique >= 2:
            warnings.append(
                f"Cluster C{c_idx:03d}: high positional spread ({pos_spread:.3f} m). "
                f"Consider re-registering scans."
            )

        # ── Label votes ───────────────────────────────────────────────────
        label_votes: dict[str, int] = {}
        for ev in cluster:
            label_votes[ev.label] = label_votes.get(ev.label, 0) + 1

        # ── Build diagnostics ─────────────────────────────────────────────
        fused_id = f"FT_{c_idx:04d}_{uuid.uuid4().hex[:6]}"

        diag = FusionDiagnostics(
            fused_id=fused_id,
            n_contributing=n_unique,
            scan_ids=unique_scan_ids,
            timestamps=[ev.timestamp for ev in cluster if ev.scan_id in unique_scan_ids],
            raw_confidences=confs,
            raw_uncertainties=uncs,
            scan_quality_weights=qws,
            mean_confidence_raw=round(mean_conf_raw, 4),
            weighted_confidence=round(weighted_conf, 4),
            uncertainty_penalty=round(unc_penalty, 4),
            repeatability_bonus=round(rep_bonus, 4),
            label_agreement=round(label_agree, 4),
            label_agreement_weight=round(label_weight, 4),
            final_confidence=round(final_conf, 4),
            confidence_gain=round(conf_gain, 4),
            fusion_tier=tier,
            position_spread_m=round(pos_spread, 6),
            depth_spread_m=round(depth_std, 6) if depth_std is not None else None,
            has_depth=has_depth,
            label_votes=label_votes,
            warnings=warnings,
        )

        dec_tier = _decision_tier(tier, round(final_conf, 4))

        target = FusedTarget(
            fused_id=fused_id,
            x=round(cx_m, 4),
            y=round(cy_m, 4),
            depth_m=round(depth_m, 4) if depth_m is not None else None,
            depth_uncertainty_m=round(depth_unc_m, 4) if depth_unc_m is not None else None,
            confidence=round(final_conf, 4),
            uncertainty=round(fused_unc, 4),
            fusion_tier=tier,
            decision_tier=dec_tier,
            supporting_scans=unique_scan_ids,
            n_scans=n_unique,
            repeatability_score=round(repeatability, 4),
            label=best_label or "UNKNOWN",
            label_agreement=round(label_agree, 4),
            timestamps=sorted({ev.timestamp for ev in cluster if ev.timestamp}),
            diagnostics=diag,
        )

        return target, warnings
