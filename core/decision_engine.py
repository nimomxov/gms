"""
GMS Core — Cross-Scan Validation & Decision Engine
Correlates anomalies across multiple independent scans.
Final decision: DIG | RESCAN | NO_DIG

Scientific constraint:
  - DIG requires ≥2 independent scan confirmations
  - Uncertainty penalizes confidence explicitly
  - No single-scan "DIG" permitted
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .anomaly_detection import DetectionResult, Anomaly

logger = logging.getLogger("gms.decision")


@dataclass
class ConfirmedAnomaly:
    """Anomaly confirmed across multiple scans."""
    group_id: str
    centroid_x: float
    centroid_y: float
    scan_confirmations: int
    contributing_scans: list[str]
    mean_confidence: float
    mean_snr: float
    mean_uncertainty: float
    best_label: str          # most common label across confirmations
    label_agreement: float   # fraction of scans agreeing on label
    combined_confidence: float
    spatial_consistency: float  # how well positions agree [0, 1]


@dataclass
class FinalReport:
    """Final analysis report for a multi-scan session."""
    session_id: str
    decision: str            # DIG | RESCAN | NO_DIG
    confirmed_anomalies: list[ConfirmedAnomaly]
    single_detections: list[Anomaly]   # not confirmed across scans
    confidence_summary: dict
    scan_quality: dict
    warnings: list[str]
    n_scans_processed: int


def _spatial_distance(a1: Anomaly, a2: Anomaly) -> float:
    """Euclidean distance between anomaly centroids (grid cells)."""
    return np.sqrt((a1.cx - a2.cx) ** 2 + (a1.cy - a2.cy) ** 2)


def _label_agreement(labels: list[str]) -> tuple[str, float]:
    """Most common label and agreement fraction."""
    if not labels:
        return "UNKNOWN", 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts[best] / len(labels)


class CrossScanValidator:
    """
    Matches anomalies across scans by spatial proximity.
    Requires positional overlap within a tolerance window.
    """

    def __init__(self, config: dict, proximity_cells: float = 8.0):
        self.proximity = proximity_cells
        self.dec_cfg = config.get("decision", {})

    def validate(self,
                 results: list[DetectionResult],
                 session_id: str = "session") -> FinalReport:

        logger.info(f"Cross-scan validation: {len(results)} scans")
        warnings = []
        all_anomalies = []

        for r in results:
            all_anomalies.extend([(r.scan_id, a) for a in r.anomalies
                                  if a.raw_label not in ("NOISE",)])

        # ── Build proximity groups ───────────────────────────────────────────
        groups = []  # list of lists of (scan_id, Anomaly)
        used = set()

        for i, (sid_i, ai) in enumerate(all_anomalies):
            if i in used:
                continue
            group = [(sid_i, ai)]
            used.add(i)
            for j, (sid_j, aj) in enumerate(all_anomalies):
                if j in used or j == i:
                    continue
                if sid_j == sid_i:
                    continue  # same scan — don't self-confirm
                dist = _spatial_distance(ai, aj)
                if dist <= self.proximity:
                    group.append((sid_j, aj))
                    used.add(j)
            groups.append(group)

        confirmed = []
        single = []

        for g_idx, group in enumerate(groups):
            scan_ids_in_group = [sid for sid, _ in group]
            unique_scans = list(set(scan_ids_in_group))
            n_confirmations = len(unique_scans)

            anomalies_in_group = [a for _, a in group]
            labels = [a.raw_label for a in anomalies_in_group]
            best_label, label_agree = _label_agreement(labels)

            mean_conf = float(np.mean([a.confidence for a in anomalies_in_group]))
            mean_snr = float(np.mean([a.snr_robust for a in anomalies_in_group]))
            mean_unc = float(np.mean([a.uncertainty for a in anomalies_in_group]))

            cx = float(np.mean([a.cx for a in anomalies_in_group]))
            cy = float(np.mean([a.cy for a in anomalies_in_group]))

            # Spatial consistency: how tightly clustered
            if len(anomalies_in_group) > 1:
                dists = [_spatial_distance(a, anomalies_in_group[0])
                         for a in anomalies_in_group[1:]]
                max_spread = max(dists) if dists else 0
                spatial_consistency = float(np.clip(1.0 - max_spread / self.proximity, 0, 1))
            else:
                spatial_consistency = 1.0

            # Combined confidence: penalize uncertainty and label disagreement
            combined_conf = mean_conf * (1 - 0.3 * mean_unc) * (0.5 + 0.5 * label_agree)

            if n_confirmations >= 2:
                confirmed.append(ConfirmedAnomaly(
                    group_id=f"G{g_idx:03d}",
                    centroid_x=round(cx, 2),
                    centroid_y=round(cy, 2),
                    scan_confirmations=n_confirmations,
                    contributing_scans=unique_scans,
                    mean_confidence=round(mean_conf, 3),
                    mean_snr=round(mean_snr, 3),
                    mean_uncertainty=round(mean_unc, 3),
                    best_label=best_label,
                    label_agreement=round(label_agree, 3),
                    combined_confidence=round(float(np.clip(combined_conf, 0, 1)), 3),
                    spatial_consistency=round(spatial_consistency, 3),
                ))
            else:
                # Not confirmed — collect original anomaly objects
                single.extend(anomalies_in_group)

        logger.info(f"  {len(confirmed)} confirmed groups, "
                    f"{len(single)} single detections")

        # ── Final Decision ───────────────────────────────────────────────────
        decision = self._make_decision(confirmed, results, warnings)

        # ── Confidence Summary ───────────────────────────────────────────────
        confidence_summary = self._build_confidence_summary(confirmed, single)

        # ── Scan Quality ─────────────────────────────────────────────────────
        scan_quality = {
            r.scan_id: {
                "quality_score": r.scan_quality_score,
                "noise_floor": round(r.noise_floor, 4),
                "n_anomalies_raw": len(r.anomalies),
                "warnings": r.warnings,
            }
            for r in results
        }

        return FinalReport(
            session_id=session_id,
            decision=decision,
            confirmed_anomalies=confirmed,
            single_detections=single,
            confidence_summary=confidence_summary,
            scan_quality=scan_quality,
            warnings=warnings,
            n_scans_processed=len(results),
        )

    def _make_decision(self, confirmed: list[ConfirmedAnomaly],
                       results: list[DetectionResult],
                       warnings: list[str]) -> str:
        dig_cfg = self.dec_cfg.get("DIG", {})
        rescan_cfg = self.dec_cfg.get("RESCAN", {})

        min_conf = dig_cfg.get("min_confidence", 0.70)
        max_unc = dig_cfg.get("max_uncertainty", 0.25)
        min_snr = dig_cfg.get("snr_min", 4.0)
        min_confirmations = dig_cfg.get("min_scan_confirmations", 2)

        # DIG candidates: confirmed + high confidence
        dig_candidates = [
            c for c in confirmed
            if (c.scan_confirmations >= min_confirmations
                and c.combined_confidence >= min_conf
                and c.mean_uncertainty <= max_unc
                and c.mean_snr >= min_snr
                and c.best_label in ("FERROUS_METAL", "CAVITY"))
        ]

        if dig_candidates:
            logger.info(f"  Decision: DIG ({len(dig_candidates)} qualifying anomalies)")
            return "DIG"

        # RESCAN: partial evidence
        rescan_min_conf = rescan_cfg.get("min_confidence", 0.45)
        rescan_min_confirm = rescan_cfg.get("min_scan_confirmations", 1)

        rescan_candidates = [
            c for c in confirmed
            if c.combined_confidence >= rescan_min_conf
        ]
        # Also check single detections with decent confidence
        strong_singles = [
            a for r in results for a in r.anomalies
            if a.confidence >= 0.50 and a.raw_label not in ("NOISE", "SOIL_VARIATION")
        ]

        if rescan_candidates or strong_singles:
            if len(results) < 2:
                warnings.append(
                    "Only 1 scan provided — DIG decision requires ≥2 independent scans"
                )
            logger.info("  Decision: RESCAN")
            return "RESCAN"

        logger.info("  Decision: NO_DIG")
        return "NO_DIG"

    def _build_confidence_summary(self,
                                   confirmed: list[ConfirmedAnomaly],
                                   singles: list[Anomaly]) -> dict:
        if not confirmed and not singles:
            return {"overall": 0.0, "max_anomaly_confidence": 0.0,
                    "n_confirmed": 0, "n_single": 0}

        all_confs = [c.combined_confidence for c in confirmed]
        all_confs += [a.confidence for a in singles]

        label_counts = {}
        for c in confirmed:
            label_counts[c.best_label] = label_counts.get(c.best_label, 0) + 1

        return {
            "overall": round(float(np.mean(all_confs)) if all_confs else 0.0, 3),
            "max_anomaly_confidence": round(float(max(all_confs)) if all_confs else 0.0, 3),
            "n_confirmed": len(confirmed),
            "n_single": len(singles),
            "label_distribution": label_counts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fusion-Aware Final Decision
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FinalDecision:
    """
    Authoritative GMS session decision — produced after fusion.

    Replaces FinalReport.decision as the definitive recommendation.

    Decision hierarchy (highest to lowest priority):
      1. FusionResult (multi-scan, registration-corrected evidence)
      2. CrossScanValidator FinalReport (grid-index proximity matching)
      3. Single-scan fallback

    Evidence source is always recorded so operators know which layer
    produced the decision.
    """
    decision:          str         # DIG | RESCAN | NO_DIG
    evidence_source:   str         # "fusion" | "cross_scan" | "single_scan"
    confidence:        float       # primary confidence value [0, 1]
    n_targets:         int         # how many targets support the decision
    primary_target_id: str         # fused_id or group_id of the top target
    decision_tier:     str         # CONFIRMED | PROBABLE | WEAK | REJECTED | N/A
    warnings:          list
    detail:            dict        # full evidence breakdown for audit trail


class FusionAwareDecisionEngine:
    """
    Derives the final DIG / RESCAN / NO_DIG recommendation by consulting
    FusionResult as the highest-priority evidence source.

    Priority chain:
      ┌─────────────────────────────────────────────────────────┐
      │ 1. FusionResult present?                                 │
      │    Yes → use FusedTarget.decision_tier + confidence      │
      │           CONFIRMED  + conf ≥ DIG_min     → DIG          │
      │           CONFIRMED  + conf < DIG_min     → RESCAN       │
      │           PROBABLE                         → RESCAN       │
      │           WEAK / REJECTED                  → NO_DIG       │
      │    No  → fall through to step 2                          │
      │                                                          │
      │ 2. CrossScanValidator FinalReport present?               │
      │    Yes → use report.decision directly (existing logic)   │
      │                                                          │
      │ 3. Neither → NO_DIG                                     │
      └─────────────────────────────────────────────────────────┘

    Scientific constraints preserved:
      - DIG requires fusion_tier == HIGH (≥3 scans) OR ≥2 scan confirmations
        in CrossScanValidator — never from a single scan alone.
      - Single-scan anomalies (LOW tier / REJECTED) cannot produce DIG.
      - A REJECTED target (LOW tier + low confidence) maps to NO_DIG, not RESCAN.
      - All confidence thresholds are read from gms_config.yaml decision section.
    """

    def __init__(self, config: dict):
        self._cfg    = config.get("decision", {})
        self._dig    = self._cfg.get("DIG",    {})
        self._rescan = self._cfg.get("RESCAN", {})

    def decide(self,
               fusion_result=None,
               cross_scan_report=None) -> "FinalDecision":
        """
        Produce the authoritative session decision.

        Parameters
        ----------
        fusion_result : FusionResult | None
            Output of MultiScanFusionEngine.fuse(). When provided, this is
            the primary evidence source.
        cross_scan_report : FinalReport | None
            Output of CrossScanValidator.validate(). Used as fallback when
            fusion_result is None.

        Returns
        -------
        FinalDecision
            The authoritative recommendation with full audit trail.
        """
        # ── Priority 1: FusionResult ──────────────────────────────────────
        if fusion_result is not None and len(fusion_result.targets) > 0:
            return self._decide_from_fusion(fusion_result)

        # ── Priority 2: CrossScanValidator report ─────────────────────────
        if cross_scan_report is not None:
            return self._decide_from_cross_scan(cross_scan_report)

        # ── Priority 3: No evidence ───────────────────────────────────────
        return FinalDecision(
            decision="NO_DIG",
            evidence_source="none",
            confidence=0.0,
            n_targets=0,
            primary_target_id="",
            decision_tier="N/A",
            warnings=["No fusion result and no cross-scan report available"],
            detail={},
        )

    # ── Fusion path ───────────────────────────────────────────────────────

    def _decide_from_fusion(self, fusion_result) -> "FinalDecision":
        """
        Derive decision from FusionResult.

        Fully backward-compatible: works with both modern FusedTarget
        (has decision_tier, to_dict(), fused_id) and old FusedTarget objects
        (fusion_tier + confidence only, no decision_tier, no to_dict()).
        All attribute access uses getattr() with safe defaults.
        """
        dig_min_conf = self._dig.get("min_confidence", 0.70)
        dig_max_unc  = self._dig.get("max_uncertainty", 0.25)

        warnings = list(getattr(fusion_result, "warnings", []))
        targets  = list(getattr(fusion_result, "targets",  []))

        # ── backward-compatible tier resolver ─────────────────────────────
        def _tier(t) -> str:
            dt = getattr(t, "decision_tier", None)
            if dt is not None:
                return dt
            ft   = getattr(t, "fusion_tier", "LOW")
            conf = getattr(t, "confidence",  0.0)
            if ft == "HIGH" and conf >= 0.70:
                return "CONFIRMED"
            if ft == "MEDIUM" or (ft == "HIGH" and conf < 0.70):
                return "PROBABLE"
            if ft == "LOW" and conf >= 0.45:
                return "WEAK"
            return "REJECTED"

        # ── helpers that avoid calling non-existent methods ───────────────
        def _n(tier: str) -> int:
            prop = {"CONFIRMED": "confirmed_targets", "PROBABLE": "probable_targets",
                    "WEAK": "weak_targets", "REJECTED": "rejected_targets"}.get(tier, "")
            if prop and hasattr(fusion_result, prop):
                return len(getattr(fusion_result, prop))
            return sum(1 for t in targets if _tier(t) == tier)

        def _target_dict(t) -> dict:
            if hasattr(t, "to_dict"):
                return t.to_dict()
            return {
                "fused_id":    getattr(t, "fused_id",    ""),
                "fusion_tier": getattr(t, "fusion_tier", ""),
                "confidence":  getattr(t, "confidence",  0.0),
                "uncertainty": getattr(t, "uncertainty", 1.0),
            }

        def _summary_detail() -> dict:
            d: dict = {}
            if hasattr(fusion_result, "summary"):
                try:
                    s = fusion_result.summary()
                    d["fusion_tier_counts"]   = s.get("tier_counts", {})
                    d["decision_tier_counts"] = s.get("decision_tier_counts", {})
                except Exception:
                    pass
            return d

        # ── DIG ────────────────────────────────────────────────────────────
        dig_target = next(
            (t for t in targets
             if _tier(t) == "CONFIRMED"
             and getattr(t, "confidence",  0.0) >= dig_min_conf
             and getattr(t, "uncertainty", 1.0) <= dig_max_unc),
            None,
        )
        if dig_target is not None:
            detail = {
                "top_target":  _target_dict(dig_target),
                "n_confirmed": _n("CONFIRMED"),
                "n_probable":  _n("PROBABLE"),
                "n_weak":      _n("WEAK"),
                "n_rejected":  _n("REJECTED"),
            }
            detail.update(_summary_detail())
            return FinalDecision(
                decision="DIG",
                evidence_source="fusion",
                confidence=getattr(dig_target, "confidence", 0.0),
                n_targets=_n("CONFIRMED"),
                primary_target_id=getattr(dig_target, "fused_id", ""),
                decision_tier=_tier(dig_target),
                warnings=warnings,
                detail=detail,
            )

        # ── RESCAN ──────────────────────────────────────────────────────────
        rescan_target = next(
            (t for t in targets
             if _tier(t) in ("CONFIRMED", "PROBABLE", "WEAK")),
            None,
        )
        if rescan_target is not None:
            reason = {
                "CONFIRMED": "CONFIRMED target below DIG confidence threshold",
                "PROBABLE":  "PROBABLE target (2 scans — cross-confirmed but below HIGH tier)",
                "WEAK":      "WEAK target (single scan — unconfirmed)",
            }.get(_tier(rescan_target), "")
            if len(targets) < 2:
                warnings.append(
                    "Only 1 scan in fusion session — "
                    "DIG requires fusion_tier=HIGH (≥3 scans)"
                )
            return FinalDecision(
                decision="RESCAN",
                evidence_source="fusion",
                confidence=getattr(rescan_target, "confidence", 0.0),
                n_targets=len(targets),
                primary_target_id=getattr(rescan_target, "fused_id", ""),
                decision_tier=_tier(rescan_target),
                warnings=warnings,
                detail={
                    "reason":      reason,
                    "top_target":  _target_dict(rescan_target),
                    "n_confirmed": _n("CONFIRMED"),
                    "n_probable":  _n("PROBABLE"),
                    "n_weak":      _n("WEAK"),
                    "n_rejected":  _n("REJECTED"),
                },
            )

        # ── NO_DIG ─────────────────────────────────────────────────────────
        top = targets[0] if targets else None
        return FinalDecision(
            decision="NO_DIG",
            evidence_source="fusion",
            confidence=getattr(top, "confidence", 0.0) if top else 0.0,
            n_targets=len(targets),
            primary_target_id=getattr(top, "fused_id", "") if top else "",
            decision_tier="REJECTED",
            warnings=warnings,
            detail={
                "reason":     "All fused targets are REJECTED (LOW tier, low confidence)",
                "n_rejected": _n("REJECTED"),
            },
        )

    # ── Cross-scan fallback path ──────────────────────────────────────────

    def _decide_from_cross_scan(self, report) -> "FinalDecision":
        """Wrap CrossScanValidator FinalReport in a FinalDecision."""
        top_conf = 0.0
        top_id   = ""
        if report.confirmed_anomalies:
            best = max(report.confirmed_anomalies,
                       key=lambda c: c.combined_confidence)
            top_conf = best.combined_confidence
            top_id   = best.group_id
        elif report.single_detections:
            best = max(report.single_detections, key=lambda a: a.confidence)
            top_conf = best.confidence
            top_id   = best.anomaly_id

        return FinalDecision(
            decision=report.decision,
            evidence_source="cross_scan",
            confidence=top_conf,
            n_targets=len(report.confirmed_anomalies),
            primary_target_id=top_id,
            decision_tier="N/A",
            warnings=report.warnings,
            detail={
                "n_confirmed":  len(report.confirmed_anomalies),
                "n_single":     len(report.single_detections),
                "n_scans":      report.n_scans_processed,
                "confidence_summary": report.confidence_summary,
            },
        )
