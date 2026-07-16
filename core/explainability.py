"""
GMS — Explainability Engine  v1.0
====================================
Generates structured, human-readable explanations for every
pipeline decision. Not a chatbot — a deterministic rule engine
that turns numeric scores into scientific prose.

Output structure:
  DecisionExplanation
    decision: str                 DIG | RESCAN | NO_DIG
    headline: str                 one-line summary
    supporting_facts: list[Fact]  each ✓ or ✗ with label + detail
    blocking_facts: list[Fact]    what is preventing a higher decision
    confidence_breakdown: dict    component → score
    reliability_narrative: str    plain-language reliability summary
    full_text: str                rendered for InspectorPanel

Scientific principle:
  Every fact maps directly to a measurable backend value.
  No inferred narratives. No vague language.
  Confidence bands are calibrated against benchmark thresholds.

Usage:
    from core.explainability import ExplainabilityEngine
    engine = ExplainabilityEngine(pipeline_config)
    exp = engine.explain(confirmed_anomaly, scan_quality, reliability)
    inspector_panel.set_explanation(exp.full_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("gms.explainability")


# ─────────────────────────────────────────────────────────────────────────────
# Fact — one supporting or blocking observation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    supports: bool          # True = supports decision, False = blocks it
    label: str              # short name, e.g. "Cross-scan confirmation"
    detail: str             # e.g. "Confirmed in 3 of 3 scans"
    metric_name: str = ""   # e.g. "scan_confirmations"
    metric_value: Any = None

    @property
    def icon(self) -> str:
        return "✓" if self.supports else "✗"

    def to_line(self) -> str:
        return f"{self.icon}  {self.label}: {self.detail}"


# ─────────────────────────────────────────────────────────────────────────────
# DecisionExplanation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionExplanation:
    decision: str
    headline: str
    target_label: str = ""
    supporting_facts: list[Fact] = field(default_factory=list)
    blocking_facts: list[Fact] = field(default_factory=list)
    confidence_breakdown: dict = field(default_factory=dict)
    reliability_narrative: str = ""
    full_text: str = ""

    def render(self) -> str:
        """Build the full inspector text block."""
        lines = [
            f"FINAL DECISION: {self.decision}",
            f"Target: {self.target_label}",
            "",
            self.headline,
            "",
        ]

        if self.supporting_facts:
            lines.append("Supporting evidence:")
            for f in self.supporting_facts:
                lines.append(f"  {f.to_line()}")

        if self.blocking_facts:
            lines.append("")
            lines.append("Limiting factors:")
            for f in self.blocking_facts:
                lines.append(f"  {f.to_line()}")

        if self.reliability_narrative:
            lines.append("")
            lines.append(f"Scan quality: {self.reliability_narrative}")

        if self.confidence_breakdown:
            lines.append("")
            lines.append("Confidence components:")
            for k, v in self.confidence_breakdown.items():
                bar = _confidence_bar(v)
                lines.append(f"  {k:<24} {bar}  {v:.0%}")

        self.full_text = "\n".join(lines)
        return self.full_text


# ─────────────────────────────────────────────────────────────────────────────
# ExplainabilityEngine
# ─────────────────────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Deterministic rule-based explainer.
    Maps numeric metrics → human-readable decision rationale.

    All thresholds mirror the PipelineConfig decision thresholds so
    the explanation is always consistent with the actual decision.
    """

    def __init__(self, pipeline_config=None):
        self._cfg = pipeline_config
        # Mirror decision thresholds from config or use defaults
        dec = {}
        if pipeline_config is not None:
            dec = getattr(pipeline_config, "decision", {})

        dig_cfg    = dec.get("DIG",    {})
        rescan_cfg = dec.get("RESCAN", {})

        self._dig_min_conf    = dig_cfg.get("min_confidence",          0.70)
        self._dig_min_snr     = dig_cfg.get("snr_min",                 4.0)
        self._dig_max_unc     = dig_cfg.get("max_uncertainty",          0.25)
        self._dig_min_confirm = dig_cfg.get("min_scan_confirmations",   2)
        self._res_min_conf    = rescan_cfg.get("min_confidence",        0.45)

    # ── Public API ─────────────────────────────────────────────────────────

    def explain_anomaly(
        self,
        anomaly,               # ConfirmedAnomaly or AnomalyInfo
        final_decision: str,
        reliability=None,      # ScanReliability
        scan_quality: dict = None,
    ) -> DecisionExplanation:
        """
        Build a complete explanation for a single anomaly.
        Safe to call with any object that has the expected attributes.
        """

        # ── Extract metrics (duck-typed for both ConfirmedAnomaly and AnomalyInfo)
        confidence    = _get(anomaly, "combined_confidence", "confidence", default=0.0)
        snr           = _get(anomaly, "mean_snr", "snr", default=0.0)
        uncertainty   = _get(anomaly, "mean_uncertainty", default=0.25)
        confirmations = _get(anomaly, "scan_confirmations", default=1)
        consistency   = _get(anomaly, "spatial_consistency", default=1.0)
        label_agree   = _get(anomaly, "label_agreement", default=1.0)
        target_label  = _get(anomaly, "best_label", "label", "target_type", default="UNKNOWN")
        fusion_boost  = _get(anomaly, "fusion_boost", default=0.0)
        topology      = _get(anomaly, "topology_status", default="")
        dipole_score  = _get(anomaly, "dipole_score", default=0.0)

        # ── Reliability metrics
        rel_score   = 1.0
        rel_label   = "GOOD"
        if reliability is not None:
            rel_score = getattr(reliability, "reliability_score",
                         getattr(reliability, "reliability", 1.0))
            rel_label = getattr(reliability, "quality_label", "UNKNOWN")

        # ── Classify facts
        supporting: list[Fact] = []
        blocking:   list[Fact] = []

        # 1. Cross-scan confirmation
        if confirmations >= self._dig_min_confirm:
            supporting.append(Fact(
                True, "Cross-scan confirmation",
                f"Detected in {confirmations} independent scan(s) "
                f"(minimum required: {self._dig_min_confirm})",
                "scan_confirmations", confirmations,
            ))
        else:
            blocking.append(Fact(
                False, "Cross-scan confirmation",
                f"Only {confirmations} scan(s) — DIG requires "
                f"≥{self._dig_min_confirm} independent scans",
                "scan_confirmations", confirmations,
            ))

        # 2. Signal-to-noise ratio
        if snr >= self._dig_min_snr:
            supporting.append(Fact(
                True, "Signal-to-noise ratio",
                f"{snr:.1f} dB — above DIG threshold ({self._dig_min_snr:.1f} dB)",
                "mean_snr", snr,
            ))
        elif snr >= 2.6:
            blocking.append(Fact(
                False, "Signal-to-noise ratio",
                f"{snr:.1f} dB — above detection floor but below DIG threshold "
                f"({self._dig_min_snr:.1f} dB)",
                "mean_snr", snr,
            ))
        else:
            blocking.append(Fact(
                False, "Signal-to-noise ratio",
                f"{snr:.1f} dB — weak signal, near noise floor",
                "mean_snr", snr,
            ))

        # 3. Confidence
        if confidence >= self._dig_min_conf:
            supporting.append(Fact(
                True, "Combined confidence",
                f"{confidence:.0%} — meets DIG threshold ({self._dig_min_conf:.0%})",
                "combined_confidence", confidence,
            ))
        elif confidence >= self._res_min_conf:
            blocking.append(Fact(
                False, "Combined confidence",
                f"{confidence:.0%} — meets RESCAN threshold but not DIG "
                f"({self._dig_min_conf:.0%})",
                "combined_confidence", confidence,
            ))
        else:
            blocking.append(Fact(
                False, "Combined confidence",
                f"{confidence:.0%} — below minimum threshold ({self._res_min_conf:.0%})",
                "combined_confidence", confidence,
            ))

        # 4. Uncertainty
        if uncertainty <= self._dig_max_unc:
            supporting.append(Fact(
                True, "Positional uncertainty",
                f"{uncertainty:.2f} — low, within acceptable range "
                f"(max: {self._dig_max_unc:.2f})",
                "mean_uncertainty", uncertainty,
            ))
        else:
            blocking.append(Fact(
                False, "Positional uncertainty",
                f"{uncertainty:.2f} — elevated, exceeds DIG limit "
                f"({self._dig_max_unc:.2f})",
                "mean_uncertainty", uncertainty,
            ))

        # 5. Spatial consistency
        if consistency >= 0.75:
            supporting.append(Fact(
                True, "Spatial consistency",
                f"{consistency:.0%} — anomaly position stable across scans",
                "spatial_consistency", consistency,
            ))
        else:
            blocking.append(Fact(
                False, "Spatial consistency",
                f"{consistency:.0%} — anomaly position varies between scans",
                "spatial_consistency", consistency,
            ))

        # 6. Topology
        if topology in ("coherent", "compact"):
            supporting.append(Fact(
                True, "Topology",
                f"Anomaly shape is {topology} — consistent with a real target",
                "topology_status", topology,
            ))
        elif topology in ("fragmented", "diffuse"):
            blocking.append(Fact(
                False, "Topology",
                f"Anomaly shape is {topology} — may indicate soil variation or noise",
                "topology_status", topology,
            ))

        # 7. Dipole score
        if dipole_score >= 0.65:
            supporting.append(Fact(
                True, "Dipole signature",
                f"Score {dipole_score:.2f} — strong dipole pattern matches ferrous target",
                "dipole_score", dipole_score,
            ))
        elif dipole_score >= 0.4:
            supporting.append(Fact(
                True, "Dipole signature",
                f"Score {dipole_score:.2f} — moderate dipole pattern detected",
                "dipole_score", dipole_score,
            ))

        # 8. Reliability penalty
        if rel_score >= 0.8:
            supporting.append(Fact(
                True, "Scan reliability",
                f"{rel_label} ({rel_score:.0%}) — low environmental noise",
                "reliability_score", rel_score,
            ))
        elif rel_score >= 0.45:
            blocking.append(Fact(
                False, "Scan reliability",
                f"{rel_label} ({rel_score:.0%}) — reliability penalty applied to confidence",
                "reliability_score", rel_score,
            ))
        else:
            blocking.append(Fact(
                False, "Scan reliability",
                f"{rel_label} ({rel_score:.0%}) — poor scan quality, "
                f"confidence heavily penalized",
                "reliability_score", rel_score,
            ))

        # ── Headline
        headline = _build_headline(final_decision, supporting, blocking)

        # ── Confidence breakdown
        confidence_breakdown = {
            "Base detection":    min(snr / 8.0, 1.0),
            "Cross-scan bonus":  min(confirmations / 3.0, 1.0),
            "Reliability":       rel_score,
            "Spatial consistency": consistency,
            "Label agreement":   label_agree,
        }

        # ── Reliability narrative
        rel_narrative = _reliability_narrative(rel_score, rel_label)

        # ── Build explanation
        exp = DecisionExplanation(
            decision=final_decision,
            headline=headline,
            target_label=target_label,
            supporting_facts=supporting,
            blocking_facts=blocking,
            confidence_breakdown=confidence_breakdown,
            reliability_narrative=rel_narrative,
        )
        exp.render()
        return exp

    def explain_session(
        self,
        result: dict,
        reliability=None,
    ) -> str:
        """
        Generate a one-paragraph session-level explanation.
        Used in status bar tooltips and report headers.
        """
        decision = result.get("decision", "UNKNOWN")
        n_conf   = len(result.get("confirmed_anomalies", []))
        n_scans  = result.get("n_scans_processed", 0)
        conf     = result.get("confidence_summary", {}).get("overall", 0.0)

        parts = [f"Decision: {decision}."]

        if decision == "DIG":
            parts.append(
                f"{n_conf} target(s) confirmed across {n_scans} independent scan(s) "
                f"with {conf:.0%} overall confidence."
            )
        elif decision == "RESCAN":
            parts.append(
                f"Anomalies detected but cross-scan confirmation is insufficient "
                f"for DIG. {n_scans} scan(s) processed."
            )
            if n_scans < 2:
                parts.append(
                    "DIG requires a minimum of 2 orthogonal scans over the same area."
                )
        else:
            parts.append(
                f"No significant targets detected across {n_scans} scan(s). "
                f"Overall signal confidence: {conf:.0%}."
            )

        if reliability is not None:
            rel_score = getattr(reliability, "reliability_score",
                         getattr(reliability, "reliability", 1.0))
            if rel_score < 0.45:
                parts.append(
                    f"WARNING: Scan reliability is poor ({rel_score:.0%}). "
                    f"Consider repeating the survey under better conditions."
                )

        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(obj, *attrs, default=None):
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
        if isinstance(obj, dict) and attr in obj:
            return obj[attr]
    return default


def _build_headline(decision: str, supporting: list[Fact], blocking: list[Fact]) -> str:
    n_sup = len(supporting)
    n_blk = len(blocking)
    if decision == "DIG":
        return (
            f"All {n_sup} DIG criteria met. "
            f"High confidence target — recommend excavation."
        )
    elif decision == "RESCAN":
        return (
            f"{n_sup} supporting factor(s), {n_blk} limiting factor(s). "
            f"Partial evidence — repeat scan recommended before excavation."
        )
    else:
        return (
            f"{n_blk} factor(s) prevent detection. "
            f"No credible targets identified in this area."
        )


def _reliability_narrative(score: float, label: str) -> str:
    if score >= 0.85:
        return f"Excellent ({label}) — results are highly trustworthy."
    elif score >= 0.70:
        return f"Good ({label}) — minor environmental noise present."
    elif score >= 0.45:
        return (
            f"Moderate ({label}) — environmental noise has reduced confidence. "
            f"Consider a repeat scan under calmer conditions."
        )
    else:
        return (
            f"Poor ({label}) — significant noise present. "
            f"Confidence values are unreliable. Repeat scan strongly recommended."
        )


def _confidence_bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"
