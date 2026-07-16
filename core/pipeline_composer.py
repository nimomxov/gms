"""
GMS — Dynamic Pipeline Composer  v2.3

Assembles the processing pipeline dynamically based on DeviceCapabilities.

BASIC device (signal only):
    SignalNormalizer → BasicDetector → LineVisualizer

STANDARD device (signal + position):
    SignalNormalizer → GridInterpolator → MedianBaseline → LoGDetector → HeatmapRenderer

ADVANCED device (+ SNR):
    + ConfidenceEngine → UncertaintyRadius

PROFESSIONAL device (+ heading + baseline + stability + noise_floor):
    + DriftCompensator → PathCorrector → AdaptiveBaseline → MatchedFilter

The composer NEVER activates a stage if the required telemetry is absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .schema.capabilities import DeviceCapabilities, TelemetryGrade

logger = logging.getLogger("gms.pipeline.composer")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Stage Descriptors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageDescriptor:
    """Describes a pipeline stage: what it needs, what it produces."""
    name: str
    requires_capabilities: list[str]   # cap attribute names that must be True
    description: str
    is_optional: bool = True           # if False: missing this stage is a pipeline error


@dataclass
class ComposedPipeline:
    """
    Result of pipeline composition: ordered list of enabled stages
    and explicit record of disabled stages with reasons.
    """
    grade: TelemetryGrade
    enabled_stages: list[str]         # ordered stage names
    disabled_stages: dict[str, str]   # stage_name → reason message
    warnings: list[str] = field(default_factory=list)

    def stage_is_active(self, name: str) -> bool:
        return name in self.enabled_stages

    def summary(self) -> str:
        active = " → ".join(self.enabled_stages)
        return f"[{self.grade.name}] {active}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage Registry
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STAGES: list[StageDescriptor] = [
    # Always required
    StageDescriptor(
        name="SignalNormalizer",
        requires_capabilities=[],
        description="Normalize raw signal values, remove global DC offset.",
        is_optional=False,
    ),

    # Requires position
    StageDescriptor(
        name="GridInterpolator",
        requires_capabilities=["can_interpolate_2d"],
        description="Interpolate scatter points onto a regular 2D grid (RBF / cubic).",
    ),

    # Requires position — runs after grid
    StageDescriptor(
        name="DriftCompensator",
        requires_capabilities=["can_drift_compensate"],
        description="Remove slow spatial drift using available baseline or position.",
    ),

    # Requires heading
    StageDescriptor(
        name="PathCorrector",
        requires_capabilities=["can_orientation_correct"],
        description="Correct scan path distortions using heading telemetry.",
    ),

    # Requires position
    StageDescriptor(
        name="AdaptiveBaseline",
        requires_capabilities=["can_adaptive_baseline"],
        description="Local adaptive baseline using MAD noise map over position grid.",
    ),

    # Always available
    StageDescriptor(
        name="MatchedFilter",
        requires_capabilities=["can_matched_filter"],
        description="NCC matched filter against dipole template bank.",
    ),

    # Requires SNR
    StageDescriptor(
        name="ConfidenceEngine",
        requires_capabilities=["can_compute_confidence"],
        description="Compute anomaly confidence scores weighted by SNR.",
    ),

    # Requires SNR
    StageDescriptor(
        name="UncertaintyRadius",
        requires_capabilities=["can_compute_uncertainty_radius"],
        description="Compute spatial uncertainty radius from SNR and stability.",
    ),

    # Always available — chooses mode based on capabilities
    StageDescriptor(
        name="AnomalyDetector",
        requires_capabilities=[],
        description="Detect anomalies: LoG (with position) or amplitude (without).",
        is_optional=False,
    ),

    # Always available — mode depends on position availability
    StageDescriptor(
        name="VisualizationRouter",
        requires_capabilities=[],
        description="Route to heatmap (with position) or line visualization (without).",
        is_optional=False,
    ),

    # Requires position
    StageDescriptor(
        name="HeatmapRenderer",
        requires_capabilities=["can_render_heatmap"],
        description="Scientific 2D heatmap with contours and anomaly markers.",
    ),

    # Requires position
    StageDescriptor(
        name="3DExplorer",
        requires_capabilities=["can_render_3d"],
        description="Interactive 3D terrain explorer with anomaly overlays.",
    ),
]

STAGE_MAP = {s.name: s for s in PIPELINE_STAGES}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Composer
# ─────────────────────────────────────────────────────────────────────────────

class DynamicPipelineComposer:
    """
    Assembles an executable pipeline description from DeviceCapabilities.

    Does NOT instantiate actual processing objects — that is the pipeline
    orchestrator's job. The composer only determines WHAT runs and WHY.
    """

    def compose(self, cap: DeviceCapabilities) -> ComposedPipeline:
        """
        Evaluate each pipeline stage against capabilities.
        Return a ComposedPipeline listing enabled and disabled stages.
        """
        enabled: list[str] = []
        disabled: dict[str, str] = {}
        warnings: list[str] = []

        for stage in PIPELINE_STAGES:
            ok, reason = self._check_stage(stage, cap)
            if ok:
                enabled.append(stage.name)
                logger.debug(f"  [+] {stage.name}")
            else:
                disabled[stage.name] = reason
                logger.debug(f"  [-] {stage.name}: {reason}")
                if not stage.is_optional:
                    warnings.append(
                        f"Required stage '{stage.name}' is disabled: {reason}"
                    )

        pipeline = ComposedPipeline(
            grade=cap.grade,
            enabled_stages=enabled,
            disabled_stages=disabled,
            warnings=warnings,
        )

        logger.info(f"[PipelineComposer] {pipeline.summary()}")
        for name, reason in disabled.items():
            logger.info(f"  [DISABLED] {name}: {reason}")

        return pipeline

    def _check_stage(
        self, stage: StageDescriptor, cap: DeviceCapabilities
    ) -> tuple[bool, str]:
        """Check if all required capabilities are present for this stage."""
        if not stage.requires_capabilities:
            return True, ""

        missing = []
        for attr in stage.requires_capabilities:
            if not getattr(cap, attr, False):
                missing.append(attr)

        if missing:
            reasons = [
                cap.disabled_reasons.get(attr.replace("can_", ""), attr)
                for attr in missing
            ]
            return False, " | ".join(reasons)

        return True, ""

    def explain(self, cap: DeviceCapabilities) -> str:
        """Return a human-readable pipeline composition explanation."""
        pipeline = self.compose(cap)
        lines = [
            f"Telemetry Grade: {cap.grade.label()}",
            "",
            "Active Pipeline Stages:",
        ]
        for name in pipeline.enabled_stages:
            desc = STAGE_MAP[name].description
            lines.append(f"  ✓ {name:<28} {desc}")

        if pipeline.disabled_stages:
            lines.append("")
            lines.append("Disabled Stages (telemetry unavailable):")
            for name, reason in pipeline.disabled_stages.items():
                lines.append(f"  ✗ {name:<28} {reason}")

        if pipeline.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in pipeline.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)
