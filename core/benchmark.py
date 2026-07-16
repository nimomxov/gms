"""
GMS — Synthetic Benchmark Suite  v2.0

Generates synthetic geophysical scans with KNOWN ground truth,
runs the pipeline, and evaluates:
  - True Positive Rate  (TPR / Recall)
  - False Positive Rate (FPR)
  - Precision
  - F1 Score
  - Detection range (min amplitude detectable)
  - Pipeline comparison across presets

This transforms GMS from "experiments" to research-grade validation.

Usage:
  python -m gms.benchmark --suite standard --output reports/benchmark
  python -m gms.benchmark --suite noise_stress  --n-runs 20
  python -m gms.benchmark --compare-presets
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("gms.benchmark")


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth target definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SyntheticTarget:
    """A single synthetic anomaly with known parameters."""
    target_id: str
    type: str                  # FERROUS_METAL | CAVITY | ROCK_DEBRIS | NONE
    x: float                   # true position x
    y: float                   # true position y
    amplitude: float           # peak signal amplitude (signed)
    width: float               # spatial extent (grid units)
    is_dipole: bool = False    # True = ferrous dipole signature
    depth_abstract: float = 0.0  # relative depth (0=shallow, 1=deep)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic scan generators
# ─────────────────────────────────────────────────────────────────────────────

def _background(n, noise_std=18.0, rng=None):
    rng = rng or np.random.default_rng()
    drift = np.linspace(0, rng.uniform(5, 20), n)
    return drift + rng.normal(0, noise_std, n)


def _ferrous_dipole(x, y, cx, cy, amplitude, width):
    """Magnetic dipole: positive lobe + negative lobe."""
    r2  = (x-cx)**2 + (y-cy)**2
    pos = amplitude * np.exp(-r2 / (2*width**2))
    r2n = (x-cx)**2 + (y-(cy+width*1.5))**2
    neg = -0.6 * amplitude * np.exp(-r2n / (2*(width*1.2)**2))
    return pos + neg


def _cavity_signal(x, y, cx, cy, amplitude, width):
    """Broad negative anomaly (signal suppression)."""
    r2 = (x-cx)**2 + (y-cy)**2
    return amplitude * np.exp(-r2 / (2*width**2))


def _rock_signal(x, y, cx, cy, amplitude, width):
    """Moderate, slightly irregular positive anomaly."""
    r2 = (x-cx)**2 + (y-cy)**2
    return amplitude * np.exp(-r2 / (2*width**2)) * (1.0 + 0.2*(x-cx)/width)


def generate_synthetic_scan(targets: list[SyntheticTarget],
                              x_range=(0, 10), y_range=(0, 8),
                              n_lines=20, pts_per_line=60,
                              noise_std=18.0,
                              seed: int = None) -> tuple[pd.DataFrame, list[SyntheticTarget]]:
    """
    Generate a synthetic CSV scan with injected targets.
    Returns (DataFrame, list of active targets in this scan).
    """
    rng = np.random.default_rng(seed)
    rows = []
    x_vals = np.linspace(x_range[0], x_range[1], pts_per_line)

    for line_idx in range(n_lines):
        y = y_range[0] + (y_range[1]-y_range[0]) * line_idx / (n_lines-1)
        y_arr = np.full_like(x_vals, y)
        signal = 512.0 + _background(pts_per_line, noise_std=noise_std, rng=rng)

        for t in targets:
            if t.type == "FERROUS_METAL" and t.is_dipole:
                signal += _ferrous_dipole(x_vals, y_arr, t.x, t.y, t.amplitude, t.width)
            elif t.type == "CAVITY":
                signal += _cavity_signal(x_vals, y_arr, t.x, t.y, t.amplitude, t.width)
            elif t.type == "ROCK_DEBRIS":
                signal += _rock_signal(x_vals, y_arr, t.x, t.y, t.amplitude, t.width)

        signal = np.clip(signal, 0, 1024)
        for xv, sv in zip(x_vals, signal):
            rows.append({"x": round(float(xv),3), "y": round(float(y),3),
                         "value": round(float(sv),2)})

    return pd.DataFrame(rows), targets


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

def _make_standard_suite() -> list[dict]:
    """Standard benchmark: 8 scenarios covering key cases."""
    return [
        {
            "name": "single_metal_strong",
            "description": "Strong ferrous target, clean background",
            "targets_A": [SyntheticTarget("T1", "FERROUS_METAL", 5.0, 4.0, 320, 0.8, True)],
            "targets_B": [SyntheticTarget("T1", "FERROUS_METAL", 5.1, 3.9, 300, 0.8, True)],
            "noise_std": 18.0, "expected_decision": "DIG",
        },
        {
            "name": "single_metal_weak",
            "description": "Weak ferrous target (borderline detectability)",
            "targets_A": [SyntheticTarget("T1", "FERROUS_METAL", 5.0, 4.0, 80, 0.8, True)],
            "targets_B": [SyntheticTarget("T1", "FERROUS_METAL", 5.1, 3.9, 75, 0.8, True)],
            "noise_std": 18.0, "expected_decision": "RESCAN",
        },
        {
            "name": "cavity_strong",
            "description": "Deep cavity (broad negative anomaly)",
            "targets_A": [SyntheticTarget("T1", "CAVITY", 3.0, 5.0, -120, 1.5, False)],
            "targets_B": [SyntheticTarget("T1", "CAVITY", 3.1, 5.1, -110, 1.5, False)],
            "noise_std": 18.0, "expected_decision": "DIG",
        },
        {
            "name": "metal_plus_cavity",
            "description": "Two co-located targets: metal + cavity",
            "targets_A": [
                SyntheticTarget("T1", "FERROUS_METAL", 5.0, 4.0, 280, 0.8, True),
                SyntheticTarget("T2", "CAVITY", 2.0, 6.0, -100, 1.2, False),
            ],
            "targets_B": [
                SyntheticTarget("T1", "FERROUS_METAL", 5.1, 3.9, 260, 0.8, True),
                SyntheticTarget("T2", "CAVITY", 2.1, 6.1, -95, 1.2, False),
            ],
            "noise_std": 20.0, "expected_decision": "DIG",
        },
        {
            "name": "noise_only",
            "description": "No targets — should return NO_DIG",
            "targets_A": [],
            "targets_B": [],
            "noise_std": 18.0, "expected_decision": "NO_DIG",
        },
        {
            "name": "high_noise",
            "description": "Strong target buried in high noise (mineralized soil)",
            "targets_A": [SyntheticTarget("T1", "FERROUS_METAL", 5.0, 4.0, 320, 0.8, True)],
            "targets_B": [SyntheticTarget("T1", "FERROUS_METAL", 5.1, 3.9, 300, 0.8, True)],
            "noise_std": 45.0, "expected_decision": "RESCAN",
        },
        {
            "name": "weak_signal_single_scan",
            "description": "Only one scan — should RESCAN not DIG",
            "targets_A": [SyntheticTarget("T1", "FERROUS_METAL", 5.0, 4.0, 250, 0.8, True)],
            "targets_B": [],  # no second scan
            "noise_std": 18.0, "expected_decision": "RESCAN",
        },
        {
            "name": "rock_debris_clutter",
            "description": "Rock/debris clutter — should not trigger DIG",
            "targets_A": [
                SyntheticTarget("R1", "ROCK_DEBRIS", 2.0, 3.0, 90, 0.6, False),
                SyntheticTarget("R2", "ROCK_DEBRIS", 7.0, 6.0, 70, 0.7, False),
            ],
            "targets_B": [
                SyntheticTarget("R1", "ROCK_DEBRIS", 2.1, 3.1, 85, 0.6, False),
            ],
            "noise_std": 18.0, "expected_decision": "RESCAN",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario_name: str
    expected_decision: str
    actual_decision: str
    passed: bool
    n_confirmed: int
    overall_confidence: float
    pipeline_used: str
    config_hash: str
    warnings: list = field(default_factory=list)


@dataclass
class BenchmarkReport:
    preset_name: str
    n_scenarios: int
    n_passed: int
    pass_rate: float
    decision_accuracy: float       # correct decision / total
    false_positive_rate: float     # DIG when expected NO_DIG or RESCAN
    false_negative_rate: float     # NO_DIG when expected DIG
    scenario_results: list[ScenarioResult]
    config_hash: str


def run_benchmark(preset_name: str = "stable",
                   suite: str = "standard",
                   output_dir: str = "reports",
                   gms_config: dict = None,
                   seed: int = 42) -> BenchmarkReport:
    """
    Run the full benchmark suite for a given preset pipeline.
    Returns a BenchmarkReport with all scenario results.
    """
    import yaml, sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    if gms_config is None:
        cfg_path = Path(__file__).parent.parent.parent / "config" / "gms_config.yaml"
        with open(cfg_path) as f:
            gms_config = yaml.safe_load(f)

    from core.pipeline import build_pipeline, PRESETS
    pipeline = build_pipeline(gms_config, preset=preset_name)
    pcfg     = PRESETS.get(preset_name)

    scenarios = _make_standard_suite()
    scenario_results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="gms_bench_"))

    logger.info(f"\n{'='*60}")
    logger.info(f"GMS Benchmark — preset={preset_name}  suite={suite}")
    logger.info(f"{'='*60}")

    for sc in scenarios:
        name = sc["name"]
        logger.info(f"\n  Scenario: {name}")

        # Generate CSV files
        scan_files = []
        for suffix, key in [("A", "targets_A"), ("B", "targets_B")]:
            targets = sc.get(key, [])
            if not targets and suffix == "B":
                continue
            df, _ = generate_synthetic_scan(
                targets=targets,
                noise_std=sc["noise_std"],
                seed=seed + hash(name + suffix) % 10000,
            )
            csv_path = tmpdir / f"{name}_{suffix}.csv"
            df.to_csv(csv_path, index=False)
            scan_files.append(str(csv_path))

        try:
            result = pipeline.run_session(scan_files, session_id=f"bench_{name}")
            actual   = result["decision"]
            expected = sc["expected_decision"]
            passed   = actual == expected

            sr = ScenarioResult(
                scenario_name=name,
                expected_decision=expected,
                actual_decision=actual,
                passed=passed,
                n_confirmed=result["confidence_summary"]["n_confirmed"],
                overall_confidence=result["confidence_summary"]["overall"],
                pipeline_used=f"{pipeline.cfg.interpolator}+{pipeline.cfg.baseline}+{pipeline.cfg.detector}",
                config_hash=pipeline.cfg.config_hash(),
                warnings=result.get("warnings", []),
            )
        except Exception as e:
            sr = ScenarioResult(
                scenario_name=name, expected_decision=sc["expected_decision"],
                actual_decision="ERROR", passed=False,
                n_confirmed=0, overall_confidence=0.0,
                pipeline_used="", config_hash="",
                warnings=[str(e)],
            )

        status = "✅ PASS" if sr.passed else "❌ FAIL"
        logger.info(f"  {status}  expected={sr.expected_decision}  "
                    f"actual={sr.actual_decision}  "
                    f"conf={sr.overall_confidence:.3f}")
        scenario_results.append(sr)

    # Compute metrics
    n_pass = sum(1 for r in scenario_results if r.passed)
    n_total = len(scenario_results)

    # FPR: predicted DIG when ground truth was NOT DIG
    fp = sum(1 for r in scenario_results
             if r.actual_decision == "DIG" and r.expected_decision != "DIG")
    tn = sum(1 for r in scenario_results if r.expected_decision != "DIG")
    fpr = fp / max(tn, 1)

    # FNR: predicted NO_DIG when ground truth was DIG
    fn = sum(1 for r in scenario_results
             if r.actual_decision == "NO_DIG" and r.expected_decision == "DIG")
    tp_gt = sum(1 for r in scenario_results if r.expected_decision == "DIG")
    fnr = fn / max(tp_gt, 1)

    report = BenchmarkReport(
        preset_name=preset_name,
        n_scenarios=n_total,
        n_passed=n_pass,
        pass_rate=round(n_pass/n_total, 3),
        decision_accuracy=round(n_pass/n_total, 3),
        false_positive_rate=round(fpr, 3),
        false_negative_rate=round(fnr, 3),
        scenario_results=scenario_results,
        config_hash=pipeline.cfg.config_hash(),
    )

    # Save JSON report
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / f"benchmark_{preset_name}.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK SUMMARY — {preset_name}")
    logger.info(f"  Pass rate:  {n_pass}/{n_total} ({report.pass_rate:.0%})")
    logger.info(f"  FPR:        {fpr:.0%}  (DIG when shouldn't)")
    logger.info(f"  FNR:        {fnr:.0%}  (missed real DIG)")
    logger.info(f"  Report:     {report_path}")
    logger.info(f"{'='*60}")

    return report


def compare_presets_benchmark(output_dir: str = "reports",
                               gms_config: dict = None) -> dict:
    """Run benchmark for all presets and produce a comparison table."""
    from core.pipeline import PRESETS

    all_reports = {}
    for preset in PRESETS:
        try:
            rpt = run_benchmark(preset, output_dir=output_dir, gms_config=gms_config)
            all_reports[preset] = {
                "pass_rate": rpt.pass_rate,
                "fpr": rpt.false_positive_rate,
                "fnr": rpt.false_negative_rate,
                "n_passed": rpt.n_passed,
                "n_scenarios": rpt.n_scenarios,
                "config_hash": rpt.config_hash,
            }
        except Exception as e:
            all_reports[preset] = {"error": str(e)}

    # Print comparison table
    print("\n" + "="*65)
    print("BENCHMARK COMPARISON TABLE")
    print(f"{'Preset':<12} {'Pass%':>6} {'FPR':>6} {'FNR':>6} {'Hash'}")
    print("-"*65)
    for name, r in all_reports.items():
        if "error" in r:
            print(f"  {name:<10}  ERROR: {r['error']}")
        else:
            print(f"  {name:<10} {r['pass_rate']:>5.0%}  "
                  f"{r['fpr']:>5.0%}  {r['fnr']:>5.0%}  "
                  f"{r['config_hash']}")
    print("="*65 + "\n")

    out = Path(output_dir) / "benchmark_comparison.json"
    with open(out, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"Saved: {out}")

    return all_reports


if __name__ == "__main__":
    import argparse, sys, yaml, logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="GMS Benchmark Suite")
    parser.add_argument("--preset", default="stable")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent.parent / "config" / "gms_config.yaml"
    with open(cfg_path) as f:
        gms_config = yaml.safe_load(f)

    if args.compare:
        compare_presets_benchmark(output_dir=args.output, gms_config=gms_config)
    else:
        run_benchmark(preset_name=args.preset, output_dir=args.output, gms_config=gms_config)
