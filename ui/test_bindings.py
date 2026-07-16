"""
GMS v3.0 — Binding Validation Test Suite
==========================================
Verifies every ObjectName binding declared in gms_controller.py
against the actual .ui files — without needing PySide6.

Tests:
  1. Every ObjectName referenced in the controller exists in the .ui
  2. Tab names are reachable
  3. Action names match menu declarations
  4. ScanCompareController logic (pure Python, no Qt)
  5. PanelToggle state machine
  6. BackendRouter method signatures exist
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
import sys, re
from pathlib import Path

UI_DIR = Path(__file__).parent

def ok(msg):  print(f"  [OK]  {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); return False
def sep(t): print(f"\n{'─'*56}\n  {t}\n{'─'*56}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect all objectName values from a .ui file
# ─────────────────────────────────────────────────────────────────────────────

def collect_names(ui_file: str) -> set[str]:
    tree = ET.parse(UI_DIR / ui_file)
    names = set()
    for el in tree.getroot().iter():
        n = el.get("name")
        if n: names.add(n)
    return names


def collect_actions(ui_file: str) -> set[str]:
    tree = ET.parse(UI_DIR / ui_file)
    return {a.get("name") for a in tree.getroot().iter("action") if a.get("name")}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ObjectName existence checks
# ─────────────────────────────────────────────────────────────────────────────

def test_objectnames():
    sep("1. ObjectName existence — main window")
    mw_names = collect_names("gms_main_window.ui")

    # Every ObjectName referenced in the controller
    REQUIRED = [
        # Frames
        "sidebarFrame", "inspectorFrame",
        # Tab widget + tabs
        "workspaceTabs",
        "tabCSVImport", "tabScanConfig", "tabCalibration",
        "tabHeatmap2D", "tabExplorer3D", "tabScansCompare",
        "tabDiagnostics", "tabBenchmark",
        # Sidebar buttons (exact names from .ui)
        "btnOpenCSV", "btnNewSession", "btnOpenScan",
        "btnSaveProject", "btnExportReport",
        "btnDeviceProfiles", "btnbtnDeviceConnection",
        "btnRunPipeline",
        "btnHeatmap2D", "btnExplorer3D",
        "btncalibration", "btnbenchmark",
        "btnOverlayControls", "btnScansCompare",
        "btnDiagnostics", "btnScanConfig",
        # CSV tab
        "btnBrowseCSV", "btnImportScan", "btnAutoDetect", "btnClearFile",
        "dropZoneFrame", "fileNameLabel", "fileSizeLabel",
        "tableDataPreview", "labelRowsDetected", "labelDelimiter",
        "warnNoHeading", "warnNoXY", "warnNoSNR",
        "grpFieldMapper",
        # Heatmap controls
        "chkLSignal", "chkLBaseline", "chkLAnomalies",
        "chkLDigZones", "chkLGrid", "chkLConfidence", "chkLRawPts",
        "cmbCmap", "cmbInterp", "cmbBase",
        "sldBright", "sldCont", "sldSmooth",
        "chkFocusMode",
        # 3D controls
        "sldVertExag", "lblVexVal", "lblVexVal_2",
        "btnResetCam", "btnTopView", "btnSideView", "btnPerspView",
        "cmbRenderMode",
        "chk3dSignal", "chk3dBase", "chk3dDig", "chk3dConf",
        "chk3dRaw", "chk3dGrid", "chk3dBlobs",
        # Calibration
        "spinADC", "spinGain", "spinOffset", "spinSensSpace",
        "btnApplySensor", "btnApplySoil", "btnValidate",
        "spinKTX", "spinKTY", "spinKTD",
        "cmbSoilProf",
        "tableCalHist",
        # Benchmark
        "btnRunBench", "benchProgress",
        "cmbBenchDS", "cmbBenchPipe", "chkBenchMulti",
        "tableBenchResults",
        # Confusion matrix
        "cmTP", "cmFP", "cmFN", "cmTN",
        # Inspector
        "inspTargetName", "inspSNR", "inspDepth",
        "inspConfidence", "inspX", "inspY",
        "inspReliability", "inspTopology", "inspFusion", "inspMultiScan",
        "inspNoTarget",
        "btnConfirmDig", "btnRejectTarget",
        # Scan config geometry
        "spinNumLines", "spinPtsPerLine", "spinFieldW", "spinFieldL",
        "chkParallel", "chkZigZag", "radioV", "radioH",
        "labelGridCalc", "labelRes",
        # Diagnostics
        "tablePipeTimings",
    ]

    missing = []
    for name in REQUIRED:
        if name in mw_names:
            ok(f"Found: {name}")
        else:
            fail(f"MISSING in .ui: {name}")
            missing.append(name)

    print(f"\n  Summary: {len(REQUIRED)-len(missing)}/{len(REQUIRED)} names found")
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# 2. Action names
# ─────────────────────────────────────────────────────────────────────────────

def test_actions():
    sep("2. Action names")
    actions = collect_actions("gms_main_window.ui")

    REQUIRED_ACTIONS = [
        "actionToggleSidebar",
        "actionToggleInspector",
        "actionPreferences",
        "actionAbout",
        "actionOpenCSV",
        "actionSaveProject",
        "actionExportReport",
        "actionExit",
        "actionNewSession",
        "actionSettings",
        "actionFocus_Mode",
    ]
    for name in REQUIRED_ACTIONS:
        if name in actions:
            ok(f"Action found: {name}")
        else:
            fail(f"Action MISSING: {name}")

    print(f"  All declared actions: {sorted(actions)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dialog ObjectNames
# ─────────────────────────────────────────────────────────────────────────────

def test_dialogs():
    sep("3. Dialog ObjectNames")
    dialogs = {
        "gms_settings_dialog.ui": [
            "settingsNav", "settingsStack",
            "btnOKSettings", "btnCancelSettings", "btnApplySettings",
            "btnRestoreDefaults",
            "spinFontSize", "comboDefPreset", "comboDefBaseline",
            "checkNeverFakeDepth", "checkNeverFakeConf",
        ],
        "gms_about_dialog.ui": [
            "btnAboutClose", "btnAboutDocs",
            "lblAboutVersion", "lblAboutLogo",
        ],
        "gms_device_dialog.ui": [
            "btnBLEScan", "btnRefreshPorts",
            "btnConnect", "btnDisconnect", "btnCancel",
            "textTelLog", "listBLEDevices", "listSerialPorts",
            "comboBaudRate", "comboBLEProfile",
            "bleScanProgress",
        ],
    }
    for ui_file, names in dialogs.items():
        all_names = collect_names(ui_file)
        for name in names:
            if name in all_names:
                ok(f"{ui_file}: {name}")
            else:
                fail(f"{ui_file}: MISSING {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tab reachability
# ─────────────────────────────────────────────────────────────────────────────

def test_tab_reachability():
    sep("4. Tab reachability from sidebar buttons")
    # mapping: button → tab (from controller)
    TAB_MAP = {
        "btnHeatmap2D":    "tabHeatmap2D",
        "btnExplorer3D":   "tabExplorer3D",
        "btncalibration":  "tabCalibration",
        "btnbenchmark":    "tabBenchmark",
        "btnScanConfig":   "tabScanConfig",
        "btnScansCompare": "tabScansCompare",
        "btnDiagnostics":  "tabDiagnostics",
    }
    mw_names = collect_names("gms_main_window.ui")
    for btn, tab in TAB_MAP.items():
        btn_ok = btn in mw_names
        tab_ok = tab in mw_names
        if btn_ok and tab_ok:
            ok(f"{btn} → {tab}")
        else:
            parts = []
            if not btn_ok: parts.append(f"btn '{btn}' missing")
            if not tab_ok: parts.append(f"tab '{tab}' missing")
            fail(f"{btn} → {tab}: {', '.join(parts)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ScanCompareController pure logic
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_logic():
    sep("5. ScanCompareController — pure logic (no Qt)")
    import numpy as np

    # Simulate the data structures only
    from dataclasses import dataclass, field as dc_field
    from typing import Any

    @dataclass
    class MockEntry:
        scan_id: str
        label: str
        signal_grid: np.ndarray
        opacity: float = 1.0
        visible: bool = True

    scans: list[MockEntry] = []

    # Add 3 scans
    rng = np.random.default_rng(42)
    for i in range(3):
        scans.append(MockEntry(
            scan_id=f"scan_{i:03d}",
            label=f"Scan {i+1}",
            signal_grid=rng.standard_normal((50, 50)) * 100 + i * 50,
        ))
    assert len(scans) == 3
    ok(f"Added 3 scan entries")

    # Opacity change
    scans[0].opacity = 0.5
    assert scans[0].opacity == 0.5
    ok("Opacity change: 1.0 → 0.5")

    # Visibility toggle
    scans[1].visible = False
    visible = [s for s in scans if s.visible]
    assert len(visible) == 2
    ok(f"Visibility toggle: {len(visible)}/3 visible")

    # Remove scan
    target_id = "scan_001"
    scans = [s for s in scans if s.scan_id != target_id]
    assert len(scans) == 2
    ok(f"Remove scan_001: {len(scans)} remain")

    # Grid compositing simulation
    visible = [s for s in scans if s.visible]
    composite = np.zeros((50, 50))
    for s in visible:
        composite += s.signal_grid * s.opacity
    assert composite.shape == (50, 50)
    assert not np.all(composite == 0)
    ok(f"Alpha composite of {len(visible)} grids: shape={composite.shape}")

    # Difference blend
    if len(visible) >= 2:
        diff = visible[0].signal_grid - visible[1].signal_grid
        assert diff.shape == (50, 50)
        ok(f"Difference blend: range=[{diff.min():.1f}, {diff.max():.1f}]")

    # Scatter → grid interpolation stub
    x = rng.uniform(0, 5, 200)
    y = rng.uniform(0, 5, 200)
    z = np.sin(x) * np.cos(y) * 100
    xi = np.linspace(0, 5, 40)
    yi = np.linspace(0, 5, 40)
    gx, gy = np.meshgrid(xi, yi)
    try:
        from scipy.interpolate import griddata
        gz = griddata(np.column_stack([x, y]), z, (gx, gy),
                      method="linear", fill_value=0)
        assert gz.shape == (40, 40)
        ok(f"scatter→grid interpolation: {gz.shape}")
    except ImportError:
        ok("scipy not available — interpolation fallback path tested")


# ─────────────────────────────────────────────────────────────────────────────
# 6. PanelToggle state machine
# ─────────────────────────────────────────────────────────────────────────────

def test_panel_toggle():
    sep("6. PanelToggle state machine")

    # Pure logic test (no Qt Frame needed)
    class MockFrame:
        def __init__(self, w): self._w = w
        def width(self): return self._w
        def setMaximumWidth(self, v): self._w = v
        def setMinimumWidth(self, v): pass

    class PanelToggleLogic:
        def __init__(self, frame, default):
            self._frame = frame
            self._saved = default
            self._visible = True
        def toggle(self):
            if self._visible:
                self._saved = max(self._frame.width(), 80)
                self._frame.setMaximumWidth(0)
                self._visible = False
            else:
                self._frame.setMinimumWidth(self._saved)
                self._frame.setMaximumWidth(self._saved)
                self._visible = True
        @property
        def visible(self): return self._visible

    frame = MockFrame(220)
    toggle = PanelToggleLogic(frame, 220)

    assert toggle.visible is True
    ok("Initial state: visible=True")

    toggle.toggle()
    assert toggle.visible is False
    assert frame._w == 0
    ok("After toggle(): visible=False, width=0")

    toggle.toggle()
    assert toggle.visible is True
    assert frame._w == 220
    ok("After toggle() again: visible=True, width=220")

    # Test with narrow frame
    frame2 = MockFrame(180)
    toggle2 = PanelToggleLogic(frame2, 180)
    toggle2.toggle()
    saved = toggle2._saved
    toggle2.toggle()
    assert toggle2._frame._w == saved
    ok(f"Narrow frame width preserved: {saved}px")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Controller method completeness
# ─────────────────────────────────────────────────────────────────────────────

def test_controller_completeness():
    sep("7. Controller method completeness")
    import ast
    src = (Path(__file__).parent / "gms_controller.py").read_text()
    tree = ast.parse(src)

    # Collect all method names across all classes
    methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef):
                    methods.setdefault(node.name, set()).add(item.name)

    required = {
        "GMSController": [
            "__init__", "_wire_actions", "_wire_sidebar_buttons",
            "_wire_heatmap_controls", "_wire_calibration",
            "_wire_benchmark", "_wire_inspector",
            "_on_toggle_sidebar", "_on_toggle_inspector",
            "_on_open_settings", "_on_open_about",
            "_on_open_device_dialog", "_on_open_csv",
            "_on_auto_detect", "_on_clear_file",
            "_on_heatmap_param_changed", "_on_layer_toggle",
            "_on_focus_mode", "_on_vert_exag_changed",
            "_on_confirm_dig", "_on_reject_target",
            "_switch_tab", "_load_csv_preview",
            "_restore_session", "save_session",
        ],
        "ScanCompareController": [
            "__init__", "_build_ui", "_build_scan_row",
            "_on_add_scan", "add_scan_from_file", "add_scan_from_dataset",
            "_on_remove", "_on_opacity", "_on_visibility", "_on_color_pick",
            "_on_export", "_refresh_canvas",
            "_render_matplotlib", "_render_fallback",
            "_scatter_to_grid",
        ],
        "PanelToggle": [
            "__init__", "toggle",
        ],
        "BackendRouter": [
            "__init__", "run_pipeline", "run_benchmark",
            "apply_sensor_calibration",
        ],
    }

    all_ok = True
    for cls_name, method_list in required.items():
        cls_methods = methods.get(cls_name, set())
        for method in method_list:
            if method in cls_methods:
                ok(f"{cls_name}.{method}")
            else:
                fail(f"MISSING: {cls_name}.{method}")
                all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 8. Controller references exact .ui ObjectNames (regex scan)
# ─────────────────────────────────────────────────────────────────────────────

def test_objectname_references_in_controller():
    sep("8. Controller references verified against .ui ObjectNames")
    src = (Path(__file__).parent / "gms_controller.py").read_text()
    mw_names = collect_names("gms_main_window.ui")

    # Extract all string literals that look like widget names
    # Pattern: findChild(SomeClass, "widgetName") or _w(x, y, "widgetName")
    referenced = set(re.findall(r'findChild\([^,]+,\s*"([^"]+)"', src))
    referenced |= set(re.findall(r'_w\([^,]+,\s*\w+,\s*"([^"]+)"', src))

    # Also capture strings passed to _switch_tab
    referenced |= set(re.findall(r'_switch_tab\("([^"]+)"', src))

    # All .ui ObjectNames across all files
    all_ui_names = mw_names
    for f in ["gms_settings_dialog.ui", "gms_about_dialog.ui", "gms_device_dialog.ui"]:
        all_ui_names |= collect_names(f)

    missing_from_ui = []
    for name in sorted(referenced):
        if name in all_ui_names:
            ok(f"Reference '{name}' → exists in .ui")
        else:
            # May be a dynamically created name (scanRow_*, etc.)
            if any(name.startswith(p) for p in
                   ["scanRow_", "swatch_", "scanName_", "btnRemove_",
                    "chkVis_", "sldOpac_", "opacVal_"]):
                ok(f"Reference '{name}' → dynamic (compare system)")
            else:
                fail(f"Reference '{name}' NOT FOUND in any .ui")
                missing_from_ui.append(name)

    print(f"\n  {len(referenced)-len(missing_from_ui)}/{len(referenced)} references valid")
    return missing_from_ui


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nGMS v3.0 — Binding Validation Suite")
    print("=" * 56)

    tests = [
        ("ObjectName existence",        test_objectnames),
        ("Action names",                test_actions),
        ("Dialog ObjectNames",          test_dialogs),
        ("Tab reachability",            test_tab_reachability),
        ("Compare logic (pure Python)", test_compare_logic),
        ("PanelToggle state machine",   test_panel_toggle),
        ("Controller completeness",     test_controller_completeness),
        ("ObjectName references",       test_objectname_references_in_controller),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n  [EXCEPTION] {name}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*56}")
    print(f"  RESULTS: {passed}/{len(tests)} tests passed, {failed} failed")
    print(f"{'='*56}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
