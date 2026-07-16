"""
GMS — Survey & Calibration Controller  v3.0
=============================================
Phase 1 (btSubmit): geometry → ScanGeometryConfig → validate → store
Phase 2a (btnApplySensor): sensor calibration → validate → store
Phase 2b (btnApplySoil): soil calibration → validate → store
Pipeline gate: validate_pipeline_ready() called before PipelineWorker.run()
"""
from __future__ import annotations
import logging
from PySide6.QtCore import QObject
from PySide6.QtWidgets import (QMainWindow, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QRadioButton, QPushButton, QMessageBox)
from .app_state import GMSApplicationState

logger = logging.getLogger("gms.survey_ctrl")

def _w(window, cls, name): return window.findChild(cls, name)
def _spin(window, name, default=0.0):
    w = window.findChild(QDoubleSpinBox, name)
    if w: return w.value()
    w = window.findChild(QSpinBox, name)
    if w: return float(w.value())
    return default

class SurveyController(QObject):
    def __init__(self, window: QMainWindow, parent=None):
        super().__init__(parent)
        self._w     = window
        self._state = GMSApplicationState.instance()
        self._reset_metadata()

        # Phase 1
        for name in ("btSubmit", "btnSubmitGeometry"):
            btn = _w(window, QPushButton, name)
            if btn: btn.clicked.connect(self._on_submit_geometry); break

        # Live field display
        for name in ("spinNumLines","spinPtsPerLine","spinLineSpacing","spinSamplesDistance"):
            w = (window.findChild(QSpinBox, name) or
                 window.findChild(QDoubleSpinBox, name))
            if w:
                try: w.valueChanged.connect(self._update_field_display)
                except Exception: pass

        # Phase 2
        btn = _w(window, QPushButton, "btnApplySensor")
        if btn: btn.clicked.connect(self._on_apply_sensor)
        btn = _w(window, QPushButton, "btnApplySoil")
        if btn: btn.clicked.connect(self._on_apply_soil)

        logger.info("[SurveyCtrl] Attached")

    def _reset_metadata(self):
        from core.geometry import SurveyMetadata
        self._state.__dict__["_survey_metadata"] = SurveyMetadata()

    def _get_meta(self):
        return self._state.__dict__.get("_survey_metadata")

    def _on_submit_geometry(self):
        from core.geometry import (ScanGeometryConfig, SurveyDirection,
                                    ScanPattern, _parse_height)
        try:
            num_lines       = int(_spin(self._w, "spinNumLines",       5))
            pts_per_line    = int(_spin(self._w, "spinPtsPerLine",    10))
            line_spacing_cm = _spin(self._w, "spinLineSpacing",   100.0)
            sample_dist_cm  = _spin(self._w, "spinSamplesDistance", 50.0)

            cmb_dir  = _w(self._w, QComboBox, "comboBox_2")
            dir_text = cmb_dir.currentText() if cmb_dir else SurveyDirection.NORTH_SOUTH.value
            direction = SurveyDirection.from_combo_text(dir_text)

            radio_h  = _w(self._w, QRadioButton, "radioH")
            pattern  = (ScanPattern.HORIZONTAL if (radio_h and radio_h.isChecked())
                        else ScanPattern.VERTICAL)

            chk_zz   = _w(self._w, QCheckBox, "chkZigZag")
            chk_par  = _w(self._w, QCheckBox, "chkParallel")
            zigzag   = chk_zz.isChecked()  if chk_zz  else True
            parallel = chk_par.isChecked() if chk_par else False

            cmb_h = _w(self._w, QComboBox, "comb_SensorProbHeight")
            height_text     = cmb_h.currentText() if cmb_h else "10-15cm"
            sensor_height_m = _parse_height(height_text)

            geo = ScanGeometryConfig(
                num_lines=num_lines, points_per_line=pts_per_line,
                line_spacing_m=line_spacing_cm/100.0,
                sample_distance_m=sample_dist_cm/100.0,
                direction=direction, scan_pattern=pattern,
                zigzag=zigzag, parallel=parallel,
                sensor_height_m=sensor_height_m,
            )
            geo.validate()

            meta = self._get_meta()
            meta.mark_phase1(geo)
            self._state.__dict__["_geometry"]         = geo
            self._state.__dict__["_survey_metadata"]  = meta

            self._fill_field_display(geo.field_width_m, geo.field_length_m)

            print(f"\n[SURVEY] Phase 1 — scan_id={meta.scan_id}\n"
                  f"  {num_lines}L × {pts_per_line}P  "
                  f"spacing={geo.line_spacing_m:.3f}m  "
                  f"step={geo.sample_distance_m:.3f}m\n"
                  f"  Field: {geo.field_width_m:.3f}m × {geo.field_length_m:.3f}m\n"
                  f"  Dir: {direction.value}  Pattern: {pattern.value}\n"
                  f"  Sensor height: {sensor_height_m:.3f}m\n"
                  f"  Timestamp: {meta.timestamp}")

            QMessageBox.information(self._w, "GMS — Geometry Submitted",
                f"Survey geometry accepted.\n\n"
                f"Field: {geo.field_width_m:.3f} m × {geo.field_length_m:.3f} m\n"
                f"Direction: {direction.value}\n"
                f"Scan ID: {meta.scan_id}\n\n"
                f"Now complete Sensor and Soil calibration,\n"
                f"then click Run Pipeline.")

        except ValueError as e:
            QMessageBox.warning(self._w, "GMS — Geometry Error", str(e))
        except Exception as e:
            logger.error(f"[SurveyCtrl] Phase 1: {e}", exc_info=True)
            QMessageBox.critical(self._w, "GMS — Error", str(e))

    def _update_field_display(self, _=None):
        try:
            n   = int(_spin(self._w, "spinNumLines",       5))
            pts = int(_spin(self._w, "spinPtsPerLine",    10))
            ls  = _spin(self._w, "spinLineSpacing",   100.0) / 100.0
            sd  = _spin(self._w, "spinSamplesDistance", 50.0) / 100.0
            if n >= 2 and pts >= 2 and ls > 0 and sd > 0:
                self._fill_field_display((pts-1)*sd, (n-1)*ls)
        except Exception:
            pass

    def _fill_field_display(self, fw: float, fl: float):
        for name, val in (("spinFieldW", fw), ("spinFieldL", fl)):
            w = self._w.findChild(QDoubleSpinBox, name)
            if w:
                w.blockSignals(True)
                w.setValue(round(val, 3))
                w.blockSignals(False)

    def _on_apply_sensor(self):
        from core.geometry import SensorCalibration, SensorOrientation
        try:
            cal = SensorCalibration(
                adc_scaling_factor=_spin(self._w, "spinADC",       1.0),
                gain              =_spin(self._w, "spinGain",      1.0),
                offset            =_spin(self._w, "spinOffset",    0.0),
                sensor_spacing_m  =_spin(self._w, "spinSensSpace", 0.5),
                orientation       =SensorOrientation.VERTICAL_GRADIENT,
            )
            cal.validate()
            meta = self._get_meta()
            meta.mark_sensor(cal)
            self._state.__dict__["_survey_metadata"]   = meta
            self._state.__dict__["_sensor_calibration"] = cal
            # Persist to calibration_state and fire calibration_changed signal
            self._state.set_calibration(meta.calibration.to_dict())
            print(f"\n[CALIBRATION] Sensor: ADC={cal.adc_scaling_factor} "
                  f"Gain={cal.gain} Offset={cal.offset} "
                  f"Spacing={cal.sensor_spacing_m}m")
            QMessageBox.information(self._w, "GMS — Sensor Calibration Applied",
                f"ADC Scale: {cal.adc_scaling_factor}\n"
                f"Gain: {cal.gain}  Offset: {cal.offset}\n"
                f"Sensor Spacing: {cal.sensor_spacing_m} m")
        except ValueError as e:
            QMessageBox.warning(self._w, "GMS — Sensor Error", str(e))
        except Exception as e:
            logger.error(f"[SurveyCtrl] Sensor: {e}", exc_info=True)

    def _on_apply_soil(self):
        from core.geometry import SoilCalibration
        try:
            cmb     = _w(self._w, QComboBox, "cmbSoilProf")
            profile = cmb.currentText() if cmb else "General Loam"
            cal = SoilCalibration(
                soil_profile              =profile,
                mineralization_correction =_spin(self._w, "spinMineral", 0.0),
                basalt_compensation       =_spin(self._w, "spinBasalt",  0.0),
            )
            cal.validate()
            meta = self._get_meta()
            meta.mark_soil(cal)
            self._state.__dict__["_survey_metadata"]  = meta
            self._state.__dict__["_soil_calibration"] = cal
            # Persist to calibration_state and fire calibration_changed signal
            self._state.set_calibration(meta.calibration.to_dict())
            print(f"\n[CALIBRATION] Soil: Profile={cal.soil_profile} "
                  f"Mineral={cal.mineralization_correction} "
                  f"Basalt={cal.basalt_compensation}")
            QMessageBox.information(self._w, "GMS — Soil Calibration Applied",
                f"Profile: {cal.soil_profile}\n"
                f"Mineralization: {cal.mineralization_correction}%\n"
                f"Basalt: {cal.basalt_compensation}%")
        except ValueError as e:
            QMessageBox.warning(self._w, "GMS — Soil Error", str(e))
        except Exception as e:
            logger.error(f"[SurveyCtrl] Soil: {e}", exc_info=True)

    def validate_pipeline_ready(self) -> bool:
        try:
            self._get_meta().validate_pipeline_ready()
            return True
        except ValueError as e:
            QMessageBox.warning(self._w, "GMS — Pipeline Blocked", str(e))
            return False

    def get_survey_metadata(self):
        return self._get_meta()