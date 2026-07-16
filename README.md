# GMS — Geophysical Modeling System
### Research-Grade Multi-Scan Analyzer v1.0.0

---

## Overview

GMS is a modular, scientifically conservative geophysical signal analysis system for processing multi-scan sensor CSV data.

**Outputs:**
- Noise-filtered geophysical heatmaps (PNG)
- Anomaly detection with classification
- Cross-scan validated confidence scoring
- Final decision: `DIG | RESCAN | NO_DIG`

**Scientific constraints:**
- No hallucinated depth estimation
- Robust MAD statistics preferred over mean/std
- DIG requires ≥2 independent scan confirmations
- Uncertainty explicitly penalizes confidence

---

## Project Structure

```
gms/
├── config/
│   └── gms_config.yaml          # All tunable parameters
├── core/
│   ├── ingestion.py             # CSV validation & loading
│   ├── signal_processing.py     # Drift removal, MAD filter, grid interpolation
│   ├── anomaly_detection.py     # Multi-scale LoG blob detector + classifier
│   └── decision_engine.py       # Cross-scan validation + DIG/RESCAN/NO_DIG
├── api/
│   └── server.py                # FastAPI REST server
├── viz/
│   └── visualization.py         # Heatmap + anomaly marker rendering
├── demo_data/
│   ├── generate_demo.py         # Synthetic scan generator
│   ├── scan_A.csv               # Demo: ferrous metal + cavity
│   ├── scan_B.csv               # Demo: same anomalies (cross-confirm)
│   └── scan_C_noise_only.csv    # Demo: noise-only reference
├── reports/                     # Output PNGs and JSON results
├── logs/                        # Log files
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
# Optional: pip install plotly uvicorn fastapi python-multipart
```

---

## Usage

### CLI — Quick Start
```bash
# Run demo (generates synthetic scans + full pipeline)
python main.py --demo

# Analyze your own scans
python main.py --scans scan1.csv scan2.csv scan3.csv --session my_survey

# Launch REST API server
python main.py --server
```

### REST API
```bash
# Start server
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Upload scans
curl -X POST http://localhost:8000/upload_scans \
  -F "files=@scan_A.csv" -F "files=@scan_B.csv"

# Analyze
curl -X POST "http://localhost:8000/analyze?session_id=<id>"

# Get results
curl http://localhost:8000/results/<id>

# Get heatmap PNG
curl http://localhost:8000/heatmap/<id> --output heatmap.png

# Get full report
curl http://localhost:8000/report/<id>
```

---

## CSV Input Format

```
x,y,value
0.0,0.0,512.4
0.1,0.0,515.2
...
```

- `x`, `y` : spatial coordinates (any unit — meters recommended)
- `value`  : ADC sensor reading [0–1024] (configurable)
- Minimum 50 samples per scan

---

## Output JSON Example

```json
{
  "decision": "DIG",
  "confidence_summary": {
    "overall": 0.763,
    "max_anomaly_confidence": 0.859,
    "n_confirmed": 1,
    "n_single": 1
  },
  "anomalies": [
    {
      "group_id": "G000",
      "label": "FERROUS_METAL",
      "centroid_x": 50.1,
      "centroid_y": 38.3,
      "combined_confidence": 0.859,
      "scan_confirmations": 2,
      "mean_snr": 8.22,
      "mean_uncertainty": 0.10,
      "label_agreement": 1.0
    }
  ],
  "scan_quality": { ... },
  "warnings": []
}
```

---

## Classification Labels

| Label | Description |
|-------|-------------|
| `FERROUS_METAL` | Strong dipole pattern — likely metallic object |
| `CAVITY` | Smooth negative anomaly — possible void/grave |
| `ROCK_DEBRIS` | Moderate amplitude, no dipole |
| `SOIL_VARIATION` | Weak broad anomaly — natural soil change |
| `NOISE` | Below SNR threshold — rejected |

---

## Decision Logic

| Decision | Criteria |
|----------|----------|
| `DIG` | ≥2 scan confirmations, confidence ≥ 0.70, uncertainty ≤ 0.25, SNR ≥ 4.0 |
| `RESCAN` | ≥1 partial detection or single-scan strong signal |
| `NO_DIG` | No statistically significant anomalies |

---

## Future Compatibility

The API is fully decoupled from the core engine for integration with:
- **Android app** (Flutter / native) — call REST endpoints
- **Windows desktop** — bundle with PyInstaller or call API
- **Web dashboard** — React/Vue frontend against FastAPI

---

## Scientific Disclaimer

> This system performs statistical signal analysis only.
> No physical depth estimation is provided without explicit sensor calibration.
> All field decisions must be validated by a qualified geophysicist.
