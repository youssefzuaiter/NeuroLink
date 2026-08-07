# NeuroLink 4.0
### *Privacy-Preserving Multi-Stream Behavioral Baseline Engine for Typing Performance*

> **Series:** NeuroLink V1 → V2 → V3 → **V4.0**  
> **Data:** 50,321 keybr.com lessons · Dec 2024 – Jul 2026  
> **Author:** Youssef Zuaiter

---

## What is NeuroLink?

NeuroLink is a personal AI system that learns your unique typing behavioral signature and detects when you drift away from it. Unlike population-based models, it is trained **exclusively on your own data** — a true N-of-1 system. Every model is personal, private, and specific to one user.

The system tracks three independent behavioral streams — **Speed**, **Accuracy**, and **Motor control** — and produces a Longitudinal Drift Index (LDI) per stream with anomaly alerting and a 90-day WPM forecast.

The long-term research direction: passive, privacy-preserving detection of neurological behavioral change through everyday typing patterns.

---

## The Series

| Version | Core Idea | Key Contribution |
|---|---|---|
| V1 | Basic analytics | Per-session WPM and error tracking |
| V2 | Statistical modeling | Rolling baselines, trend detection |
| V3 | N-of-1 autoencoder | DP-SGD training, LDI anomaly detection |
| **V4** | **Multi-stream decomposition** | **3 decoder heads, per-stream LDI, 90-day forecast** |

---

## V4 Architecture

```
Input (15 features)
        │
        ▼
┌─────────────────────────────┐
│      Shared Encoder         │
│   15 → 32 → 16 → 10 (Tanh) │  ← Muscle Memory Signature
└─────────────────────────────┘
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
Speed     Accuracy        Motor
Head      Head            Head
10→16→5   10→16→5         10→16→5
   │         │              │
Speed LDI  Accuracy LDI  Motor LDI
```

### 15 Features across 3 Streams

| Stream | Features |
|---|---|
| **Speed** | EV, RSV, wpm_vs_50roll, improvement_rate, warmup_effect |
| **Accuracy** | error_entropy, speed_var_wl, worst_char_improvement, miss_ratio_top3 *(new)*, error_rate_trend *(new)* |
| **Motor** | CTA, CTA_cos, hand_asymmetry, time_of_day, speed_dispersion *(new)* |

### Privacy
- **Differential Privacy via Opacus** (DP-SGD)
- ε = 10.0, δ = 1e-5
- RDP accountant
- All training done locally — raw data never leaves your machine

---

## Results

| Metric | Value |
|---|---|
| Validation R² | **0.9751 PASS** |
| Test R² | 0.8922 |
| Best epoch | 112 |
| DP budget | ε = 9.997, δ = 1e-5 |
| HIGH anomaly days | 18.5% |
| NORMAL days | 63.2% |

### Per-Feature R²
All 15 features above **0.89**, 13 of 15 above **0.95**.

### 90-Day WPM Forecast
- Current avg WPM (last 7 days): **112.7**
- LDI-adjusted 90-day projection: **122.9 WPM**

---

## What's New in V4 vs V3

| | V3 | V4 |
|---|---|---|
| Decoder | Single head | 3 stream-specific heads |
| LDI | 1 global signal | 3 independent streams |
| Anomaly output | Overall tier | Decomposed: which stream drifted |
| Loss function | Weighted MSE | Huber loss + head-size normalization |
| New features | — | miss_ratio_top3, error_rate_trend, speed_dispersion |
| Forecasting | — | 90-day log-growth curve + LDI adjustment |

---

## How to Run

### 1. Export your keybr data
Go to [keybr.com](https://www.keybr.com) → Profile → Export → download `typing-data.json`

### 2. Generate the feature CSV
Run the data preparation cell at the top of the notebook:
```python
# Cell [1] in the notebook
# Converts typing-data.json → transformed_typing_features_v4.csv
```

### 3. Run in Google Colab
Upload `NeuroLink_V4_0.ipynb` and `transformed_typing_features_v4.csv` to Colab.  
Set runtime to **GPU (T4)**. Run all cells in order.

### Requirements
```
torch
opacus>=1.4.0
numpy
pandas
scikit-learn
scipy
matplotlib
```

---

## Project Structure

```
NeuroLink V4.0/
├── NeuroLink_V4_0.ipynb        # Main notebook — all code end to end
├── README.md
└── .gitignore
```

> `typing-data.json` and `transformed_typing_features_v4.csv` are excluded from the repo — they contain personal behavioral data.  
> Trained model weights (`.pt`) and scaler (`.pkl`) are excluded due to file size — regenerate by running the notebook.

---

## Roadmap

- [ ] **V5** — Sequential encoder (GRU over sliding window of last 30 lessons)
- [ ] Character-level mastery submodel
- [ ] Comparison baselines vs Isolation Forest, One-Class SVM, statistical process control
- [ ] Multi-user validation (N-of-1 framework, multiple participants)
- [ ] Browser extension for passive keystroke collection
- [ ] arXiv preprint

---

## Research Direction

The long-term hypothesis: **typing behavioral drift is a passive, early signal for neurological change**.

Clinical literature establishes that fine motor degradation precedes clinical symptoms in conditions such as early Parkinson's disease and MS. Everyday typing is fine motor behavior. A system that monitors your personal baseline continuously — privately, on-device, without behavior change — could surface anomaly patterns worth discussing with a doctor long before traditional screening would.

*This is a research direction, not a medical device. Nothing in this system constitutes medical advice.*

---

## Citation

If you use or reference this work:

```
Zuaiter, Y. (2026). NeuroLink: A Privacy-Preserving Multi-Stream Behavioral
Baseline Engine for Personal Typing Performance Monitoring.
GitHub: https://github.com/youssefzuaiter/neurolink
```

---

## License

MIT License — see `LICENSE` for details.
