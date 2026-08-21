# NeuroLink v4 — Multi-Stream Behavioral Baseline Engine

### *Privacy-Preserving Multi-Stream Behavioral Baseline Engine for Typing Performance*

> **Data:** 50,321 keybr.com lessons · Dec 2024 – Jul 2026
> **Author:** Youssef Zuaiter

## What is NeuroLink?

NeuroLink is a personal AI system that learns your unique typing behavioral signature and detects when you drift away from it. Unlike population-based models, it is trained **exclusively on your own data** — a true N-of-1 system.

v4 decomposes the single-stream v3 autoencoder into three independent behavioral streams — **Speed**, **Accuracy**, and **Motor control** — each with its own decoder head, producing a per-stream Longitudinal Drift Index (LDI) with anomaly alerting and a 90-day WPM forecast.

## Architecture

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
```

### 15 Features across 3 Streams

| Stream | Features |
|---|---|
| **Speed** | EV, RSV, wpm_vs_50roll, improvement_rate, warmup_effect |
| **Accuracy** | error_entropy, speed_var_wl, worst_char_improvement, miss_ratio_top3, error_rate_trend |
| **Motor** | CTA, CTA_cos, hand_asymmetry, time_of_day, speed_dispersion |

### Privacy
Differential Privacy via **Opacus** (DP-SGD), ε = 10.0, δ = 1e-5, RDP accountant. All training done locally — raw data never leaves your machine.

## Results

| Metric | Value |
|---|---|
| Validation R² | 0.9751 |
| Test R² | 0.8922 |
| Best epoch | 112 |
| DP budget | ε = 9.997, δ = 1e-5 |

## What's new vs v3

| | v3 | v4 |
|---|---|---|
| Decoder | Single head | 3 stream-specific heads |
| LDI | 1 global signal | 3 independent streams |
| Loss function | Weighted MSE | Huber loss + head-size normalization |
| Forecasting | — | 90-day log-growth curve + LDI adjustment |

## How to run

1. Export your keybr.com data: Profile → Export → `typing-data.json`.
2. Run the data-prep cell at the top of `NeuroLink_V4_0.ipynb` to build `transformed_typing_features_v4.csv`.
3. Upload notebook + CSV to Google Colab, set runtime to GPU (T4), run all cells.

Requirements: `torch`, `opacus>=1.4.0`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.

`typing-data.json` and `transformed_typing_features_v4.csv` are excluded — personal behavioral data. Trained weights (`.pt`) and scaler (`.pkl`) are excluded — regenerate by running the notebook.

## Research direction

The long-term hypothesis: typing behavioral drift is a passive, early signal for neurological change. Fine motor degradation precedes clinical symptoms in conditions such as early Parkinson's disease and MS; everyday typing is fine motor behavior. *This is a research direction, not a medical device — nothing here constitutes medical advice.*

## License

MIT License — see the repo-level `LICENSE`.
