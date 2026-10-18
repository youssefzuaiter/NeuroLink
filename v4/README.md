# NeuroLink — Privacy-Preserving Behavioral Baseline Engine

**Author:** Youssef Zuaiter  
**Current version:** V4  
**Data:** 50,317 keybr.com lessons · 492 days (Dec 2024 – Jul 2026)  
**Website:** [youssefzuaiter.github.io/NeuroLink](https://youssefzuaiter.github.io/NeuroLink)  
**Dataset:** [Kaggle — NeuroLink Personal Typing Behavioral Features](https://www.kaggle.com/datasets/youssefsufianzuaiter/neurolink-personal-typing-behavioral-features)

---

## What is NeuroLink?

**NeuroLink** is a personal AI system that learns your unique typing behavioral signature and detects when you deviate from it. It is trained exclusively on your own data — no population averages, no other people's baselines. The model is a true **N-of-1 system**: your baseline is yours alone.

The name comes from the idea of linking the model directly to your neurology. Typing is a fine motor behavior controlled by the brain and nervous system. The hypothesis behind NeuroLink is that changes in how you type — not just how fast, but your accuracy patterns, warmup characteristics, hand coordination, and time-of-day rhythms — can serve as a passive, continuous window into neurological state. NeuroLink builds a mathematical model of who you are as a typist, then watches for the day you start drifting away from it.

---

## The Research Direction

The long-term hypothesis: **typing behavioral drift is a passive, early signal for neurological change.**

Fine motor degradation precedes clinical symptoms in conditions such as early Parkinson's disease, multiple sclerosis, and other neurodegenerative disorders. Everyday typing is fine motor behavior — a task most knowledge workers perform thousands of times a day, without thinking about it. If a model can learn the subtle statistical fingerprint of your typing, it may be able to detect deviations long before they become noticeable.

NeuroLink does not diagnose anything. It is a research project, not a medical device. But the detection results from December 2024 — a real period when typing performance dropped sharply and the model flagged it at 89% — show that the core idea is empirically traceable.

> *Nothing in this project constitutes medical advice.*

---

## Version History

NeuroLink went through four major versions. Each version was a response to a concrete limitation discovered in the previous one.

---

### V1 — NeuroLink Analytics (2024)

**What it was:** A statistical dashboard, not a model. V1 had no machine learning — it was a data analysis and visualization pipeline that processed raw keybr.com exports and produced a performance summary with a linear WPM forecast.

**Architecture:**
```
Raw typing-data.json
        │
        ▼
Session aggregation (WPM, accuracy per session)
        │
        ▼
Linear Regression forecast (toward 130 WPM target)
        │
        ▼
Dashboard: WPM trend · accuracy · goal forecast · circadian efficiency
```

**Features used:** WPM per session, raw accuracy, session count.

**What it could do:**
- Plot WPM and accuracy trends over time
- Forecast when you would hit a target WPM using linear regression
- Show circadian efficiency (WPM by hour of day)
- Compute a mastery matrix: volatility, error penalty, longest perfect streak

**What it could not do:** Detect anomalies. It had no concept of "normal vs abnormal" — it only tracked trend.

**Why it was replaced:** A linear forecast toward a goal WPM misses the entire point. Progress is not linear. The interesting signal is not slope — it is deviation from your own behavioral norm. V2 introduced the concept of a behavioral signature.

---

### V2 — NeuroLink 2.0 + Spectre Engine (2025)

**What it was:** A two-class architecture — `NeuroLinkEngine` (behavioral metrics) and `SpectreEngine` (deep telemetry visualization). V2 introduced more behavioral features and a Ghost simulator — a predicted version of yourself that you could race against.

**Architecture:**
```
CSV session data (30k+ lessons)
        │
        ▼
NeuroLinkEngine
  ├── Neural metrics: avg WPM, error penalty (linear fit)
  ├── Mastery matrix: volatility, streak, recovery
  └── Ghost simulator: predict time for a text given your neural metrics

SpectreEngine
  ├── Hexbin density plot (WPM progression)
  ├── Circadian WPM by hour (boxplot)
  ├── Fatigue curve (WPM vs session-of-day)
  └── Error heat map per character
```

**Features used:** WPM, accuracy, errors, session-of-day, hour-of-day, character-level hit/miss counts.

**Ghost Simulator:** Given a text, predict how long you would take to type it based on your recent WPM average and error penalty (modeled as `base_time + errors × penalty / 10`).

**What it could do:**
- Detect circadian patterns (what time of day you type fastest)
- Model fatigue across a session (how WPM changes as sessions accumulate)
- Simulate a "ghost" of yourself from your own statistics

**What it could not do:** Still no anomaly detection. The Ghost scale factor (`0.05`) was a hardcoded guess, not derived from data. There was no formal notion of a behavioral baseline.

**Why it was replaced:** Statistical dashboards and simple linear models cannot capture the structure of typing behavior over time. The behavioral fingerprint is multivariate — speed, accuracy, and motor patterns are correlated in complex ways. V3 introduced the first neural model: an autoencoder.

---

### V3 — NeuroLink 3.0 (2025–2026)

**What it was:** The first real machine learning version. V3 introduced a single-stream autoencoder trained with differential privacy, a Longitudinal Drift Index (LDI), and Bootstrap confidence intervals for anomaly alerting.

**Architecture:**
```
Input (12 features)
        │
        ▼
Encoder: 12 → 32 → 16 → 10 (Tanh)   ← Muscle Memory Signature (latent space)
        │
        ▼
Decoder: 10 → 16 → 32 → 12           ← Reconstruction
        │
        ▼
Reconstruction error (MSE per feature)
        │
        ▼
LDI (7-day rolling mean of per-lesson MSE)
        │
        ▼
3-tier Bootstrap CI alerting: HIGH / MARGINAL / NORMAL
```

**Total parameters:** 2,230  
**Privacy:** DP-SGD via Opacus · ε ≤ 8.0, δ = 1e-5

**12 Features:**

| Feature | What It Captures |
|---------|-----------------|
| `EV` | Residual: WPM minus 7-lesson rolling mean (execution surprise) |
| `RSV` | Rolling std of WPM over 30 lessons (speed volatility) |
| `SSF` | Sequential stamina fatigue |
| `ESRP` | Residual: error penalty minus 5-lesson rolling mean (accuracy surprise) |
| `CTA` | Sin of circadian time (time-of-day rhythm) |
| `CTA_cos` | Cos of circadian time |
| `wpm_vs_50roll` | WPM deviation from 50-lesson mean |
| `lesson_dur_z` | log1p(session duration) |
| `active_char_frac` | Curriculum depth (fraction of alphabet active) |
| `hand_asymmetry` | Left vs right hand speed ratio |
| `error_entropy` | Spread of errors across characters |
| `warmup_effect` | WPM change from session 1 to session 2 each day |

**Loss function:** Weighted MSE (hand-tuned feature weights)  
**Scheduler:** ReduceLROnPlateau (v5.1) — drops LR × 0.5 after 5 stagnant epochs  
**Anomaly method:** Bootstrap resampling (2,000 iterations) on training LDI to derive 95% CI thresholds  
**Ghost V3:** Empirically calibrated — the LDI-to-WPM scale factor is derived from a linear fit across all training sessions instead of a hardcoded guess  
**Adaptive baseline:** 90-day rolling window that re-derives the threshold from recent stable behavior, preventing long-term drift from being absorbed into the baseline

**What it could do:**
- Train a differentially private autoencoder on your personal typing data
- Compute a daily Longitudinal Drift Index (LDI) as average reconstruction error
- Alert at three tiers: HIGH (above upper CI), MARGINAL (between CI bounds), NORMAL
- Adapt the threshold over time to account for genuine skill improvement
- Predict WPM given a target LDI ratio using an empirically calibrated Ghost

**What it could not do:** The single decoder head mixed Speed, Accuracy, and Motor signals together. A bad accuracy day and a bad motor day produced the same reconstruction error signal — you could not tell which stream was responsible. V4 separates these.

**Why it was replaced:** Stream-level explainability. When an anomaly fires, you want to know *which* behavioral dimension changed — was it speed, accuracy, or motor coordination? That requires separate decoder heads.

---

### V4 — NeuroLink V4 (Current)

**What it is:** A multi-head autoencoder with three independent behavioral stream decoders, trained with differential privacy. Each decoder is responsible for its own subset of features, producing a per-stream LDI with independent anomaly thresholds and multi-scale alerting.

**Architecture:**
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

**Privacy:** DP-SGD via Opacus · ε = 9.997, δ = 1e-5, RDP accountant, MAX_GRAD_NORM = 1.0

**15 Features across 3 Streams:**

| Stream | Features | What It Captures |
|--------|----------|-----------------|
| **Speed** | `EV`, `RSV`, `wpm_vs_50roll`, `improvement_rate`, `warmup_effect` | Pace, volatility, trend, warmup behavior |
| **Accuracy** | `error_entropy`, `speed_var_wl`, `worst_char_improvement`, `miss_ratio_top3`, `error_rate_trend` | Error spread, hardest characters, improvement over time |
| **Motor** | `CTA`, `CTA_cos`, `hand_asymmetry`, `time_of_day`, `speed_dispersion` | Time-of-day rhythm, left/right balance, within-session variability |

**Loss function:** Huber loss normalized by head size (`stream_loss`) — reduces sensitivity to outlier sessions compared to MSE  
**Anomaly detection:** Per-stream LDI with Bootstrap CI thresholds at three temporal scales: 7-day, 30-day, 90-day  
**Forecasting:** 90-day log-growth WPM forecast adjusted by per-stream LDI

**Results:**

| Metric | Value |
|--------|-------|
| Validation R² | 0.9751 |
| Test R² | 0.8922 |
| Best epoch | 112 |
| DP budget | ε = 9.997, δ = 1e-5 |

**Baseline comparison — Dec 2024 anomaly period:**

| Method | Anomaly Rate | WPM Gap | Dec 2024 Detection |
|--------|-------------|---------|-------------------|
| SPC (rolling ±2σ) | 0.0% | +0.0 WPM | 0% |
| Isolation Forest | 9.6% | +39.8 WPM | 67% |
| One-Class SVM | 7.9% | +51.0 WPM | 78% |
| Non-DP Autoencoder | 8.5% | +46.6 WPM | 100% |
| **NeuroLink V4 (DP)** | **8.5%** | **+45.9 WPM** | **89%** |

---

## Version Comparison

| | V1 | V2 | V3 | V4 |
|--|----|----|----|----|
| **Model type** | Linear regression | Statistical engine | Single-stream AE | Multi-head AE |
| **Features** | 2 (WPM, accuracy) | ~6 (WPM, errors, circadian) | 12 | 15 |
| **Streams** | None | None | 1 global | 3 independent |
| **Anomaly detection** | None | None | LDI + Bootstrap CI | Per-stream LDI + multi-scale CI |
| **Differential privacy** | No | No | ε ≤ 8.0 | ε = 9.997 |
| **Loss function** | MSE | N/A | Weighted MSE | Huber + head-size normalization |
| **Forecasting** | Linear toward goal WPM | Ghost (statistical) | Ghost V3 (empirical) | 90-day log-growth + LDI |
| **Parameters** | ~3 | ~0 | 2,230 | ~3,500 |
| **Explainability** | None | None | Feature-level MSE | Stream-level LDI |

---

## What the Website Shows

The research website at [youssefzuaiter.github.io/NeuroLink](https://youssefzuaiter.github.io/NeuroLink) presents the project as a public research paper. It covers the full V4 results, all five detection method comparisons, the feature table, a 492-day WPM timeline, and a step-by-step guide for running NeuroLink on your own keybr.com data.

---

## How to Run V4 on Your Own Data

1. Export your keybr.com data: Profile → Export → `typing-data.json`
2. Run the data-prep cell at the top of `NeuroLink_V4_0.ipynb` to build `transformed_typing_features_v4.csv`
3. Upload notebook + CSV to Google Colab, set runtime to GPU (T4), run all cells

**Requirements:** `torch`, `opacus>=1.4.0`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`

> `typing-data.json` and `transformed_typing_features_v4.csv` are excluded — personal behavioral data. Trained weights (`.pt`) and scaler (`.pkl`) are excluded — regenerate by running the notebook.

---

## The Kaggle Challenge

The 15-feature dataset is public on Kaggle. Can you build a better anomaly detector than NeuroLink V4?

**Baseline to beat:** 89% detection on Dec 2024, WPM gap −45.9 (p<0.001)

A starter notebook reproduces all baselines with a standardized evaluation protocol.

---

## V5 Roadmap

V5 will address the ε = 10 privacy budget by pre-training on large public keystroke datasets (CMU Keystroke, CLARKSON II) before DP fine-tuning on personal data — targeting ε ≤ 3. A sequential encoder (GRU/LSTM) will replace the feedforward encoder to capture temporal structure across sessions. Planned for early 2027 (~100k personal lessons).

---

## License

MIT License — see `LICENSE`.  
Dataset: CC BY 4.0 (Kaggle).
