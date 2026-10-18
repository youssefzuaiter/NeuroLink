# NeuroLink V4 — Privacy-Preserving Multi-Stream Behavioral Baseline Engine

**Author:** Youssef Zuaiter  
**Data:** 50,317 keybr.com lessons · 492 days (Dec 2024 – Jul 2026)  
**Website:** [youssefzuaiter.github.io/NeuroLink](https://youssefzuaiter.github.io/NeuroLink)  
**Dataset:** [Kaggle — NeuroLink Personal Typing Behavioral Features](https://www.kaggle.com/datasets/youssefsufianzuaiter/neurolink-personal-typing-behavioral-features)

---

## What is NeuroLink?

**NeuroLink** is a personal AI system that learns your unique typing behavioral signature and detects when you deviate from it. It is trained exclusively on your own data — no population averages, no other people's baselines. The model is a true **N-of-1 system**: your baseline is yours alone.

The name comes from the idea of linking the model directly to your neurology. Typing is a fine motor behavior controlled by the brain and nervous system. The hypothesis behind NeuroLink is that changes in how you type — not just how fast, but your accuracy patterns, warmup characteristics, hand coordination, and time-of-day rhythms — can serve as a passive, continuous window into neurological state. NeuroLink builds a mathematical model of who you are as a typist, then watches for the day you start drifting away from it.

---

## The Research Direction

The long-term hypothesis is this: **typing behavioral drift is a passive, early signal for neurological change.**

Fine motor degradation precedes clinical symptoms in conditions such as early Parkinson's disease, multiple sclerosis, and other neurodegenerative disorders. Everyday typing is fine motor behavior — a task most knowledge workers perform thousands of times a day, without thinking about it. If a model can learn the subtle statistical fingerprint of your typing, it may be able to detect deviations long before they become noticeable.

NeuroLink does not diagnose anything. It is a research project, not a medical device. But the detection results from December 2024 — a real period when typing performance dropped sharply and the model flagged it at 89% — show that the core idea is empirically traceable.

> *Nothing in this project constitutes medical advice.*

---

## What the Website Shows

The research website at [youssefzuaiter.github.io/NeuroLink](https://youssefzuaiter.github.io/NeuroLink) presents the project as a public research paper. It covers:

- The abstract and key findings in plain language
- The three behavioral streams (Speed, Accuracy, Motor control) and what each one captures
- An animated timeline of WPM drift over 492 days
- The full feature table with plain-language descriptions
- A comparison of all five detection methods (SPC, Isolation Forest, One-Class SVM, Non-DP Autoencoder, NeuroLink V4)
- A step-by-step guide for running NeuroLink on your own keybr.com data
- The V5 research roadmap

The website is designed in the style of a Distill.pub research paper — clean, typeset, readable by both ML researchers and people with no technical background.

---

## What V4 Does

V4 decomposes the single-stream autoencoder from earlier versions into three independent behavioral streams, each with its own decoder head. This lets the model separately track Speed, Accuracy, and Motor control — producing a per-stream **Longitudinal Drift Index (LDI)** with Bootstrap confidence intervals and multi-scale anomaly alerting (7, 30, and 90-day windows).

### Architecture

```
Input (15 features)
        │
        ▼
┌─────────────────────────────┐
│      Shared Encoder         │
│   15 → 32 → 16 → 10 (Tanh) │  ← Behavioral Signature (latent space)
└─────────────────────────────┘
        │
   ┌────┴────┬──────────────┐
   ▼         ▼              ▼
Speed     Accuracy        Motor
Head      Head            Head
10→16→5   10→16→5         10→16→5
```

The shared encoder compresses 15 behavioral features into a 10-dimensional latent vector — a compact representation of your typing signature at that point in time. Each decoder head reconstructs its own subset of features. High reconstruction error in a specific head means that stream has drifted from baseline.

### 15 Features across 3 Streams

| Stream | Features | What It Captures |
|--------|----------|-----------------|
| **Speed** | EV, RSV, wpm_vs_50roll, improvement_rate, warmup_effect | Overall pace, volatility, trend, how fast you warm up |
| **Accuracy** | error_entropy, speed_var_wl, worst_char_improvement, miss_ratio_top3, error_rate_trend | Which characters you struggle with, error spread, improvement over time |
| **Motor** | CTA, CTA_cos, hand_asymmetry, time_of_day, speed_dispersion | Time-of-day rhythm, left/right hand balance, within-session speed variability |

### Privacy

Differential Privacy via **Opacus** (DP-SGD): ε = 9.997, δ = 1e-5, RDP accountant, MAX_GRAD_NORM = 1.0.

The ε = 10 budget is appropriate for this threat model. The data is yours alone — there is no population to leak against. Even if the trained model were extracted, it could only reveal information about one person who already knows their own typing behavior. The privacy guarantee is still meaningful: it bounds the influence of any single session on the model's weights, preventing a bad day from permanently skewing your baseline.

All training is done locally. Raw data never leaves your machine.

---

## Results

| Metric | Value |
|--------|-------|
| Validation R² | 0.9751 |
| Test R² | 0.8922 |
| DP budget | ε = 9.997, δ = 1e-5 |
| Best epoch | 112 |

### Baseline Comparison — Dec 2024 Anomaly Period

| Method | Anomaly Rate | WPM Gap | Dec 2024 Detection |
|--------|-------------|---------|-------------------|
| SPC (rolling ±2σ) | 0.0% | +0.0 WPM | 0% |
| Isolation Forest | 9.6% | +39.8 WPM | 67% |
| One-Class SVM | 7.9% | +51.0 WPM | 78% |
| Non-DP Autoencoder | 8.5% | +46.6 WPM | 100% |
| **NeuroLink V4 (DP)** | **8.5%** | **+45.9 WPM** | **89%** |

NeuroLink V4 achieves 89% detection on the Dec 2024 anomaly period while maintaining formal differential privacy — losing only 11 percentage points compared to the non-private autoencoder.

---

## What Changed from V3

| | V3 | V4 |
|--|----|----|
| Decoder | Single head | 3 stream-specific heads |
| LDI | 1 global signal | 3 independent streams |
| Loss function | Weighted MSE | Huber loss + head-size normalization |
| Forecasting | — | 90-day log-growth curve + LDI adjustment |
| Anomaly scales | Single window | 7 / 30 / 90-day multi-scale |

---

## How to Run It on Your Own Data

1. Export your keybr.com data: Profile → Export → `typing-data.json`
2. Run the data-prep cell at the top of `NeuroLink_V4_0.ipynb` to build `transformed_typing_features_v4.csv`
3. Upload the notebook + CSV to Google Colab, set runtime to GPU (T4), run all cells

**Requirements:** `torch`, `opacus>=1.4.0`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`

> `typing-data.json` and `transformed_typing_features_v4.csv` are excluded from this repo — personal behavioral data. Trained weights (`.pt`) and scaler (`.pkl`) are excluded — regenerate by running the notebook.

---

## The Kaggle Challenge

The 15-feature daily dataset is publicly available on Kaggle. The challenge: can you build a better anomaly detector than NeuroLink V4?

**Baseline to beat:** 89% detection on the Dec 2024 anomaly period, WPM gap −45.9 (p<0.001)

A starter notebook is included that reproduces all baselines with a standardized evaluation protocol so every submission is comparable.

---

## V5 Roadmap

The main limitation of V4 is the ε = 10 privacy budget. With only ~50k personal lessons, the model cannot be trained to lower ε without destroying utility. V5 will address this with:

1. **Public pre-training** on large open keystroke datasets (CMU Keystroke, CLARKSON II), which do not consume personal privacy budget
2. **DP fine-tuning** on personal data to personalize the pre-trained model — targeting ε ≤ 3
3. **Sequential encoder** (GRU/LSTM) to capture temporal structure across sessions
4. **Multi-user validation** — 3–5 people running V4 on their own data independently

V5 is planned for early 2027 (~100k personal lessons accumulated).

---

## License

MIT License — see `LICENSE`.  
Dataset: CC BY 4.0 (Kaggle).
