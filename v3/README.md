# NeuroLink v3

**A Privacy-Preserving N-of-1 Deep Autoencoder for Longitudinal Behavioral Drift Detection in Keystroke Dynamics**

Youssef Zuaiter · Bahçeşehir University

> Paper preprint: link forthcoming — update once the arXiv submission is live. Draft LaTeX source and PDF are in `patent-proposal/`.

## Overview

NeuroLink v3 learns a personalized baseline of an individual's own keystroke behavior and flags sessions that deviate from it, using reconstruction error from an undercomplete autoencoder. Unlike population-based keystroke authentication systems, it follows an **N-of-1** design: the model is trained exclusively on one person's longitudinal typing history, so it detects *within-person* behavioral drift (fatigue, stress, change over time) rather than *cross-user* identity discrimination.

- **Dataset:** 42,501 typing lessons from [keybr.com](https://www.keybr.com), spanning 17 months (Dec 2024 – May 2026), a single participant.
- **Architecture:** symmetric undercomplete autoencoder, `12 → 32 → 16 → 10 → 16 → 32 → 12`, 2,230 trainable parameters, 10-dimensional *Muscle Memory Signature* bottleneck.
- **Privacy:** trained with DP-SGD via [Opacus](https://github.com/pytorch/opacus) (RDP accounting), achieving **ε = 7.9983** at **δ = 10⁻⁵**.
- **Result:** variance-weighted validation **R² = 0.9586**, test **R² = 0.9604**.

## Key contributions

1. **N-of-1 undercomplete autoencoder** — trained on 17 months of personal keystroke data with a 10-dimensional bottleneck, learning individual behavioral regularity with no population-norm reference.
2. **Residual feature reformulation** — replacing rolling-mean Execution Velocity (EV) and Error-Speed Recovery Penalty (ESRP) with their *residual* (deviation-from-trend) form resolves a structural reconstruction ceiling. Raises EV R² from 0.234 → 0.954 and ESRP R² from 0.113 → 0.854.
3. **Differential privacy** via Opacus DP-SGD with RDP accounting.
4. **Bootstrap confidence-interval alerting** — 2,000 bootstrap resamples of training-period drift values produce a 95% CI band, enabling three-tier anomaly classification (HIGH / MARGINAL / NORMAL).
5. **Empirical Ghost V3 predictor** — a drift-adjusted typing-pace estimator whose scale factor is fit directly from data via linear regression.

## Ablation results

| Variant | Val R² | EV R² | ESRP R² | Best Epoch |
|---|---|---|---|---|
| v4 (rolling-mean EV/ESRP) | 0.9502 | 0.234 | 0.113 | 47 |
| v5 (residual + LR 5e-3) | 0.8860 | 0.948 | 0.730 | 10 |
| **v5.1 (+ ReduceLROnPlateau + weights)** | **0.9586** | **0.954** | **0.854** | **54** |

All metrics evaluated on the held-out test split (final 7.5% of lessons, chronological split).

## Contents

| Path | Description |
|---|---|
| `NeuroLink_3_0_v5_1_FINAL.ipynb` | End-to-end notebook: data loading, feature engineering, DP-SGD training, evaluation, LDI/anomaly detection, Ghost V3, full dashboard. Outputs pre-computed — no re-execution needed to inspect results. |
| `NeuroLink V3.0 Results.pdf` | Standalone results writeup. |
| `patent-proposal/` | LaTeX source + compiled `Paper.pdf` for the associated paper/patent draft (abstract, intro, related work, method, experiments, conclusion). |
| `requirements.txt` | Exact Python dependencies. |
| `LICENSE` | MIT License. |

`transformed_typing_features.csv` (the 12 engineered behavioral features per lesson used to train/evaluate) is excluded from the repo — personal data / repo size.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook NeuroLink_3_0_v5_1_FINAL.ipynb
```

## Citation

```bibtex
@article{zuaiter2026neurolink,
  title={NeuroLink 3.0: A Privacy-Preserving N-of-1 Deep Autoencoder for Longitudinal Behavioral Drift Detection in Keystroke Dynamics},
  author={Zuaiter, Youssef},
  journal={arXiv preprint},
  year={2026},
  note={Link forthcoming}
}
```

## License

MIT License — see `LICENSE`.
