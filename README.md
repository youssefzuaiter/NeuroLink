# NeuroLink

NeuroLink is a personal AI system that learns your unique typing behavioral signature from [keybr.com](https://www.keybr.com) session history and detects when you drift away from it. It's an **N-of-1** project: every model is trained exclusively on one person's own data, not a population baseline. The project has gone through four public iterations, each in its own subfolder below, with growing sophistication from simple regression analytics to a differentially-private multi-stream autoencoder.

→ **New here? Start with [`v4/`](v4/)** for the current iteration, or [`v3/`](v3/) for the version with the highest reported test R² and the accompanying paper draft.

## Versions

| Version | Core idea | Key result |
|---|---|---|
| [v1](v1/) | Linear-regression analytics on raw session history | Projects sessions-to-target-WPM from 20k+ sessions |
| [v2](v2/) | Statistical "Ghost" typing simulator | Predicts completion time for any text from a personal WPM/error profile, 30k+ sessions |
| [v3](v3/) | N-of-1 undercomplete autoencoder + DP-SGD | Test R² = 0.9604, ε = 7.9983 differential privacy; includes paper/patent draft |
| [v4](v4/) | Multi-stream autoencoder (Speed / Accuracy / Motor heads) | Validation R² = 0.9751; per-stream drift detection + 90-day WPM forecast |

Note: v3's headline R² (0.9604, test split) and v4's headline R² (0.9751, validation split) aren't measured on the same split, so they aren't directly comparable at face value — read each subfolder's README for the full methodology before treating one as strictly "better."

## Repository layout

```
neurolink/
├── v1/   NeuroLink Analytics — regression-based forecasting
├── v2/   Ghost Prediction Engine — typing-time simulator
├── v3/   N-of-1 autoencoder + DP-SGD, incl. patent-proposal/ (paper draft)
├── v4/   Multi-stream autoencoder, per-stream drift + forecasting
└── README.md
```

Each subfolder has its own README with that version's method, results, and how to run it.

## Data privacy

Raw typing session exports (`typing-data.json`, `*.csv` feature files) are excluded from every version — they contain one person's personal behavioral data and, for the larger versions, exceed a size that belongs in git history. Regenerate them by exporting your own [keybr.com](https://www.keybr.com) history and running the data-prep cell at the top of the relevant notebook.

## License

MIT — see [`v3/LICENSE`](v3/LICENSE).

## TRY IT YOURSELF, TEST YOUR DATA 
(https://youssefzuaiter.github.io/NeuroLink/)
