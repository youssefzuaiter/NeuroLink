# NeuroLink v1 — NeuroLink Analytics

**Predictive Performance Modeling & Motor Skill Forecasting**

The first version of NeuroLink. Treats typing as a high-frequency data stream and models the acquisition of muscle memory from raw session history exported from [keybr.com](https://www.keybr.com).

- **Data:** 20,000+ personal typing sessions.
- **Method:** Linear Regression (scikit-learn) to project the session count required to hit a target WPM/accuracy; character-level latency and error "hotspot" analysis; statistical detection of a warm-up effect.
- **Output:** projected milestone (e.g. 130 WPM) with an estimated date and required practice volume.
- **Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

See `README.pdf` for the original write-up. Raw session CSV is excluded (personal data / repo size) — regenerate by exporting your own keybr.com history and pointing the notebook at it.

**Files:** `NeuroLink_Analytics.ipynb` — full analysis notebook.
