# NeuroLink v2 — Ghost Prediction Engine

**Personal Typing Performance Analytics & Ghost Prediction Engine**

Builds on v1 by turning the analytics into a simulator: a personalized "Ghost" that predicts how long *you specifically* would take to type any given text.

- **Data:** 30,000+ typing sessions.
- **Pipeline:** load/sort session history chronologically → compute a personal profile (recent average WPM over the last 500 sessions, an error penalty fit via linear regression) → simulate typing of arbitrary input text using that profile (base time from WPM, simulated errors, individualized time penalty) → output a predicted completion time.
- **Extra:** a built-in practice-sentence generator so the tool doubles as an interactive typing trainer, not just an offline analysis script.
- **Stack:** Python, Pandas, NumPy, Scikit-learn (Linear Regression).

See `README.pdf` for the original write-up. Raw session CSV (`cleaned_typing_data_v30k.csv`) is excluded (personal data / repo size).

**Files:** `NeuroLink_V2_0.ipynb` — full analysis + Ghost simulator notebook.
