"""
DataProcessing.py
-----------------
Generates synthetic keyboard/text-input behavioural data and splits it
into per-node training files that simulate a federated setting.

Privacy note: No raw text is stored. Only behavioural statistics are used.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

N_NORMAL  = 1400
N_ANOMALY = 600
N_NODES   = 9   # one training file per MPI worker

# ── Synthetic normal sessions ──────────────────────────────────────────────
normal = pd.DataFrame({
    "keypress_duration": rng.normal(80, 15, N_NORMAL).clip(20, 300),
    "inter_key_delay":   rng.normal(120, 30, N_NORMAL).clip(10, 600),
    "error_rate":        rng.beta(2, 20, N_NORMAL),
    "session_length":    rng.integers(50, 500, N_NORMAL).astype(float),
    "swipe_speed":       rng.normal(1.2, 0.3, N_NORMAL).clip(0, 5),
    "label":             np.zeros(N_NORMAL, dtype=int),
})

# ── Synthetic anomalous sessions ──────────────────────────────────────────
anomaly = pd.DataFrame({
    "keypress_duration": rng.normal(30, 10, N_ANOMALY).clip(5, 120),
    "inter_key_delay":   rng.normal(20, 8, N_ANOMALY).clip(2, 100),
    "error_rate":        rng.beta(1, 3, N_ANOMALY),
    "session_length":    rng.integers(5, 50, N_ANOMALY).astype(float),
    "swipe_speed":       rng.normal(3.5, 0.5, N_ANOMALY).clip(0, 6),
    "label":             np.ones(N_ANOMALY, dtype=int),
})

# ── Combine & shuffle ─────────────────────────────────────────────────────
df = pd.concat([normal, anomaly], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ── Train / test split ────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

# Save test data
test_df.to_csv("test_dataset.txt", index=False)
print(f"[DataProcessing] Test set  : {len(test_df)} rows -> test_dataset.txt")

# ── Split training data across nodes (FIXED) ──────────────────────────────
splits = np.array_split(train_df, N_NODES)

for i, split in enumerate(splits):
    fname = f"train_dataset_{i}.txt"

    # Ensure split is always a DataFrame (important fix)
    split_df = pd.DataFrame(split, columns=train_df.columns)

    split_df.to_csv(fname, index=False)
    print(f"[DataProcessing] Node {i} data: {len(split_df)} rows -> {fname}")

print("\n Dataset creation complete.")
print("Features : keypress_duration, inter_key_delay, error_rate, session_length, swipe_speed")
print("Label    : 0 = normal session | 1 = anomalous / bot-like session")