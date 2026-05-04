"""
NormalML.py
-----------
Centralised (non-federated) baseline for keyboard-behaviour anomaly detection.
"""

import glob
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROUNDS = 1


def load_csv(filepath: str):
    df = pd.read_csv(filepath)
    y = df["label"].values.astype(int)
    X = df.drop(columns=["label"]).values.astype(float)
    return X, y


def print_line(width=60):
    print("=" * width)


def print_subline(width=60):
    print("-" * width)


def print_kv(key, value):
    print(f"{key:<20}: {value}")


def print_table(headers, rows):
    col_widths = [max(len(h), 12) for h in headers]

    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(f"{val:.4f}" if isinstance(val, float) else str(val)))

    print_subline(sum(col_widths) + len(headers) * 3)

    header_row = ""
    for i, h in enumerate(headers):
        header_row += f"{h:<{col_widths[i]}} | "
    print(header_row.rstrip())

    print_subline(sum(col_widths) + len(headers) * 3)

    for row in rows:
        row_str = ""
        for i, val in enumerate(row):
            if isinstance(val, float):
                row_str += f"{val:<{col_widths[i]}.4f} | "
            else:
                row_str += f"{val:<{col_widths[i]}} | "
        print(row_str.rstrip())

    print_subline(sum(col_widths) + len(headers) * 3)


# ── Load Data ─────────────────────────────────────────────
train_files = sorted(glob.glob("train_dataset_*.txt"))
if not train_files:
    raise FileNotFoundError("Run DataProcessing.py first.")

Xs, ys = zip(*[load_csv(f) for f in train_files])
X_train = np.vstack(Xs)
y_train = np.concatenate(ys)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test_raw, y_test = load_csv("test_dataset.txt")
X_test = scaler.transform(X_test_raw)

# ── Header ───────────────────────────────────────────────
print_line()
print("CENTRALISED MACHINE LEARNING BASELINE".center(60))
print_line()

print_kv("Training Files", len(train_files))
print_kv("Training Samples", len(y_train))
print_kv("Test Samples", len(y_test))
print_kv("Rounds", ROUNDS)

# ── Training ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, class_weight="balanced")

round_results = []
total_start = time.time()

for r in range(ROUNDS):
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()

    round_results.append([r + 1, t1 - t0])

total_time = time.time() - total_start

# ── Round Table ──────────────────────────────────────────
print("\nTRAINING SUMMARY")
headers = ["Round", "Training Time (s)"]
print_table(headers, round_results)

# ── Evaluation ───────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# ── Final Results ────────────────────────────────────────
print("\nFINAL RESULTS")
print_subline()
print_kv("Accuracy", f"{acc:.4f}")
print_kv("F1 Score", f"{f1:.4f}")
print_kv("Total Time (s)", f"{total_time:.4f}")

# ── Confusion Matrix ─────────────────────────────────────
print("\nCONFUSION MATRIX")
headers = ["", "Pred 0", "Pred 1"]
rows = [
    ["Actual 0", cm[0][0], cm[0][1]],
    ["Actual 1", cm[1][0], cm[1][1]]
]
print_table(headers, rows)

# ── Save Model ───────────────────────────────────────────
joblib.dump(model, "normal_model.pkl")

print("\nOUTPUT FILES")
print_subline()
print("Model saved   : normal_model.pkl")
print_line()