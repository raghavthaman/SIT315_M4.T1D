import glob
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

# ====================== CONFIGURATION ======================
ROUNDS      = 3
EPSILON     = 1.0       # differential privacy budget
SENSITIVITY = 1.0       # Laplace mechanism sensitivity

FEATURE_NAMES = [
    "keypress_duration",
    "inter_key_delay",
    "error_rate",
    "session_length",
    "swipe_speed",
]


# ====================== HELPERS ======================

def print_line(char="=", width=65):
    print(char * width)

def print_header(title):
    print("\n" + "=" * 65)
    print(title.center(65))
    print("=" * 65)

def print_kv(key, value):
    print(f"  {key:<28}: {value}")

def print_table(headers, rows):
    col_widths = [max(len(str(h)), 12) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(
                col_widths[i],
                len(f"{val:.4f}" if isinstance(val, float) else str(val))
            )
    sep = "-+-".join("-" * w for w in col_widths)
    print("  " + sep)
    print("  " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)))
    print("  " + sep)
    for row in rows:
        print("  " + " | ".join(
            f"{v:<{col_widths[i]}.4f}" if isinstance(v, float)
            else f"{v:<{col_widths[i]}}"
            for i, v in enumerate(row)
        ))
    print("  " + sep)

def add_laplace_noise(values, sensitivity, epsilon):
    """Laplace mechanism for (epsilon, 0)-differential privacy."""
    scale = sensitivity / epsilon
    return values + np.random.laplace(0, scale, size=values.shape)

def load_csv(filepath):
    df = pd.read_csv(filepath)
    y  = df["label"].values.astype(int)
    X  = df.drop(columns=["label"]).values.astype(float)
    return X, y, df

def print_dataset_preview(df, node_id, filepath):
    """Print first 10 rows of a node's dataset."""
    n_total   = len(df)
    n_normal  = int((df["label"] == 0).sum())
    n_anomaly = int((df["label"] == 1).sum())

    print_header(f"NODE {node_id}  —  DATASET PREVIEW")
    print_kv("File",           filepath)
    print_kv("Total samples",  n_total)
    print_kv("Normal  (0)",    n_normal)
    print_kv("Anomalous (1)",  n_anomaly)
    print_kv("Class balance",  f"{n_normal/n_total*100:.1f}% normal  |  "
                                f"{n_anomaly/n_total*100:.1f}% anomalous")

    col_w = [18, 16, 12, 14, 12, 7]
    header = (f"{'keypress_dur':>{col_w[0]}} "
              f"{'inter_key_dly':>{col_w[1]}} "
              f"{'error_rate':>{col_w[2]}} "
              f"{'session_len':>{col_w[3]}} "
              f"{'swipe_spd':>{col_w[4]}} "
              f"{'label':>{col_w[5]}}")
    divider = "-" * (sum(col_w) + len(col_w))

    print(f"\n  First 10 rows:\n")
    print("  " + header)
    print("  " + divider)
    for _, row in df.head(10).iterrows():
        line = (f"{row['keypress_duration']:>{col_w[0]}.4f} "
                f"{row['inter_key_delay']:>{col_w[1]}.4f} "
                f"{row['error_rate']:>{col_w[2]}.4f} "
                f"{row['session_length']:>{col_w[3]}.1f} "
                f"{row['swipe_speed']:>{col_w[4]}.4f} "
                f"{int(row['label']):>{col_w[5]}}")
        print("  " + line)
    print("  " + divider)
    print(f"  ... ({n_total - 10} more rows)\n")


# ====================== LOAD DATA ======================

print_header("SEQUENTIAL FEDERATED LEARNING BASELINE")
print_kv("Paradigm",       "Sequential (single-process loop)")
print_kv("Rounds",         ROUNDS)
print_kv("Epsilon (ε)",    EPSILON)
print_kv("Sensitivity",    SENSITIVITY)
print_kv("Privacy Method", "Laplace Differential Privacy")

train_files = sorted(glob.glob("train_dataset_*.txt"))
if not train_files:
    raise FileNotFoundError("Dataset files not found. Run DataProcessing.py first.")

print_kv("Nodes (datasets)", len(train_files))

# Load test set
_, y_test, test_df = load_csv("test_dataset.txt")
X_test_raw = test_df.drop(columns=["label"]).values.astype(float)

# Fit a global scaler on all training data combined
all_X = []
for f in train_files:
    _, _, df = load_csv(f)
    all_X.append(df.drop(columns=["label"]).values.astype(float))
global_scaler = StandardScaler()
global_scaler.fit(np.vstack(all_X))
X_test = global_scaler.transform(X_test_raw)

print_header("TEST DATASET PREVIEW")
print_kv("File",          "test_dataset.txt")
print_kv("Total samples", len(test_df))
print_kv("Normal  (0)",   int((test_df["label"] == 0).sum()))
print_kv("Anomalous (1)", int((test_df["label"] == 1).sum()))

col_w = [18, 16, 12, 14, 12, 7]
header = (f"{'keypress_dur':>{col_w[0]}} "
          f"{'inter_key_dly':>{col_w[1]}} "
          f"{'error_rate':>{col_w[2]}} "
          f"{'session_len':>{col_w[3]}} "
          f"{'swipe_spd':>{col_w[4]}} "
          f"{'label':>{col_w[5]}}")
divider = "-" * (sum(col_w) + len(col_w))
print(f"\n  First 10 rows:\n")
print("  " + header)
print("  " + divider)
for _, row in test_df.head(10).iterrows():
    line = (f"{row['keypress_duration']:>{col_w[0]}.4f} "
            f"{row['inter_key_delay']:>{col_w[1]}.4f} "
            f"{row['error_rate']:>{col_w[2]}.4f} "
            f"{row['session_length']:>{col_w[3]}.1f} "
            f"{row['swipe_speed']:>{col_w[4]}.4f} "
            f"{int(row['label']):>{col_w[5]}}")
    print("  " + line)
print("  " + divider)


# ====================== SEQUENTIAL TRAINING LOOP ======================

global_model   = LogisticRegression(max_iter=1000, class_weight="balanced")
round_results  = []
total_start    = time.time()

for r in range(ROUNDS):
    print_header(f"SEQUENTIAL ROUND  {r + 1} / {ROUNDS}")
    print(f"  Processing {len(train_files)} nodes one by one ...\n")

    node_packets   = []
    round_start    = time.time()

    # ── Sequential node loop (key paradigm difference) ──────────
    for node_idx, filepath in enumerate(train_files):
        node_id = node_idx + 1

        X_raw, y_node, df_node = load_csv(filepath)
        X_node = global_scaler.transform(X_raw)

        # Print dataset preview only on the first round
        if r == 0:
            print_dataset_preview(df_node, node_id, filepath)

        # ── Local training (sequential, not parallel) ────────────
        node_start = time.time()
        local_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        local_model.fit(X_node, y_node)
        node_time = time.time() - node_start

        # ── Differential privacy — Laplace noise ─────────────────
        noisy_coef      = add_laplace_noise(local_model.coef_[0],
                                            SENSITIVITY, EPSILON)
        noisy_intercept = add_laplace_noise(
            np.array([local_model.intercept_[0]]),
            SENSITIVITY, EPSILON
        )[0]

        # Local evaluation before noise
        y_local_pred = local_model.predict(X_node)
        local_acc    = accuracy_score(y_node, y_local_pred)
        local_f1     = f1_score(y_node, y_local_pred, zero_division=0)

        print(f"  [Node {node_id:>2}]  "
              f"samples={len(y_node):<5}  "
              f"normal={int((y_node==0).sum()):<5}  "
              f"anomalous={int((y_node==1).sum()):<5}  "
              f"train_time={node_time:.4f}s  "
              f"local_acc={local_acc:.4f}  "
              f"local_f1={local_f1:.4f}")

        node_packets.append({
            "node_id":    node_id,
            "coef":       noisy_coef,
            "intercept":  noisy_intercept,
            "time":       node_time,
            "n_samples":  len(y_node),
            "n_normal":   int((y_node == 0).sum()),
            "n_anomaly":  int((y_node == 1).sum()),
            "local_acc":  local_acc,
            "local_f1":   local_f1,
        })

    round_elapsed = time.time() - round_start

    # ── FedAvg aggregation (same as federated) ───────────────────
    avg_coef      = np.mean([p["coef"]      for p in node_packets], axis=0)
    avg_intercept = np.mean([p["intercept"] for p in node_packets])

    global_model.coef_          = np.array([avg_coef])
    global_model.intercept_     = np.array([avg_intercept])
    global_model.classes_       = np.array([0, 1])
    global_model.n_features_in_ = avg_coef.shape[0]

    # ── Global evaluation ─────────────────────────────────────────
    y_pred = global_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, zero_division=0)

    round_results.append([r + 1, round_elapsed, acc, f1])

    print(f"\n  ── Round {r+1} Aggregation ──")
    print_kv("  Total round time",    f"{round_elapsed:.4f}s")
    print_kv("  Nodes processed",     len(node_packets))
    print_kv("  Global Accuracy",     f"{acc:.4f}")
    print_kv("  Global F1 Score",     f"{f1:.4f}")
    print_kv("  Avg node train time", f"{np.mean([p['time'] for p in node_packets]):.4f}s")

total_time = time.time() - total_start


# ====================== FINAL ANALYSIS ======================

print_header("SEQUENTIAL — ROUND SUMMARY")
print_table(
    ["Round", "Round Time (s)", "Accuracy", "F1 Score"],
    round_results
)

# Final metrics
y_pred_final = global_model.predict(X_test)
acc_final    = accuracy_score(y_test, y_pred_final)
f1_final     = f1_score(y_test, y_pred_final, zero_division=0)
cm           = confusion_matrix(y_test, y_pred_final)
report       = classification_report(y_test, y_pred_final,
                                     target_names=["Normal", "Anomalous"],
                                     zero_division=0)

print_header("SEQUENTIAL — FINAL PERFORMANCE")
print_kv("Final Accuracy",        f"{acc_final:.4f}")
print_kv("Final F1 Score",        f"{f1_final:.4f}")
print_kv("Total Execution Time",  f"{total_time:.4f}s")
print_kv("Total Rounds",          ROUNDS)
print_kv("Total Nodes",           len(train_files))
print_kv("Privacy Budget (ε)",    EPSILON)
print_kv("Test Samples",          len(y_test))

print_header("SEQUENTIAL — CONFUSION MATRIX")
cm_headers = ["", "Predicted Normal", "Predicted Anomalous"]
cm_rows    = [
    ["Actual Normal",    cm[0][0], cm[0][1]],
    ["Actual Anomalous", cm[1][0], cm[1][1]],
]
print_table(cm_headers, cm_rows)

tn, fp, fn, tp = cm.ravel()
precision   = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
recall      = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
specificity = tn / (tn + fp)  if (tn + fp)  > 0 else 0.0

print_header("SEQUENTIAL — DETAILED METRICS")
print_kv("True Positives  (TP)",  tp)
print_kv("True Negatives  (TN)",  tn)
print_kv("False Positives (FP)",  fp)
print_kv("False Negatives (FN)",  fn)
print_kv("Precision",             f"{precision:.4f}")
print_kv("Recall (Sensitivity)",  f"{recall:.4f}")
print_kv("Specificity",           f"{specificity:.4f}")
print_kv("F1 Score",              f"{f1_final:.4f}")

print_header("SEQUENTIAL — CLASSIFICATION REPORT")
print(report)

print_header("SEQUENTIAL — MODEL COEFFICIENTS (Aggregated)")
for fname, coef in zip(FEATURE_NAMES, global_model.coef_[0]):
    direction = "↑ anomaly" if coef > 0 else "↓ anomaly"
    print(f"  {fname:<28}: {coef:>8.4f}  ({direction})")
print(f"  {'intercept':<28}: {global_model.intercept_[0]:>8.4f}")

print_header("SEQUENTIAL — PARADIGM COMPARISON SUMMARY")
print(f"  {'Paradigm':<22} {'Execution':<20} {'Privacy':<18} {'Speed'}")
print("  " + "-" * 62)
print(f"  {'Centralised':<22} {'Single model':<20} {'None':<18} {'Fastest'}")
print(f"  {'Sequential (this)':<22} {'Node-by-node loop':<20} {'Laplace DP':<18} {'Slowest'}")
print(f"  {'Federated (MPI)':<22} {'Parallel workers':<20} {'Laplace DP':<18} {'Fast'}")
print()
print_kv("  Sequential total time",  f"{total_time:.4f}s  "
         "(vs MPI: all nodes run in parallel)")
print_kv("  Key difference",
         "Sequential waits for each node;\n"
         "                              "
         "MPI runs all nodes simultaneously")

# Save model
joblib.dump(global_model, "sequential_model.pkl")

print_header("SEQUENTIAL — FILES SAVED")
print("  • sequential_model.pkl")
print_line()