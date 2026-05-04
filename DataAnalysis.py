"""
DataAnalysis.py
---------------
Generates evaluation plots for Federated vs Centralised ML.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

BASELINE = 0.2129  # single-process baseline time (seconds)


def save(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[DataAnalysis] Saved {fname}")


# ── Load data ─────────────────────────────────────────────
XLS = "FederatedML_Evaluation.xlsx"

if os.path.exists(XLS):
    sheet1 = pd.read_excel(XLS, sheet_name="Implementation#1 Evaluation")
    sheet2 = pd.read_excel(XLS, sheet_name="Implementation#2 Evaluation")
else:
    print(f"[DataAnalysis] {XLS} not found – using demo data.")

    sheet1 = pd.DataFrame({
        "Total Processes": [2, 3, 5, 9],
        "Execution Time": [0.31, 0.22, 0.18, 0.21],
    })

    sheet2 = pd.DataFrame({
        "Total Processes": [2, 3, 5, 9],
        "Execution Time": [0.34, 0.25, 0.20, 0.23],
    })

# Compute speedup
sheet1["Speedup"] = BASELINE / sheet1["Execution Time"]
sheet2["Speedup"] = BASELINE / sheet2["Execution Time"]


# ── Implementation 1 ──────────────────────────────────────
plt.figure()
sns.barplot(x="Total Processes", y="Execution Time", data=sheet1, hue="Total Processes", legend=False)
plt.title("Implementation 1 (Epsilon = 1.0): Execution Time")
plt.xlabel("Total MPI Processes")
plt.ylabel("Execution Time (seconds)")
save("impl1_execution_time.png")

plt.figure()
sns.lineplot(x="Total Processes", y="Speedup", data=sheet1, marker="o")
plt.axhline(1.0, linestyle="--")
plt.title("Implementation 1: Speedup")
plt.xlabel("Total MPI Processes")
plt.ylabel("Speedup")
save("impl1_speedup.png")


# ── Implementation 2 ──────────────────────────────────────
plt.figure()
sns.barplot(x="Total Processes", y="Execution Time", data=sheet2, hue="Total Processes", legend=False)
plt.title("Implementation 2 (Epsilon = 0.5): Execution Time")
plt.xlabel("Total MPI Processes")
plt.ylabel("Execution Time (seconds)")
save("impl2_execution_time.png")

plt.figure()
sns.lineplot(x="Total Processes", y="Speedup", data=sheet2, marker="o")
plt.axhline(1.0, linestyle="--")
plt.title("Implementation 2: Speedup")
plt.xlabel("Total MPI Processes")
plt.ylabel("Speedup")
save("impl2_speedup.png")


# ── Combined Comparison ───────────────────────────────────
combined = pd.concat([
    sheet1.assign(Type="Impl 1 (eps=1.0)"),
    sheet2.assign(Type="Impl 2 (eps=0.5)")
])

plt.figure()
sns.barplot(x="Total Processes", y="Execution Time", hue="Type", data=combined)
plt.title("Execution Time Comparison")
save("combined_execution_time.png")

plt.figure()
sns.lineplot(x="Total Processes", y="Speedup", hue="Type", data=combined, marker="o")
plt.axhline(1.0, linestyle="--")
plt.title("Speedup Comparison")
save("combined_speedup.png")


# ── Accuracy per round (FIXED ROBUST VERSION) ─────────────
if os.path.exists("federated_results.csv"):
    res = pd.read_csv("federated_results.csv")

    # Normalize column names
    res.columns = [c.strip().lower() for c in res.columns]

    round_col = None
    acc_col = None
    f1_col = None

    for col in res.columns:
        if "round" in col:
            round_col = col
        elif "accuracy" in col:
            acc_col = col
        elif "f1" in col:
            f1_col = col

    if round_col and acc_col and f1_col:
        plt.figure()
        plt.plot(res[round_col], res[acc_col], marker="o", label="Accuracy")
        plt.plot(res[round_col], res[f1_col], marker="s", label="F1 Score")

        plt.title("Federated Model Performance per Round")
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.legend()

        save("accuracy_per_round.png")
    else:
        print("[DataAnalysis] Accuracy plot skipped (columns not found)")

print("[DataAnalysis] All plots generated successfully.")