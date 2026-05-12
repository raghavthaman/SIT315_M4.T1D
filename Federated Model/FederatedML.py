import argparse
import csv
import time

import joblib
import numpy as np
from mpi4py import MPI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


# ====================== COMMAND LINE ARGUMENTS ======================
# Parse command-line arguments for federated learning parameters
parser = argparse.ArgumentParser(description="Privacy-Preserving Federated Machine Learning using MPI")
parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
parser.add_argument("--eps", type=float, default=1.0, help="Privacy budget (epsilon) for differential privacy")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity parameter for Laplace noise")

# Parse known args (ignores unknown arguments from MPI)
args, _ = parser.parse_known_args()

ROUNDS = args.rounds
EPSILON = args.eps
SENSITIVITY = args.sensitivity


# ====================== MPI SETUP ======================
# Initialize MPI for distributed computing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Current process rank (0 = master, others = workers)
size = comm.Get_size()      # Total number of processes


# ====================== HELPER FUNCTIONS ======================

def add_laplace_noise(values, sensitivity, epsilon):
    """
    Adds Laplace noise to values for differential privacy.
    This is the core mechanism for achieving (ε, 0)-differential privacy.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=values.shape)
    return values + noise


def load_data(filepath):
    """
    Loads dataset from CSV, separates features and target, 
    and applies standard scaling (zero mean, unit variance).
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


def print_line(width=60):
    """Prints a horizontal line for better output formatting."""
    print("-" * width)


def print_header(title):
    """Prints a nicely formatted header for different sections of output."""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)


def print_kv(key, value):
    """Prints key-value pairs in a clean aligned format."""
    print(f"{key:<20}: {value}")


def print_table(headers, rows):
    """
    Prints a formatted table with dynamic column width adjustment.
    Handles both string and float values gracefully.
    """
    # Calculate optimal column widths
    col_widths = [max(len(str(h)), 12) for h in headers]

    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], 
                               len(f"{val:.4f}" if isinstance(val, float) else str(val)))

    # Print header
    print_line(sum(col_widths) + len(headers) * 3)
    header_row = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(header_row)
    print_line(sum(col_widths) + len(headers) * 3)

    # Print data rows
    for row in rows:
        row_str = " | ".join(
            f"{val:<{col_widths[i]}.4f}" if isinstance(val, float) 
            else f"{val:<{col_widths[i]}}" 
            for i, val in enumerate(row)
        )
        print(row_str)

    print_line(sum(col_widths) + len(headers) * 3)


# ====================== MASTER NODE (rank == 0) ======================
if rank == 0:
    n_workers = size - 1

    print_header("FEDERATED PRIVACY-PRESERVING ML SYSTEM")
    print_kv("Workers", n_workers)
    print_kv("Rounds", ROUNDS)
    print_kv("Epsilon (ε)", EPSILON)
    print_kv("Sensitivity", SENSITIVITY)
    print_kv("Master Node", MPI.Get_processor_name())

    # Initialize global model
    global_model = LogisticRegression(max_iter=1000, class_weight="balanced")

    # Load test dataset (used for evaluation after every round)
    X_test, y_test = load_data("test_dataset.txt")

    round_results = []
    total_start = time.time()

    for r in range(ROUNDS):
        print(f"\n--- Starting Federated Round {r + 1}/{ROUNDS} ---")

        # Notify all workers to start training for this round
        for w in range(1, size):
            comm.isend(f"round_{r}", dest=w, tag=10)

        # Collect model updates from all workers
        packets = [comm.recv(source=w, tag=20) for w in range(1, size)]

        # Aggregate model parameters using simple averaging (Federated Averaging)
        avg_coef = np.mean([p["coef"] for p in packets], axis=0)
        avg_intercept = np.mean([p["intercept"] for p in packets])
        avg_time = np.mean([p["time"] for p in packets])

        # Update global model with aggregated parameters
        global_model.coef_ = np.array([avg_coef])
        global_model.intercept_ = np.array([avg_intercept])
        global_model.classes_ = np.array([0, 1])
        global_model.n_features_in_ = avg_coef.shape[0]

        # Evaluate global model on test set
        y_pred = global_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        round_results.append([r + 1, avg_time, acc, f1])

        print_kv(f"Round {r+1} Accuracy", f"{acc:.4f}")
        print_kv(f"Round {r+1} F1 Score", f"{f1:.4f}")

    total_time = time.time() - total_start

    # ====================== RESULTS DISPLAY ======================
    print_header("ROUND SUMMARY")
    headers = ["Round", "Avg Time (s)", "Accuracy", "F1 Score"]
    print_table(headers, round_results)

    # Final evaluation
    y_pred_final = global_model.predict(X_test)
    acc_final = accuracy_score(y_test, y_pred_final)
    f1_final = f1_score(y_test, y_pred_final)
    cm = confusion_matrix(y_test, y_pred_final)

    print_header("FINAL RESULTS")
    print_kv("Final Accuracy", f"{acc_final:.4f}")
    print_kv("Final F1 Score", f"{f1_final:.4f}")
    print_kv("Total Training Time (s)", f"{total_time:.4f}")

    # Confusion Matrix
    print_header("CONFUSION MATRIX")
    cm_headers = ["", "Predicted 0", "Predicted 1"]
    cm_rows = [
        ["Actual 0", cm[0][0], cm[0][1]],
        ["Actual 1", cm[1][0], cm[1][1]]
    ]
    print_table(cm_headers, cm_rows)

    # Save final model and results
    joblib.dump(global_model, "federated_model.pkl")

    with open("federated_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(round_results)

    print("\n Model and results saved successfully!")
    print("   • Model   → federated_model.pkl")
    print("   • Results → federated_results.csv")

    # Gracefully shutdown all worker nodes
    for w in range(1, size):
        comm.isend("terminate", dest=w, tag=10)


# ====================== WORKER NODES (rank > 0) ======================
else:
    # Each worker loads its own local private dataset
    file_path = f"train_dataset_{rank - 1}.txt"
    X, y = load_data(file_path)

    print(f"Worker {rank} loaded dataset: {file_path} | Samples: {len(y)}")

    while True:
        # Wait for instruction from master
        msg = comm.recv(source=0, tag=10)

        if msg == "terminate":
            print(f"Worker {rank} shutting down...")
            break

        # Start local training
        start = time.time()

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)

        end = time.time()
        training_time = end - start

        # Apply differential privacy by adding Laplace noise to model parameters
        noisy_coef = add_laplace_noise(model.coef_[0], SENSITIVITY, EPSILON)
        noisy_intercept = add_laplace_noise(
            np.array([model.intercept_[0]]), SENSITIVITY, EPSILON
        )[0]

        # Prepare packet to send back to master
        packet = {
            "coef": noisy_coef,
            "intercept": noisy_intercept,
            "time": training_time
        }

        # Send noisy model parameters to master
        comm.send(packet, dest=0, tag=20)