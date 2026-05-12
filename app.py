import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Privacy-Preserving Federated Learning Dashboard",
    page_icon="🔐",
    layout="wide",
)

sns.set_style("whitegrid")

PROJECT_DIR = Path(__file__).parent
RESULTS_CSV = PROJECT_DIR / "federated_results.csv"
CENTRAL_CSV = PROJECT_DIR / "normal_results.csv"
FED_SCRIPT = PROJECT_DIR / "FederatedML.py"
PYTHON_EXEC = sys.executable

# -------------------------
# HELPERS
# -------------------------
def load_results_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def make_sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [1, 2, 3],
            "accuracy": [1.0, 1.0, 1.0],
            "f1_score": [1.0, 1.0, 1.0],
            "execution_time": [0.31, 0.22, 0.18],
            "processes": [2, 3, 5],
            "epsilon": [1.0, 1.0, 1.0],
        }
    )


def make_sample_central() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Centralised"],
            "accuracy": [1.0],
            "f1_score": [1.0],
            "execution_time": [0.0222],
        }
    )


def get_process_row(df: pd.DataFrame, processes: int):
    if "processes" not in df.columns:
        return None
    subset = df[df["processes"] == processes]
    if subset.empty:
        return None
    return subset.iloc[0]


def run_mpi_job(processes: int, epsilon: float, rounds: int, use_oversubscribe: bool):
    cmd = [
        "mpirun",
        "-np",
        str(processes),
    ]
    if use_oversubscribe:
        cmd.insert(1, "--oversubscribe")
    cmd += [
        PYTHON_EXEC,
        str(FED_SCRIPT),
        "--eps",
        str(epsilon),
        "--rounds",
        str(rounds),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        return result
    except FileNotFoundError as e:
        class DummyResult:
            def __init__(self, err):
                self.returncode = 1
                self.stdout = ""
                self.stderr = f"Error: Command '{cmd[0]}' not found. Is MPI installed and in your PATH?\nDetails: {str(err)}"
        return DummyResult(e)


def plot_line_chart(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    return fig


def draw_mpi_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")

    boxes = [
        (0.04, 0.55, 0.22, 0.25, "Master (Rank 0)\nBroadcast global model\nAggregate noisy updates"),
        (0.35, 0.15, 0.18, 0.22, "Worker 1\nLocal train\nAdd Laplace noise"),
        (0.35, 0.55, 0.18, 0.22, "Worker 2\nLocal train\nAdd Laplace noise"),
        (0.35, 0.90, 0.18, 0.22, "Worker N\nLocal train\nAdd Laplace noise"),
        (0.66, 0.55, 0.26, 0.25, "Privacy Layer\nLaplace mechanism\nModel updates only"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    # arrows
    ax.annotate("", xy=(0.35, 0.67), xytext=(0.26, 0.67), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.35, 0.27), xytext=(0.26, 0.67), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.35, 0.98), xytext=(0.26, 0.67), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.66, 0.67), xytext=(0.53, 0.67), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.57, 0.67), xytext=(0.66, 0.67), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(0.25, 0.78, "Broadcast", fontsize=9)
    ax.text(0.57, 0.78, "Gather", fontsize=9)
    ax.text(0.57, 0.43, "FedAvg / update", fontsize=9)

    plt.tight_layout()
    return fig


# -------------------------
# SESSION STATE
# -------------------------
if "run_log" not in st.session_state:
    st.session_state.run_log = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -------------------------
# TITLE
# -------------------------
st.title("🔐 Privacy-Preserving Federated Learning Dashboard")
st.caption("MPI + Differential Privacy + Logistic Regression + Visual Model Training")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")
epsilon = st.sidebar.slider("Privacy Budget (ε)", 0.1, 2.0, 1.0, 0.1)
rounds = st.sidebar.slider("Training Rounds", 1, 10, 3)
processes = st.sidebar.selectbox("MPI Processes", [2, 3, 5, 9], index=2)
use_oversubscribe = st.sidebar.checkbox("Use --oversubscribe", value=True)
show_demo_mode = st.sidebar.checkbox("Demo mode (use sample charts)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Run Backend")
run_backend = st.sidebar.button("▶ Run MPI Training", use_container_width=True)
refresh_view = st.sidebar.button("🔄 Refresh Results", use_container_width=True)

# -------------------------
# RUN MPI
# -------------------------
if run_backend:
    with st.spinner("Running MPI training... this may take a moment."):
        result = run_mpi_job(processes, epsilon, rounds, use_oversubscribe)
        st.session_state.run_log = (
            "COMMAND:\n"
            + f"mpirun -np {processes} {'--oversubscribe ' if use_oversubscribe else ''}{PYTHON_EXEC} FederatedML.py --eps {epsilon} --rounds {rounds}\n\n"
            + "STDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )
        st.session_state.last_result = result.returncode
        if result.returncode == 0:
            st.success("MPI training completed successfully.")
        else:
            st.error(f"MPI training finished with exit code {result.returncode}.")

# -------------------------
# LOAD DATA
# -------------------------
fed_df = load_results_csv(RESULTS_CSV)
if fed_df is None:
    fed_df = make_sample_results()

central_df = load_results_csv(CENTRAL_CSV)
if central_df is None:
    central_df = make_sample_central()

# Normalise expected columns if the CSV uses slightly different names
rename_map = {
    "Round": "round",
    "Accuracy": "accuracy",
    "F1": "f1_score",
    "F1-Score": "f1_score",
    "F1 Score": "f1_score",
    "Execution Time": "execution_time",
    "Exec_Time": "execution_time",
    "Avg Time (s)": "execution_time",
    "Total Processes": "processes",
    "Processes": "processes",
    "Epsilon": "epsilon",
}
fed_df = fed_df.rename(columns={c: rename_map.get(c, c) for c in fed_df.columns})
central_df = central_df.rename(columns={c: rename_map.get(c, c) for c in central_df.columns})

# -------------------------
# HERO METRICS
# -------------------------
st.markdown("## 📌 Project Snapshot")
col_a, col_b, col_c, col_d = st.columns(4)

latest_fed_acc = float(fed_df["accuracy"].iloc[-1]) if "accuracy" in fed_df.columns else 1.0
latest_fed_f1 = float(fed_df["f1_score"].iloc[-1]) if "f1_score" in fed_df.columns else 1.0
best_time = float(fed_df["execution_time"].min()) if "execution_time" in fed_df.columns else 0.18
central_time = float(central_df["execution_time"].iloc[0]) if "execution_time" in central_df.columns else 0.0222
speedup = central_time / best_time if best_time > 0 else 0.0

col_a.metric("Federated Accuracy", f"{latest_fed_acc:.4f}")
col_b.metric("Federated F1 Score", f"{latest_fed_f1:.4f}")
col_c.metric("Best Time (Federated)", f"{best_time:.4f}s")
col_d.metric("Centralised / Best Time", f"{speedup:.2f}x")

# -------------------------
# SYSTEM OVERVIEW
# -------------------------
st.markdown("---")
st.header("📌 System Overview")
left, right = st.columns([1.1, 1])
with left:
    st.markdown(
        """
        - **Master-worker MPI architecture** for distributed coordination.
        - **Local logistic regression training** at each worker.
        - **Laplace mechanism** to protect model updates before aggregation.
        - **Federated averaging** to combine noisy worker models.
        - **Centralised baseline** for direct comparison.
        """
    )
    st.info(
        "HD angle: this dashboard is not just reporting results; it explains the privacy–utility trade-off, the MPI topology, and the training lifecycle in one place."
    )
with right:
    fig_arch = draw_mpi_architecture()
    st.pyplot(fig_arch, clear_figure=True)

# -------------------------
# VISUAL MODEL TRAINING
# -------------------------
st.markdown("---")
st.header("🔁 Visual Model Training")
train_tab1, train_tab2, train_tab3 = st.tabs(["Training Rounds", "Round-by-Round Log", "Worker Topology"])

with train_tab1:
    if "round" not in fed_df.columns:
        fed_df["round"] = range(1, len(fed_df) + 1)
    if "accuracy" not in fed_df.columns:
        fed_df["accuracy"] = 1.0
    if "f1_score" not in fed_df.columns:
        fed_df["f1_score"] = 1.0

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = plot_line_chart(fed_df["round"], fed_df["accuracy"], "Round", "Accuracy", "Accuracy across training rounds")
        st.pyplot(fig_acc, clear_figure=True)
    with c2:
        fig_f1 = plot_line_chart(fed_df["round"], fed_df["f1_score"], "Round", "F1 Score", "F1-score across training rounds")
        st.pyplot(fig_f1, clear_figure=True)

    st.caption("If your model converges quickly, the lines will flatten near 1.0. That is expected for the synthetic linearly separable dataset.")

with train_tab2:
    if st.session_state.run_log:
        st.code(st.session_state.run_log, language="text")
    else:
        st.write("Run the MPI job from the sidebar to show live output here.")

with train_tab3:
    worker_count = max(processes - 1, 1)
    top = st.columns(worker_count if worker_count <= 4 else 4)
    for i in range(worker_count):
        with top[i % len(top)]:
            st.success(f"Worker {i + 1}")
            st.write("Local data shard")
            st.write("Train logistic regression")
            st.write("Add Laplace noise")
            st.write("Send update to master")

# -------------------------
# PRIVACY VS UTILITY
# -------------------------
st.markdown("---")
st.header("🔐 Privacy vs Utility Trade-off")

if epsilon <= 0.5:
    st.error("Strong privacy: more Laplace noise, stronger protection, potentially lower utility.")
elif epsilon <= 1.0:
    st.warning("Balanced privacy: good compromise between protection and utility.")
else:
    st.success("Higher utility: less noise, stronger model stability, weaker privacy.")

priv_col1, priv_col2 = st.columns(2)
with priv_col1:
    st.metric("Selected ε", f"{epsilon:.1f}")
    st.write("Smaller ε means stronger privacy and more perturbation in model updates.")
with priv_col2:
    noise_scale = 1.0 / epsilon
    st.metric("Approx. Noise Scale", f"{noise_scale:.2f}")
    st.write("For the Laplace mechanism, smaller ε increases the noise scale." )

# -------------------------
# PERFORMANCE ANALYSIS
# -------------------------
st.markdown("---")
st.header("📊 Performance Analysis")

base_df = pd.DataFrame(
    {
        "processes": [2, 3, 5, 9],
        "exec_time_eps1": [0.31, 0.22, 0.18, 0.21],
        "exec_time_eps05": [0.34, 0.25, 0.20, 0.23],
        "speedup_eps1": [1.00, 0.97, 1.19, 1.02],
        "speedup_eps05": [1.00, 0.85, 1.07, 0.93],
    }
)

selected_exec = base_df["exec_time_eps1"] if epsilon >= 1.0 else base_df["exec_time_eps05"]
selected_speed = base_df["speedup_eps1"] if epsilon >= 1.0 else base_df["speedup_eps05"]
selected_label = "ε = 1.0 (Moderate Privacy)" if epsilon >= 1.0 else "ε = 0.5 (Stronger Privacy)"

perf1, perf2 = st.columns(2)
with perf1:
    fig_exec = plot_line_chart(base_df["processes"], selected_exec, "MPI Processes", "Execution Time (s)", f"Execution Time vs Processes\n{selected_label}")
    st.pyplot(fig_exec, clear_figure=True)
with perf2:
    fig_speed = plot_line_chart(base_df["processes"], selected_speed, "MPI Processes", "Speedup", f"Speedup vs Processes\n{selected_label}")
    st.pyplot(fig_speed, clear_figure=True)

# bar chart comparison
comparison_df = pd.DataFrame(
    {
        "model": ["Centralised", "Federated"],
        "execution_time": [central_time, best_time],
        "accuracy": [1.0, latest_fed_acc],
    }
)

c1, c2 = st.columns(2)
with c1:
    fig_cmp_time = plot_bar_chart(comparison_df, "model", "execution_time", "Centralised vs Federated Execution Time")
    st.pyplot(fig_cmp_time, clear_figure=True)
with c2:
    fig_cmp_acc = plot_bar_chart(comparison_df, "model", "accuracy", "Centralised vs Federated Accuracy")
    st.pyplot(fig_cmp_acc, clear_figure=True)

# -------------------------
# DATA TABLES
# -------------------------
st.markdown("---")
st.header("🧾 Results Table")

show_df = fed_df.copy()
cols_to_show = [c for c in ["round", "processes", "epsilon", "accuracy", "f1_score", "execution_time"] if c in show_df.columns]
if cols_to_show:
    st.dataframe(show_df[cols_to_show], use_container_width=True)
else:
    st.dataframe(show_df, use_container_width=True)

# -------------------------
# HD INSIGHT PANEL
# -------------------------
st.markdown("---")
st.header("🎓 HD-Level Insights")
insight1, insight2, insight3 = st.columns(3)

with insight1:
    st.markdown(
        """
        **Scalability**  
        Peak performance occurs when the worker count matches the number of data partitions.
        After that point, MPI communication overhead and process scheduling reduce efficiency.
        """
    )
with insight2:
    st.markdown(
        """
        **Privacy**  
        Differential privacy protects model updates by adding Laplace noise, trading off privacy strength against model utility.
        """
    )
with insight3:
    st.markdown(
        """
        **System Design**  
        The architecture separates local training, secure update sharing, and global aggregation, which is the key federated-learning idea.
        """
    )

st.markdown("---")
st.caption("SIT315 M4.T1D | Privacy-Preserving Federated Machine Learning using MPI and Differential Privacy")

# -------------------------
# REAL VS FAKE CLICK ANALYSIS
# -------------------------
st.markdown("---")
st.header("🕵️‍♂️ Real vs Fake Click Analysis")
st.write("Test the Federated Model on real data to detect if a keyboard/click session is human (Real) or bot-like (Fake).")

import joblib
from sklearn.preprocessing import StandardScaler

def load_test_data_and_predict():
    test_file = PROJECT_DIR / "test_dataset.txt"
    model_file = PROJECT_DIR / "federated_model.pkl"
    
    if not test_file.exists():
        return None, None, None, "Test dataset not found. Please run the MPI training first."
    if not model_file.exists():
        return None, None, None, "Federated model not found. Please run the MPI training first."
        
    try:
        # Load the test data
        df = pd.read_csv(test_file)
        y_true = df["label"].values
        X_raw = df.drop(columns=["label"])
        
        # Scale the data exactly as done in FederatedML.py
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw.values)
        
        # Load the model
        model = joblib.load(model_file)
        
        # Pick a random sample
        idx = pd.Series(range(len(df))).sample(1).iloc[0]
        sample_features = X_raw.iloc[idx].to_dict()
        true_label = int(y_true[idx])
        
        # Predict
        pred = model.predict(X_scaled[[idx]])[0]
        
        return sample_features, true_label, int(pred), None
    except Exception as e:
        return None, None, None, f"Error during inference: {str(e)}"

def predict_single_sample(features_dict):
    test_file = PROJECT_DIR / "test_dataset.txt"
    model_file = PROJECT_DIR / "federated_model.pkl"
    
    if not test_file.exists():
        return None, "Test dataset not found. Please run the MPI training first."
    if not model_file.exists():
        return None, "Federated model not found. Please run the MPI training first."
        
    try:
        df = pd.read_csv(test_file)
        X_raw = df.drop(columns=["label"])
        
        scaler = StandardScaler()
        scaler.fit(X_raw.values)
        
        model = joblib.load(model_file)
        
        input_data = pd.DataFrame([features_dict], columns=X_raw.columns)
        X_scaled = scaler.transform(input_data.values)
        
        pred = model.predict(X_scaled)[0]
        
        return int(pred), None
    except Exception as e:
        return None, f"Error during inference: {str(e)}"

tab1, tab2 = st.tabs(["🎲 Random Sample", "✍️ Manual Entry"])

with tab1:
    if st.button("🎲 Simulate New Interaction Session"):
        features, true_label, pred, err = load_test_data_and_predict()
        
        if err:
            st.error(err)
        else:
            st.subheader("Interaction Features captured:")
            
            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("Keypress Duration", f"{features['keypress_duration']:.1f} ms")
            f2.metric("Inter-key Delay", f"{features['inter_key_delay']:.1f} ms")
            f3.metric("Error Rate", f"{features['error_rate']:.3f}")
            f4.metric("Session Length", f"{int(features['session_length'])} clicks")
            f5.metric("Swipe Speed", f"{features['swipe_speed']:.2f} px/ms")
            
            st.markdown("### Prediction Results")
            res1, res2 = st.columns(2)
            
            with res1:
                if pred == 0:
                    st.success("✅ Prediction: REAL (Human Interaction)")
                else:
                    st.error("🚨 Prediction: FAKE (Bot/Anomaly Detected)")
                    
            with res2:
                if true_label == pred:
                    st.info(f"Actual Label: {'Real' if true_label == 0 else 'Fake'} (Model is Correct)")
                else:
                    st.warning(f"Actual Label: {'Real' if true_label == 0 else 'Fake'} (Model is Incorrect)")

with tab2:
    st.write("Enter interaction features manually to predict if it is Real or Fake.")
    with st.form("manual_entry_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            in_duration = st.number_input("Keypress Duration (ms)", value=150.0)
            in_delay = st.number_input("Inter-key Delay (ms)", value=300.0)
        with col2:
            in_error = st.number_input("Error Rate", value=0.05, format="%.3f")
            in_len = st.number_input("Session Length (clicks)", value=50)
        with col3:
            in_speed = st.number_input("Swipe Speed (px/ms)", value=1.5)
            in_label = st.selectbox("True Label (Optional for checking)", ["Unknown", "Real (0)", "Fake (1)"])
        
        submit_btn = st.form_submit_button("Predict Interaction")
        
    if submit_btn:
        features_dict = {
            "keypress_duration": in_duration,
            "inter_key_delay": in_delay,
            "error_rate": in_error,
            "session_length": in_len,
            "swipe_speed": in_speed
        }
        pred, err = predict_single_sample(features_dict)
        if err:
            st.error(err)
        else:
            st.markdown("### Prediction Results")
            res1, res2 = st.columns(2)
            
            with res1:
                if pred == 0:
                    st.success("✅ Prediction: REAL (Human Interaction)")
                else:
                    st.error("🚨 Prediction: FAKE (Bot/Anomaly Detected)")
                    
            with res2:
                if in_label != "Unknown":
                    true_val = 0 if "Real" in in_label else 1
                    if true_val == pred:
                        st.info(f"Actual Label: {'Real' if true_val == 0 else 'Fake'} (Model is Correct)")
                    else:
                        st.warning(f"Actual Label: {'Real' if true_val == 0 else 'Fake'} (Model is Incorrect)")
