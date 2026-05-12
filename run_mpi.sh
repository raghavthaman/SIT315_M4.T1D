#!/bin/bash
# run_federated.sh  —  clean, correct MPI launcher
#
# Usage:
#   bash run_federated.sh [num_workers] [epsilon]
#
#   Examples:
#     bash run_federated.sh 4 1.0    # 4 workers (5 total procs), epsilon=1.0
#     bash run_federated.sh 4 0.5    # stricter differential privacy
#     bash run_federated.sh 8 1.0    # stress test – needs 8 train_dataset_*.txt files

NUM_WORKERS=${1:-4}
EPSILON=${2:-1.0}
TOTAL=$((NUM_WORKERS + 1))

NODE=$(hostname)
case "$NODE" in
  Master) PYTHON=/home/vboxuser/fedenv/bin/python3 ;;
  Head)   PYTHON=/home/explorer/fedenv/bin/python3 ;;
  *)      PYTHON=$(which python3) ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Privacy-Preserving Federated ML"
echo "  Node: $NODE | Python: $PYTHON"
echo "  Workers: $NUM_WORKERS | Total procs: $TOTAL"
echo "  DP epsilon: $EPSILON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "[1/3] Generating dataset..."
$PYTHON DataProcessing.py || exit 1

echo "[2/3] Running federated training..."
mpirun -np $TOTAL $PYTHON FederatedML.py --eps $EPSILON --rounds 3 || exit 1

echo "[3/3] Generating charts..."
$PYTHON DataAnalysis.py

echo "✓ Complete. See federated_model.pkl and *.png"