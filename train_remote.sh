#!/bin/bash
# ──────────────────────────────────────────────────────────────
# train_remote.sh — Run training in tmux (survives SSH disconnect)
#
# Usage:
#   ./train_remote.sh              # Fresh start, 2000 batches
#   ./train_remote.sh --resume     # Resume from checkpoint
#   ./train_remote.sh --selfplay   # Self-play training
#
# The training runs inside a tmux session called "catan".
# You can disconnect from SSH and it keeps running.
#
# Reconnect with:
#   tmux attach -t catan
#
# ──────────────────────────────────────────────────────────────
set -e

BATCHES=${BATCHES:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${WORKERS:-4}
EVAL_EVERY=${EVAL_EVERY:-10}
SAVE_PATH=${SAVE_PATH:-"checkpoints/a2c_gpu.pt"}
SEED=${SEED:-42}
RESUME=""
MODE="train"

# Parse args
for arg in "$@"; do
    case $arg in
        --resume) RESUME="--resume" ;;
        --selfplay) MODE="selfplay" ;;
        *) ;;
    esac
done

# Check if tmux session already exists
if tmux has-session -t catan 2>/dev/null; then
    echo "Training session 'catan' already running!"
    echo "  Attach: tmux attach -t catan"
    echo "  Kill:   tmux kill-session -t catan"
    exit 1
fi

# Build the training command
if [ "$MODE" = "selfplay" ]; then
    SAVE_PATH="checkpoints/a2c_selfplay_gpu.pt"
    CMD="python3 -m training.train_selfplay \
        --batches $BATCHES \
        --batch-size $BATCH_SIZE \
        --workers $WORKERS \
        --eval-every 25 \
        --save-path $SAVE_PATH \
        --seed $SEED \
        2>&1 | tee training.log"
else
    CMD="python3 -m training.train \
        --algo a2c \
        --batches $BATCHES \
        --batch-size $BATCH_SIZE \
        --workers $WORKERS \
        --eval-every $EVAL_EVERY \
        --save-path $SAVE_PATH \
        --seed $SEED \
        $RESUME \
        2>&1 | tee training.log"
fi

echo "=== Starting Catan Training ==="
echo "  Mode:       $MODE"
echo "  Batches:    $BATCHES x $BATCH_SIZE"
echo "  Workers:    $WORKERS"
echo "  Save path:  $SAVE_PATH"
echo "  Resume:     ${RESUME:-no}"
echo ""
echo "  Training will run in tmux session 'catan'"
echo "  Detach:     Ctrl+B, then D"
echo "  Reattach:   tmux attach -t catan"
echo "  View log:   tail -f training.log"
echo ""

# Start in tmux
tmux new-session -d -s catan "$CMD; echo ''; echo 'Training complete! Press Enter to exit.'; read"

echo "Training started! Attaching to session..."
echo "(Press Ctrl+B then D to detach without stopping training)"
echo ""
sleep 1
tmux attach -t catan
