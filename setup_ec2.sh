#!/bin/bash
# ──────────────────────────────────────────────────────────────
# setup_ec2.sh — One-command setup for training on EC2 GPU
#
# Launch a g4dn.xlarge spot instance with Deep Learning AMI,
# then SSH in and run:
#
#   git clone <your-repo-url> catanEngine && cd catanEngine
#   chmod +x setup_ec2.sh && ./setup_ec2.sh
#
# ──────────────────────────────────────────────────────────────
set -e

echo "=== Catan Engine — EC2 GPU Setup ==="
echo ""

# ── 1. System packages ──
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop > /dev/null 2>&1

# ── 2. Python environment ──
echo "[2/4] Setting up Python environment..."
# Use conda if available (Deep Learning AMI), otherwise system python
if command -v conda &> /dev/null; then
    echo "  Using conda..."
    conda activate pytorch 2>/dev/null || conda activate base
    pip install -q gymnasium numpy
else
    echo "  Using system Python..."
    python3 -m pip install --upgrade pip -q
    python3 -m pip install torch gymnasium numpy -q
fi

# ── 3. Verify setup ──
echo "[3/4] Verifying setup..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Verify game engine
python3 -c "
from catan.board import Board
from catan.game import Game
board = Board.build(seed=0)
game = Game(board, num_players=4, seed=0)
print(f'  Engine:  OK ({board.num_vertices} vertices, {board.num_edges} edges)')
"

# ── 4. Create checkpoints dir ──
echo "[4/4] Creating directories..."
mkdir -p checkpoints replays

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  ./train_remote.sh"
echo ""
echo "Or manually:"
echo "  python3 -m training.train --algo a2c --batches 2000 --batch-size 64 --workers 4 --save-path checkpoints/a2c_gpu.pt"
echo ""
