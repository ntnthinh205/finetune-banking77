#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "BANKING77 Intent Detection - Training Pipeline"
echo "=============================================="

# Step 1: Install requirements
echo ""
echo "[Step 1/3] Installing requirements..."
pip install -r requirements.txt

# Step 2: Preprocess data
echo ""
echo "[Step 2/3] Preprocessing data..."
python scripts/preprocess_data.py --config configs/train.yaml

# Step 3: Train model
echo ""
echo "[Step 3/3] Fine-tuning model..."
python scripts/train.py --config configs/train.yaml

echo ""
echo "=============================================="
echo "Training pipeline complete!"
echo "=============================================="
