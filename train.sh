#!/bin/bash
set -e  # Exit on error

echo "=============================================="
echo "BANKING77 Intent Detection - Training"
echo "=============================================="

# Train model
echo ""
echo "Fine-tuning model..."
python scripts/train.py --config configs/train.yaml

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
