#!/bin/bash
# ==============================================================================
# inference.sh - Inference for BANKING77 Intent Detection
# ==============================================================================
# This script runs inference using the fine-tuned model.
#
# Usage:
#   # Single message prediction:
#   bash inference.sh "I am still waiting on my card"
#
#   # Evaluate on test set:
#   bash inference.sh --evaluate
#
#   # Interactive mode:
#   bash inference.sh --interactive
#
#   # Demo mode (default):
#   bash inference.sh
# ==============================================================================

set -e  # Exit on error

CONFIG="configs/inference.yaml"

if [ "$1" == "--evaluate" ]; then
    echo "Evaluating on test set..."
    python scripts/inference.py --config "$CONFIG" --evaluate "sample_data/test.csv"
elif [ "$1" == "--interactive" ]; then
    echo "Starting interactive mode..."
    python scripts/inference.py --config "$CONFIG" --interactive
elif [ -n "$1" ]; then
    echo "Predicting intent for: $1"
    python scripts/inference.py --config "$CONFIG" --message "$1"
else
    echo "Running demo..."
    python scripts/inference.py --config "$CONFIG"
fi
