# Banking Intent Detection with Unsloth

Fine-tuning a language model for intent detection on the **BANKING77** dataset using **Unsloth** with LoRA.

## 📋 Project Overview

This project implements a banking intent classification system that can predict **30 intent categories** from customer messages. The model is fine-tuned using the **Unsloth** library with **LoRA (Low-Rank Adaptation)** for efficient training on limited GPU resources.

### Key Features

- **Model**: Qwen2.5-1.5B-Instruct (4-bit quantized)
- **Dataset**: BANKING77 (subset of 30 intents, ~5,000 samples)
- **Training**: LoRA fine-tuning with Unsloth + SFTTrainer
- **Approach**: Generative classification (LLM outputs intent label as text)

## 📁 Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py    # Data preprocessing pipeline
│   ├── train.py              # Model fine-tuning with Unsloth
│   └── inference.py          # Standalone inference class
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference configuration
├── sample_data/
│   ├── train.csv             # Training data (generated)
│   ├── test.csv              # Test data (generated)
│   └── label_map.json        # Label ID ↔ name mapping
├── checkpoints/              # Saved model (generated)
├── train.sh                  # Training shell script
├── inference.sh              # Inference shell script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Setup & Installation

### Prerequisites

- **Python** 3.9+
- **GPU**: NVIDIA GPU with at least 8GB VRAM (Google Colab T4 or better)
- **Platform**: Google Colab (recommended), Kaggle, or local machine with NVIDIA GPU

### Step 1: Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> **For Google Colab**, run the following in the first cell:
> ```python
> !pip install unsloth
> ```

## 📦 Data Preparation

### Download and Preprocess

```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

This will:
1. Download the BANKING77 dataset from HuggingFace
2. Sample 30 intents (out of 77) for manageable training
3. Normalize text data
4. Split into 80% train / 20% test
5. Save to `sample_data/train.csv` and `sample_data/test.csv`

### Dataset Statistics

| Split | Samples | Intents |
|-------|---------|---------|
| Train | ~4,000  | 30      |
| Test  | ~1,000  | 30      |

## 🏋️ Training

### Run Training

```bash
# Full pipeline (preprocess + train)
bash train.sh

# Or train only (if data is already preprocessed)
python scripts/train.py --config configs/train.yaml
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Max Sequence Length | 512 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 32 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Weight Decay | 0.01 |
| Epochs | 3 |
| LR Scheduler | Linear |
| Precision | FP16 |
| Warmup Steps | 10 |

### Training Approach

The model is trained using a **generative classification** approach where the classification task is framed as text generation:

```
### Instruction:
Classify the following banking customer message into the correct intent category.
Only respond with the intent label, nothing else.

### Message:
I am still waiting on my card?

### Intent:
card_arrival
```

## 🔮 Inference

### Using the Python Class

```python
from scripts.inference import IntentClassification

# Initialize (loads model from checkpoint)
classifier = IntentClassification("configs/inference.yaml")

# Predict intent for a message
result = classifier("I am still waiting on my card?")
print(result)  # → "card_arrival"
```

### Command Line Usage

```bash
# Single message prediction
bash inference.sh "I want to change my PIN"

# Evaluate on test set
bash inference.sh --evaluate

# Interactive mode
bash inference.sh --interactive

# Demo with example messages
bash inference.sh
```

### Inference Class Interface

```python
class IntentClassification:
    def __init__(self, model_path):
        """
        Load configuration, tokenizer, and model checkpoint.

        Args:
            model_path: Path to inference config YAML file
        """
        pass

    def __call__(self, message):
        """
        Predict the intent label for a customer message.

        Args:
            message: Banking customer message string

        Returns:
            predicted_label: The predicted intent label string
        """
        return predicted_label
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | TBD (fill after training) |
| Training Time | ~15-20 min on T4 GPU |
| Peak GPU Memory | ~8 GB |

## 🎥 Video Demo

> **Video demonstration link:** [Google Drive Link](YOUR_LINK_HERE)
>
> The video shows:
> - How the inference script is executed
> - Example input messages and predicted intents
> - Test set accuracy

## ⚙️ Configuration

All hyperparameters can be modified in:
- `configs/train.yaml` — Training settings
- `configs/inference.yaml` — Inference settings

## 📚 References

- [BANKING77 Dataset](https://huggingface.co/datasets/PolyAI/banking77) — PolyAI
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) — Official guides

## 👤 Author

- **Name**: [Your Name]
- **Student ID**: [Your ID]
- **Course**: Applications of Natural Language Processing in Industry
- **Lecturer**: Dr. Nguyen Hong Buu Long

---
*University of Science - Vietnam National University Ho Chi Minh City*
# finetune-banking77
