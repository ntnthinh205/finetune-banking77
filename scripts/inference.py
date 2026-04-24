"""
inference.py - Standalone Inference for BANKING77 Intent Detection
===================================================================
This script provides the IntentClassification class that loads a
fine-tuned model checkpoint and predicts intent labels for input messages.

Required interface (per project specification):
    - __init__(self, model_path): Load config, tokenizer, and model
    - __call__(self, message): Predict intent label for a message

Usage:
    # As a module:
    from scripts.inference import IntentClassification
    classifier = IntentClassification("configs/inference.yaml")
    result = classifier("I want to know my card balance")
    print(result)  # "balance_not_updated_after_bank_transfer"

    # From command line:
    python scripts/inference.py --config configs/inference.yaml \
        --message "I want to know my card balance"
"""

import os
import sys
import json
import argparse
import yaml
import torch


# ==============================================================================
# Prompt Template (must match the one used during training)
# ==============================================================================
PROMPT_TEMPLATE = """### Instruction:
Classify the following banking customer message into the correct intent category.
Only respond with the intent label, nothing else.

### Message:
{}

### Intent:
"""


class IntentClassification:
    """
    Banking intent classification using a fine-tuned language model.

    This class loads a saved checkpoint (fine-tuned with Unsloth) and
    predicts the intent label for a given customer message.

    Attributes:
        model: The fine-tuned language model
        tokenizer: The tokenizer for the model
        label_map: Mapping between label IDs and label names
        device: The device (cuda/cpu) used for inference
    """

    def __init__(self, model_path: str):
        """
        Initialize the IntentClassification model.

        Args:
            model_path: Path to the configuration YAML file that contains
                       the model checkpoint path and other settings.
        """
        # Load configuration
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)

        checkpoint_dir = self.config["model_checkpoint"]
        max_seq_length = self.config.get("max_seq_length", 512)
        load_in_4bit = self.config.get("load_in_4bit", True)

        print(f"Loading model from: {checkpoint_dir}")

        # Load label mapping
        label_map_path = self.config.get(
            "label_map_path",
            os.path.join(checkpoint_dir, "label_map.json"),
        )
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        self.id2label = self.label_map["id2label"]
        self.label2id = self.label_map["label2id"]
        self.num_labels = self.label_map["num_labels"]
        self.valid_labels = set(
            name.lower() for name in self.label_map["label2id"].keys()
        )

        # Load model with Unsloth
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )

        # Enable fast inference
        FastLanguageModel.for_inference(self.model)

        self.max_seq_length = max_seq_length
        self.device = self.model.device

        print(f"Model loaded successfully!")
        print(f"Number of intent classes: {self.num_labels}")
        print(f"Device: {self.device}")

    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a given customer message.

        Args:
            message: The customer banking message to classify.

        Returns:
            predicted_label: The predicted intent label as a string.
        """
        # Format the prompt
        prompt = PROMPT_TEMPLATE.format(message.strip())

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        # Suppress warnings
        import transformers
        import warnings
        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )

        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        # Clean up the predicted label
        predicted_label = generated_text.strip()

        # Try to match with valid labels (case-insensitive)
        predicted_lower = predicted_label.lower()
        if predicted_lower in self.valid_labels:
            # Return the correctly-cased version
            for original_name in self.label_map["label2id"].keys():
                if original_name.lower() == predicted_lower:
                    return original_name

        # If no exact match, return the raw prediction
        return predicted_label

    def predict_with_confidence(self, message: str) -> dict:
        """
        Predict intent with additional metadata (for debugging/analysis).

        Args:
            message: The customer banking message to classify.

        Returns:
            Dictionary with 'label', 'raw_output', and 'valid' keys.
        """
        # Format the prompt
        prompt = PROMPT_TEMPLATE.format(message.strip())

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        predicted_label = self.__call__(message)
        is_valid = predicted_label.lower() in self.valid_labels

        return {
            "label": predicted_label,
            "raw_output": generated_text,
            "valid": is_valid,
        }


def evaluate_test_set(classifier: IntentClassification, test_csv_path: str):
    """
    Evaluate the model on the test set and print accuracy.

    Args:
        classifier: IntentClassification instance
        test_csv_path: Path to test CSV file
    """
    import pandas as pd

    test_df = pd.read_csv(test_csv_path)

    correct = 0
    total = len(test_df)

    print(f"\nEvaluating on {total} test samples...")
    print("-" * 50)

    for idx, row in test_df.iterrows():
        predicted = classifier(row["text"])
        true_label = row["label_name"]

        is_correct = predicted.lower().strip() == true_label.lower().strip()
        correct += int(is_correct)

        # Print progress every 50 samples
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{total} ({100 * correct / (idx + 1):.1f}% so far)")

    accuracy = 100 * correct / total
    print("-" * 50)
    print(f"\n✅ Test Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for BANKING77 intent detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="A banking customer message to classify",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to test CSV for evaluation (e.g., sample_data/test.csv)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (type messages one by one)",
    )
    args = parser.parse_args()

    # Initialize classifier
    print("=" * 60)
    print("BANKING77 Intent Detection - Inference")
    print("=" * 60)

    classifier = IntentClassification(args.config)

    # Mode 1: Single message prediction
    if args.message:
        print(f"\nInput message: {args.message}")
        result = classifier(args.message)
        print(f"Predicted intent: {result}")

        # Also show detailed prediction
        detailed = classifier.predict_with_confidence(args.message)
        print(f"Raw model output: {detailed['raw_output']}")
        print(f"Valid label: {detailed['valid']}")

    # Mode 2: Evaluate test set
    elif args.evaluate:
        evaluate_test_set(classifier, args.evaluate)

    # Mode 3: Interactive mode
    elif args.interactive:
        print("\n💬 Interactive Mode (type 'quit' to exit)")
        print("-" * 50)

        while True:
            message = input("\nYour message: ").strip()
            if message.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not message:
                continue

            result = classifier(message)
            print(f"🏷️  Predicted intent: {result}")

    # Mode 4: Demo with example messages
    else:
        print("\n📋 Demo with example messages:")
        print("-" * 50)

        examples = [
            "I am still waiting on my card?",
            "What can I do if my card is lost?",
            "I need to change my PIN number",
            "Why was I charged a fee for my transaction?",
            "How do I transfer money to another account?",
        ]

        for msg in examples:
            result = classifier(msg)
            print(f"\n  Message: \"{msg}\"")
            print(f"  Intent:  {result}")

        print("\n" + "-" * 50)
        print("Use --message, --evaluate, or --interactive for more options.")


if __name__ == "__main__":
    main()
