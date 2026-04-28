import os
import sys
import json
import argparse
import yaml
import torch
from unsloth import FastLanguageModel

PROMPT_TEMPLATE = """### Instruction:
Classify the following banking customer message into the correct intent category.
Only respond with the intent label, nothing else.
### Message:
{}
### Intent:
"""


class IntentClassification:
    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
        checkpoint_dir = self.config["model_checkpoint"]
        max_seq_length = self.config.get("max_seq_length", 512)
        load_in_4bit = self.config.get("load_in_4bit", True)
        print(f"Loading model from: {checkpoint_dir}")
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

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )
        FastLanguageModel.for_inference(self.model)
        self.max_seq_length = max_seq_length
        self.device = self.model.device
        print(f"Model loaded successfully!")
        print(f"Number of intent classes: {self.num_labels}")
        print(f"Device: {self.device}")

    def __call__(self, message: str) -> str:
        prompt = PROMPT_TEMPLATE.format(message.strip())
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        import transformers
        import warnings

        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore")
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
        predicted_label = generated_text.strip()
        predicted_lower = predicted_label.lower()
        if predicted_lower in self.valid_labels:
            for original_name in self.label_map["label2id"].keys():
                if original_name.lower() == predicted_lower:
                    return original_name
        return predicted_label

    def predict_with_confidence(self, message: str) -> dict:
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
    import pandas as pd

    test_df = pd.read_csv(test_csv_path)
    correct = 0
    total = len(test_df)
    print(f"\nEvaluating on {total} test samples...")
    print("-" * 50)
    for idx, row in test_df.iterrows():
        predicted = classifier(row["text"])
        true_label = row["label_name"]
        predicted_clean = (
            predicted.strip().split("\n")[0].strip().lower().rstrip(". ,;!?")
        )
        true_label_lower = true_label.strip().lower()
        is_correct = (predicted_clean == true_label_lower) or (
            true_label_lower in predicted.lower()
        )
        correct += int(is_correct)
        if (idx + 1) % 50 == 0:
            print(
                f"   Processed {idx + 1}/{total} ({100 * correct / (idx + 1):.1f}% so far)"
            )
    accuracy = 100 * correct / total
    print("-" * 50)
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")
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
    print("=" * 60)
    print("BANKING77 Intent Detection - Inference")
    print("=" * 60)
    classifier = IntentClassification(args.config)
    if args.message:
        print(f"\nInput message: {args.message}")
        result = classifier(args.message)
        print(f"Predicted intent: {result}")
        detailed = classifier.predict_with_confidence(args.message)
        print(f"Raw model output: {detailed['raw_output']}")
        print(f"Valid label: {detailed['valid']}")
    elif args.evaluate:
        evaluate_test_set(classifier, args.evaluate)
    elif args.interactive:
        print("\nInteractive Mode (type 'quit' to exit)")
        print("-" * 50)
        while True:
            message = input("\nYour message: ").strip()
            if message.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not message:
                continue
            result = classifier(message)
            print(f"Predicted intent: {result}")
    else:
        print("\nDemo with example messages:")
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
            print(f'\n  Message: "{msg}"')
            print(f"  Intent:  {result}")
        print("\n" + "-" * 50)
        print("Use --message, --evaluate, or --interactive for more options.")


if __name__ == "__main__":
    main()
