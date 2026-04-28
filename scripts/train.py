import os
import sys
import json
import argparse
import yaml
import torch
import pandas as pd
from datasets import Dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


PROMPT_TEMPLATE = """### Instruction:
Classify the following banking customer message into the correct intent category.
Only respond with the intent label, nothing else.
### Message:
{}
### Intent:
{}"""


def format_prompt(text: str, label_name: str = "", for_inference: bool = False) -> str:
    if for_inference:
        return PROMPT_TEMPLATE.split("### Intent:\n")[0] + "### Intent:\n"
    return PROMPT_TEMPLATE.format(text, label_name)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune intent detection model with Unsloth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    lora_config = config["lora"]
    train_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]
    print("=" * 60)
    print("BANKING77 Intent Detection - Fine-tuning with Unsloth")
    print("=" * 60)
    print("\n[1/6] Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config["max_seq_length"],
        load_in_4bit=model_config["load_in_4bit"],
        dtype=None,
    )
    print(f"   Model: {model_config['name']}")
    print(f"   Max seq length: {model_config['max_seq_length']}")
    print(f"   Quantization: {'4-bit' if model_config['load_in_4bit'] else 'None'}")
    print("\n[2/6] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
        use_rslora=lora_config["use_rslora"],
        loftq_config=lora_config["loftq_config"],
        random_state=train_config["seed"],
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   LoRA rank (r): {lora_config['r']}")
    print(f"   LoRA alpha: {lora_config['lora_alpha']}")
    print(f"   Target modules: {lora_config['target_modules']}")
    print(
        f"   Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    print("\n[3/6] Loading and formatting dataset...")
    train_df = pd.read_csv(data_config["train_path"])
    test_df = pd.read_csv(data_config["test_path"])
    label_map_path = os.path.join(
        os.path.dirname(data_config["train_path"]), "label_map.json"
    )
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    num_labels = label_map["num_labels"]
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"   Number of intents: {num_labels}")

    def format_dataset(row):
        return {
            "text": format_prompt(row["text"], row["label_name"]) + tokenizer.eos_token,
        }

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        format_dataset,
        remove_columns=train_dataset.column_names,
    )
    print("\n   Example formatted prompt:")
    print("   " + "-" * 50)
    example = train_dataset[0]["text"]
    for line in example.split("\n"):
        print(f"   {line}")
    print("   " + "-" * 50)
    print("\n[4/6] Configuring SFTTrainer...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    os.makedirs(output_config["checkpoint_dir"], exist_ok=True)
    os.makedirs(output_config.get("log_dir", "logs"), exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_config["checkpoint_dir"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        num_train_epochs=train_config["num_train_epochs"],
        learning_rate=train_config["learning_rate"],
        optim=train_config["optimizer"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_steps"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        logging_steps=train_config["logging_steps"],
        save_strategy=train_config["save_strategy"],
        seed=train_config["seed"],
        max_grad_norm=train_config["max_grad_norm"],
        logging_dir=output_config.get("log_dir", "logs"),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=model_config["max_seq_length"],
        args=training_args,
    )
    print(f"   Batch size: {train_config['per_device_train_batch_size']}")
    print(f"   Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(
        f"   Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}"
    )
    print(f"   Epochs: {train_config['num_train_epochs']}")
    print(f"   Learning rate: {train_config['learning_rate']}")
    print(f"   Optimizer: {train_config['optimizer']}")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\n   GPU: {gpu_stats.name}")
        print(f"   GPU Memory: {start_gpu_memory} GB / {max_memory} GB")
    print("\n[5/6] Starting training...")
    print("=" * 60)
    trainer_stats = trainer.train()
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"   Training time: {trainer_stats.metrics['train_runtime']:.1f} seconds")
    print(
        f"   Training time: {trainer_stats.metrics['train_runtime'] / 60:.1f} minutes"
    )
    print(f"   Training loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        print(f"   Peak GPU memory: {used_memory} GB")
        print(f"   GPU memory for training: {used_memory_for_lora} GB")
    print("\n[6/6] Saving model...")
    print("\n   Saving model checkpoint...")
    model.save_pretrained(output_config["checkpoint_dir"])
    tokenizer.save_pretrained(output_config["checkpoint_dir"])
    checkpoint_label_map_path = os.path.join(
        output_config["checkpoint_dir"], "label_map.json"
    )
    with open(checkpoint_label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    summary = {
        "model": model_config["name"],
        "num_intents": num_labels,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "training_time_seconds": trainer_stats.metrics["train_runtime"],
        "training_loss": trainer_stats.metrics.get("train_loss", None),
        "hyperparameters": {
            "lora_r": lora_config["r"],
            "lora_alpha": lora_config["lora_alpha"],
            "batch_size": train_config["per_device_train_batch_size"],
            "gradient_accumulation": train_config["gradient_accumulation_steps"],
            "learning_rate": train_config["learning_rate"],
            "epochs": train_config["num_train_epochs"],
            "optimizer": train_config["optimizer"],
            "max_seq_length": model_config["max_seq_length"],
        },
    }
    summary_path = os.path.join(
        output_config["checkpoint_dir"], "training_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved checkpoint to: {output_config['checkpoint_dir']}")
    print(f"   Saved training summary to: {summary_path}")
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"   Model saved at: {output_config['checkpoint_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
