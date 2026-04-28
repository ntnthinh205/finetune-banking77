import os
import sys
import json
import argparse
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_banking77_label_names() -> list:
    return [
        "activate_my_card",
        "age_limit",
        "apple_pay_or_google_pay",
        "atm_support",
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "country_support",
        "declined_card_payment",
        "declined_cash_withdrawal",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "disposable_card_limits",
        "edit_personal_details",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "extra_charge_on_statement",
        "failed_transfer",
        "fiat_currency_support",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "order_physical_card",
        "passcode_forgotten",
        "pending_card_payment",
        "pending_cash_withdrawal",
        "pending_top_up",
        "pending_transfer",
        "pin_blocked",
        "receiving_money",
        "Refund_not_showing_up",
        "request_refund",
        "reverted_card_payment?",
        "supported_cards_and_currencies",
        "terminate_account",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "verify_top_up",
        "virtual_card_not_working",
        "visa_or_mastercard",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]


def normalize_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BANKING77 dataset for intent classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    data_config = config["data"]

    num_intents = data_config.get("num_intents", 30)
    test_size = data_config.get("test_size", 0.2)
    random_state = data_config.get("random_state", 42)
    train_path = data_config.get("train_path", "sample_data/train.csv")
    test_path = data_config.get("test_path", "sample_data/test.csv")

    print("=" * 60)
    print("BANKING77 Data Preprocessing")
    print("=" * 60)

    # ----- Step 1: Load dataset -----
    print("\n[1/5] Loading BANKING77 dataset from HuggingFace...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Combine train and test for re-splitting
    all_texts = train_data["text"] + test_data["text"]
    all_labels = train_data["label"] + test_data["label"]

    label_names = get_banking77_label_names()
    print(f"   Total samples: {len(all_texts)}")
    print(f"   Total intents: {len(set(all_labels))}")

    # ----- Step 2: Sample subset of intents -----
    print(f"\n[2/5] Sampling {num_intents} intents out of 77...")

    # Select intents deterministically
    import random

    random.seed(random_state)
    unique_labels = sorted(set(all_labels))
    selected_labels = sorted(random.sample(unique_labels, min(num_intents, len(unique_labels))))

    # Filter data
    filtered_texts = []
    filtered_labels = []
    filtered_label_names = []

    for text, label in zip(all_texts, all_labels):
        if label in selected_labels:
            filtered_texts.append(text)
            filtered_labels.append(label)
            filtered_label_names.append(label_names[label])

    print(f"   Selected intents: {len(selected_labels)}")
    print(f"   Total samples after filtering: {len(filtered_texts)}")

    # ----- Step 3: Normalize text -----
    print("\n[3/5] Normalizing text...")
    filtered_texts = [normalize_text(t) for t in filtered_texts]

    # Create label remapping (0, 1, 2, ..., num_intents-1)
    label_remap = {old_label: new_idx for new_idx, old_label in enumerate(selected_labels)}
    remapped_labels = [label_remap[l] for l in filtered_labels]
    selected_label_names = [label_names[l] for l in selected_labels]

    # ----- Step 4: Split train/test -----
    print(f"\n[4/5] Splitting data (train: {1 - test_size:.0%}, test: {test_size:.0%})...")

    X_train, X_test, y_train, y_test, y_names_train, y_names_test = train_test_split(
        filtered_texts,
        remapped_labels,
        filtered_label_names,
        test_size=test_size,
        random_state=random_state,
        stratify=remapped_labels,
    )

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # ----- Step 5: Save data -----
    print("\n[5/5] Saving data...")

    # Create output directories
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # Save train CSV
    train_df = pd.DataFrame(
        {"text": X_train, "label": y_train, "label_name": y_names_train}
    )
    train_df.to_csv(train_path, index=False)
    print(f"   Saved training data to: {train_path}")

    # Save test CSV
    test_df = pd.DataFrame(
        {"text": X_test, "label": y_test, "label_name": y_names_test}
    )
    test_df.to_csv(test_path, index=False)
    print(f"   Saved test data to: {test_path}")

    # Save label mapping
    label_map = {
        "id2label": {str(i): name for i, name in enumerate(selected_label_names)},
        "label2id": {name: i for i, name in enumerate(selected_label_names)},
        "num_labels": len(selected_label_names),
    }
    label_map_path = os.path.join(os.path.dirname(train_path), "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"   Saved label mapping to: {label_map_path}")

    # Print label distribution summary
    print("\n" + "=" * 60)
    print("Label Distribution (Training Set):")
    print("=" * 60)
    label_counts = train_df["label_name"].value_counts().sort_index()
    for name, count in label_counts.items():
        print(f"   {name}: {count}")

    print(f"   - {len(X_train)} training samples")
    print(f"   - {len(X_test)} test samples")
    print(f"   - {len(selected_label_names)} intent classes")


if __name__ == "__main__":
    main()
