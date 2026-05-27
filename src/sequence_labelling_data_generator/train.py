#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa for Sequence Labeling using LoRA and AMP")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="daominhwysi/synthetic-seq-labelling-vi-exam",
        help="Hugging Face Dataset repository ID"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Hugging Face base model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save checkpoint results and models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for trainable parameters (LoRA + classification head)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (requires compatible GPU like A100+)"
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable float16 mixed precision (defaults to True otherwise on CUDA)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Max number of checkpoints to retain"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final trained model/adapters back to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face authentication token (or set HF_TOKEN env var)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Hugging Face Authentication & Token Setup
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        # Avoid prompt blocking in Colab
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged into Hugging Face Hub successfully.")
    
    # 2. Check GPU/Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set mixed precision defaults
    fp16_enabled = torch.cuda.is_available() and not args.no_fp16
    bf16_enabled = torch.cuda.is_available() and args.use_bf16
    if bf16_enabled:
        fp16_enabled = False
        print("Automatic Mixed Precision (AMP) enabled: bfloat16")
    elif fp16_enabled:
        print("Automatic Mixed Precision (AMP) enabled: float16 (standard GPU)")
    else:
        print("Automatic Mixed Precision (AMP) disabled")

    # 3. Download Label Mapping & Dataset
    print(f"Downloading dataset and label mapping from HF: '{args.repo_id}'...")
    try:
        from datasets import load_dataset
        # Load the custom split jsonl files
        dataset = load_dataset(
            args.repo_id,
            data_files={
                "train": "train.jsonl",
                "validation": "val.jsonl",
                "test": "test.jsonl"
            },
            token=hf_token
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you specify the correct --repo_id and provide a valid token if private.")
        sys.exit(1)
        
    try:
        from huggingface_hub import hf_hub_download
        label_mapping_path = hf_hub_download(
            repo_id=args.repo_id,
            filename="label_mapping.json",
            repo_type="dataset",
            token=hf_token
        )
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
            
        tag_to_id = label_mapping["tag_to_id"]
        id_to_tag = {int(k): v for k, v in label_mapping["id_to_tag"].items()}
        print(f"Loaded label mapping from Hub. Found {len(tag_to_id)} labels.")
    except Exception as e:
        print(f"Warning: Could not download 'label_mapping.json' from the repository: {e}")
        print("Building label mapping dynamically from training dataset tags...")
        # Fallback dynamic mapping builder
        unique_labels = set()
        for split in ["train", "validation"]:
            for sample in dataset[split]:
                unique_labels.update(sample["labels"])
        # Remove ignored index
        unique_labels.discard(-100)
        # Sort labels to be deterministic
        sorted_labels = sorted(list(unique_labels))
        
        # Build standard mappings (assuming standard schema tags)
        # Note: If label_mapping.json is missing, we try to reconstruct labels
        print(f"Found unique label IDs in dataset: {sorted_labels}")
        # Standard tag labels used for logging validation
        id_to_tag = {l: f"LABEL_{l}" for l in sorted_labels}
        id_to_tag[0] = "O" # Ensure label 0 is marked "O"
        tag_to_id = {v: k for k, v in id_to_tag.items()}
        
    num_labels = len(tag_to_id)
    label_list = [id_to_tag[i] for i in sorted(id_to_tag.keys())]

    # 4. Tokenizer Setup (necessary for Data Collator padding)
    print(f"Loading Tokenizer: '{args.model_name}'...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    
    # Check if we need to add special LaTeX token to token embeddings if dataset used it
    has_latex_placeholder = False
    for sample in dataset["train"]:
        if "[LATEX]" in sample.get("tokens", []):
            has_latex_placeholder = True
            break
            
    if has_latex_placeholder:
        print("Detected '[LATEX]' token in dataset. Adding it to the tokenizer...")
        tokenizer.add_special_tokens({"additional_special_tokens": ["[LATEX]"]})
        
    # 5. Initialize Model
    print(f"Loading Model: '{args.model_name}'...")
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: id_to_tag[i] for i in id_to_tag},
        label2id=tag_to_id,
        token=hf_token
    )
    
    # Resize token embeddings if we added special tokens
    if has_latex_placeholder:
        model.resize_token_embeddings(len(tokenizer))

    # 6. Apply LoRA (PEFT)
    print("Applying Low-Rank Adaptation (LoRA)...")
    # Bypass torchao compatibility check bug on older pre-installed versions in Google Colab
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    from peft import LoraConfig, get_peft_model, TaskType
    
    # Standard XLM-RoBERTa self-attention target modules are query and value
    target_modules = ["query", "value"]
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["classifier"]  # Ensures classifier head is trained fully (not frozen)
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 7. Metrics Definition
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (-100)
        true_predictions = [
            [label_list[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]

        try:
            from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
                "accuracy": np.mean([p_v == l_v for p_seq, l_seq in zip(true_predictions, true_labels) for p_v, l_v in zip(p_seq, l_seq)])
            }
        except ImportError:
            # Fallback to token-level evaluation if seqeval is not installed
            from sklearn.metrics import f1_score, accuracy_score
            flat_preds = [p_v for p_seq in true_predictions for p_v in p_seq]
            flat_labels = [l_v for l_seq in true_labels for l_v in l_seq]
            return {
                "accuracy": accuracy_score(flat_labels, flat_preds),
                "f1_macro": f1_score(flat_labels, flat_preds, average="macro")
            }

    # 8. Data Collator
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # 9. Training Arguments
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if "seqeval" in sys.modules or "seqeval" in globals() else "accuracy",
        greater_is_better=True,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        save_total_limit=args.save_total_limit,
        report_to="none",  # Change to "wandb" or "tensorboard" if configured
        push_to_hub=args.push_to_hub,
        hub_token=hf_token
    )

    # 10. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 11. Run Training
    print("Starting training...")
    trainer.train()
    
    # 12. Run final test split evaluation
    print("Evaluating on test split...")
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    print(f"\nFinal Test Set Results:\n{json.dumps(test_results, indent=2)}")
    
    # Save the final adapter model
    print(f"Saving final model adapter to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    
    if args.push_to_hub:
        print(f"Pushing model adapters to HF Hub...")
        trainer.push_to_hub(commit_message="Add trained XLM-RoBERTa LoRA sequence labeler adapters")

if __name__ == "__main__":
    main()
