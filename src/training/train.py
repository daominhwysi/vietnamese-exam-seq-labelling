#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import torch
import inspect

# Setup local import paths
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.training_pipeline import (
    parse_args,
    setup_device,
    get_compute_metrics_fn,
    WeightedTrainer,
    EMACallback
)

def main():
    args = parse_args()
    run_train(args)

def run_train(args):
    # 1. Hugging Face Authentication & Token Setup
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged into Hugging Face Hub successfully.")

    # 2. Check GPU/Device
    device = setup_device(args)

    gpu_supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    # Load model config to check its native precision/dtype
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model_name, token=hf_token)
        config_dtype = getattr(config, "torch_dtype", None)
    except Exception as e:
        print(f"Warning: Could not load model configuration: {e}")
        config_dtype = None

    # Handle bfloat16 compatibility and automatic selection
    if args.use_bf16:
        bf16_enabled = True
        fp16_enabled = False
    elif config_dtype in [torch.bfloat16, "bfloat16", "bf16"] and gpu_supports_bf16 and not args.no_fp16:
        bf16_enabled = True
        fp16_enabled = False
        print("Model configuration specifies bfloat16 and GPU supports it. Automatically enabling bfloat16 training.")
    else:
        bf16_enabled = False
        fp16_enabled = torch.cuda.is_available() and not args.no_fp16

    # Select the model loading torch_dtype based on precision settings
    if bf16_enabled:
        load_dtype = torch.bfloat16
        print("Automatic Mixed Precision (AMP) enabled: bfloat16")
    elif fp16_enabled:
        load_dtype = torch.float32
        print("Automatic Mixed Precision (AMP) enabled: float16 (standard GPU)")
    else:
        load_dtype = torch.float32
        print("Automatic Mixed Precision (AMP) disabled (training in float32)")

    # 3. Download Label Mapping & Dataset
    print(f"Downloading dataset and label mapping from HF: '{args.repo_id}'...")
    try:
        from datasets import load_dataset
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
        unique_labels = set()
        for split in ["train", "validation"]:
            for sample in dataset[split]:
                unique_labels.update(sample["labels"])
        unique_labels.discard(-100)
        sorted_labels = sorted(list(unique_labels))

        print(f"Found unique label IDs in dataset: {sorted_labels}")
        id_to_tag = {l: f"LABEL_{l}" for l in sorted_labels}
        id_to_tag[0] = "O"
        tag_to_id = {v: k for k, v in id_to_tag.items()}

    num_labels = len(tag_to_id)
    label_list = [id_to_tag[i] for i in sorted(id_to_tag.keys())]

    # 4. Tokenizer Setup
    print(f"Loading Tokenizer: '{args.model_name}'...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)

    special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
    print(f"Adding additional special tokens to the tokenizer: {special_tokens}")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 5. Initialize Model
    print(f"Loading Model: '{args.model_name}' with dtype: {load_dtype}...")
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: id_to_tag[i] for i in id_to_tag},
        label2id=tag_to_id,
        token=hf_token,
        torch_dtype=load_dtype
    )

    model.resize_token_embeddings(len(tokenizer))

    # 6. Apply LoRA (PEFT) if enabled
    if not getattr(args, "no_lora", False):
        print("Applying Low-Rank Adaptation (LoRA)...")
        try:
            import peft.import_utils
            peft.import_utils.is_torchao_available = lambda: False
        except Exception:
            pass

        from peft import LoraConfig, get_peft_model, TaskType

        model_name_lower = args.model_name.lower()
        if "modernbert" in model_name_lower or "mmbert" in model_name_lower:
            target_modules = ["Wqkv", "Wo"]
            print(f"Detected ModernBERT/mmBERT architecture. Targeting modules: {target_modules}")
        else:
            target_modules = ["query", "value"]
            print(f"Targeting standard attention modules: {target_modules}")

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["classifier"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("LoRA is disabled. Preparing for Full Fine-Tuning...")

    # 7. Metrics Definition
    compute_metrics = get_compute_metrics_fn(label_list)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if "seqeval" in sys.modules or "seqeval" in globals() else "accuracy",
        greater_is_better=True,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        save_total_limit=args.save_total_limit,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_token=hf_token,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        seed=getattr(args, "seed", 42),
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "linear"),
        warmup_ratio=getattr(args, "warmup_ratio", 0.0),
        warmup_steps=getattr(args, "warmup_steps", 0),
    )

    # 9.5 Calculate class weights if enabled
    class_weights = None
    if not getattr(args, "no_class_weights", False):
        print("Calculating class weights from training dataset...")
        from collections import Counter
        label_counts = Counter()
        for sample in dataset["train"]:
            label_counts.update([l for l in sample["labels"] if l != -100])
        
        weights = np.ones(num_labels, dtype=np.float32)
        total_count = sum(label_counts.values())
        
        if total_count > 0:
            for label_id in range(num_labels):
                count = label_counts.get(label_id, 0)
                if count > 0:
                    weights[label_id] = total_count / (num_labels * np.sqrt(count))
            weights = weights / weights.mean()
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Computed class weights: {weights.tolist()}")
            for label_name, label_id in tag_to_id.items():
                print(f"  {label_name} (ID: {label_id}): Weight = {weights[label_id]:.4f}")
        else:
            print("Warning: No labels found in training dataset. Skipping class weights.")

    real_upsample_factor = getattr(args, "real_upsample_factor", 1.0)

    # 10. Instantiate Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(
        class_weights=class_weights,
        real_upsample_factor=real_upsample_factor,
        **trainer_kwargs
    )

    # 10.5 Apply EMA Callback if enabled
    if getattr(args, "ema_decay", 0.0) > 0.0:
        trainer.add_callback(EMACallback(decay=args.ema_decay))
        print(f"EMA (Exponential Moving Average) enabled with decay rate: {args.ema_decay}")

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
