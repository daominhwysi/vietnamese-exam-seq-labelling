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
        default="daominhwysi/synthetic-seq-labelling-vi-exam-v2",
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
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and perform full fine-tuning"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training reproducibility"
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (linear, cosine, cosine_with_restarts, constant, etc.)"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Decay rate for Exponential Moving Average (EMA). Set > 0.0 (e.g. 0.999) to enable."
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights for cross-entropy loss penalty"
    )
    parser.add_argument(
        "--real-upsample-factor",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for real exam samples relative to synthetic ones. "
             "e.g. 5.0 means real samples are drawn 5x more often per epoch. Default: 1.0 (no upsampling)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    run_train(args)

def run_train(args):

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
    # bfloat16 requires hardware support (compute capability >= 8.0, i.e., Ampere or newer) to run fast.
    # On older architectures like Turing (e.g., T4 with compute capability 7.5), BF16 runs extremely slowly via emulation.
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

    # Add the exact same special tokens in the exact same order as during dataset preparation
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

    # Resize token embeddings to match tokenizer with added special tokens
    model.resize_token_embeddings(len(tokenizer))

    # 6. Apply LoRA (PEFT) if enabled
    if not getattr(args, "no_lora", False):
        print("Applying Low-Rank Adaptation (LoRA)...")
        # Bypass torchao compatibility check bug on older pre-installed versions in Google Colab
        try:
            import peft.import_utils
            peft.import_utils.is_torchao_available = lambda: False
        except Exception:
            pass

        from peft import LoraConfig, get_peft_model, TaskType

        # Select target modules dynamically based on the model architecture
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
            modules_to_save=["classifier"]  # Ensures classifier head is trained fully (not frozen)
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("LoRA is disabled. Preparing for Full Fine-Tuning...")

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
        report_to="none",  # Change to "wandb" or "tensorboard" if configured
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
    if getattr(args, "use_class_weights", False):
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
                    # Smoothed inverse frequency weighting
                    weights[label_id] = total_count / (num_labels * np.sqrt(count))
            # Normalize so mean weight is 1.0
            weights = weights / weights.mean()
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Computed class weights: {weights.tolist()}")
            # Print mapping with weights for debugging
            for label_name, label_id in tag_to_id.items():
                print(f"  {label_name} (ID: {label_id}): Weight = {weights[label_id]:.4f}")
        else:
            print("Warning: No labels found in training dataset. Skipping class weights.")

    # Define custom Trainer with class weights + real-sample upsampling support
    real_upsample_factor = getattr(args, "real_upsample_factor", 1.0)

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, real_upsample_factor=1.0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.real_upsample_factor = real_upsample_factor

        def get_train_dataloader(self):
            """Override to inject WeightedRandomSampler for real vs synthetic upsampling."""
            if self.real_upsample_factor <= 1.0:
                # No upsampling requested — use the default dataloader
                return super().get_train_dataloader()

            train_dataset = self.train_dataset
            n_real = 0
            n_synth = 0
            sample_weights = []

            # Build per-sample weights from raw dataset BEFORE column removal,
            # since metadata (which holds is_real) is stripped afterwards.
            for sample in train_dataset:
                meta = sample.get("metadata", {})
                # HuggingFace datasets may deserialize nested dicts as plain dicts
                is_real = meta.get("is_real", False) if isinstance(meta, dict) else False
                if is_real:
                    sample_weights.append(self.real_upsample_factor)
                    n_real += 1
                else:
                    sample_weights.append(1.0)
                    n_synth += 1

            print(
                f"[WeightedSampler] Synth samples: {n_synth}, Real samples: {n_real} "
                f"(effective weight: synth=1.0, real={self.real_upsample_factor})"
            )

            if n_real == 0:
                print("[WeightedSampler] Warning: no real samples found in train split (is_real=False for all). "
                      "Falling back to uniform sampling. Re-run prepare-dataset to propagate is_real metadata.")
                return super().get_train_dataloader()

            from torch.utils.data import WeightedRandomSampler, DataLoader

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(self.args.seed),
            )

            # Mirror what Trainer.get_train_dataloader() does internally:
            # strip non-model columns (tokens, tags, metadata) so the collator
            # only sees tensor-compatible fields (input_ids, attention_mask, labels).
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)

            # Save past state if required (e.g. for evaluation metrics)
            if getattr(self.args, "past_index", -1) >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None and self.class_weights is not None:
                logits = outputs.get("logits")
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    # 10. Instantiate Trainer (support both processing_class and tokenizer dynamically)
    import inspect
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

    trainer = WeightedTrainer(class_weights=class_weights, real_upsample_factor=real_upsample_factor, **trainer_kwargs)

    # 10.5 Apply EMA Callback if enabled
    if getattr(args, "ema_decay", 0.0) > 0.0:
        from transformers import TrainerCallback
        class EMACallback(TrainerCallback):
            def __init__(self, decay=0.999):
                self.decay = decay
                self.shadow = {}
                self.backup = {}
                self.ema_active = False

            def on_step_end(self, args, state, control, model=None, **kwargs):
                if model is None:
                    return
                active_model = model.module if hasattr(model, "module") else model
                for name, param in active_model.named_parameters():
                    if param.requires_grad:
                        if name not in self.shadow:
                            self.shadow[name] = param.data.clone()
                        else:
                            self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

            def on_epoch_begin(self, args, state, control, model=None, **kwargs):
                self._restore_regular_weights(model)

            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                self._apply_ema_weights(model)

            def on_train_end(self, args, state, control, model=None, **kwargs):
                self._apply_ema_weights(model)
                print("Final EMA weights permanently applied to the model.")

            def _apply_ema_weights(self, model):
                if model is None or self.ema_active:
                    return
                active_model = model.module if hasattr(model, "module") else model
                for name, param in active_model.named_parameters():
                    if param.requires_grad and name in self.shadow:
                        self.backup[name] = param.data.clone()
                        param.data.copy_(self.shadow[name])
                self.ema_active = True

            def _restore_regular_weights(self, model):
                if model is None or not self.ema_active:
                    return
                active_model = model.module if hasattr(model, "module") else model
                for name, param in active_model.named_parameters():
                    if param.requires_grad and name in self.backup:
                        param.data.copy_(self.backup[name])
                self.backup.clear()
                self.ema_active = False

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
