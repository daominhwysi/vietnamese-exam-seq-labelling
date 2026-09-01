#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add project root directory to sys.path so sub-processes spawned by torchrun find 'src'
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import time
import datetime
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

try:
    import seqeval
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-RoBERTa for Sequence Labeling using LoRA and AMP")
    parser.add_argument(
        "--repo_id",
        "--repo-id",
        type=str,
        default="daominhwysi/synthetic-seq-labelling-vi-exam-v2",
        help="Hugging Face Dataset repository ID"
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        type=str,
        default=None,
        help="Local directory containing offline dataset splits (train.jsonl, val.jsonl, label_mapping.json)"
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="Hugging Face base model name"
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
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
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--eval_batch_size",
        "--eval-batch-size",
        type=int,
        default=8,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        "--eval-accumulation-steps",
        type=int,
        default=10,
        help="Number of evaluation steps to accumulate outputs before moving to CPU (default: 10). Prevents GPU OOM during validation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for trainable parameters (LoRA + classification head)"
    )
    parser.add_argument(
        "--lora_r",
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )
    parser.add_argument(
        "--use_bf16",
        "--use-bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (requires compatible GPU like A100+)"
    )
    parser.add_argument(
        "--no_fp16",
        "--no-fp16",
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
        "--hub_model_id",
        "--hub-model-id",
        type=str,
        default=None,
        help="Target model repository ID on Hugging Face Hub (e.g. 'username/model-name')"
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
        "--enhanced-head",
        action="store_true",
        default=True,
        help="Enable Enhanced Token Classification Head (Layer Pooling + Dense MLP + MSD + Focal Loss, default: True)"
    )
    parser.add_argument(
        "--no-enhanced-head",
        action="store_false",
        dest="enhanced_head",
        help="Disable Enhanced Head and use standard default linear head"
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Gamma focusing parameter for Focal Loss (default: 1.5)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for loss computation (default: 0.0)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights for cross-entropy loss penalty"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder (e.g. './results_enhanced_v3/checkpoint-3564' or 'auto' or 'True') to resume training from."
    )
    parser.add_argument(
        "--real-upsample-factor",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for real exam samples relative to synthetic ones. "
             "e.g. 5.0 means real samples are drawn 5x more often per epoch. Default: 1.0 (no upsampling)."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Integration to report logs to ('wandb', 'tensorboard', 'none')"
    )
    parser.add_argument(
        "--wandb_project",
        "--wandb-project",
        type=str,
        default="vietnamese-exam-seq-labelling",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--logs_per_epoch",
        "--logs-per-epoch",
        type=int,
        default=10,
        help="Number of log outputs per epoch (default: 10). Dynamically calculates logging_steps."
    )
    parser.add_argument(
        "--logging_steps",
        "--logging-steps",
        type=int,
        default=None,
        help="Explicit number of update steps between logging metrics (overrides logs_per_epoch if specified)"
    )
    parser.add_argument(
        "--online_augmentation",
        "--online-augmentation",
        action="store_true",
        default=False,
        help="Enable dynamic on-the-fly online data augmentation during training (default: False)"
    )
    parser.add_argument(
        "--train_ratio",
        "--train-ratio",
        type=float,
        default=0.85,
        help="Ratio of training set for online splitting (default: 0.85)"
    )
    parser.add_argument(
        "--val_ratio",
        "--val-ratio",
        type=float,
        default=0.10,
        help="Ratio of validation set for online splitting (default: 0.10)"
    )
    parser.add_argument(
        "--test_ratio",
        "--test-ratio",
        type=float,
        default=0.05,
        help="Ratio of test set for online splitting (default: 0.05)"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker subprocesses for data loading & online augmentation (default: 4)"
    )
    parser.add_argument(
        "--raw_data_dir",
        "--raw-data-dir",
        type=str,
        default="output",
        help="Directory containing raw question JSONs or XMLs for online augmentation (default: 'output')"
    )
    return parser.parse_args()

class OnlineAugmentedDataset(torch.utils.data.Dataset):
    """
    Dynamic Online PyTorch Dataset that reconstructs, augments, tokenizes,
    and aligns character spans on-the-fly inside the DataLoader multi-worker processes.
    """
    def __init__(
        self,
        raw_items: list,
        tokenizer: Any,
        tag_to_id: Dict[str, int],
        is_train: bool = True,
        max_length: int = 1024
    ):
        self.raw_items = raw_items
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.is_train = is_train
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_items)

    def __getitem__(self, idx):
        import random
        from src.generation.reconstructor import reconstruct_question, reconstruct_exam, ReconstructorConfig
        from src.data.prepare import align_tokens_to_spans, mask_latex_in_real_data
        
        item = self.raw_items[idx]
        
        if self.is_train:
            aug_config = ReconstructorConfig(
                question_prefix_template=random.choice([
                    "Câu {num}: ", "Câu {num}. ", "Câu {num}:", "Câu {num} - ", "{num}. ", "{num}) ", "Question {num}: "
                ]),
                option_prefix_style=random.choice([
                    "capital_dot", "lowercase_paren", "capital_paren", "lowercase_dot", "bold_capital_dot", "bold_lowercase_paren"
                ]),
                separator_stem_options=random.choice(["\n", " ", "\n\n"]),
                separator_options=random.choice(["\n", "    ", "\t\t", "   "]),
                option_drop_prob=0.10,
                space_noise_rate=0.15,
                formatting_noise_prob=0.15,
                casing_noise_prob=0.10,
                typo_rate=0.02,
                latex_mask_prob=0.50,
                enable_permutations=True,
                inline_option_prob=0.35,
                min_inline_spaces=1,
                max_inline_spaces=30,
                grid_2x2_prob=0.15,
                same_line_stem_options_prob=0.20,
                flatten_newlines_prob=0.15,
                collapse_whitespace_prob=0.50,
                randomize_q_num=True
            )
        else:
            aug_config = ReconstructorConfig(randomize_q_num=False)

        # Handle real OCR exams vs synthetic exams/questions
        if item.get("is_real", False) and "raw_text" in item and "spans" in item:
            raw_text = item["raw_text"]
            spans = item["spans"]
            if self.is_train and aug_config.latex_mask_prob > 0.0:
                raw_text, spans = mask_latex_in_real_data(
                    raw_text, spans, aug_config.latex_placeholder, aug_config.latex_mask_prob, random
                )
        elif "sections" in item:
            rec = reconstruct_exam(item, aug_config)
            raw_text = rec["raw_text"]
            spans = rec["spans"]
        else:
            rec = reconstruct_question(item, aug_config)
            raw_text = rec["raw_text"]
            spans = rec["spans"]

        tokenized = self.tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )

        labels = align_tokens_to_spans(
            tokenized["offset_mapping"],
            spans,
            self.tag_to_id,
            raw_text=raw_text
        )

        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def main():
    args = parse_args()
    run_train(args)

def run_train(args):
    print("=" * 60)
    print("Starting training with the following arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key:<30}: {value}")
    print("=" * 60)

    # Suppress all third-party and download tqdm progress bars to avoid terminal clutter
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars as hf_hub_disable_progress_bars
        hf_hub_disable_progress_bars()
    except Exception:
        pass

    try:
        from datasets.utils.logging import disable_progress_bar as datasets_disable_progress_bar
        datasets_disable_progress_bar()
    except Exception:
        pass

    try:
        from transformers.utils.logging import disable_progress_bar as transformers_disable_progress_bar
        transformers_disable_progress_bar()
    except Exception:
        pass

    # Set up Weights & Biases project environment variable if configured
    if getattr(args, "report_to", "none") == "wandb":
        os.environ["WANDB_PROJECT"] = getattr(args, "wandb_project", "vietnamese-exam-seq-labelling")

    # ── PEFT MONKEYPATCH FOR EMBEDDINGS ──────────────────────────────────────
    # Some PEFT versions define AuxiliaryTrainingWrapper.forward(self, x, ...)
    # which crashes with a TypeError when the embedding layer is called with
    # keyword-only arguments like self.embeddings(input_ids=input_ids).
    try:
        import peft.utils.other
        original_forward = peft.utils.other.AuxiliaryTrainingWrapper.forward

        def patched_forward(self, x=None, *args, **kwargs):
            if x is None:
                for possible_key in ["input_ids", "inputs_embeds", "input", "hidden_states"]:
                    if possible_key in kwargs:
                        x = kwargs.pop(possible_key)
                        break
            if x is None and len(args) > 0:
                x = args[0]
                args = args[1:]
            return original_forward(self, x, *args, **kwargs)

        peft.utils.other.AuxiliaryTrainingWrapper.forward = patched_forward
        print("PEFT AuxiliaryTrainingWrapper monkeypatch applied successfully.")
    except Exception as e:
        print(f"Warning: Could not patch PEFT AuxiliaryTrainingWrapper: {e}")

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
    train_dataset_obj = None
    eval_dataset_obj = None
    
    if getattr(args, "online_augmentation", False):
        print(f"Online Dynamic Augmentation enabled! Loading raw question/exam files from '{args.raw_data_dir}'...")
        from src.data.prepare import get_tag_mappings, parse_xml_annotations, infer_metadata_from_path
        tag_to_id, id_to_tag = get_tag_mappings()
        
        raw_path = Path(args.raw_data_dir)
        raw_items = []
        
        # 1. Fast path: load directly from consolidated raw_exams.jsonl if available
        jsonl_candidates = []
        if raw_path.is_file() and raw_path.name.endswith(".jsonl"):
            jsonl_candidates.append(raw_path)
        elif (raw_path / "raw_exams.jsonl").exists():
            jsonl_candidates.append(raw_path / "raw_exams.jsonl")
        elif (raw_path / "dataset" / "raw_exams.jsonl").exists():
            jsonl_candidates.append(raw_path / "dataset" / "raw_exams.jsonl")

        if jsonl_candidates:
            target_jsonl = jsonl_candidates[0]
            print(f"Loading consolidated raw exams from '{target_jsonl}'...")
            with open(target_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            raw_items.append(json.loads(line))
                        except Exception:
                            pass

        if not raw_items:
            # Walk directory following symlinks (e.g. output/real_annotated -> sequence_labelling_annotated)
            for dirpath, _, filenames in os.walk(raw_path, followlinks=True):
                dp = Path(dirpath)
                for fn in filenames:
                    fp = dp / fn
                    
                    # 1. Real annotated XML files (e.g. merged.xml)
                    if fn == "merged.xml" or fn.endswith("_annotated.xml"):
                        audit_file = dp / "audit_report.json"
                        if audit_file.exists():
                            try:
                                audit = json.loads(audit_file.read_text(encoding="utf-8"))
                                if audit.get("decision", "").strip().upper() != "PASS" or audit.get("is_malfunctioned", False):
                                    continue
                            except Exception:
                                pass
                        try:
                            content = fp.read_text(encoding="utf-8")
                            raw_text, spans = parse_xml_annotations(content)
                            if spans:
                                meta = infer_metadata_from_path(fp, raw_path)
                                raw_items.append({
                                    "exam_id": meta.get("exam_id", fp.stem),
                                    "is_real": True,
                                    "raw_text": raw_text,
                                    "spans": spans,
                                    "subject": meta.get("subject", "general"),
                                    "grade": meta.get("grade", 12)
                                })
                        except Exception:
                            pass
                    
                    # 2. JSON files (synthetic questions, compiled exams, real JSONs)
                    elif fn.endswith(".json") and not fn.startswith("chunk_") and fn not in ["label_mapping.json", "audit_report.json", "train_stats.json", "val_stats.json", "test_stats.json"]:
                        if fn.startswith("question_") or fn.startswith("exam_") or fn.startswith("real_") or fn == "merged.json":
                            try:
                                with open(fp, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                if isinstance(data, list):
                                    raw_items.extend(data)
                                elif isinstance(data, dict):
                                    if data.get("is_real", False) and "raw_text" in data and "spans" in data:
                                        raw_items.append(data)
                                    elif "sections" in data:
                                        raw_items.append(data)
                                    else:
                                        raw_items.append(data)
                            except Exception:
                                pass
                
        if not raw_items and args.repo_id:
            print(f"No local raw items found. Attempting to download 'raw_exams.jsonl' from Hugging Face dataset '{args.repo_id}'...")
            try:
                from huggingface_hub import hf_hub_download
                downloaded_file = hf_hub_download(
                    repo_id=args.repo_id,
                    filename="raw_exams.jsonl",
                    repo_type="dataset",
                    token=hf_token
                )
                print(f"Downloaded '{downloaded_file}'. Loading raw exams...")
                with open(downloaded_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                raw_items.append(json.loads(line))
                            except Exception:
                                pass
            except Exception as e:
                print(f"Notice: Could not download 'raw_exams.jsonl' from Hub ({e}).")

        if not raw_items:
            print(f"Warning: No raw questions found in '{args.raw_data_dir}' or Hub. Falling back to offline dataset from Hub.")
        else:
            import random as rand_mod
            print(f"Successfully loaded {len(raw_items)} raw items for online dynamic augmentation.")
            rng_split = rand_mod.Random(args.seed)
            rng_split.shuffle(raw_items)
            val_size = max(5, int(len(raw_items) * getattr(args, "val_ratio", 0.10)))
            test_size = max(5, int(len(raw_items) * getattr(args, "test_ratio", 0.05)))
            if len(raw_items) <= val_size + test_size:
                val_size = max(1, len(raw_items) // 10)
                test_size = max(1, len(raw_items) // 20)
            train_items = raw_items[val_size + test_size:]
            val_items = raw_items[:val_size]
            test_items = raw_items[val_size:val_size + test_size]
            
            train_dataset_obj = OnlineAugmentedDataset(train_items, None, tag_to_id, is_train=True)
            eval_dataset_obj = OnlineAugmentedDataset(val_items, None, tag_to_id, is_train=False)
            test_dataset_obj = OnlineAugmentedDataset(test_items, None, tag_to_id, is_train=False)
            
            # Dataset wrapper supporting train, validation, and test splits
            dataset = {"train": train_dataset_obj, "validation": eval_dataset_obj, "test": test_dataset_obj}

    if train_dataset_obj is None:
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
                    if isinstance(sample, dict) and "labels" in sample:
                        unique_labels.update(sample["labels"])
            # Remove ignored index
            unique_labels.discard(-100)
            # Sort labels to be deterministic
            sorted_labels = sorted(list(unique_labels))

            # Build standard mappings (assuming standard schema tags)
            print(f"Found unique label IDs in dataset: {sorted_labels}")
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

    if train_dataset_obj is not None:
        train_dataset_obj.tokenizer = tokenizer
        eval_dataset_obj.tokenizer = tokenizer
        if test_dataset_obj is not None:
            test_dataset_obj.tokenizer = tokenizer

    # 5. Initialize Model
    use_enhanced = getattr(args, "enhanced_head", True)
    if use_enhanced:
        print(f"Loading Model with Enhanced Head: '{args.model_name}' (Layer Pooling + Dense MLP + MSD + Focal Loss)...")
        from src.model.head import EnhancedTokenClassifierModel
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            id2label={i: id_to_tag[i] for i in id_to_tag},
            label2id=tag_to_id,
            token=hf_token,
        )
        config.output_hidden_states = True
        base_backbone = AutoModel.from_pretrained(
            args.model_name,
            config=config,
            token=hf_token,
            torch_dtype=load_dtype
        )
        model = EnhancedTokenClassifierModel(
            config=config,
            base_model=base_backbone,
            num_layers_to_fuse=4,
            focal_gamma=getattr(args, "focal_gamma", 2.0),
            label_smoothing=getattr(args, "label_smoothing", 0.05)
        )
    else:
        print(f"Loading Model: '{args.model_name}' with standard Linear head and dtype: {load_dtype}...")
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

        if use_enhanced:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
            model.base_model = get_peft_model(model.base_model, peft_config)
            model.base_model.print_trainable_parameters()
        else:
            modules_to_save = ["classifier"]
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        print(f"{'Enhanced Head' if use_enhanced else 'Standard'} Full Fine-Tuning enabled...")

    # 7. Metrics & Preprocessing Definition
    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocesses model output logits on GPU per batch to only retain argmax predictions.
        Reduces evaluation memory footprint by >13x and prevents CUDA OOM during validation loops.
        """
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if hasattr(logits, "argmax"):
            return logits.argmax(dim=-1)
        return logits

    def compute_metrics(p):
        predictions, labels = p
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        # Convert predictions to class indices if raw 3D logits were received
        if hasattr(predictions, "ndim") and predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        elif isinstance(predictions, torch.Tensor) and predictions.ndim == 3:
            predictions = predictions.argmax(dim=-1).cpu().numpy()
        elif isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

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
                "accuracy": np.mean([p_v == l_v for p_seq, l_seq in zip(true_predictions, true_labels) for p_v, l_v in zip(p_seq, l_seq)]) if any(len(s) > 0 for s in true_labels) else 0.0
            }
        except ImportError:
            # Fallback to token-level evaluation if seqeval is not installed
            from sklearn.metrics import f1_score, accuracy_score
            flat_preds = [p_v for p_seq in true_predictions for p_v in p_seq]
            flat_labels = [l_v for l_seq in true_labels for l_v in l_seq]
            if len(flat_labels) == 0:
                return {"accuracy": 0.0, "f1_macro": 0.0}
            return {
                "accuracy": accuracy_score(flat_labels, flat_preds),
                "f1_macro": f1_score(flat_labels, flat_preds, average="macro")
            }

    # 8. Data Collator
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # 9. Training Arguments (robust across all transformers versions)
    import inspect
    from transformers import TrainingArguments, Trainer

    ta_sig = inspect.signature(TrainingArguments.__init__)

    # Dynamically compute logging_steps from logs_per_epoch unless explicitly overridden
    effective_batch = args.batch_size * max(1, getattr(args, "gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, len(dataset["train"]) // effective_batch)

    if getattr(args, "logging_steps", None) is not None and args.logging_steps > 0:
        dynamic_logging_steps = args.logging_steps
    else:
        logs_per_epoch = max(1, getattr(args, "logs_per_epoch", 10))
        dynamic_logging_steps = max(1, steps_per_epoch // logs_per_epoch)

    print(
        f"[Logger Config] Steps per epoch: {steps_per_epoch}, "
        f"Logs per epoch: {getattr(args, 'logs_per_epoch', 10)}, "
        f"Effective logging_steps: {dynamic_logging_steps}"
    )

    eval_accumulation_steps = getattr(args, "eval_accumulation_steps", 10)

    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "eval_accumulation_steps": eval_accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": dynamic_logging_steps,
        "disable_tqdm": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1" if HAS_SEQEVAL else "f1_macro",
        "greater_is_better": True,
        "fp16": fp16_enabled,
        "bf16": bf16_enabled,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": getattr(args, "hub_model_id", None),
        "hub_token": hf_token,
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "seed": getattr(args, "seed", 42),
        "lr_scheduler_type": getattr(args, "lr_scheduler_type", "linear"),
    }

    # Handle evaluation strategy naming differences
    if "eval_strategy" in ta_sig.parameters:
        training_args_dict["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_sig.parameters:
        training_args_dict["evaluation_strategy"] = "epoch"

    # Handle warmup_ratio vs warmup_steps across transformers versions
    warmup_ratio_val = getattr(args, "warmup_ratio", 0.0)
    warmup_steps_val = getattr(args, "warmup_steps", 0)

    if "warmup_ratio" in ta_sig.parameters and warmup_ratio_val > 0.0:
        training_args_dict["warmup_ratio"] = warmup_ratio_val
    elif warmup_steps_val > 0:
        training_args_dict["warmup_steps"] = warmup_steps_val
    elif warmup_ratio_val > 0.0:
        # Convert warmup_ratio to warmup_steps for versions without warmup_ratio
        total_steps = steps_per_epoch * args.epochs
        training_args_dict["warmup_steps"] = max(1, int(total_steps * warmup_ratio_val))

    # Filter only arguments supported by the installed transformers version
    valid_ta_kwargs = {k: v for k, v in training_args_dict.items() if k in ta_sig.parameters}
    training_args = TrainingArguments(**valid_ta_kwargs)

    # 9.5 Calculate class weights if enabled (Tying B- and I- tags together per entity)
    class_weights = None
    if not getattr(args, "no_class_weights", False):
        print("Calculating entity-tied class weights from training dataset...")
        from collections import Counter
        label_counts = Counter()
        if train_dataset_obj is not None:
            # Online mode: estimate entity frequencies from raw spans in train_items
            for item in getattr(train_dataset_obj, "raw_items", []):
                for span in item.get("spans", []):
                    tag_name = span.get("label", "")
                    approx_tokens = max(1, len(span.get("text", "").split()))
                    b_id = tag_to_id.get(f"B-{tag_name}")
                    i_id = tag_to_id.get(f"I-{tag_name}")
                    if b_id is not None:
                        label_counts[b_id] += 1
                    if i_id is not None and approx_tokens > 1:
                        label_counts[i_id] += (approx_tokens - 1)
        else:
            for sample in dataset["train"]:
                label_counts.update([l for l in sample["labels"] if l != -100])
        
        # 1. Aggregate token counts by base entity name (e.g., 'stimulus', 'stem', 'question_label')
        # This prevents 1-token boundary B-tags from receiving an artificially high weight compared to multi-token I-tags.
        entity_counts = Counter()
        for label_id, count in label_counts.items():
            tag_str = id_to_tag.get(label_id, "O")
            entity_name = tag_str[2:] if (tag_str.startswith("B-") or tag_str.startswith("I-")) else tag_str
            entity_counts[entity_name] += count
            
        total_count = sum(label_counts.values())
        num_entities = len(entity_counts)
        
        if total_count > 0:
            # 2. Compute base entity weights using smoothed inverse frequency
            entity_weights = {}
            for entity_name, count in entity_counts.items():
                if count > 0:
                    entity_weights[entity_name] = total_count / (num_entities * np.sqrt(count))
                else:
                    entity_weights[entity_name] = 1.0

            # 3. Assign the tied entity weight to both B- and I- tags
            weights = np.ones(num_labels, dtype=np.float32)
            for label_id in range(num_labels):
                tag_str = id_to_tag.get(label_id, "O")
                entity_name = tag_str[2:] if (tag_str.startswith("B-") or tag_str.startswith("I-")) else tag_str
                weights[label_id] = entity_weights.get(entity_name, 1.0)

            # Normalize so mean weight across all classes is 1.0
            weights = weights / weights.mean()
            target_device = training_args.device if "training_args" in locals() else device
            class_weights = torch.tensor(weights, dtype=torch.float32).to(target_device)

            # Update model's loss function with tied class weights if available
            if hasattr(model, "loss_fct") and hasattr(model.loss_fct, "weight"):
                model.loss_fct.weight = class_weights

            print("Computed Entity-Tied Class Weights:")
            for label_name, label_id in tag_to_id.items():
                print(f"  {label_name:<20} (ID: {label_id:>2}): Weight = {weights[label_id]:.4f}")
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
                      "Falling back to uniform sampling. Re-run prepare-offline-dataset to propagate is_real metadata.")
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

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """Override evaluate to clear CUDA cache before and after validation."""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            output = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return output

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)

            # Save past state if required (e.g. for evaluation metrics)
            if getattr(self.args, "past_index", -1) >= 0:
                self._past = outputs[self.args.past_index]

            # 1. Prefer model's built-in loss if computed (e.g. EnhancedTokenClassifierModel's FocalLoss)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
                loss = outputs["loss"]
            elif labels is not None and self.class_weights is not None:
                logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    # 10. Instantiate Trainer (support both processing_class and tokenizer dynamically)
    import inspect
    
    if train_dataset_obj is not None:
        train_dataset_final = train_dataset_obj
        eval_dataset_final = eval_dataset_obj
    else:
        train_dataset_final = dataset["train"]
        eval_dataset_final = dataset["validation"]
        
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset_final,
        "eval_dataset": eval_dataset_final,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "preprocess_logits_for_metrics" in trainer_signature.parameters:
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(class_weights=class_weights, real_upsample_factor=real_upsample_factor, **trainer_kwargs)

    # 10.8 Replace default progress/printer callbacks with clean discrete iteration logger
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    trainer.remove_callback(ProgressCallback)
    trainer.remove_callback(PrinterCallback)

    from transformers import TrainerCallback

    class IterLoggerCallback(TrainerCallback):
        """
        Discrete iteration-based logger replacing tqdm to eliminate browser lag
        from frequent ANSI carriage returns (\r) and terminal repaints.
        """
        def __init__(self, logs_per_epoch: int = 10):
            self.logs_per_epoch = max(1, logs_per_epoch)
            self.train_start_time = None
            self.last_log_time = None
            self.last_log_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self.train_start_time = time.time()
                self.last_log_time = self.train_start_time
                self.last_log_step = 0
                print("=" * 80)
                print(
                    f"Starting Training: Total Steps = {state.max_steps} | "
                    f"Epochs = {args.num_train_epochs} | "
                    f"Batch Size = {args.per_device_train_batch_size} | "
                    f"Logging Steps = {args.logging_steps}"
                )
                print("=" * 80, flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or logs is None:
                return

            now = time.time()
            is_eval = any(k.startswith("eval_") for k in logs)
            is_final_train = ("train_loss" in logs or "train_runtime" in logs) and not is_eval

            if is_eval:
                epoch_val = logs.get("epoch", state.epoch if state.epoch is not None else 0.0)
                eval_metrics = []
                if "eval_loss" in logs:
                    eval_metrics.append(f"loss: {logs['eval_loss']:.4f}")
                for metric in ["eval_f1", "eval_accuracy", "eval_precision", "eval_recall"]:
                    if metric in logs:
                        eval_metrics.append(f"{metric.replace('eval_', '')}: {logs[metric]:.4f}")
                if "eval_runtime" in logs:
                    eval_metrics.append(f"time: {logs['eval_runtime']:.2f}s")
                
                if eval_metrics:
                    print(f">>> [Evaluation @ Step {state.global_step:>5} | Epoch {epoch_val:5.2f}] " + " | ".join(eval_metrics), flush=True)
                self.last_log_time = time.time()
            elif is_final_train:
                train_loss_val = logs.get("train_loss", "N/A")
                loss_str = f"{train_loss_val:.4f}" if isinstance(train_loss_val, (int, float)) else str(train_loss_val)
                runtime = logs.get("train_runtime", 0.0)
                steps_per_sec = logs.get("train_steps_per_second", 0.0)
                print(f">>> [Training Summary] Train Loss: {loss_str} | Runtime: {runtime:.2f}s | Speed: {steps_per_sec:.2f} steps/s", flush=True)
            elif "loss" in logs or "learning_rate" in logs:
                step = state.global_step
                max_steps = max(1, state.max_steps)
                pct = (step / max_steps) * 100.0
                epoch_val = logs.get("epoch", state.epoch if state.epoch is not None else 0.0)

                elapsed_sec = int(now - self.train_start_time) if self.train_start_time else 0
                step_delta = step - self.last_log_step
                time_delta = now - self.last_log_time if self.last_log_time else 0.0

                it_speed = (step_delta / time_delta) if time_delta > 0 else ((step / elapsed_sec) if elapsed_sec > 0 else 0.0)
                overall_speed = (step / elapsed_sec) if elapsed_sec > 0 else 0.0
                remaining_steps = max(0, max_steps - step)
                eta_sec = int(remaining_steps / overall_speed) if overall_speed > 0 else 0

                elapsed_fmt = str(datetime.timedelta(seconds=elapsed_sec))
                eta_fmt = str(datetime.timedelta(seconds=eta_sec))

                loss_val = logs.get("loss", "N/A")
                loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
                lr_val = logs.get("learning_rate", None)
                lr_str = f"{lr_val:.2e}" if isinstance(lr_val, (int, float)) else "N/A"

                self.last_log_step = step
                self.last_log_time = now

                print(
                    f"[Step {step:>5}/{max_steps} | Epoch {epoch_val:5.2f}/{args.num_train_epochs:.2f} ({pct:5.1f}%)] "
                    f"Loss: {loss_str} | LR: {lr_str} | Speed: {it_speed:5.2f} it/s | "
                    f"Elapsed: {elapsed_fmt} | ETA: {eta_fmt}",
                    flush=True
                )

        def on_evaluate(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_log_time = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state.is_world_process_zero:
                current_epoch = int(round(state.epoch)) if state.epoch is not None else 1
                total_epochs = int(args.num_train_epochs)
                print(f"--- Epoch {current_epoch}/{total_epochs} completed ---", flush=True)
            self.last_log_time = time.time()

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero and self.train_start_time:
                total_sec = int(time.time() - self.train_start_time)
                total_fmt = str(datetime.timedelta(seconds=total_sec))
                print("=" * 80)
                print(f"Training Completed: Total Steps = {state.global_step} | Total Time = {total_fmt}")
                print("=" * 80, flush=True)

    trainer.add_callback(IterLoggerCallback(logs_per_epoch=getattr(args, "logs_per_epoch", 10)))

    # 11. Run Training
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt in ["auto", "True", "true"]:
        # Check if output_dir directly holds checkpoint state (optimizer.pt or trainer_state.json)
        if os.path.exists(os.path.join(args.output_dir, "trainer_state.json")) or os.path.exists(os.path.join(args.output_dir, "optimizer.pt")):
            resume_ckpt = args.output_dir
        else:
            try:
                from transformers.trainer_utils import get_last_checkpoint
                last_ckpt = get_last_checkpoint(args.output_dir)
                resume_ckpt = last_ckpt if last_ckpt is not None else None
            except Exception:
                resume_ckpt = None
    elif resume_ckpt and not os.path.exists(resume_ckpt):
        print(f"Notice: Specified resume checkpoint '{resume_ckpt}' not found. Starting from scratch.")
        resume_ckpt = None

    print(f"Starting training (resume_from_checkpoint={resume_ckpt})...")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 12. Run final test split evaluation
    print("Evaluating on test split...")
    test_eval_set = test_dataset_obj if (test_dataset_obj is not None) else dataset.get("test", eval_dataset_final)
    test_results = trainer.evaluate(eval_dataset=test_eval_set)
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS:")
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.4f}")
        else:
            print(f"  {k:<28}: {v}")
    print("=" * 60)

    # Save the final model & label mapping
    print(f"Saving final model to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    try:
        mapping_file = os.path.join(args.output_dir, "label_mapping.json")
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2, ensure_ascii=False)
        print(f"Saved label mapping to '{mapping_file}'.")
    except Exception as e:
        print(f"Notice saving label mapping: {e}")

    if args.push_to_hub:
        target_repo = args.hub_model_id or getattr(trainer.args, "hub_model_id", None) or args.output_dir
        print(f"Pushing model adapters to HF Hub ({target_repo})...")
        trainer.push_to_hub(commit_message="Add trained sequence labeler model")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=mapping_file,
                path_in_repo="label_mapping.json",
                repo_id=target_repo,
            )
            head_cfg = os.path.join(args.output_dir, "enhanced_head_config.json")
            if os.path.exists(head_cfg):
                api.upload_file(
                    path_or_fileobj=head_cfg,
                    path_in_repo="enhanced_head_config.json",
                    repo_id=target_repo,
                )
            print(f"Successfully uploaded label mapping and head config to HF Hub.")
        except Exception as e:
            print(f"Notice uploading auxiliary files to hub: {e}")

if __name__ == "__main__":
    main()
