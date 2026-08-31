import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Default configuration fallback
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "jhu-clsp/mmBERT-base",
        "tokenizer_name": "jhu-clsp/mmBERT-base",
        "max_sequence_length": 8192,
        "latex_placeholder": "[LATEX]",
        "mask_latex_during_inference": True,
    },
    "augmentation": {
        "enabled": True,
        "typo_rate": 0.02,
        "space_noise_rate": 0.15,
        "latex_mask_prob": 0.50,
        "option_drop_prob": 0.05,
        "casing_noise_prob": 0.10,
        "synonym_swap_prob": 0.10,
        "formatting_noise_prob": 0.10,
        "randomize_blanks": True,
        "inline_options": {
            "probability": 0.25,
            "min_spaces": 5,
            "max_spaces": 30,
            "min_tabs": 1,
            "max_tabs": 3,
        },
    },
    "data_preparation": {
        "input_dir": "output",
        "output_dir": "output/dataset",
        "train_ratio": 0.80,
        "val_ratio": 0.10,
        "test_ratio": 0.10,
        "seed": 42,
        "exam_level": True,
        "sliding_windows": {
            "max_lengths": [512, 768, 1024, 2048],
            "strides": [128, 192, 256, 512],
        },
    },
    "training": {
        "repo_id": "daominhwysi/synthetic-seq-labelling-vi-exam-v2",
        "output_dir": "./results",
        "epochs": 3,
        "batch_size": 8,
        "eval_batch_size": 8,
        "learning_rate": 5.0e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.10,
        "warmup_steps": 0,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "use_bf16": True,
        "no_fp16": False,
        "seed": 42,
        "save_total_limit": 2,
        "use_class_weights": False,
        "real_upsample_factor": 1.0,
        "lora": {
            "enabled": True,
            "r": 16,
            "alpha": 32,
            "dropout": 0.10,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "Wqkv", "out_proj"],
        },
        "logging": {
            "report_to": "none",
            "wandb_project": "vietnamese-exam-seq-labelling",
            "logs_per_epoch": 10,
        },
    },
    "providers": {
        "codex": {
            "base_url": "",
            "api_key_env": "OPENAI_API_KEY",
        },
        "xah": {
            "base_url": "https://api.xah.io/v1",
            "api_key_env": "XAH_API_KEY",
        },
        "vilao": {
            "base_url": "https://api.vilao.ai/v1",
            "api_key_env": "LLM_API_KEY",
        },
        "nvidia": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "commandcode": {
            "base_url": "http://127.0.0.1:3050/v1",
            "api_key_env": "CMD_API_KEY",
        },
    },
    "generation": {
        "model": "gpt-5.6-luna",
        "provider": "codex",
        "thinking": "low",
        "concurrency": 2,
        "num_exams": 50,
        "output_dir": "output/exams",
    },
    "inference": {
        "model_dir": "./results",
        "window_size": 2048,
        "stride": 512,
        "batch_size": 4,
        "device": "cuda",
        "export_onnx": True,
    },
    "webapp": {
        "host": "127.0.0.1",
        "viewer_port": 8000,
        "inference_port": 8001,
        "reload": True,
    },
    "huggingface": {
        "dataset_repo_id": "daominhwysi/synthetic-seq-labelling-vi-exam-v2",
        "model_repo_id": "daominhwysi/mmbert-base-vi-exam-seq-labeling",
        "private": False,
    },
}

def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merges dictionary update into base."""
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Loads project YAML configuration file with fallbacks.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return DEFAULT_CONFIG

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        return deep_merge(DEFAULT_CONFIG, user_config)
    except ImportError:
        # Simple YAML key: value line fallback if PyYAML is not installed
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"Warning: Failed to parse '{path}': {e}. Using default config.")
        return DEFAULT_CONFIG
