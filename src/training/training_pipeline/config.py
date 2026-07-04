import os
import argparse
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
        help="Sampling weight multiplier for real exam samples relative to synthetic ones."
    )
    return parser.parse_args()

def setup_device(args) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device
