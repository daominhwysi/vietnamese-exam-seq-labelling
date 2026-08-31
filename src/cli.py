import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import concurrent.futures
from tqdm import tqdm

from src.generation.curriculum import generate_curriculum
from src.generation.reconstructor import (
    reconstruct_question,
    reconstruct_exam,
    ReconstructorConfig,
    OPTION_PREFIX_STYLES
)

def run_reconstructor_on_existing(input_directory: str, dest_directory: Optional[str], config: ReconstructorConfig, max_workers: int = 4):
    in_dir = Path(input_directory)
    if not in_dir.exists():
        print(f"Error: Input directory '{input_directory}' does not exist.")
        sys.exit(1)

    json_files = list(in_dir.glob("question_*.json")) + list(in_dir.glob("**/exam_*.json"))
    if not json_files:
        print(f"No question JSON or exam JSON files found in '{input_directory}'.")
        return
        
    if dest_directory:
        dest_path = Path(dest_directory)
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(json_files)} file(s) in '{input_directory}'. Saving reconstructed files to '{dest_directory}'...")
    else:
        print(f"WARNING: Modifying {len(json_files)} file(s) in '{input_directory}' in-place...")

    def process_file(file_path: Path) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "sections" in data:
                updated_data = reconstruct_exam(data, config)
            else:
                updated_data = reconstruct_question(data, config)

            if not dest_directory:
                target_path = file_path
            else:
                target_path = Path(dest_directory) / file_path.name

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            tqdm.write(f"Error processing {file_path.name}: {e}")
            return False

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in json_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(json_files), desc="Reconstructing"):
            if future.result():
                success_count += 1

    if dest_directory:
        print(f"Completed: Saved {success_count}/{len(json_files)} reconstructed file(s) to '{dest_directory}'.")
    else:
        print(f"Completed: Reconstructed {success_count}/{len(json_files)} file(s) in-place.")

def main():
    parser = argparse.ArgumentParser(description="Mock Exam Question Generator & Text Reconstructor")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Refactored subcommands")

    # 1. curriculum
    p_curr = subparsers.add_parser("curriculum", help="Stage 1: Generate curriculum JSON files")
    p_curr.add_argument("--all", action="store_true", help="Generate curricula for all subjects & grades concurrently")
    p_curr.add_argument("--subject", type=str, help="Subject slug (e.g. 'physics')")
    p_curr.add_argument("--grade", type=int, help="Grade level (e.g. 11)")
    p_curr.add_argument("--model", type=str, default="deepseek-v4-pro", help="LLM model to use")
    p_curr.add_argument("--provider", type=str, choices=["deepseek", "nvidia", "vilao"], default=None, help="LLM provider to use")
    p_curr.add_argument("--thinking", type=str, default="high", choices=["high", "max", "low", "medium", "none"], help="Thinking effort level")
    p_curr.add_argument("-c", "--concurrency", type=int, default=4, help="Number of parallel workers for concurrent generation")

    # 3. reconstruct
    p_rec = subparsers.add_parser("reconstruct", help="Stage 3: Reconstruct raw text and track spans")
    p_rec.add_argument("-i", "--input-dir", type=str, default="output", help="Input directory containing questions")
    p_rec.add_argument("--in-place", action="store_true", help="Overwrite existing files directly")
    p_rec.add_argument("--reconstruct-dest", type=str, default="output/reconstructed", help="Destination folder if not in-place")
    p_rec.add_argument("-c", "--concurrency", type=int, default=4, help="Concurrency for reconstruction")
    p_rec.add_argument("--q-prefix", type=str, help="Override question prefix template")
    p_rec.add_argument("--opt-style", type=str, choices=list(OPTION_PREFIX_STYLES.keys()), help="Override option prefix style")
    p_rec.add_argument("--ord-item-style", type=str, choices=["char", "index"], help="Override ordering item style")
    p_rec.add_argument("--ord-item-template", type=str, help="Override ordering item template")
    
    # Reconstruct augmentations
    p_rec.add_argument("--typo-rate", type=float, default=0.0, help="Spelling mistake typo injection rate")
    p_rec.add_argument("--space-noise-rate", type=float, default=0.0, help="Spacing noise injection rate")
    p_rec.add_argument("--latex-mask-prob", type=float, default=0.0, help="LaTeX masking probability")
    p_rec.add_argument("--latex-placeholder", type=str, default="[LATEX]", help="LaTeX masking placeholder")
    p_rec.add_argument("--enable-permutations", action="store_true", help="Enable random permutations")
    p_rec.add_argument("--option-drop-prob", type=float, default=0.0, help="Option drop probability")
    p_rec.add_argument("--casing-noise-prob", type=float, default=0.0, help="Casing noise probability")
    p_rec.add_argument("--synonym-swap-prob", type=float, default=0.0, help="Synonym swap probability")
    p_rec.add_argument("--formatting-noise-prob", type=float, default=0.0, help="Formatting tag noise probability")
    p_rec.add_argument("--inline-option-prob", type=float, default=0.0, help="Probability of formatting options inline")
    p_rec.add_argument("--min-inline-spaces", type=int, default=5, help="Minimum random spaces to inject between inline options")
    p_rec.add_argument("--max-inline-spaces", type=int, default=30, help="Maximum random spaces to inject between inline options")
    p_rec.add_argument("--min-inline-tabs", type=int, default=1, help="Minimum random tabs to inject between inline options")
    p_rec.add_argument("--max-inline-tabs", type=int, default=3, help="Maximum random tabs to inject between inline options")

    # 4. exam
    p_exam = subparsers.add_parser("exam", help="Stage 4: Generate mock exams as compiled JSON")
    p_exam.add_argument("-n", "--num-exams", type=int, default=300, help="Number of exams to generate")
    p_exam.add_argument("-o", "--output-dir", type=str, default="output/exams", help="Output directory path")
    p_exam.add_argument("--model", type=str, default="deepseek-v4-pro", help="LLM model to use")
    p_exam.add_argument("--provider", type=str, choices=["deepseek", "nvidia", "vilao"], default=None, help="LLM provider to use")
    p_exam.add_argument("--thinking", type=str, default="high", choices=["high", "max", "low", "medium", "none"], help="Thinking effort level")
    p_exam.add_argument("-c", "--concurrency", type=int, default=8, help="Number of concurrent threads per exam")
    p_exam.add_argument("--subject", type=str, help="Filter generation for a specific subject")
    p_exam.add_argument("--grade", type=int, help="Filter generation for a specific grade")

    # 5. prepare
    p_prep = subparsers.add_parser("prepare", help="Stage 5: Prepare tokenized datasets for training")
    p_prep.add_argument("-i", "--input-dir", type=str, default="output", help="Input folder of question files")
    p_prep.add_argument("-o", "--output-dir", type=str, default="output/dataset", help="Output folder for training dataset split")
    p_prep.add_argument("--model", type=str, default="jhu-clsp/mmBERT-base", help="Base model/tokenizer name")
    p_prep.add_argument("--latex-placeholder", type=str, default="[LATEX]", help="Placeholder for LaTeX equations")
    p_prep.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training set")
    p_prep.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation set")
    p_prep.add_argument("--seed", type=int, default=42, help="Seed for splitting")
    p_prep.add_argument("--exam-level", action="store_true", help="Process at exam level")
    p_prep.add_argument("--max-len", type=str, default="512,768,1024,2048", help="Sequence lengths")
    p_prep.add_argument("--stride", type=str, default="128,192,256,512", help="Strides")
    p_prep.add_argument("--typo-rate", type=float, default=0.02, help="Typo rate")
    p_prep.add_argument("--space-noise-rate", type=float, default=0.15, help="Space noise rate")
    p_prep.add_argument("--latex-mask-prob", type=float, default=0.5, help="LaTeX mask probability")
    p_prep.add_argument("--enable-permutations", action="store_true", help="Enable permutations")
    p_prep.add_argument("--option-drop-prob", type=float, default=0.05, help="Option drop probability")
    p_prep.add_argument("--casing-noise-prob", type=float, default=0.10, help="Casing noise probability")
    p_prep.add_argument("--synonym-swap-prob", type=float, default=0.10, help="Synonym swap probability")
    p_prep.add_argument("--formatting-noise-prob", type=float, default=0.10, help="Formatting tag noise probability")
    p_prep.add_argument("--inline-option-prob", type=float, default=0.0, help="Probability of formatting options inline")
    p_prep.add_argument("--min-inline-spaces", type=int, default=5, help="Minimum random spaces to inject between inline options")
    p_prep.add_argument("--max-inline-spaces", type=int, default=30, help="Maximum random spaces to inject between inline options")
    p_prep.add_argument("--min-inline-tabs", type=int, default=1, help="Minimum random tabs to inject between inline options")
    p_prep.add_argument("--max-inline-tabs", type=int, default=3, help="Maximum random tabs to inject between inline options")
    p_prep.add_argument("--only-passed", action="store_true", default=True, help="Only include documents that passed quality audit (default: True)")
    p_prep.add_argument("--include-all", action="store_true", help="Include all documents regardless of audit status")

    # 6. train
    p_train = subparsers.add_parser("train", help="Stage 6: Train XLM-RoBERTa model with LoRA")
    p_train.add_argument("--repo_id", type=str, default="daominhwysi/synthetic-seq-labelling-vi-exam-v2", help="HF Dataset repository ID")
    p_train.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base", help="HF base model name")
    p_train.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints")
    p_train.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p_train.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    p_train.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    p_train.add_argument("--eval_accumulation_steps", type=int, default=10, help="Number of evaluation steps before moving outputs to CPU (default: 10)")
    p_train.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p_train.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p_train.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p_train.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    p_train.add_argument("--use_bf16", action="store_true", help="Use bfloat16 mixed precision")
    p_train.add_argument("--no_fp16", action="store_true", help="Disable float16 mixed precision")
    p_train.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p_train.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    p_train.add_argument("--no-lora", action="store_true", help="Disable LoRA and perform full fine-tuning")
    p_train.add_argument("--enhanced-head", action="store_true", default=True, help="Enable Enhanced Token Classification Head (default: True)")
    p_train.add_argument("--no-enhanced-head", action="store_false", dest="enhanced_head", help="Disable Enhanced Head")
    p_train.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma focusing parameter for Focal Loss (default: 2.0)")
    p_train.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor for loss computation (default: 0.05)")
    p_train.add_argument("--no-class-weights", action="store_true", help="Disable class weights for cross-entropy loss penalty")
    p_train.add_argument("--real-upsample-factor", type=float, default=1.0, help="Sampling weight multiplier for real exam samples")
    p_train.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of update steps to accumulate before performing a backward pass")
    p_train.add_argument("--seed", type=int, default=42, help="Seed for training reproducibility")
    p_train.add_argument("--lr-scheduler-type", type=str, default="linear", help="Learning rate scheduler type (linear, cosine, constant, etc.)")
    p_train.add_argument("--warmup-ratio", type=float, default=0.0, help="Warmup ratio for scheduler")
    p_train.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for scheduler")
    p_train.add_argument("--report_to", type=str, default="none", help="Log to 'wandb', 'tensorboard', or 'none'")
    p_train.add_argument("--wandb_project", type=str, default="vietnamese-exam-seq-labelling", help="Weights & Biases project name")
    p_train.add_argument("--logs_per_epoch", type=int, default=10, help="Number of log outputs per epoch (default: 10). Dynamically calculates logging_steps.")
    p_train.add_argument("--logging_steps", type=int, default=None, help="Explicit number of update steps between logging metrics (overrides logs_per_epoch if specified)")
    p_train.add_argument("--push_to_hub", action="store_true", help="Push to Hugging Face Hub")
    p_train.add_argument("--hf_token", type=str, help="Hugging Face authentication token")

    # 7. inference
    p_inf = subparsers.add_parser("inference", help="Stage 7: Run inference on sample inputs using trained model")
    p_inf.add_argument("--model_dir", "-m", type=str, default="./results", help="Model adapter/checkpoint directory or HF repo")
    p_inf.add_argument("--base_model_name", type=str, default=None, help="HF base model name")
    p_inf.add_argument("--text", "-t", type=str, default=None, help="Direct text input string to segment")
    p_inf.add_argument("--file", "-f", type=str, default=None, help="Input text file path to segment")
    p_inf.add_argument("--output", "-o", type=str, default=None, help="Output file path (.json, .xml, or .txt)")
    p_inf.add_argument("--max-length", type=int, default=1024, help="Sliding window token length (default: 1024)")
    p_inf.add_argument("--stride", type=int, default=256, help="Sliding window stride (default: 256)")

    # 8. upload (dataset)
    p_upl = subparsers.add_parser("upload", help="Upload dataset splits to Hugging Face Hub")
    p_upl.add_argument("--token", type=str, help="HF Token")
    p_upl.add_argument("--repo-id", type=str, help="HF dataset repository target path")
    p_upl.add_argument(
        "--dataset-dir",
        type=str,
        default="output/dataset",
        help="Local folder with train/val/test JSONL splits and xml/ subfolder (default: 'output/dataset')",
    )

    # 8b. upload-model
    p_upl_m = subparsers.add_parser("upload-model", help="Upload trained model/adapter checkpoint to Hugging Face Hub")
    p_upl_m.add_argument(
        "--model-dir",
        type=str,
        default="./results",
        help="Path to the trained model/adapter directory (default: './results')",
    )
    p_upl_m.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF model repository ID, e.g. 'username/repo-name' (default: 'daominhwysi/vi-exam-seq-labeller')",
    )
    p_upl_m.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (falls back to HF_TOKEN env var)",
    )
    p_upl_m.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public)",
    )
    p_upl_m.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )
    p_upl_m.add_argument(
        "--dataset-repo",
        type=str,
        default="daominhwysi/synthetic-seq-labelling-vi-exam-v2",
        help="HF dataset repo to reference in the model card (default: 'daominhwysi/synthetic-seq-labelling-vi-exam-v2')",
    )

    # 9. visualize
    p_vis = subparsers.add_parser("visualize", help="Generate an HTML interactive visualizer for token span labels")
    p_vis.add_argument("-i", "--input-file", type=str, default="output/dataset/train.jsonl", help="Path to jsonl file to visualize")
    p_vis.add_argument("-o", "--output-html", type=str, default="output/dataset/sample_visualization.html", help="Output path for HTML")
    p_vis.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to embed in HTML")

    args = parser.parse_args()

    # Route commands
    if args.command == "curriculum":
        from src.generation.curriculum import generate_all_curricula
        thinking_val = None if args.thinking == "none" else args.thinking
        if args.all:
            generate_all_curricula(model=args.model, thinking=thinking_val, concurrency=args.concurrency, provider=args.provider)
        else:
            if not args.subject or not args.grade:
                print("Error: Single curriculum generation requires both --subject and --grade parameters, or pass --all.")
                sys.exit(1)
            generate_curriculum(args.subject, args.grade, model=args.model, thinking=thinking_val, provider=args.provider)
            print("Curriculum generation completed successfully.")

    elif args.command == "reconstruct":
        reconstruct_config = ReconstructorConfig(
            question_prefix_template=args.q_prefix,
            option_prefix_style=args.opt_style,
            ordering_item_label_style=args.ord_item_style,
            ordering_item_prefix_template=args.ord_item_template,
            typo_rate=args.typo_rate,
            space_noise_rate=args.space_noise_rate,
            latex_mask_prob=args.latex_mask_prob,
            latex_placeholder=args.latex_placeholder,
            enable_permutations=args.enable_permutations,
            option_drop_prob=args.option_drop_prob,
            casing_noise_prob=args.casing_noise_prob,
            synonym_swap_prob=args.synonym_swap_prob,
            formatting_noise_prob=args.formatting_noise_prob,
            inline_option_prob=getattr(args, "inline_option_prob", 0.0),
            min_inline_spaces=getattr(args, "min_inline_spaces", 5),
            max_inline_spaces=getattr(args, "max_inline_spaces", 30),
            min_inline_tabs=getattr(args, "min_inline_tabs", 1),
            max_inline_tabs=getattr(args, "max_inline_tabs", 3)
        )
        dest_dir = None if args.in_place else args.reconstruct_dest
        run_reconstructor_on_existing(
            input_directory=args.input_dir,
            dest_directory=dest_dir,
            config=reconstruct_config,
            max_workers=args.concurrency
        )

    elif args.command == "exam":
        from src.generation.exam_compiler import run_batch_exams_generator
        thinking_val = None if args.thinking == "none" else args.thinking
        run_batch_exams_generator(
            num_exams=args.num_exams,
            output_dir=args.output_dir,
            model=args.model,
            thinking=thinking_val,
            concurrency=args.concurrency,
            subject=args.subject,
            grade=args.grade,
            provider=args.provider
        )

    elif args.command == "prepare":
        from src.data.prepare import run_prepare_dataset
        run_prepare_dataset(args)

    elif args.command == "train":
        from src.model.train import run_train
        run_train(args)

    elif args.command == "inference":
        from src.inference.predict import run_inference
        run_inference(
            model_dir=args.model_dir,
            base_model_name=args.base_model_name,
            text=args.text,
            file_path=args.file,
            output_path=args.output,
            max_length=args.max_length,
            stride=args.stride
        )

    elif args.command == "upload":
        from src.data.upload import upload_dataset
        upload_dataset(
            token=args.token,
            repo_id=args.repo_id,
            dataset_dir=args.dataset_dir,
        )

    elif args.command == "upload-model":
        from src.model.upload import upload_model
        upload_model(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            dataset_repo=args.dataset_repo,
        )

    elif args.command == "visualize":
        from src.utils.visualize import generate_visualization
        generate_visualization(
            jsonl_path=args.input_file,
            output_html_path=args.output_html,
            max_embed_samples=args.max_samples
        )

if __name__ == "__main__":
    main()
