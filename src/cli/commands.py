import sys
import json
import concurrent.futures
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.generation.curriculum import generate_curriculum, generate_all_curricula
from src.generation.reconstructor import (
    reconstruct_question,
    reconstruct_exam,
    ReconstructorConfig
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

def execute_command(args):
    if args.command == "curriculum":
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
            max_inline_tabs=getattr(args, "max_inline_tabs", 3),
            answer_table_format=getattr(args, "answer_table_format", None),
            answer_table_direction=getattr(args, "answer_table_direction", None),
            answer_table_chunk_size=getattr(args, "answer_table_chunk_size", None),
            paraphrase_section_titles=getattr(args, "paraphrase_section_titles", False)
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
        from src.training.prepare_dataset import run_prepare_dataset
        run_prepare_dataset(args)

    elif args.command == "train":
        from src.training.train import run_train
        run_train(args)

    elif args.command == "inference":
        from src.training.inference import run_inference
        run_inference(model_dir=args.model_dir, base_model_name=args.base_model_name)

    elif args.command == "upload":
        from src.training.upload_dataset import upload_dataset
        upload_dataset(
            token=args.token,
            repo_id=args.repo_id,
            dataset_dir=args.dataset_dir,
        )

    elif args.command == "upload-model":
        from src.training.upload_model import upload_model
        upload_model(
            model_dir=args.model_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            dataset_repo=args.dataset_repo,
        )

    elif args.command == "visualize":
        from src.training.visualize_samples import generate_visualization
        generate_visualization(
            jsonl_path=args.input_file,
            output_html_path=args.output_html,
            max_embed_samples=args.max_samples
        )
