import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import concurrent.futures
from tqdm import tqdm

from sequence_labelling_data_generator.generator import run_generator
from sequence_labelling_data_generator.reconstructor import (
    reconstruct_question,
    ReconstructorConfig,
    OPTION_PREFIX_STYLES
)

def run_reconstructor_on_existing(input_directory: str, dest_directory: Optional[str], config: ReconstructorConfig, max_workers: int = 4):
    in_dir = Path(input_directory)
    if not in_dir.exists():
        print(f"Error: Input directory '{input_directory}' does not exist.")
        sys.exit(1)
        
    json_files = list(in_dir.glob("question_*.json"))
    if not json_files:
        print(f"No question JSON files found in '{input_directory}'.")
        return
        
    if dest_directory:
        dest_path = Path(dest_directory)
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(json_files)} question file(s) in '{input_directory}'. Saving reconstructed files to '{dest_directory}'...")
    else:
        print(f"WARNING: Modifying {len(json_files)} question file(s) in '{input_directory}' in-place...")
        
    def process_file(file_path: Path) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Reconstruct (add raw_text and spans)
            updated_data = reconstruct_question(data, config)
            
            # Target path
            target_path = file_path if not dest_directory else Path(dest_directory) / file_path.name
            
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
    
    # Mode selection
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="If set, reconstructs raw text and tracks spans for all existing JSON files in the output directory instead of generating new ones."
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="If set, overwrites the files in the output directory directly. Otherwise, saves them to --reconstruct-dest."
    )
    parser.add_argument(
        "--reconstruct-dest",
        type=str,
        default="output_reconstructed",
        help="Output directory path for reconstructed files if not in-place (default: 'output_reconstructed')"
    )
    
    # Standard generation arguments
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=1,
        help="Number of questions to generate (default: 1)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory path for generation (default: 'output')"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent workers for parallel generation/reconstruction (default: 4)"
    )
    
    # Custom reconstructor configurations
    parser.add_argument(
        "--q-prefix",
        type=str,
        default=None,
        help="Override question prefix template (e.g. 'Câu {num}: ', 'Question {num}. ')"
    )
    parser.add_argument(
        "--opt-style",
        type=str,
        default=None,
        choices=list(OPTION_PREFIX_STYLES.keys()),
        help="Override option prefix style for multiple choice/true-false questions."
    )
    parser.add_argument(
        "--ord-item-style",
        type=str,
        default=None,
        choices=["char", "index"],
        help="Override item label style for ordering questions ('char' or 'index')."
    )
    parser.add_argument(
        "--ord-item-template",
        type=str,
        default=None,
        help="Override ordering item prefix template (e.g. '{label}. ', '* {label}. ')"
    )
    
    args = parser.parse_args()
    
    # Construct ReconstructorConfig from CLI arguments
    reconstruct_config = ReconstructorConfig(
        question_prefix_template=args.q_prefix,
        option_prefix_style=args.opt_style,
        ordering_item_label_style=args.ord_item_style,
        ordering_item_prefix_template=args.ord_item_template
    )
    
    if args.reconstruct:
        dest_dir = None if args.in_place else args.reconstruct_dest
        run_reconstructor_on_existing(
            input_directory=args.output,
            dest_directory=dest_dir,
            config=reconstruct_config,
            max_workers=args.concurrency
        )
    else:
        if args.num <= 0:
            print("Error: Number of questions must be greater than 0.")
            sys.exit(1)
            
        if args.concurrency <= 0:
            print("Error: Concurrency must be greater than 0.")
            sys.exit(1)
            
        print(f"Starting parallel generation of {args.num} question(s) to directory: {args.output} with concurrency: {args.concurrency}")
        run_generator(num_questions=args.num, output_dir=args.output, max_workers=args.concurrency)

if __name__ == "__main__":
    main()
