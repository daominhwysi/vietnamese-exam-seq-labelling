"""
scratch/migrate_real_exams.py
─────────────────────────────
Migrates existing flat real_exam_*.json / *.xml files in output/real_exams/
into a subfolder structure that mirrors real_data_annotator/out/.

The hash in each filename is md5(relative_path_from_input_root)[:8], so we
can reverse-look-up which source .md file produced each output file.

Usage:
    pixi run python scratch/migrate_real_exams.py [--dry-run]
"""

import hashlib
import shutil
import argparse
from pathlib import Path


def build_hash_map(input_root: Path) -> dict[str, Path]:
    """Walk input_root and return {hash: relative_md_path} for every .md file."""
    hash_map = {}
    for md_file in input_root.rglob("*.md"):
        rel = md_file.relative_to(input_root)
        h = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:8]
        hash_map[h] = rel
    return hash_map


def main():
    parser = argparse.ArgumentParser(
        description="Migrate flat real_exam_* files into mirrored subfolders."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without actually moving anything.",
    )
    parser.add_argument(
        "--input",
        default="real_data_annotator/out",
        help="Root of the OCR input tree (default: real_data_annotator/out)",
    )
    parser.add_argument(
        "--output",
        default="output/real_exams",
        help="Root of the flat output directory (default: output/real_exams)",
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        print(f"[ERROR] Input root not found: {input_root}")
        return
    if not output_root.exists():
        print(f"[ERROR] Output root not found: {output_root}")
        return

    # Build hash → relative source path mapping
    hash_map = build_hash_map(input_root)
    print(f"Found {len(hash_map)} source .md file(s) in '{input_root}'.\n")

    moved = 0
    skipped = 0
    unknown = 0

    # Process every file directly under output_root (not in any subfolder yet)
    flat_files = [f for f in output_root.iterdir() if f.is_file()]
    print(f"Found {len(flat_files)} flat file(s) in '{output_root}'.\n")

    for flat_file in sorted(flat_files):
        name = flat_file.name  # e.g. real_exam_bc27a60a.json

        # Extract hash from filename
        stem = flat_file.stem  # e.g. real_exam_bc27a60a
        parts = stem.split("_")
        if len(parts) < 3 or parts[0] != "real" or parts[1] != "exam":
            print(f"  [SKIP]  Unexpected filename format: {name}")
            skipped += 1
            continue

        file_hash = parts[2]  # e.g. bc27a60a

        if file_hash not in hash_map:
            print(f"  [UNKNOWN] Hash '{file_hash}' not found in source tree: {name}")
            unknown += 1
            continue

        rel_md = hash_map[file_hash]          # e.g. ta/ta-10-hp-26-27.md
        target_subdir = output_root / rel_md.parent  # e.g. output/real_exams/ta
        target_file = target_subdir / name

        if target_file == flat_file:
            # Already in the right place (shouldn't happen for flat root files)
            skipped += 1
            continue

        if target_file.exists():
            print(f"  [SKIP]  Target already exists, skipping: {target_file.relative_to(output_root)}")
            skipped += 1
            continue

        print(f"  [MOVE]  {name}")
        print(f"          -> {target_file.relative_to(output_root)}")

        if not args.dry_run:
            target_subdir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(flat_file), str(target_file))

        moved += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    print(f"  Moved   : {moved}")
    print(f"  Skipped : {skipped}")
    print(f"  Unknown : {unknown}")


if __name__ == "__main__":
    main()
