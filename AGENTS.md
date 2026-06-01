Say hallo to user everytime they say hello

# Agent Instructions: Updating Project Structure
You MUST update the project structure section in this file (`AGENTS.md`) every time you make changes to the files or directories in the project.

# Project Structure

- `pyproject.toml` - Project configuration, dependencies, and Pixi task script registration.
- `pixi.lock` - Lockfile tracking platform dependencies.
- `DATA_STRUCTURE.md` - Documentation of output JSON schemas, enums, and API prompt flow.
- `AGENTS.md` - Rules and detailed project structure for assistant agents (this file).
- `tests/` - Directory housing all unittest suites.
  - `test_curriculum.py` - Tests for curriculum loader, parser, and level mapping.
  - `test_exam_compiler.py` - Tests for exam generator and section compilation.
  - `test_prepare_dataset.py` - Tests for XLM-RoBERTa dataset tokenizer alignments.
  - `test_reconstructor.py` - Tests for question and span reconstruction logic.
- `src/` - Main source package directory.
  - `cli.py` - Main CLI console entry point handling all 9 subcommands.
  - `generation/` - Core Data Generation Subpackage.
    - `curriculum.py` - Handles subject & grade curriculum loading and generation.
    - `deepseek_client.py` - Client wrapping DeepSeek completions API reasoning models.
    - `exam_compiler.py` - Compiles multiple questions into section-grouped mock exams.
    - `generator.py` - Orchestrates question generation with AI prompting.
    - `parser.py` - Parses standard and group question elements from LLM XML output.
    - `reconstructor.py` - Rebuilds raw text from structured objects and maps offset character spans.
  - `training/` - Downstream Training & Evaluation Subpackage.
    - `prepare_dataset.py` - Formats, tokenizes, and splits synthetic questions into train/val/test splits.
    - `train.py` - Performs token-classification training with LoRA adapters.
    - `inference.py` - Local inference utility using trained LoRA adapter models.
    - `upload_dataset.py` - Uploads processed dataset splits to the Hugging Face hub.
    - `visualize_samples.py` - Generates HTML page for token-span alignment visualization.
