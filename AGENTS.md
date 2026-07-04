# Project Purpose

This project is a complete synthetic data generation and downstream model training pipeline designed to produce and train **Sequence Labeling** models (like XLM-RoBERTa) on Vietnamese educational exam papers.

The pipeline generates high-quality curriculum-aligned synthetic exam questions using LLMs (DeepSeek), compiles them into mock exams, reconstructs raw exam texts with character offset span mappings, tokenizes and labels them, and trains/evaluates LoRA token classification adapter models to automatically segment and extract exam question structures (such as question stems, option text, option labels, and contexts).

# Agent Instructions: Updating Project Structure

You MUST update the project structure section in this file (`AGENTS.md`) every time you make changes to the files or directories in the project.

# Project Structure

- `pyproject.toml` - Project configuration, dependencies, and Pixi task script registration.
- `pixi.lock` - Lockfile tracking platform dependencies.
- `AGENTS.md` - Rules, project structure, schemas, and developer workflows for assistant agents (this file).
- `README.md` - Comprehensive developer guide, pipeline workflow description, execution tasks, and data contract specifications.
- `tests/` - Directory housing all unittest suites.
  - `test_curriculum.py` - Tests for curriculum loader, parser, and level mapping.
  - `test_exam_compiler.py` - Tests for exam generator and section compilation.
  - `test_prepare_dataset.py` - Tests for XLM-RoBERTa dataset tokenizer alignments.
  - `test_parser.py` - Tests for question XML parsing and option prefix cleaning.
  - `test_reconstructor.py` - Tests for question and span reconstruction logic.
- `scratch/` - Used for temporary utility scripts, tagging helpers, and debugging utilities.
  - `annotate_math_files.py` - Helper script to tag, align, save, and verify math exam files.
  - `automatic_tagger.py` - Script to automatically tag OCR text using inference heuristics or models.
  - `check_xml_errors.py` - Script validating XML annotations and alignment.
  - `find_hash.py` - Script to find math files by hash.
  - `parse_xml_to_json.py` - Parses manually or subagent annotated XML files into spans and writes JSON outputs.
  - `process_all_vl.py` - Script to tag, align, parse, and verify the physics exam files.
  - `process_exams.py` - Processes exams and prepares annotation mappings.
- `logs/` - Runtime API usage logs directory (gitignored JSONL files; `.gitkeep` keeps the folder tracked).
  - `token_usage_<YYYY-MM-DD>.jsonl` - Daily append-only log; one JSON record per DeepSeek API call containing token counts and response text.
- `output/` - Output directory containing generated exams, curricula, and datasets (gitignored).
  - `dataset/` - Prepared tokenized sequence labeling dataset splits (`train.jsonl`, `val.jsonl`, `test.jsonl`).
  - `real-exams/` - Directory containing annotated real OCR exams as JSON/XML files.
- `real_data_annotator/` - Directory for real OCR data processing and annotation.
  - `pdf_converter.py` - Runs OCR on raw PDF files and extracts raw markdown text.
  - `annotate_ocr.py` - Annotates raw OCR markdown files with entity tags and generates JSON exam structures.
  - `out/` - Generated OCR markdown outputs and intermediate annotated files used for cleanup and review.
- `src/` - Main source package directory.
  - `cli.py` - Main CLI console entry point (wrapper module importing from the `cli` package).
  - `cli/` - Command-line interface package containing argument parsing and subcommand execution.
    - `parser.py` - CLI options and subcommand config.
    - `commands.py` - Action routers executing curricula, reconstruct, exam compilers, preparing, training, and uploading.
    - `__init__.py` - Package entry point.
  - `token_tracker.py` - Thread-safe token-usage logger.
  - `generation/` - Core Data Generation Subpackage.
    - `curriculum.py` - Handles subject & grade curriculum loading and generation.
    - `deepseek_client.py` - Client wrapping DeepSeek completions API reasoning models.
    - `exam_compiler.py` - Compiles multiple questions into section-grouped mock exams.
    - `generator.py` - Orchestrates question generation with AI prompting.
    - `parser.py` - Parses standard and group question elements from LLM XML output.
    - `reconstructor.py` - Rebuilds raw text from structured objects and maps offset character spans (wrapper module importing from the `reconstruction` package).
    - `reconstruction/` - Core Text Reconstruction Subpackage.
      - `config.py` - Constants, configurations (`ReconstructorConfig` with OCR bullet noise parameters), and stable random helpers.
      - `augment.py` - Typo injection, blank tokens randomization, LaTeX masking, formatting wrappers, and OCR bullet noise injection.
      - `layout.py` - Whitespace layout separators, section title paraphrasing, answer table formatting, and ordering choices.
      - `core.py` - Core reconstruction functions (`reconstruct_question`, `reconstruct_exam`).
      - `__init__.py` - Package entry point.
  - `training/` - Downstream Training & Evaluation Subpackage.
    - `prepare_dataset.py` - Formats, tokenizes, and splits synthetic questions into dataset splits (wrapper module importing from the `dataset` package).
    - `dataset/` - Dataset Preparation and Alignment Subpackage.
      - `alignment.py` - Token-to-span offset alignments, XML tag utilities.
      - `processing.py` - Multi-scale sliding window tokenization, LaTeX masking.
      - `io.py` - Dataset loading, directory scanning, and JSONL saving splits.
      - `__init__.py` - Package entry point.
    - `train.py` - Performs token-classification training with LoRA adapters (wrapper module importing from the `training_pipeline` package).
    - `training_pipeline/` - Downstream Training Pipeline Subpackage.
      - `config.py` - Training configurations, CLI arguments (including Focal Loss, Cosine LR Warmup, and LoRA params), environment setups, and devices.
      - `metrics.py` - Token-level evaluation and entity F1 metrics using seqeval.
      - `trainer.py` - Custom WeightedTrainer supporting class-weight, real-sample upsampling, and Focal Loss.
      - `callbacks.py` - Exponential Moving Average (EMA) callback tracker.

      - `__init__.py` - Package entry point.
    - `inference.py` - Local inference utility using trained LoRA adapter models.
    - `inference_folder.py` - Batch inference utility for segmenting raw text files.
    - `export_onnx.py` - Merges LoRA adapters and exports the sequence labeling model to ONNX.
    - `upload_dataset.py` - Uploads processed dataset splits to the Hugging Face hub.
    - `upload_model.py` - Uploads a trained LoRA adapter or full fine-tuned model checkpoint to HF hub.
    - `visualize_samples.py` - Generates HTML page for token-span alignment visualization.
  - `webapp/` - Web Application Subpackage.
    - `main.py` - FastAPI application entry point, routing layer (imports from the `webapp/backend` package).
    - `backend/` - Web Application Backend logic.
      - `stats.py` - Computes and caches training split dataset statistics.
      - `exams.py` - Handles JSON file discovery, parsing, and retrieval metadata.
      - `__init__.py` - Package entry point.
    - `inference_helper.py` - Helper utilities for sequence labeling predictions (wrapper module importing from the `inference` package).
    - `inference/` - Model Sequence Labeling Inference subpackage.
      - `latex.py` - LaTeX math validators, formula detection, and offset mapper.
      - `bio.py` - Transition correctors resolving BIO boundary violations.
      - `parser.py` - Converts span segments to inline XML and groups entities to questions.
      - `manager.py` - Manages backend PyTorch and ONNX inference runtime sessions.
      - `core.py` - Integrates tokenizer, sliding windows, and label outputs.
      - `__init__.py` - Package entry point.
    - `inference_app.py` - Standalone FastAPI web application running the model inference interface on port 8001.
    - `frontend/` - Decoupled React + TypeScript + Vite SPA frontend application.
    - `templates/` - Jinja2 HTML templates.

---

# Developer & Agent Workflows

## Environment & Dependency Management

This project uses **Pixi** for environment and dependency management.

- To run the unit tests: `pixi run test`
- To generate synthetic mock exams: `pixi run generate -n <num_exams>`
- To prepare the tokenized dataset: `pixi run prepare-dataset`
- To train the LoRA model: `pixi run train`
- To visualize token spans: `pixi run visualize`
- To upload the dataset to HF Hub: `pixi run upload-dataset`
- To run the web exam viewer: `pixi run view-exams`
- To run the standalone web inference app: `pixi run view-inference`

## Environment Configuration

The project utilizes the following environment variables:

- `DEEPSEEK_API_KEY`: API key for calling the DeepSeek reasoning and flash models.
- `HF_TOKEN`: Hugging Face write token for pulling/pushing datasets and LoRA adapter weights.
- **Important:** Do NOT try to read `.env` to fetch these keys. Prompt the user to provide them if they are missing from the environment context, or use standard environment retrieval.

---

# Data Contracts & Schemas

## 1. Domain Enums

- **Subjects:** `economics_law`, `geography`, `history`, `math_algebra`, `math_geometry`, `physics`, `chemistry`, `english`, `literature`.
- **Question Types:** `multiple_choice`, `true_false`, `short_answer`, `ordering`, `group_multiple_choice`, `group_short_answer`.
- **Cognitive/Difficulty Levels:** `recognize`, `comprehend`, `low_application`, `application`, `high_application`.

## 2. Output Question JSON Schemas

Synthetic question outputs are saved under `output/` as: `question_{subject}_g{grade}_{timestamp}_{uuid}.json`.

### Standard Question (`is_group: false`)

```json
{
  "is_group": false,
  "stem": "string — question stem (LaTeX formulas enclosed in $...$)",
  "options": ["string", "string", "..."],
  "answer": "string — correct answer letter, value, list, or sequence",
  "explanation": "string — explanation in Vietnamese",
  "subject": "chemistry | physics | ...",
  "grade": 8 | 9 | 10 | 11 | 12,
  "question_type": "multiple_choice | true_false | short_answer | ordering",
  "difficulty": "recognize | comprehend | ..."
}
```

### Group Question (`is_group: true`)

```json
{
  "is_group": true,
  "context": "string — shared passage/context text (LaTeX formulas in $...$)",
  "questions": [
    {
      "stem": "string — sub-question stem",
      "options": ["string", "string", "..."],
      "answer": "string — correct answer for sub-question",
      "explanation": "string — explanation for sub-question"
    }
  ],
  "subject": "chemistry | physics | ...",
  "grade": 8 | 9 | 10 | 11 | 12,
  "question_type": "group_multiple_choice | group_short_answer",
  "difficulty": "recognize | comprehend | ..."
}
```

## 3. Sequence Labeling Tag Set

During dataset preparation (Stage 5), text tokens are aligned to character spans and labeled using the following entity categories:

- `question_label`: Labels prefix indicators (e.g. "Câu 1:")
- `stem`: Labels the main text body of a question/sub-question (including ordering items if present).
- `option_label`: Labels options letters/prefixes (e.g. "A.", "B.", "a)")
- `option_text`: Labels the textual content of options.
- `context`: Labels the shared passage/context block in group questions.
- `section`: Labels section headers, directions, and reference titles (e.g., "ĐÁP ÁN THAM KHẢO").
- `explanation`: Labels reference explanations and solutions for questions (e.g., "**Câu 1: B** ...").

# Privacy and Security Rules

- NEVER attempt to read or access the `.env` file under any circumstances, as it contains sensitive credentials.

# RTK - Rust Token Killer (Google Antigravity)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk` to minimize token consumption.

Examples:

```bash
rtk git status
rtk cargo test
rtk ls src/
rtk grep "pattern" src/
rtk find "*.rs" .
rtk docker ps
rtk gh pr list
```

## Meta Commands

```bash
rtk gain              # Show token savings
rtk gain --history    # Command history with savings
rtk discover          # Find missed RTK opportunities
rtk proxy <cmd>       # Run raw (no filtering, for debugging)
```

## Why

RTK filters and compresses command output before it reaches the LLM context, saving 60-90% tokens on common operations. Always use `rtk <cmd>` instead of raw commands.
