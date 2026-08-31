Say hallo to user everytime they say hello

# Project Purpose

This project is a complete synthetic data generation and downstream model training pipeline designed to produce and train **Sequence Labeling** models (like mmBERT-base / ModernBERT) on Vietnamese educational exam papers.

The pipeline generates high-quality curriculum-aligned synthetic exam questions using LLMs (DeepSeek), compiles them into mock exams, reconstructs raw exam texts with character offset span mappings, tokenizes and labels them, and trains/evaluates LoRA token classification adapter models to automatically segment and extract exam question structures (such as question stems, option text, option labels, and contexts).

# Agent Instructions: Updating Project Structure

You MUST update the project structure section in this file (`AGENTS.md`) every time you make changes to the files or directories in the project.

# Project Structure

- `pyproject.toml` - Project configuration, dependencies, and Pixi task script registration.
- `pixi.lock` - Lockfile tracking platform dependencies.
- `config.yaml` - Centralized pipeline configuration (model selection, augmentation params, train/inference settings).
- `AGENTS.md` - Rules, project structure, schemas, and developer workflows for assistant agents (this file).
- `README.md` - Comprehensive developer guide, pipeline workflow description, execution tasks, and data contract specifications.
- `samples/` - Sample raw image and OCR visualization files.
- `tests/` - Directory housing all unittest suites.
  - `test_codex_provider.py` - Tests for OpenAI Codex API provider, reasoning effort mapping, and token tracking.
  - `test_curriculum.py` - Tests for curriculum loader, parser, and level mapping.
  - `test_exam_compiler.py` - Tests for exam generator and section compilation.
  - `test_prepare_dataset.py` - Tests for mmBERT / ModernBERT dataset tokenizer alignments.
  - `test_parser.py` - Tests for question XML parsing and option prefix cleaning.
  - `test_reconstructor.py` - Tests for question and span reconstruction logic.
  - `test_iter_logger.py` - Tests for iteration-based training logger and epoch-proportional step calculation.
  - `test_enhanced_head.py` - Tests for Enhanced Token Classification Head, Multi-Sample Dropout, Layer Pooling, and Focal Loss.
- `scratch/` - Directory housing temporary and utility debug scripts.
  - `generate_bio_lit_50.py` - Batch generation script for 25 Literature and 25 Biology exams using Codex.
  - `benchmark_augmentation.py` - Benchmarks online data augmentation latency and CPU vs GPU throughput.
  - `test_raw_exams/` - Sample raw exam text documents for batch inference testing.
  - `inference_output/` - Output directory containing segmented JSON, predictions TXT, and tagged XML outputs from batch inference.
  - `inference_results/` - Model predictions, XML tagged files, and segmented JSONs from exam audits.
- `logs/` - Runtime API usage logs directory (gitignored JSONL files; `.gitkeep` keeps the folder tracked).
  - `token_usage_<YYYY-MM-DD>.jsonl` - Daily append-only log; one JSON record per DeepSeek API call containing token counts and response text.
- `output/` - Output directory containing generated exams, curricula, and datasets (gitignored).
  - `dataset/` - Prepared tokenized sequence labeling dataset splits (`train.jsonl`, `val.jsonl`, `test.jsonl`, `label_mapping.json`, and `xml/`).
  - `ocr_input/` - Raw input PDFs for real OCR data processing (gitignored).
  - `ocr_out/` - Extracted OCR markdown files (gitignored).
  - `real_exams/` - Generated real OCR exam structures in JSON format.
- `src/` - Main source package directory.
  - `cli.py` - Main CLI console entry point handling the pipeline subcommands (curriculum, reconstruct, exam, prepare, train, inference, upload, visualize).
  - `data/` - Data Preparation & Real OCR Pipeline.
    - `prepare.py` - Formats, tokenizes, and splits synthetic questions and audited real XML exams into train/val/test splits with multi-scale sliding windows.
    - `upload.py` - Uploads processed dataset splits and inline-tagged XML files to Hugging Face Hub.
    - `pdf_converter.py` - Runs OCR on raw PDF files using LLM vision models and extracts raw markdown.
    - `annotate_ocr.py` - Annotates raw OCR markdown files with entity tags and generates JSON exam structures.
  - `generation/` - Core Data Generation Subpackage.
    - `curriculum.py` - Handles subject & grade curriculum loading and generation.
    - `deepseek_client.py` - Client wrapping DeepSeek completions API reasoning models.
    - `exam_compiler.py` - Compiles multiple questions into section-grouped mock exams.
    - `generator.py` - Orchestrates question generation with AI prompting.
    - `parser.py` - Parses standard and group question elements from LLM XML output.
    - `reconstructor.py` - Rebuilds raw text from structured questions and maps span offsets with layout, whitespace collapsing, and noise augmentations.
  - `model/` - Downstream Training & Export Subpackage.
    - `head.py` - Enhanced Token Classification Head (Weighted Layer Pooling, Dense MLP, Multi-Sample Dropout) and Focal Loss module.
    - `train.py` - Performs token-classification training (with online/offline dynamic data augmentations, LoRA adapters, and FP16/BF16 support).
    - `export.py` - Merges LoRA adapters and exports the sequence labeling model to ONNX.
    - `upload.py` - Uploads trained LoRA adapters or full fine-tuned model checkpoints to HF Hub.
  - `inference/` - Downstream Inference & Batch Predictions.
    - `predict.py` - Local sequence labeling inference utility.
    - `predict_folder.py` - Batch inference utility for segmenting raw text files into structured JSON segments.
  - `utils/` - Utility & Helper Subpackage.
    - `config.py` - Loads and parses `config.yaml` with schema fallbacks across the pipeline.
    - `token_tracker.py` - Thread-safe token-usage logger; writes daily logs to `logs/token_usage_*.jsonl`.
    - `visualize.py` - Generates HTML page for token-span alignment visualization.
  - `webapp/` - Web Application Subpackage.
    - `main.py` - FastAPI application entry point, routing, exam/dataset stats computation.
    - `inference_helper.py` - Helper utilities for sequence labeling predictions (sliding window aggregation, LaTeX masking, offset mapping).
    - `inference_app.py` - Standalone FastAPI web application running the model inference interface on port 8001.
    - `templates/` - Jinja2 HTML templates for dashboards, exam viewer, and inference website.

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
  "stimulus": "string — shared reading passage/stimulus text (LaTeX formulas in $...$)",
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
- `stimulus`: Labels the shared reading passage / stimulus block in group questions.
- `section`: Labels section headers and test instructions.

# Privacy and Security Rules

- NEVER attempt to read or access the `.env` file under any circumstances, as it contains sensitive credentials.
