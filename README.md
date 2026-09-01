# Vietnamese Exam Sequence Labelling — Data Generation & Model Training Pipeline

An end-to-end pipeline for generating curriculum-aligned synthetic Vietnamese exam data, annotating real OCR exam papers, building token-classification datasets, and fine-tuning LoRA-adapted sequence labelling models (mmBERT-base, ModernBERT) on the Hugging Face ecosystem.

---

## 🌟 Key Features

| Feature | Details |
|---|---|
| **LLM-driven data generation** | DeepSeek reasoning & flash models generate curriculum-aligned exam questions across 9 subjects and grades 8–12 |
| **Real OCR annotation** | Semi-automated OCR pipeline annotates real exam PDFs with ground-truth XML entity tags |
| **Rich data augmentation** | Typo injection, spacing noise, LaTeX masking, casing noise, prefix synonym swaps, inline option formatting, option drop simulation |
| **Multi-scale sliding window** | Tokenizes long documents at 512 / 768 / 1024 / 2048 token windows with configurable stride overlap |
| **Real sample upsampling** | `WeightedRandomSampler` at training time balances scarce real exam samples vs. abundant synthetic ones |
| **LoRA / full fine-tuning** | PEFT LoRA adapters or full fine-tune via Hugging Face `Trainer` with BF16/FP16/FP32 auto-detection |
| **HF Hub integration** | Auto-generated dataset cards & model cards; uploads splits + XML annotation files to HF datasets and models |
| **Batch inference + XML output** | `inference_folder.py` segments raw `.txt` files and produces structured JSON, token prediction tables, and inline-tagged XML |
| **Interactive web dashboard** | FastAPI + Jinja2 exam viewer with color-coded span highlighting |

---

## 📁 Project Structure

```text
.
├── pyproject.toml                  # Project config, Pixi task definitions, and dependencies
├── pixi.lock                       # Exact cross-platform lockfile (conda-forge + PyPI)
├── AGENTS.md                       # Agent instructions, schemas, and project structure reference
├── README.md                       # This file
│
├── tests/                          # Unit test suite
│   ├── test_curriculum.py
│   ├── test_exam_compiler.py
│   ├── test_prepare_dataset.py
│   ├── test_parser.py
│   └── test_reconstructor.py
│
├── scratch/                        # Debug & utility scripts (not part of main pipeline)
│   ├── check_alignment.py
│   ├── debug_alignment.py
│   ├── debug_model.py
│   ├── count_tokens.py
│   ├── incremental_train.py
│   ├── inspect_exam_spans.py
│   └── ...
│
├── logs/                           # API token-usage logs (gitignored JSONL)
│   └── token_usage_<YYYY-MM-DD>.jsonl
│
├── output/                         # Generated artifacts (gitignored)
│   ├── curriculum/                 # Cached curriculum JSON files
│   ├── exams/                      # Generated mock exam JSON files
│   │   ├── exam_*.json             # Synthetic compiled exams
│   │   └── real_exam_*.json        # Real OCR-annotated exams
│   └── dataset/                    # Prepared tokenized dataset
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       ├── label_mapping.json
│       └── xml/                    # Ground-truth inline-tagged XML per source exam
│           └── *_annotated.xml
│
├── real_data_annotator/            # Real exam OCR + annotation tools
│   ├── pdf_converter.py            # OCR PDF → markdown text
│   ├── annotate_ocr.py             # LLM-annotates OCR text → real_exam_*.json + XML
│   └── out/                        # Intermediate annotation outputs
│
└── src/                            # Main source package
    ├── cli.py                      # Unified CLI entry point (all pipeline subcommands)
    ├── token_tracker.py            # Thread-safe DeepSeek token usage logger
    │
    ├── generation/                 # Stage 1–3: Data Generation
    │   ├── curriculum.py           # Subject/grade curriculum generation & caching
    │   ├── deepseek_client.py      # DeepSeek API client wrapper
    │   ├── exam_compiler.py        # Compiles questions into section-grouped mock exams
    │   ├── generator.py            # Orchestrates LLM prompting & question generation
    │   ├── parser.py               # Parses LLM XML output into structured question dicts
    │   └── reconstructor.py        # Rebuilds raw text from JSON + computes character-level spans
    │
    ├── training/                   # Stage 4–8: Dataset + Training + Inference + Upload
    │   ├── prepare_dataset.py      # Tokenizes exams, aligns BIO labels, splits dataset, generates XML
    │   ├── train.py                # LoRA/full fine-tune training with WeightedRandomSampler
    │   ├── inference.py            # Single-input inference with trained adapter
    │   ├── inference_folder.py     # Batch inference → JSON + TXT + annotated XML per file
    │   ├── upload_dataset.py       # Uploads dataset + auto-generated dataset card to HF Hub
    │   ├── upload_model.py         # Uploads model + auto-generated model card to HF Hub
    │   └── visualize_samples.py    # Generates standalone HTML token-span visualizer
    │
    └── webapp/                     # FastAPI exam viewer
        ├── main.py
        └── templates/              # Jinja2 templates (dashboard, exam viewer, dataset browser)
```

---

## ⚙️ Setup

This project uses **[Pixi](https://pixi.sh)** for hermetic, cross-platform environment management.

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 2. Install dependencies

```bash
pixi install
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
DEEPSEEK_API_KEY=sk-...      # DeepSeek API key for question generation
HF_TOKEN=hf_...              # Hugging Face write token for upload/download
```

> [!WARNING]
> Never commit `.env` or any file containing API keys to version control.

---

## 🚀 Pipeline Workflow

The full pipeline runs in 8 stages. Each stage is a `pixi run` task.

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  1. curriculum  │───▶│  2. exam/generate │───▶│  3. (reconstruct)│
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
               ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
               │  4. prepare     │───▶│  5. train    │───▶│  6. inference│
               └─────────────────┘    └──────────────┘    └──────────────┘
                        │                    │
                        ▼                    ▼
               ┌──────────────────┐  ┌────────────────┐
               │ upload-dataset   │  │  upload-model  │
               └──────────────────┘  └────────────────┘
```

### Stage 1 — Curriculum Generation

Generate subject/grade curricula (chapters, units, problem types) used to steer question generation:

```bash
# Single subject/grade
pixi run curriculum --subject physics --grade 11

# All subjects & grades concurrently
pixi run curriculum --all --concurrency 8
```

Output: `output/curriculum/{subject}_g{grade}.json`

---

### Stage 2 — Mock Exam Generation

Synthesize section-grouped mock exams. Each exam contains multiple questions across question types (multiple choice, true/false, short answer, ordering, group questions):

```bash
pixi run generate -n 300 --concurrency 8

# Filter to a specific subject/grade
pixi run generate -n 50 --subject history --grade 12
```

Output: `output/exams/exam_*.json`

---

### Stage 3 — (Optional) Text Reconstruction

Re-reconstruct raw text and offset spans from existing question JSON files (useful after schema changes):

```bash
pixi run reconstruct -i output --reconstruct-dest output/reconstructed
```

---

### Stage 4 — Offline Dataset Preparation

Tokenizes all exams (synthetic + real), aligns BIO sequence labels to token offsets using multi-scale sliding windows, applies data augmentation, and splits into train/val/test. Also generates ground-truth annotated XML files per source exam.

```bash
pixi run prepare-offline-dataset -i output/exams -o output/dataset
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `jhu-clsp/mmBERT-base` | Tokenizer to use |
| `--exam-level` | off | Process at exam level (vs. question level) |
| `--max-len` | `512,768,1024,2048` | Sliding window sizes |
| `--stride` | `128,192,256,512` | Overlap strides |
| `--typo-rate` | `0.02` | Typo injection rate |
| `--space-noise-rate` | `0.15` | Whitespace noise rate |
| `--latex-mask-prob` | `0.5` | LaTeX `$...$` masking probability |
| `--casing-noise-prob` | `0.10` | Random casing noise |
| `--synonym-swap-prob` | `0.10` | Prefix synonym swap |
| `--inline-option-prob` | `0.0` | Inline option formatting probability |
| `--option-drop-prob` | `0.05` | Simulate truncated options (OCR cuts) |

Output:

```
output/dataset/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── raw_exams.jsonl
├── label_mapping.json
└── xml/
    ├── exam_abc123_annotated.xml
    ├── real_exam_xyz_annotated.xml
    └── ...
```

---

### Stage 5 — Train Token Classifier

Fine-tune with LoRA adapters (or full fine-tuning) using HF `Trainer`. Supports weighted sampling to upsample real exam data:

```bash
pixi run train \
  --repo_id daominhwysi/synthetic-seq-labelling-vi-exam-v2 \
  --epochs 5 \
  --batch_size 8 \
  --real-upsample-factor 5.0
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--repo_id` | `daominhwysi/synthetic-seq-labelling-vi-exam-v2` | HF dataset repo to load |
| `--model_name` | `jhu-clsp/mmBERT-base` | Base model |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--real-upsample-factor` | `1.0` | Weight multiplier for real samples (e.g. `5.0` = 5× more frequent) |
| `--no-lora` | off | Full fine-tune instead of LoRA |
| `--report_to` | `none` | Log integration to report metrics to (`wandb`, `tensorboard`, `none`) |
| `--wandb_project` | `vietnamese-exam-seq-labelling` | Weights & Biases project name |
| `--warmup-ratio` | `0.0` | LR warmup ratio |
| `--gradient-checkpointing` | off | Save GPU memory |
| `--no-class-weights` | off | Disable inverse-frequency class weighting |

Output: `./results/` (LoRA adapter weights + `label_mapping.json`)

---

### Stage 6 — Batch Inference

Run the trained model over a folder of raw `.txt` files. Produces three output files per input:

```bash
pixi run batch-inference \
  -i path/to/txt_files/ \
  -o inference_output/ \
  --model-dir ./results \
  --max-length 1024 \
  --stride 256
```

Per input file `exam.txt`, outputs:

| File | Content |
|---|---|
| `exam_structured.json` | Structured segment list with labels and text |
| `exam_predictions.txt` | Token-level tabular prediction table |
| `exam_annotated.xml` | Inline-tagged XML matching `annotate_ocr.py` format |

**XML example:**
```xml
<question_label>Câu 1.</question_label> <stem>Cho phương trình $x^2 - 5x + 6 = 0$.</stem>
<option_label>A.</option_label> <option_text>x = 2 hoặc x = 3</option_text>
<option_label>B.</option_label> <option_text>x = 1 hoặc x = 6</option_text>
```

---

### Stage 7 — Upload Dataset to HF Hub

Uploads `output/dataset/` (splits + `label_mapping.json` + `xml/` subfolder) and auto-generates a dataset card:

```bash
pixi run upload-dataset

# Custom target repo
sequence-labelling-generator upload \
  --repo-id myuser/my-dataset \
  --dataset-dir output/dataset
```

---

### Stage 8 — Upload Model to HF Hub

Uploads `./results/` and auto-generates a model card from `adapter_config.json` + `label_mapping.json`:

```bash
pixi run upload-model

# Custom options
sequence-labelling-generator upload-model \
  --model-dir ./results \
  --repo-id myuser/vi-exam-seq-labeller \
  --private
```

---

### Visualization & Web Dashboard

```bash
# Standalone HTML token-span alignment visualizer
pixi run visualize -i output/dataset/train.jsonl -o output/dataset/viz.html

# FastAPI interactive exam viewer (http://127.0.0.1:8000)
pixi run view-exams
```

---

## 📊 Data Contracts

### Tag Set (BIO Schema)

| Tag | Description | Example |
|---|---|---|
| `question_label` | Question/sub-question number prefix | `"Câu 1:"`, `"Question 2."` |
| `stem` | Main question body text | `"Cho hàm số $y = f(x)$..."` |
| `option_label` | Option letter prefix | `"A."`, `"B."`, `"a)"` |
| `option_text` | Option content text | `"Hàm số đồng biến trên $(0; +∞)"` |
| `stimulus` | Shared reading passage / stimulus for group questions | `"Đọc đoạn văn sau..."` |
| `section` | Section headers and exam directions | `"PHẦN I. Trắc nghiệm"` |

All tags use the BIO prefix scheme: `B-<tag>` for the first token, `I-<tag>` for continuation, `O` for outside.

### Dataset Sample Schema

Each line in `train.jsonl`, `val.jsonl`, `test.jsonl`:

```json
{
  "tokens":         ["▁Câu", "▁1", ".", "▁Cho", "..."],
  "input_ids":      [0, 12, 34, 56, ...],
  "attention_mask": [1, 1, 1, 1, ...],
  "labels":         [1, 2, 0, 3, ...],
  "tags":           ["B-question_label", "I-question_label", "O", "B-stem", "..."],
  "metadata": {
    "subject":       "physics",
    "grade":         11,
    "is_real":       false,
    "exam_id":       "exam_abc123",
    "question_type": "multiple_choice",
    "difficulty":    "comprehend",
    "max_len":       512,
    "stride":        128,
    "chunk_idx":     0,
    "total_chunks":  1
  }
}
```

### Subjects & Grades

**Subjects:** `economics_law` · `geography` · `history` · `math_algebra` · `math_geometry` · `physics` · `chemistry` · `english` · `literature`

**Grades:** 8 · 9 · 10 · 11 · 12

**Question types:** `multiple_choice` · `true_false` · `short_answer` · `ordering` · `group_multiple_choice` · `group_short_answer`

**Difficulty levels:** `recognize` · `comprehend` · `low_application` · `application` · `high_application`

### Question JSON Schema

#### Standard Question

```json
{
  "is_group": false,
  "stem": "Nguyên tố nào có tính kim loại mạnh nhất?",
  "options": ["Na", "K", "Li", "Cs"],
  "answer": "B",
  "explanation": "Kali (K) có năng lượng ion hoá thứ nhất thấp nhất...",
  "subject": "chemistry",
  "grade": 10,
  "question_type": "multiple_choice",
  "difficulty": "comprehend"
}
```

#### Group Question

```json
{
  "is_group": true,
  "stimulus": "Đọc đoạn văn sau và trả lời các câu hỏi...",
  "questions": [
    {
      "stem": "Ý chính của đoạn văn là gì?",
      "options": ["A...", "B...", "C...", "D..."],
      "answer": "A",
      "explanation": "..."
    }
  ],
  "subject": "english",
  "grade": 12,
  "question_type": "group_multiple_choice",
  "difficulty": "application"
}
```

---

## 🛠️ Development

```bash
# Run all unit tests
pixi run test

# Check API token balance
python scratch/get_balance.py

# Inspect generated dataset splits
python scratch/inspect_local_data.py

# Verify span alignment accuracy
python scratch/check_xml_accuracy.py
```

---

## 🔗 Related Resources

- **Dataset:** [daominhwysi/synthetic-seq-labelling-vi-exam-v2](https://huggingface.co/datasets/daominhwysi/synthetic-seq-labelling-vi-exam-v2)
- **Model:** [daominhwysi/mmbert-base-vi-exam-seq-labeling](https://huggingface.co/daominhwysi/mmbert-base-vi-exam-seq-labeling)
- **Base model:** [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)
