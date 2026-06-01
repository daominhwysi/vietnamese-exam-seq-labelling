# Vietnamese Exam Sequence Labeling Data Generator & Model Training Pipeline

An end-to-end framework for curriculum-guided synthetic data generation, mock exam compilation, sequence labeling dataset alignment, downstream Hugging Face token-classification training with LoRA adapters, and interactive web-based visualization.

This project is tailored for processing Vietnamese high school educational exam papers (Grades 8–12) across multiple subjects, automatically segmenting structures such as stems, option labels, option texts, context passages, and question prefixes.

---

## 🌟 Key Features

1. **Curriculum-Guided Data Generation**: Orchestrates DeepSeek LLMs (Reasoning & Flash models) to generate detailed subjects/grade curricula, which are then used to feed specific topic contexts and problem types (Nhận biết, Thông hiểu, Vận dụng, Vận dụng cao) directly into the question generation prompt.
2. **Support for Diverse Question Formats**:
   - **Standard Questions**: Multiple Choice, True/False, Short Answer, and Ordering (with custom-generated distractors based on step permutations).
   - **Group Questions**: Passage-based/context-shared question clusters with multiple sub-questions.
3. **Deterministic Text Reconstruction & Offset-Span Tracking**: Reconstructs standard, formatted exam-style texts from structured JSON elements while calculating exact character-level offsets (`start` and `end` indices) for each semantic entity.
4. **Sequence Labeling Dataset Builder**:
   - Tokenizes raw exam text with models like `FacebookAI/xlm-roberta-base`.
   - Optionally masks LaTeX formulas (`$...$`) with a special `[LATEX]` token.
   - Automatically aligns tokens to character spans to produce token-level labels (BIO schema).
   - Splits the generated corpus into Train, Validation, and Test sets in `.jsonl` format.
5. **Parameter-Efficient Finetuning (LoRA)**: Train token classification models using the Hugging Face `Trainer` API, applying PEFT/LoRA adapters, automatic mixed precision (AMP with `float16`/`bfloat16`), and integration with Hugging Face Hub.
6. **Premium FastAPI Dashboard & Visualizers**:
   - Standalone HTML interactive sample alignment page (`pixi run visualize`).
   - Premium Tailwind CSS styled FastAPI web dashboard with search, stats, and a mock exam layout viewer highlighting annotated token spans (`pixi run view-exams`).

---

## 📁 Project Structure

```text
├── pyproject.toml         # Project config, metadata, dependency definitions, and Pixi tasks
├── pixi.lock              # Lockfile tracking exact conda-forge & PyPI dependencies
├── AGENTS.md              # Instructions, schemas, and structure reference for agent developers
├── tests/                 # Unit testing suite
│   ├── test_curriculum.py     # Tests for curriculum loaders, parsing, and mappings
│   ├── test_exam_compiler.py  # Tests for mock exam compilation and section grouping
│   ├── test_prepare_dataset.py# Tests for tokenizers and BIO sequence labeling span alignments
│   └── test_reconstructor.py  # Tests for text reconstruction and offset index mapping
└── src/                   # Core pipeline codebase
    ├── cli.py             # Global command-line entrypoint managing all workflow stages
    ├── generation/        # Core Data Generation Modules
    │   ├── curriculum.py      # Generates, caches, and filters subject-level learning paths
    │   ├── deepseek_client.py # Client wrapper around DeepSeek's chat API
    │   ├── exam_compiler.py   # Assembles independent questions into structured mock exams
    │   ├── generator.py       # Manages prompt engineering, LLM queries, and XML parsing
    │   ├── parser.py          # Extractor logic for structured XML tags returned by the LLM
    │   └── reconstructor.py   # Renders structured inputs into raw text with character-span tokens
    ├── training/          # Model Training and Fine-Tuning Modules
    │   ├── prepare_dataset.py # Formats, tokenizes, and aligns character-spans to model tokens
    │   ├── train.py           # Training loop leveraging Hugging Face Trainer and LoRA adapters
    │   ├── inference.py       # Inference script for running sequence prediction locally
    │   ├── upload_dataset.py  # Script for pushing local dataset splits to HF Hub
    │   └── visualize_samples.py # Generates standalone interactive tag visualization HTML pages
    └── webapp/            # FastAPI Web Interface
        ├── main.py            # FastAPI router, statistics aggregator, and endpoint handlers
        └── templates/         # Jinja2 template definitions (dashboard, base frame, and exam viewer)
```

---

## ⚙️ Setup and Configuration

This project uses **Pixi** for cross-platform environment and dependency management.

### Prerequisites

Ensure you have Pixi installed:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Environment Variables

Configure authentication credentials by copying `.env.example` into a `.env` file in the root directory:
```bash
cp .env.example .env
```

Define the following environment variables:
- `DEEPSEEK_API_KEY`: API key for calling the DeepSeek reasoning and flash models.
- `HF_TOKEN`: Hugging Face write token for pulling/pushing datasets and LoRA adapter weights.

> [!WARNING]
> Never check `.env` or any files containing sensitive credentials into version control.

---

## 🚀 Execution Workflow (Step-by-Step)

The entire pipeline is mapped into Pixi tasks, making execution direct and reproducible:

### 1. Run Unit Tests
Verify the environment and code integrity before running the generation steps:
```bash
pixi run test
```

### 2. Stage 1: Curriculum Generation
Compile the educational curriculum outline (Chapters, Units, and specific Dạng bài tập/Problem Types) for a target subject and grade:
```bash
# Generate a specific curriculum
pixi run curriculum --subject physics --grade 11

# Generate curricula for all subjects and grades (10-12) concurrently
pixi run curriculum --all
```
Generated curricula are cached as JSON files under `output/curriculum/`.

### 3. Stage 2: Question Generation
Synthesize question items based on curricula in the cache or through random generation:
```bash
# Generate 10 mock questions
pixi run generate -n 10

# Generate 50 questions for a specific subject/grade
pixi run generate -n 50 --subject chemistry --grade 12 --concurrency 8
```
Generated raw question files are stored in `output/question_*.json`.

### 4. Stage 3: Text Reconstruction & Span Alignment
Reconstruct plain text from questions and map character offsets. This step runs automatically during generation, but you can re-run or configure it separately:
```bash
pixi run reconstruct -i output --reconstruct-dest output_reconstructed
```

### 5. Stage 4: Mock Exam Compilation
Combine individual generated questions into section-grouped mock exams (e.g. matching realistic layouts with specific question structures):
```bash
pixi run compile-exams -n 50 --concurrency 8
```
Mock exams are saved in `output/exams/`.

### 6. Stage 5: Dataset Preparation
Preprocess the questions, replace LaTeX math blocks if specified, tokenize, assign BIO sequence labels, and split the dataset into `train.jsonl`, `val.jsonl`, and `test.jsonl`:
```bash
pixi run prepare-dataset -i output -o dataset_output --latex-placeholder "[LATEX]"
```

### 7. Stage 6: Train Token Classifier
Fine-tune the `FacebookAI/xlm-roberta-base` model using LoRA adapters on the prepared dataset:
```bash
pixi run train --repo_id your_hf_username/repo_name --epochs 3 --batch_size 8
```

### 8. Stage 7: Local Inference
Load the trained LoRA adapter weights and run a sequence labeling model against sample input text:
```bash
pixi run inference --model_dir ./results --base_model_name FacebookAI/xlm-roberta-base
```

### 9. Stage 8: HTML Visualization
Generate an offline HTML visualizer that highlights the sequence tag alignments for verification:
```bash
pixi run visualize -i dataset_output/train.jsonl -o dataset_output/sample_visualization.html
```

### 10. Web Interface: Interactive Exam Viewer
Start the FastAPI server to search, view statistics, and display generated exams with highlighted, color-coded token spans:
```bash
pixi run view-exams
```
Open your browser at `http://127.0.0.1:8000` to interact with the dashboard.

---

## 📊 Data Contracts & Labeling Schemas

### Sequence Labeling Tag Set

During token alignment, the tokens are mapped into BIO sequence tags matching these categories:

| Entity Tag | Description | Example Text |
|---|---|---|
| `question_label` | Prefix markers for question numbering | `"Câu 1:"`, `"Câu 12."`, `"C1."` |
| `stem` | Core content body of the question or sub-question | `"Cho hàm số $y=f(x)$ liên tục trên $\mathbb{R}$..."` |
| `option_label` | Choice indices/prefixes for options | `"A."`, `"B."`, `"a)"`, `"b."` |
| `option_text` | Textual content of a question choice | `"Hàm số đồng biến trên khoảng $(0; +\infty)$"` |
| `context` | Shared passage/passage blocks in group questions | `"Đọc đoạn thông tin sau đây và trả lời các câu hỏi..."` |

### Output Question JSON Format

Each generated question fits either a **Standard** or a **Group** structure.

#### Standard Question (`is_group: false`)
```json
{
  "is_group": false,
  "subject": "physics",
  "grade": 11,
  "question_type": "multiple_choice",
  "difficulty": "comprehend",
  "stem": "Câu hỏi của bạn...",
  "options": [
    "Phương án A",
    "Phương án B",
    "Phương án C",
    "Phương án D"
  ],
  "chapter": "Sóng",
  "unit": "Giao thoa ánh sáng",
  "problem_type_id": "physics_11_optics_dang1",
  "problem_type_name": "Tên dạng bài",
  "problem_type_level": "NB_TH",
  "raw_text": "Câu 1: Câu hỏi của bạn...\nA. Phương án A\nB. Phương án B...",
  "spans": [
    { "start": 0, "end": 7, "label": "question_label", "text": "Câu 1: " },
    { "start": 7, "end": 26, "label": "stem", "text": "Câu hỏi của bạn..." }
    // ... option_labels and option_texts spans
  ]
}
```

#### Group Question (`is_group: true`)
```json
{
  "is_group": true,
  "subject": "history",
  "grade": 12,
  "question_type": "group_multiple_choice",
  "difficulty": "application",
  "context": "Đoạn thông tin ngữ cảnh chung cho các câu hỏi...",
  "questions": [
    {
      "stem": "Câu hỏi phụ 1...",
      "options": ["A...", "B...", "C...", "D..."]
    },
    {
      "stem": "Câu hỏi phụ 2...",
      "options": ["A...", "B...", "C...", "D..."]
    }
  ],
  "raw_text": "Đoạn thông tin ngữ cảnh chung...\nCâu 1: Câu hỏi phụ 1...\nA. A...\n...",
  "spans": [
    { "start": 0, "end": 33, "label": "context", "text": "Đoạn thông tin ngữ cảnh chung..." }
    // ... sub-question labels, stems, option_labels and option_texts spans
  ]
}
```

---

## 🛠️ Verification & Development

To run unit tests or contribute to the repository:
- Run test suites: `pixi run test`
- Make sure code modifications preserve schema definitions (as described in `AGENTS.md` and `README.md`).
