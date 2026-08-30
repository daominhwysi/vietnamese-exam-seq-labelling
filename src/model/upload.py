#!/usr/bin/env python3
"""
Upload a trained model (LoRA adapter or full fine-tuned checkpoint) to Hugging Face Hub.
Automatically generates and uploads a model card (README.md).

Usage:
    pixi run upload-model
    python -m src.model.upload --model-dir ./results --repo-id <username>/<repo-name>
"""

import os
import sys
import json
import tempfile
import argparse
from pathlib import Path


DEFAULT_MODEL_REPO = "daominhwysi/vi-exam-seq-labeller"
DEFAULT_DATASET_REPO = "daominhwysi/synthetic-seq-labelling-vi-exam-v2"


# ---------------------------------------------------------------------------
# Model card (README.md)
# ---------------------------------------------------------------------------


def _generate_model_card(
    repo_id: str,
    base_model: str,
    checkpoint_type: str,
    label_list: list,
    dataset_repo: str,
) -> str:
    label_table_rows = "\n".join(
        f"| {i} | `{label}` |" for i, label in enumerate(label_list)
    )

    lora_note = (
        "This is a **LoRA / PEFT adapter** checkpoint. Load it on top of the base model:"
        if checkpoint_type == "LoRA adapter"
        else "This is a **fully fine-tuned** model checkpoint."
    )

    lora_load_snippet = (
        f"""\
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForTokenClassification

base_model = AutoModelForTokenClassification.from_pretrained(
    "{base_model}",
    num_labels={len(label_list)},
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
"""
        if checkpoint_type == "LoRA adapter"
        else f"""\
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
"""
    )

    return f"""\
---
language:
- vi
license: apache-2.0
base_model: {base_model}
task_categories:
- token-classification
task_ids:
- named-entity-recognition
tags:
- vietnamese
- exam
- sequence-labeling
- lora
- peft
- educational
datasets:
- {dataset_repo}
---

# Vietnamese Exam Sequence Labeller

A **token classification** model for automatically segmenting Vietnamese educational exam papers into structured components (question labels, stems, option labels, option texts, stimuli, and section headers).

{lora_note}

**Base model:** [`{base_model}`](https://huggingface.co/{base_model})
**Dataset:** [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})

## Label Set

| ID | Label |
|---|---|
{label_table_rows}

## Usage

```python
{lora_load_snippet}
import torch

text = "Câu 1. Nguyên tố nào có tính kim loại mạnh nhất? A. Na  B. K  C. Li  D. Cs"

# Add same special tokens used during training
special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
tokenizer.add_special_tokens({{"additional_special_tokens": special_tokens}})
model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
offset_mapping = inputs.pop("offset_mapping")

with torch.no_grad():
    logits = model(**inputs).logits

predictions = logits.argmax(-1)[0].tolist()
id2label = model.config.id2label

for token_id, pred, (start, end) in zip(
    inputs["input_ids"][0], predictions, offset_mapping[0]
):
    if start == end:   # special token
        continue
    token = tokenizer.convert_ids_to_tokens(token_id.item())
    print(f"{{token:20}} → {{id2label[pred]}}")
```

## Training

Trained with **LoRA** (rank 16, alpha 32) on top of `{base_model}` using the
[sequence-labelling-data-generator](https://github.com/daominhwysi/vietnamese-exam-seq-labelling) pipeline.

Training features:
- Synthetic exam generation via DeepSeek LLMs
- Real OCR-annotated exam data with upsampling (`WeightedRandomSampler`)
- Multi-scale sliding-window tokenization (512 / 768 / 1024 / 2048 tokens)
- Data augmentation: typo injection, spacing noise, LaTeX masking, casing noise, synonym swaps

## Citation

```bibtex
@misc{{vi-exam-seq-labeller,
  title  = {{Vietnamese Exam Sequence Labeller}},
  author = {{daominhwysi}},
  year   = {{2025}},
  url    = {{https://huggingface.co/{repo_id}}}
}}
```
"""


# ---------------------------------------------------------------------------
# Main upload function
# ---------------------------------------------------------------------------


def upload_model(
    model_dir: str = "./results",
    repo_id: str = None,
    token: str = None,
    private: bool = False,
    commit_message: str = None,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    onnx_dir: str = "./output/onnx",
    onnx_only: bool = False,
):
    """
    Upload the contents of `model_dir` to a Hugging Face model repository.
    Generates and uploads a model card (README.md) automatically.

    Supports both:
    - LoRA / PEFT adapter checkpoints (detected by adapter_config.json)
    - Full fine-tuned model checkpoints
    - ONNX model files (uploaded under 'onnx/' folder if present)
    """
    # -- Resolve HF token --------------------------------------------------
    if not token:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
        token = os.getenv("HF_TOKEN")

    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except ImportError:
            pass

    if not token:
        print("Error: HF_TOKEN not found in environment or huggingface-cli cache. Please log in with `hf auth login` or set HF_TOKEN.")
        sys.exit(1)

    # -- Validate model directory ------------------------------------------
    model_path = Path(model_dir)
    if not model_path.exists() or not model_path.is_dir():
        print(f"Error: Model directory '{model_dir}' does not exist.")
        sys.exit(1)

    # -- Detect checkpoint type & base model name --------------------------
    adapter_cfg_path = model_path / "adapter_config.json"
    is_lora = adapter_cfg_path.exists()
    checkpoint_type = "LoRA adapter" if is_lora else "full fine-tuned model"
    print(f"Detected checkpoint type: {checkpoint_type}")

    base_model = "jhu-clsp/mmBERT-base"  # sensible default
    if is_lora:
        try:
            with open(adapter_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path", base_model)
            print(f"  Base model (from adapter_config): {base_model}")
        except Exception:
            pass

    # -- Load label list from label_mapping.json ---------------------------
    label_list = []
    label_mapping_path = model_path / "label_mapping.json"
    if label_mapping_path.exists():
        try:
            with open(label_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            id_to_tag = {int(k): v for k, v in mapping.get("id_to_tag", {}).items()}
            label_list = [id_to_tag[i] for i in sorted(id_to_tag.keys())]
            print(f"  Loaded {len(label_list)} labels from label_mapping.json")
        except Exception as e:
            print(f"  Warning: Could not load label_mapping.json: {e}")

    if not label_list:
        # Fallback to standard tag schema
        base_tags = [
            "question_label",
            "stem",
            "option_label",
            "option_text",
            "stimulus",
            "section",
        ]
        label_list = ["O"] + [
            f"{prefix}-{tag}" for tag in base_tags for prefix in ("B", "I")
        ]

    # -- Resolve repo_id ---------------------------------------------------
    if not repo_id:
        repo_id = DEFAULT_MODEL_REPO

    # -- Import HF Hub -----------------------------------------------------
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: 'huggingface_hub' is not installed.")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"\nEnsuring model repository '{repo_id}' exists on Hugging Face Hub...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
            exist_ok=True,
        )
        print("Repository is ready.")
    except Exception as e:
        print(f"Notice during repository creation: {e}")

    # -- Generate and upload README.md (model card) ------------------------
    print("\nGenerating model card (README.md)...")
    readme_content = _generate_model_card(
        repo_id=repo_id,
        base_model=base_model,
        checkpoint_type=checkpoint_type,
        label_list=label_list,
        dataset_repo=dataset_repo,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(readme_content)
        tmp_readme_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=tmp_readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card (README.md)",
        )
        print("  Uploaded README.md")
    except Exception as e:
        print(f"  Warning: Could not upload README.md: {e}")
    finally:
        Path(tmp_readme_path).unlink(missing_ok=True)

    # -- Upload model files ------------------------------------------------
    if not onnx_only:
        if not commit_message:
            commit_message = (
                f"Upload {checkpoint_type} for Vietnamese exam sequence labelling"
            )

        print(f"\nUploading '{model_dir}' -> '{repo_id}' ...")
        print(f"  Commit message: {commit_message}")

        try:
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["checkpoint-*", "**/checkpoint-*/**"],
            )
            print(f"Success! PyTorch/LoRA model uploaded.")
        except Exception as e:
            print(f"Error uploading model: {e}")
            sys.exit(1)
    else:
        print(
            "\nSkipping PyTorch/LoRA model checkpoint upload (--onnx-only specified)."
        )

    # -- Upload ONNX model files if present --------------------------------
    if onnx_dir:
        onnx_path = Path(onnx_dir)
        if onnx_path.exists() and onnx_path.is_dir():
            print(f"\nDetected ONNX model directory at '{onnx_dir}'.")
            print(f"Uploading ONNX model files -> '{repo_id}/onnx'...")
            try:
                api.upload_folder(
                    folder_path=str(onnx_path),
                    path_in_repo="onnx",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Upload ONNX model format",
                )
                print("Success! ONNX model uploaded under 'onnx/' folder.")
            except Exception as e:
                print(f"Warning: Could not upload ONNX model: {e}")
        else:
            print(
                f"\nONNX model directory '{onnx_dir}' not found. Skipping ONNX upload."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained model checkpoint to Hugging Face Hub"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./results",
        help="Path to the trained model/adapter directory (default: './results')",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=f"HF model repository ID (default: '{DEFAULT_MODEL_REPO}')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (falls back to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=DEFAULT_DATASET_REPO,
        help=f"HF dataset repo to reference in the model card (default: '{DEFAULT_DATASET_REPO}')",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="./output/onnx",
        help="Path to the exported ONNX model directory (default: './output/onnx')",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Upload only the ONNX model files and README.md (skips PyTorch weights/adapter checkpoint)",
    )
    args = parser.parse_args()

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
        dataset_repo=args.dataset_repo,
        onnx_dir=args.onnx_dir,
        onnx_only=args.onnx_only,
    )


if __name__ == "__main__":
    main()
