#!/usr/bin/env python3
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Bypass torchao compatibility check bug on older pre-installed versions in Google Colab
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Sample test cases if no custom text is passed
TEST_SAMPLES = [
    {
        "type": "Hóa học - Multiple Choice with Formula",
        "text": "Câu 1: Trong phòng thí nghiệm, khí oxi ($O_2$) được điều chế bằng cách nhiệt phân chất nào sau đây?\nA. $KMnO_4$\nB. $NaCl$\nC. $CaCO_3$\nD. $H_2O$"
    },
    {
        "type": "Vật lý - True/False Statements",
        "text": "Câu 2: Cho các phát biểu sau về các kim loại kiềm:\na) Kim loại kiềm có nhiệt độ nóng chảy thấp và độ cứng nhỏ.\nb) Trong tự nhiên, kim loại kiềm tồn tại ở cả dạng đơn chất và hợp chất.\nc) Các kim loại kiềm đều được bảo quản bằng cách ngâm trong dầu hỏa.\nd) Tất cả các kim loại kiềm đều phản ứng mãnh liệt với nước ở nhiệt độ thường."
    },
    {
        "type": "Toán học / Vật lý - Ordering Sequence",
        "text": "Câu 3: Hãy sắp xếp trình tự đúng các bước xác định tiêu cự của thấu kính hội tụ:\n1. Đặt thấu kính và màn ảnh trên giá quang học thẳng hàng.\n2. Bật đèn chiếu sáng nguồn sáng hướng vào thấu kính.\n3. Di chuyển màn ảnh từ từ để nhận được ảnh rõ nét trên màn.\n4. Đo khoảng cách từ thấu kính đến màn ảnh và ghi nhận tiêu cự.\nA. 1 – 2 – 3 – 4\nB. 2 – 1 – 3 – 4\nC. 3 – 2 – 1 – 4\nD. 4 – 3 – 2 – 1"
    },
    {
        "type": "Đọc hiểu - Group Context & Questions",
        "text": "# PHẦN 2: TƯ DUY KHOA HỌC\n\nDựa vào thông tin sau để trả lời câu hỏi 4 và 5:\nSự quang hợp ở thực vật diễn ra chủ yếu ở lục lạp nhờ sắc tố diệp lục hấp thụ năng lượng ánh sáng mặt trời để chuyển hóa $CO_2$ và $H_2O$ thành hợp chất hữu cơ ($C_6H_{12}O_6$) đồng thời giải phóng khí $O_2$.\n\nCâu 4: Bào quan nào thực hiện chức năng quang hợp ở tế bào thực vật?\nA. Ty thể\nB. Lục lạp\nC. Ribosome\nD. Không bào\n\nCâu 5: Khí nào sau đây được giải phóng trong quá trình quang hợp?\nA. Khí nitơ ($N_2$)\nB. Khí cacbonic ($CO_2$)\nC. Khí oxi ($O_2$)\nD. Khí hiđro ($H_2$)"
    }
]

def load_label_mapping(model_dir: str):
    # 1. Try local model directory
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}

    # 2. Try default dataset output folder
    default_dataset_mapping = "output/dataset/label_mapping.json"
    if os.path.exists(default_dataset_mapping):
        with open(default_dataset_mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}

    # 3. Try Hugging Face Hub if model_dir is a repo id
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=model_dir, filename="label_mapping.json")
        with open(downloaded, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
    except Exception:
        pass

    # 4. Standard base tags fallback
    base_tags = ["question_label", "stem", "option_label", "option_text", "stimulus", "section"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def is_valid_latex(content: str) -> bool:
    content_stripped = content.strip()
    if not content_stripped:
        return False
    if len(content_stripped) == 1:
        return content_stripped.isalnum()
    # Case 2: Mathematical interval notation like (-\infty; 0] or [8; +\infty) or (1; 2]
    if re.match(r'^[\[\(][^\[\]\(\)]+[\]\)]$', content_stripped) and (';' in content_stripped or ',' in content_stripped):
        return True
    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []
    for char in content_stripped:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            last = stack.pop()
            if brackets[last] != char:
                return False
    if stack:
        return False
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True
    return False

def get_latex_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        content = match.group(0)[2:-2]
        if is_valid_latex(content):
            spans.append(match.span())
    for match in re.finditer(r"\$(?!\s)[^\$\n]+?(?<!\s)\$", text):
        span = match.span()
        content = match.group(0)[1:-1]
        if not is_valid_latex(content):
            continue
        overlap = False
        for d_start, d_end in spans:
            if not (span[1] <= d_start or span[0] >= d_end):
                overlap = True
                break
        if not overlap:
            spans.append(span)
    spans.sort(key=lambda x: x[0])
    return spans

def build_tagged_xml(raw_text: str, segments: List[Dict[str, Any]]) -> str:
    """
    Constructs inline-tagged XML string from segments.
    """
    sorted_segs = sorted(segments, key=lambda x: (x.get("start", 0), -x.get("end", 0)))
    result = []
    cursor = 0
    for seg in sorted_segs:
        start = seg.get("start", -1)
        end = seg.get("end", -1)
        label = seg["label"]
        text = seg["text"]
        if start >= 0 and end >= 0:
            if start < cursor:
                continue
            if start > cursor:
                result.append(raw_text[cursor:start])
            result.append(f"<{label}>{text}</{label}>")
            cursor = end
        else:
            idx = raw_text.find(text, cursor)
            if idx == -1:
                continue
            if idx > cursor:
                result.append(raw_text[cursor:idx])
            result.append(f"<{label}>{text}</{label}>")
            cursor = idx + len(text)
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])
    return "".join(result)

def predict_text(
    text: str,
    model,
    tokenizer,
    id_to_tag: Dict[int, str],
    device: str,
    max_length: int = 1024,
    stride: int = 256
) -> Dict[str, Any]:
    """
    Runs sequence labeling on a raw text string with sliding window and LaTeX equation masking.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    latex_spans = get_latex_spans(text)
    processed_text = ""
    last_idx = 0
    for start, end in latex_spans:
        processed_text += text[last_idx:start] + "[LATEX]"
        last_idx = end
    processed_text += text[last_idx:]

    def map_idx(idx):
        mod_pos = 0
        orig_pos = 0
        for o_start, o_end in latex_spans:
            segment_len = o_start - orig_pos
            if idx <= mod_pos + segment_len:
                return orig_pos + (idx - mod_pos)
            orig_pos = o_end
            mod_pos += segment_len + len("[LATEX]")
            if idx < mod_pos:
                return o_start
        return orig_pos + (idx - mod_pos)

    tokenized = tokenizer(
        processed_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True
    )

    span_logits = {}
    span_tokens = {}
    num_chunks = len(tokenized["input_ids"])

    for chunk_idx in range(num_chunks):
        chunk_input_ids = tokenized["input_ids"][chunk_idx]
        chunk_attention_mask = tokenized["attention_mask"][chunk_idx]
        chunk_mod_offsets = tokenized["offset_mapping"][chunk_idx]

        chunk_offsets = []
        for start, end in chunk_mod_offsets:
            if start == 0 and end == 0:
                chunk_offsets.append((0, 0))
            else:
                chunk_offsets.append((map_idx(start), map_idx(end)))

        inputs = {
            "input_ids": torch.tensor([chunk_input_ids]).to(device),
            "attention_mask": torch.tensor([chunk_attention_mask]).to(device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        chunk_logits = outputs.logits[0].cpu()
        chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_input_ids)

        for i, (token, offset, mask) in enumerate(zip(chunk_tokens, chunk_offsets, chunk_attention_mask)):
            start, end = offset
            if (start == 0 and end == 0) or mask == 0:
                continue
            span = (start, end)
            if span not in span_logits:
                span_logits[span] = []
                span_tokens[span] = token
            span_logits[span].append(chunk_logits[i])

    sorted_spans = sorted(span_logits.keys(), key=lambda x: (x[0], x[1]))
    predictions = []
    offsets = []
    tokens = []

    for span in sorted_spans:
        start, end = span
        avg_logits = torch.mean(torch.stack(span_logits[span]), dim=0)
        pred_id = torch.argmax(avg_logits).item()
        predictions.append(pred_id)
        offsets.append(span)
        tokens.append(span_tokens[span])

    segments = []
    current_label = None
    current_start = -1
    current_end = -1

    for idx, (pred_id, offset) in enumerate(zip(predictions, offsets)):
        start, end = offset
        label = id_to_tag.get(pred_id, "O")
        if label.startswith("B-"):
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": text[current_start:current_end].strip()
                })
            current_label = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-"):
            tag_name = label[2:]
            if current_label == tag_name:
                current_end = end
            else:
                if current_label and current_start < current_end:
                    segments.append({
                        "label": current_label,
                        "start": current_start,
                        "end": current_end,
                        "text": text[current_start:current_end].strip()
                    })
                # Auto-promote orphaned I-tag to start a new segment
                current_label = tag_name
                current_start = start
                current_end = end
        else:
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": text[current_start:current_end].strip()
                })
                current_label = None
                current_start = -1
                current_end = -1

    if current_label and current_start < current_end:
        segments.append({
            "label": current_label,
            "start": current_start,
            "end": current_end,
            "text": text[current_start:current_end].strip()
        })

    # Filter out empty and spurious non-alphanumeric label segments (e.g. '*' or '-' tagged as option_label)
    cleaned_segments = []
    for s in segments:
        txt = s["text"].strip()
        label = s["label"]
        if not txt:
            continue
        if label in ["option_label", "question_label"] and not any(c.isalnum() for c in txt):
            continue
        cleaned_segments.append(s)
    segments = cleaned_segments
    xml_content = build_tagged_xml(text, segments)

    return {
        "raw_text": text,
        "segments": segments,
        "xml_content": xml_content,
        "token_count": len(tokens)
    }

def print_result(res: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("EXTRACTED ENTITY SEGMENTS:")
    print("=" * 70)
    for seg in res["segments"]:
        label = seg["label"]
        txt = seg["text"]
        tag_badge = f"[{label.upper()}]"
        print(f"  {tag_badge:<18} : {txt}")

    print("\n" + "-" * 70)
    print("INLINE-TAGGED XML PREVIEW:")
    print("-" * 70)
    print(res["xml_content"])
    print("=" * 70)

def load_inference_model(model_dir: str, base_model_name: str = None, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect default model path if default "./results" doesn't exist
    if model_dir.startswith("./content/"):
        alt_path = model_dir[1:]  # /content/...
        if os.path.exists(alt_path):
            model_dir = alt_path
        elif os.path.exists(model_dir[10:]):  # stripped ./content/
            model_dir = model_dir[10:]

    if model_dir == "./results" and not os.path.exists("./results"):
        if os.path.exists("./results_enhanced_v3"):
            model_dir = "./results_enhanced_v3"
        elif os.path.exists("./results_full"):
            model_dir = "./results_full"
        else:
            model_dir = "daominhwysi/results_full"

    if not os.path.exists(model_dir) and (model_dir.startswith((".", "/", "\\")) or "/" in model_dir and os.path.exists(os.path.abspath(model_dir))):
        if os.path.exists(os.path.abspath(model_dir)):
            model_dir = os.path.abspath(model_dir)

    print(f"Model Identifier: {model_dir}")
    print(f"Inference Device: {device}")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        print(f"Loaded tokenizer from: {model_dir}")
    except Exception:
        fallback_base = base_model_name or "jhu-clsp/mmBERT-base"
        print(f"Local tokenizer not found, loading base: {fallback_base}...")
        tokenizer = AutoTokenizer.from_pretrained(fallback_base, use_fast=True)
        special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 2. Load Label Mappings
    tag_to_id, id_to_tag = load_label_mapping(model_dir)

    # 3. Detect and Load Model (Enhanced Head vs. LoRA adapter vs. Full Fine-Tuned Checkpoint)
    is_lora = False
    has_enhanced_head = False

    if os.path.exists(os.path.join(model_dir, "enhanced_head_config.json")):
        has_enhanced_head = True
    elif os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        is_lora = True
    else:
        try:
            from transformers.utils.hub import cached_file
            st_file = os.path.join(model_dir, "model.safetensors") if os.path.isdir(model_dir) else cached_file(model_dir, "model.safetensors")
            if st_file and os.path.exists(st_file):
                from safetensors import safe_open
                with safe_open(st_file, framework="pt") as f:
                    st_keys = f.keys()
                    if any("head.layer_weights" in k or "head.dense" in k for k in st_keys):
                        has_enhanced_head = True
        except Exception:
            pass
        if not is_lora:
            try:
                from huggingface_hub import file_exists
                if file_exists(repo_id=model_dir, filename="adapter_config.json"):
                    is_lora = True
            except Exception:
                pass

    if has_enhanced_head:
        print("Detected Enhanced Head model checkpoint. Loading with EnhancedTokenClassifierModel...")
        from src.model.head import EnhancedTokenClassifierModel
        model = EnhancedTokenClassifierModel.from_pretrained(
            model_dir,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        if hasattr(model.config, "id2label") and model.config.id2label:
            id_to_tag = {int(k): v for k, v in model.config.id2label.items()}
            tag_to_id = {v: k for k, v in id_to_tag.items()}
    elif is_lora:
        print("Detected PEFT/LoRA adapter model.")
        from peft import PeftModel
        detected_base = base_model_name or "jhu-clsp/mmBERT-base"
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            try:
                with open(os.path.join(model_dir, "adapter_config.json"), "r", encoding="utf-8") as f:
                    detected_base = json.load(f).get("base_model_name_or_path", detected_base)
            except Exception:
                pass
        base_model = AutoModelForTokenClassification.from_pretrained(
            detected_base,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print("Loading full fine-tuned model checkpoint...")
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        # If model config has id2label, sync mappings
        if hasattr(model.config, "id2label") and model.config.id2label:
            id_to_tag = {int(k): v for k, v in model.config.id2label.items()}
            tag_to_id = {v: k for k, v in id_to_tag.items()}

    model.to(device)
    model.eval()
    print(f"Model successfully loaded with {len(id_to_tag)} labels.")
    return model, tokenizer, id_to_tag, device

def run_inference(
    model_dir: str = "./results",
    base_model_name: str = None,
    text: str = None,
    file_path: str = None,
    output_path: str = None,
    max_length: int = 1024,
    stride: int = 256
):
    model, tokenizer, id_to_tag, device = load_inference_model(model_dir, base_model_name)

    if file_path:
        p = Path(file_path)
        if not p.exists():
            print(f"Error: Input file '{file_path}' does not exist.")
            sys.exit(1)
        print(f"\nReading input from file: {file_path}")
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        res = predict_text(content, model, tokenizer, id_to_tag, device, max_length, stride)
        print_result(res)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            elif out_p.suffix.lower() in [".xml", ".html"]:
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(res["xml_content"])
            else:
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(res["xml_content"])
            print(f"\nSaved inference output to: '{output_path}'")

    elif text:
        print(f"\nRunning inference on custom text string:")
        print("-" * 50)
        print(text)
        print("-" * 50)
        res = predict_text(text, model, tokenizer, id_to_tag, device, max_length, stride)
        print_result(res)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".json":
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
            else:
                with open(out_p, "w", encoding="utf-8") as f:
                    f.write(res["xml_content"])
            print(f"\nSaved inference output to: '{output_path}'")

    else:
        print("\nNo custom input provided. Running evaluation on built-in test samples...")
        for i, sample in enumerate(TEST_SAMPLES):
            print(f"\n=======================================================")
            print(f"SAMPLE {i+1}: {sample['type']}")
            print("=======================================================")
            res = predict_text(sample["text"], model, tokenizer, id_to_tag, device, max_length, stride)
            print_result(res)

def main():
    parser = argparse.ArgumentParser(description="Exam Document Sequence Labeling Inference")
    parser.add_argument("-m", "--model-dir", type=str, default="./results", help="Path to model checkpoint or HF repo ID (default: './results' with fallback to HF)")
    parser.add_argument("--base-model-name", type=str, default="jhu-clsp/mmBERT-base", help="Base model identifier")
    parser.add_argument("-t", "--text", type=str, default=None, help="Direct text input string to segment")
    parser.add_argument("-f", "--file", type=str, default=None, help="Input text file path to segment")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path (.json, .xml, or .txt)")
    parser.add_argument("--max-length", type=int, default=1024, help="Sliding window token length (default: 1024)")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride (default: 256)")

    args = parser.parse_args()
    run_inference(
        model_dir=args.model_dir,
        base_model_name=args.base_model_name,
        text=args.text,
        file_path=args.file,
        output_path=args.output,
        max_length=args.max_length,
        stride=args.stride
    )

if __name__ == "__main__":
    main()
