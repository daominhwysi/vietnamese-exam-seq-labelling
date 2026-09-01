import os
import re
import json
import threading
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Global instance of the ModelManager
model_manager_lock = threading.Lock()

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.base_model_name = None
        self.device = None
        self.tag_to_id = None
        self.id_to_tag = None
        self.is_onnx = False
        
        self.status = "idle"  # idle, loading, loaded, error
        self.error_message = ""
        self.load_lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_path": self.model_path,
            "base_model_name": self.base_model_name,
            "device": self.device,
            "error_message": self.error_message,
            "labels": list(self.tag_to_id.keys()) if self.tag_to_id else []
        }

    def load_model_in_background(self, model_path: str, base_model_name: str, device_choice: str):
        thread = threading.Thread(
            target=self.load_model,
            args=(model_path, base_model_name, device_choice)
        )
        thread.daemon = True
        thread.start()

    def load_model(self, model_path: str, base_model_name: str, device_choice: str):
        with self.load_lock:
            if self.model_path == model_path and self.base_model_name == base_model_name and self.model is not None:
                self.status = "loaded"
                self.error_message = ""
                return
            
            self.status = "loading"
            self.error_message = ""
            
            try:
                # 1. Determine device
                if device_choice == "auto":
                    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
                else:
                    device = device_choice
                
                # 2. Load tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
                    special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
                    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

                # 3. Load label mapping
                tag_to_id, id_to_tag = self._load_label_mapping(model_path)

                # 4. Determine if ONNX or PyTorch/LoRA model
                onnx_file = None
                local_onnx_options = [
                    os.path.join(model_path, "model.onnx"),
                    os.path.join(model_path, "onnx", "model.onnx")
                ]
                for opt in local_onnx_options:
                    if os.path.exists(opt):
                        onnx_file = opt
                        break

                # If not found locally, check if it's a Hugging Face repository (contains '/' and is not a local path)
                is_hf_repo = "/" in model_path and not model_path.startswith((".", "/", "\\")) and not os.path.exists(model_path)
                if not onnx_file and is_hf_repo:
                    try:
                        from huggingface_hub import snapshot_download
                        import shutil
                        
                        # Resolve HF token
                        token = os.getenv("HF_TOKEN")
                        if not token:
                            try:
                                from dotenv import load_dotenv
                                from pathlib import Path
                                load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
                                token = os.getenv("HF_TOKEN")
                            except Exception:
                                pass

                        print(f"HF_TOKEN found in environment: {'Yes' if token else 'No'}")
                        print(f"Downloading ONNX model and configuration files from HF Hub '{model_path}'...")
                        # Download only the onnx/ folder and label_mapping.json to save bandwidth
                        downloaded_dir = snapshot_download(
                            repo_id=model_path,
                            allow_patterns=["onnx/*", "label_mapping.json"],
                            token=token
                        )
                        onnx_folder = os.path.join(downloaded_dir, "onnx")
                        if os.path.exists(onnx_folder):
                            # Copy label_mapping.json into the onnx directory if downloaded at root
                            root_mapping = os.path.join(downloaded_dir, "label_mapping.json")
                            dest_mapping = os.path.join(onnx_folder, "label_mapping.json")
                            if os.path.exists(root_mapping) and not os.path.exists(dest_mapping):
                                shutil.copy(root_mapping, dest_mapping)
                            
                            # Redirect model_path to the local downloaded ONNX folder
                            model_path = onnx_folder
                            onnx_file = os.path.join(model_path, "model.onnx")
                            print(f"Successfully downloaded ONNX model package to: {model_path}")
                    except Exception as hf_err:
                        print(f"Could not download ONNX folder from Hugging Face Hub: {hf_err}")

                if onnx_file:
                    print(f"ONNX model detected at: {onnx_file}")
                    import onnxruntime as ort
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
                    model = ort.InferenceSession(onnx_file, providers=providers)
                    self.is_onnx = True
                else:
                    if not HAS_TORCH:
                        raise ImportError("PyTorch ('torch') is not installed in the current environment. Please export/load an ONNX model instead, or install torch.")
                    self.is_onnx = False
                    # Determine if LoRA adapter
                    is_lora = False
                    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                        is_lora = True
                    else:
                        try:
                            from huggingface_hub import file_exists
                            is_lora = file_exists(repo_id=model_path, filename="adapter_config.json")
                        except Exception:
                            is_lora = (model_path != base_model_name)

                    # 5. Load PyTorch model (Check for Enhanced Head vs. LoRA vs. full fine-tune)
                    has_enhanced = False
                    if os.path.exists(os.path.join(model_path, "enhanced_head_config.json")):
                        has_enhanced = True
                    else:
                        try:
                            from huggingface_hub import file_exists
                            has_enhanced = file_exists(repo_id=model_path, filename="enhanced_head_config.json")
                        except Exception:
                            pass

                    if has_enhanced:
                        print(f"Loading Enhanced Head PyTorch model from: {model_path}...")
                        from src.model.head import EnhancedTokenClassifierModel
                        model = EnhancedTokenClassifierModel.from_pretrained(
                            model_path,
                            num_labels=len(tag_to_id),
                            id2label=id_to_tag,
                            label2id=tag_to_id
                        )
                    elif is_lora:
                        detected_base = base_model_name
                        # Read base model from config if possible
                        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                            try:
                                with open(os.path.join(model_path, "adapter_config.json"), "r", encoding="utf-8") as f:
                                    cfg = json.load(f)
                                    detected_base = cfg.get("base_model_name_or_path", detected_base)
                            except Exception:
                                pass
                        else:
                            try:
                                from huggingface_hub import hf_hub_download
                                config_file = hf_hub_download(repo_id=model_path, filename="adapter_config.json")
                                with open(config_file, "r", encoding="utf-8") as f:
                                    cfg = json.load(f)
                                    detected_base = cfg.get("base_model_name_or_path", detected_base)
                            except Exception:
                                pass

                        base_model = AutoModelForTokenClassification.from_pretrained(
                            detected_base,
                            num_labels=len(tag_to_id),
                            id2label=id_to_tag,
                            label2id=tag_to_id
                        )
                        base_model.resize_token_embeddings(len(tokenizer))
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(base_model, model_path)
                    else:
                        model = AutoModelForTokenClassification.from_pretrained(model_path)

                    model.to(device)
                    model.eval()

                # Save to manager attributes
                self.model = model
                self.tokenizer = tokenizer
                self.model_path = model_path
                self.base_model_name = base_model_name
                self.device = device
                self.tag_to_id = tag_to_id
                self.id_to_tag = id_to_tag
                
                self.status = "loaded"
                self.error_message = ""
                
            except Exception as e:
                self.status = "error"
                self.error_message = str(e)
                import traceback
                traceback.print_exc()

    def _load_label_mapping(self, model_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Try local folder
        mapping_path = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}

        # Try Hugging Face Hub
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=model_dir, filename="label_mapping.json")
            with open(downloaded, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
        except Exception:
            pass

        # Default fallback standard tag mapping
        base_tags = ["question_label", "stem", "option_label", "option_text", "stimulus", "section", "explanation"]
        tag_to_id = {"O": 0}
        for tag in base_tags:
            tag_to_id[f"B-{tag}"] = len(tag_to_id)
            tag_to_id[f"I-{tag}"] = len(tag_to_id)
        id_to_tag = {v: k for k, v in tag_to_id.items()}
        return tag_to_id, id_to_tag

# Global model manager
model_manager = ModelManager()

def is_valid_latex(content: str) -> bool:
    content_stripped = content.strip()
    if not content_stripped:
        return False
    
    # Case 1: Single variable or number (e.g. $x$, $a$, $1$)
    if len(content_stripped) == 1:
        return content_stripped.isalnum()
        
    # Case 2: Mathematical interval notation like (-\infty; 0] or [8; +\infty) or (1; 2]
    if re.match(r'^[\[\(][^\[\]\(\)]+[\]\)]$', content_stripped) and (';' in content_stripped or ',' in content_stripped):
        return True

    # Case 3: Balanced braces, brackets, and parentheses
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
        
    # Case 4: Contains standard math/latex character indicators
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
        
    # Case 5: Short alphanumeric math terms without spaces (e.g. $2a$, $x1$, $100$)
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True
        
    return False

def get_latex_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    # 1. Matches $$...$$ (display math)
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        # Verify content inside $$
        content = match.group(0)[2:-2]
        if is_valid_latex(content):
            spans.append(match.span())
        
    # 2. Matches $...$ (inline math)
    for match in re.finditer(r"\$(?!\s)[^\$\n]+?(?<!\s)\$", text):
        span = match.span()
        content = match.group(0)[1:-1]
        if not is_valid_latex(content):
            continue
            
        # Avoid overlapping with display math
        overlap = False
        for d_start, d_end in spans:
            if not (span[1] <= d_start or span[0] >= d_end):
                overlap = True
                break
        if not overlap:
            spans.append(span)
            
    spans.sort(key=lambda x: x[0])
    return spans

def run_model_inference(
    raw_text: str,
    max_length: int = 1024,
    stride: int = 256
) -> Dict[str, Any]:
    """
    Main inference worker. Tokenizes raw text with sliding window, aggregates overlapping
    logits, recovers character offsets, maps spans, and outputs structures.
    """
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("No model loaded. Please load a model first via /api/load-model.")

    # Normalize line endings to LF (\n) to match training data and standard file reading behavior
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    device = model_manager.device
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    id_to_tag = model_manager.id_to_tag

    # 1. LaTeX replacement
    latex_spans = get_latex_spans(raw_text)
    processed_text = ""
    last_idx = 0
    for start, end in latex_spans:
        processed_text += raw_text[last_idx:start] + "[LATEX]"
        last_idx = end
    processed_text += raw_text[last_idx:]

    # Offset mapper helper back to original raw_text
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

    # 2. Tokenize sliding window
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
        
        # Map modified offsets back to original raw text
        chunk_offsets = []
        for start, end in chunk_mod_offsets:
            if start == 0 and end == 0:
                chunk_offsets.append((0, 0))
            else:
                chunk_offsets.append((map_idx(start), map_idx(end)))
        
        # Model evaluation / inference
        if model_manager.is_onnx:
            import numpy as np
            onnx_inputs = {
                "input_ids": np.array([chunk_input_ids], dtype=np.int64),
                "attention_mask": np.array([chunk_attention_mask], dtype=np.int64)
            }
            outputs = model.run(["logits"], onnx_inputs)
            chunk_logits = outputs[0][0]
        else:
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

    # 3. Average overlapping logits and generate sequence predictions
    sorted_spans = sorted(span_logits.keys(), key=lambda x: (x[0], x[1]))
    
    predictions = []
    offsets = []
    tokens = []
    attention_mask = []
    
    for span in sorted_spans:
        start, end = span
        if model_manager.is_onnx:
            import numpy as np
            avg_logits = np.mean(np.stack(span_logits[span]), axis=0)
            pred_id = int(np.argmax(avg_logits))
        else:
            avg_logits = torch.mean(torch.stack(span_logits[span]), dim=0)
            pred_id = torch.argmax(avg_logits).item()
        
        predictions.append(pred_id)
        offsets.append(span)
        tokens.append(span_tokens[span])
        attention_mask.append(1)

    # 4. Extract contiguous labeled spans from tokens
    segments = []
    current_label = None
    current_start = -1
    current_end = -1
    
    for idx, (pred_id, offset) in enumerate(zip(predictions, offsets)):
        start, end = offset
        label = id_to_tag[pred_id]
        
        if label.startswith("B-"):
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
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
                        "text": raw_text[current_start:current_end]
                    })
                # Auto-promote orphaned I-tag to start a new segment
                current_label = tag_name
                current_start = start
                current_end = end
        else:  # O tag
            if current_label and current_start < current_end:
                segments.append({
                    "label": current_label,
                    "start": current_start,
                    "end": current_end,
                    "text": raw_text[current_start:current_end]
                })
                current_label = None
                current_start = -1
                current_end = -1
                
    if current_label and current_start < current_end:
        segments.append({
            "label": current_label,
            "start": current_start,
            "end": current_end,
            "text": raw_text[current_start:current_end]
        })

    # Filter empty or whitespace-only segments
    filtered_spans = []
    for seg in segments:
        text_content = seg["text"]
        filtered_spans.append({
            "label": seg["label"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text_content
        })

    # 5. Build inline-tagged XML representation
    xml_content = build_xml(raw_text, filtered_spans)

    # 6. Build structured JSON representation of the exam questions
    structured_exam = parse_segments_to_questions(filtered_spans)

    # 7. Format details of each token
    token_details = []
    for token, pred_id, offset in zip(tokens, predictions, offsets):
        start, end = offset
        tag = id_to_tag[pred_id]
        readable_token = token.replace(" ", " ").replace("▁", "")
        token_details.append({
            "token": readable_token,
            "tag": tag,
            "start": start,
            "end": end
        })

    return {
        "raw_text": raw_text,
        "spans": filtered_spans,
        "xml_content": xml_content,
        "structured_exam": structured_exam,
        "token_details": token_details
    }

def build_xml(raw_text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Reconstructs the full raw text with inline XML tags around each labeled span.
    Uses sorting to avoid overlapping issues and insert XML tags correctly.
    """
    valid_spans = []
    for span in spans:
        label = span["label"]
        text = span["text"].strip()
        if not text:
            continue
        if label in ["option_label", "question_label"] and not any(c.isalnum() for c in text):
            continue
        valid_spans.append(span)

    sorted_spans = sorted(valid_spans, key=lambda x: (x["start"], -x["end"]))
    
    result = []
    cursor = 0
    
    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]
        text = span["text"]
        
        if start < cursor:
            continue
            
        if start > cursor:
            result.append(raw_text[cursor:start])
            
        result.append(f"<{label}>{text}</{label}>")
        cursor = end
        
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])
        
    return "".join(result)

def parse_segments_to_questions(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Heuristic parser that groups sequence labeling spans into question structures.
    Recognizes question labels, contexts, stems, option labels, and option texts.
    """
    questions = []
    current_context = ""
    current_question = None
    
    for span in spans:
        label = span["label"]
        text = span["text"].strip()
        if not text:
            continue
        if label in ["option_label", "question_label"] and not any(c.isalnum() for c in text):
            continue
            
        if label in ["stimulus", "context"]:
            current_context = text
            
        elif label == "question_label":
            if current_question:
                questions.append(current_question)
            current_question = {
                "question_label": text,
                "stimulus": current_context,
                "context": current_context,
                "stem": "",
                "options": [],
                "current_option_label": None
            }
            
        elif label == "stem":
            if current_question is None:
                current_question = {
                    "question_label": "",
                    "context": current_context,
                    "stem": text,
                    "options": [],
                    "current_option_label": None
                }
            else:
                if current_question["stem"]:
                    current_question["stem"] += "\n" + text
                else:
                    current_question["stem"] = text
                    
        elif label == "option_label":
            if current_question:
                current_question["current_option_label"] = text
                
        elif label == "option_text":
            if current_question:
                opt_lbl = current_question.get("current_option_label", "")
                prefix = f"{opt_lbl} " if opt_lbl else ""
                current_question["options"].append({
                    "label": opt_lbl,
                    "text": text,
                    "full": prefix + text
                })
                current_question["current_option_label"] = None
                
        elif label == "section":
            current_context = ""
            
    if current_question:
        questions.append(current_question)
        
    cleaned_questions = []
    for q in questions:
        q_label = q["question_label"]
        stem = q["stem"]
        stimulus = q.get("stimulus") or q.get("context", "")

        # Fallback: Recover question label if it leaked into stem or was unclassified
        if not q_label and stem:
            prefix_match = re.match(r'^(?:\*\*)?(?:Question|Câu|Bài|Q)\s*\d+[\s:.)-]*(?:\*\*)?\s*', stem, re.IGNORECASE)
            if prefix_match:
                q_label = prefix_match.group(0).strip()
                stem = stem[prefix_match.end():].strip()

        cleaned_options = [opt["full"] for opt in q["options"]]
        cleaned_questions.append({
            "label": q_label,
            "stimulus": stimulus,
            "context": stimulus,
            "stem": stem,
            "options": cleaned_options,
            "raw_options": q["options"]
        })
        
    return cleaned_questions
