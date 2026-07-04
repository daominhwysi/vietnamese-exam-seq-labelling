import os
import json
import threading
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

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
                if device_choice == "auto":
                    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
                else:
                    device = device_choice
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
                    special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
                    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

                tag_to_id, id_to_tag = self._load_label_mapping(model_path)

                onnx_file = None
                local_onnx_options = [
                    os.path.join(model_path, "model.onnx"),
                    os.path.join(model_path, "onnx", "model.onnx")
                ]
                for opt in local_onnx_options:
                    if os.path.exists(opt):
                        onnx_file = opt
                        break

                is_hf_repo = "/" in model_path and not model_path.startswith((".", "/", "\\")) and not os.path.exists(model_path)
                if not onnx_file and is_hf_repo:
                    try:
                        from huggingface_hub import snapshot_download
                        import shutil
                        
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
                        downloaded_dir = snapshot_download(
                            repo_id=model_path,
                            allow_patterns=["onnx/*", "label_mapping.json"],
                            token=token
                        )
                        onnx_folder = os.path.join(downloaded_dir, "onnx")
                        if os.path.exists(onnx_folder):
                            root_mapping = os.path.join(downloaded_dir, "label_mapping.json")
                            dest_mapping = os.path.join(onnx_folder, "label_mapping.json")
                            if os.path.exists(root_mapping) and not os.path.exists(dest_mapping):
                                shutil.copy(root_mapping, dest_mapping)
                            
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
                    is_lora = False
                    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                        is_lora = True
                    else:
                        try:
                            from huggingface_hub import file_exists
                            is_lora = file_exists(repo_id=model_path, filename="adapter_config.json")
                        except Exception:
                            is_lora = (model_path != base_model_name)

                    if is_lora:
                        detected_base = base_model_name
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
        mapping_path = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}

        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=model_dir, filename="label_mapping.json")
            with open(downloaded, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
        except Exception:
            pass

        base_tags = ["question_label", "stem", "option_label", "option_text", "context", "section", "explanation"]
        tag_to_id = {"O": 0}
        for tag in base_tags:
            tag_to_id[f"B-{tag}"] = len(tag_to_id)
            tag_to_id[f"I-{tag}"] = len(tag_to_id)
        id_to_tag = {v: k for k, v in tag_to_id.items()}
        return tag_to_id, id_to_tag

# Global instance
model_manager = ModelManager()

