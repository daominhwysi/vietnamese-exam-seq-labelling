import os
import sys
import json
import argparse
import tempfile
import subprocess
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

def load_label_mapping(model_dir):
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
            
    base_tags = ["question_label", "stem", "option_label", "option_text", "stimulus", "section"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def main():
    parser = argparse.ArgumentParser(description="Export LoRA Sequence Labeling model to ONNX using HF Optimum.")
    parser.add_argument("--model_dir", type=str, default="./results", help="Directory containing LoRA adapter weights.")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base model name/path. Auto-detected if not specified.")
    parser.add_argument("--output_dir", type=str, default="./output/onnx", help="Directory to save the exported ONNX model.")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    # 1. Auto-detect base model name
    base_model_name = args.base_model_name
    if base_model_name is None:
        adapter_config_path = os.path.join(model_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path")
                    print(f"Auto-detected base model name from adapter config: {base_model_name}")
            except Exception as e:
                print(f"Error reading adapter config: {e}")
        
    if not base_model_name:
        base_model_name = "jhu-clsp/mmBERT-small"
        print(f"Fallback to default base model: {base_model_name}")

    # 2. Load tokenizer correctly (bypassing TokenizersBackend error)
    print(f"Loading tokenizer from base model: {base_model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Re-add custom special tokens
    special_tokens = ["<blank />", "<blank/>", "[BLANK]", "[LATEX]"]
    print(f"Adding special tokens to tokenizer: {special_tokens}")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 3. Load model (Check if it's a LoRA checkpoint or a full fine-tuned checkpoint)
    tag_to_id, id_to_tag = load_label_mapping(model_dir)
    print(f"Loaded {len(tag_to_id)} labels.")
    
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"LoRA adapter detected. Loading base model '{base_model_name}' and wrapping with PEFT...")
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        base_model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(base_model, model_dir)
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
    else:
        print(f"No LoRA adapter config found at {adapter_config_path}.")
        print(f"Loading full fine-tuned model directly from: {model_dir}...")
        merged_model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(tag_to_id),
            id2label=id_to_tag,
            label2id=tag_to_id
        )
        merged_model.resize_token_embeddings(len(tokenizer))
        
    merged_model.eval()

    # 5. Create a temporary folder to save the merged model & tokenizer
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Saving merged model to temporary folder: {tmp_dir}...")
        merged_model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        
        # Copy label_mapping.json to the temp dir to include it with the config
        label_mapping_src = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(label_mapping_src):
            shutil.copy(label_mapping_src, os.path.join(tmp_dir, "label_mapping.json"))

        # 6. Run optimum export using the Python API
        print("Checking if optimum is installed...")
        try:
            from optimum.exporters.onnx import main_export
        except ImportError:
            print("HF Optimum is not installed. Installing optimum[onnxruntime] via pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "optimum[onnxruntime]"], check=True)
            from optimum.exporters.onnx import main_export

        print(f"Exporting model to ONNX at {output_dir} using Optimum Python API...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            main_export(
                model_name_or_path=tmp_dir,
                output=output_dir,
                task="token-classification"
            )
            print(f"Success! Model exported to {output_dir}")
            
            # Also copy label_mapping.json to output directory
            if os.path.exists(label_mapping_src):
                shutil.copy(label_mapping_src, os.path.join(output_dir, "label_mapping.json"))
                print(f"Copied label_mapping.json to {output_dir}")
        except Exception as e:
            print("Export failed!")
            print(e)
            sys.exit(1)

if __name__ == "__main__":
    main()
