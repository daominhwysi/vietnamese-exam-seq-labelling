import os
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    model_dir = "./results"
    
    # 1. Load label mappings
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id_to_tag = {int(k): v for k, v in mapping["id_to_tag"].items()}
    else:
        # fallback
        id_to_tag = {
            0: "O",
            1: "B-question_label", 2: "I-question_label",
            3: "B-stem", 4: "I-stem",
            5: "B-option_label", 6: "I-option_label",
            7: "B-option_text", 8: "I-option_text",
            9: "B-context", 10: "I-context"
        }
        
    print(f"Loaded label mapping: {id_to_tag}")
    
    # 2. Load Tokenizer
    print("Loading tokenizer from local results directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # 3. Load Model
    print("Loading model/adapter from local results directory...")
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if is_lora:
        # Read base model from config
        with open(os.path.join(model_dir, "adapter_config.json"), "r") as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "aisingapore/SEA-LION-ModernBERT-300M")
        print(f"Detected LoRA adapter. Loading base model: {base_model_name}")
        from peft import PeftModel
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=len(id_to_tag),
            id2label=id_to_tag,
            label2id={v: k for k, v in id_to_tag.items()}
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        print("Loading full fine-tuned model...")
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
    model.eval()
    
    # Test text (use raw string to prevent \n in \neq from becoming a literal newline)
    text = r"Câu 1. Cho $f(x) = ax^2 + bx + c \:(a \neq 0)$. Điều kiện để f(x) >= 0 là chọn đáp án A."
    print(f"\nTest text: {text}")
    
    # Pre-process LaTeX equations in text (replace with '[LATEX]') using DOTALL flag
    processed_text = re.sub(r'\$\$.*?\$\$|\$.*?\$', '[LATEX]', text, flags=re.DOTALL)
    print(f"Processed text: {processed_text}")
    
    # Tokenize
    inputs = tokenizer(processed_text, return_tensors="pt")
    input_ids = inputs["input_ids"][0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    preds = torch.argmax(outputs.logits, dim=-1)[0].numpy()
    
    print("\nModel Predictions:")
    for t, p in zip(tokens, preds):
        label = id_to_tag[p]
        print(f"Token: {t:20} -> Predicted Label: {label}")

if __name__ == "__main__":
    main()
