#!/usr/bin/env python3
import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel

# Bypass torchao compatibility check bug on older pre-installed versions in Google Colab
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

# Hardcoded test samples of different question types
TEST_SAMPLES = [
    {
        "type": "multiple_choice",
        "text": "Câu 1: Trong phòng thí nghiệm, khí oxi ($O_2$) được điều chế bằng cách nhiệt phân chất nào sau đây?\nA. $KMnO_4$\nB. $NaCl$\nC. $CaCO_3$\nD. $H_2O$"
    },
    {
        "type": "true_false",
        "text": "Câu 2: Cho các phát biểu sau về các kim loại kiềm:\na. Kim loại kiềm có nhiệt độ nóng chảy thấp và độ cứng nhỏ.\nb. Trong tự nhiên, kim loại kiềm tồn tại ở cả dạng đơn chất và hợp chất.\nc. Các kim loại kiềm đều được bảo quản bằng cách ngâm trong dầu hỏa.\nd. Tất cả các kim loại kiềm đều phản ứng mãnh liệt với nước ở nhiệt độ thường."
    },
    {
        "type": "ordering",
        "text": "Câu 3: Hãy sắp xếp trình tự đúng các bước xác định tiêu cự của thấu kính hội tụ:\n1. Đặt thấu kính và màn ảnh trên giá quang học thẳng hàng.\n2. Bật đèn chiếu sáng nguồn sáng hướng vào thấu kính.\n3. Di chuyển màn ảnh từ từ để nhận được ảnh rõ nét trên màn.\n4. Đo khoảng cách từ thấu kính đến màn ảnh và ghi nhận tiêu cự."
    }
]

def load_label_mapping(model_dir):
    # Try loading from local config if copied there, or default label list
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            return mapping["tag_to_id"], {int(k): v for k, v in mapping["id_to_tag"].items()}
            
    # Fallback to standard base tags mapping
    base_tags = ["question_label", "stem", "option_label", "option_text", "context", "ordering_item_label", "ordering_item_text"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag

def run_inference(model_dir="./results", base_model_name="FacebookAI/xlm-roberta-base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load tokenizer
    print(f"Loading tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Check if latex token is needed
    tokenizer.add_special_tokens({"additional_special_tokens": ["[LATEX]"]})
    
    # 2. Load label mapping
    tag_to_id, id_to_tag = load_label_mapping(model_dir)
    print(f"Loaded {len(tag_to_id)} labels.")
    
    # 3. Load base model and wrap with LoRA adapters
    print(f"Loading base model: {base_model_name}...")
    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapters from: {model_dir}...")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()
    
    print("\n" + "="*50)
    print("RUNNING PREDICTIONS ON SAMPLES")
    print("="*50)
    
    for i, sample in enumerate(TEST_SAMPLES):
        text = sample["text"]
        print(f"\n--- Sample {i+1} ({sample['type'].upper()}) ---")
        print("Raw text:")
        print(text)
        print("-" * 30)
        
        # Pre-process LaTeX equations in text if your training dataset had it
        # (This replacement is optional and depends on whether you used --latex-placeholder in prepare_dataset.py)
        import re
        processed_text = re.sub(r'\$[^$]+\$', '[LATEX]', text)
        
        # Tokenize
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        input_ids = inputs["input_ids"][0].cpu().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Align predictions and print labelled spans
        current_span_label = None
        current_span_tokens = []
        
        print("Model Predicted Segments:")
        for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
            # Skip special tokens
            if token in ["<s>", "</s>", "<pad>"]:
                continue
                
            label = id_to_tag[pred_id]
            clean_token = token.replace(" ", " ").replace("▁", "")  # cleanup sentencepiece character
            
            # Simple boundary alignment
            if label.startswith("B-"):
                if current_span_label:
                    print(f"  [{current_span_label}]: {''.join(current_span_tokens).strip()}")
                current_span_label = label[2:]
                current_span_tokens = [clean_token]
            elif label.startswith("I-") and current_span_label == label[2:]:
                current_span_tokens.append(clean_token)
            else:
                if current_span_label:
                    print(f"  [{current_span_label}]: {''.join(current_span_tokens).strip()}")
                    current_span_label = None
                    current_span_tokens = []
                if label == "O":
                    # Optionally track non-entity tokens or skip printing O to keep it clean
                    pass
        if current_span_label:
            print(f"  [{current_span_label}]: {''.join(current_span_tokens).strip()}")

if __name__ == "__main__":
    model_path = "./results"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    run_inference(model_dir=model_path)
