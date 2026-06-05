import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_local_or_hub_sample():
    # Try loading from local output/dataset/train.jsonl first
    local_path = "output/dataset/train.jsonl"
    if os.path.exists(local_path):
        print(f"Loading sample from local file: {local_path}")
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        return json.loads(line)
        except Exception as e:
            print(f"Error reading local file: {e}")
            
    # Try loading from Hugging Face hub
    print("Local train.jsonl not found or failed to load. Loading from Hugging Face Hub...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("daominhwysi/synthetic-seq-labelling-vi-exam-v2", split="train")
        return dataset[0]
    except Exception as e:
        print(f"Error loading from Hub: {e}")
        return None

def main():
    model_dir = "./results"
    
    # 1. Load label mappings
    mapping_path = os.path.join(model_dir, "label_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id_to_tag = {int(k): v for k, v in mapping["id_to_tag"].items()}
    else:
        id_to_tag = {
            0: "O",
            1: "B-question_label", 2: "I-question_label",
            3: "B-stem", 4: "I-stem",
            5: "B-option_label", 6: "I-option_label",
            7: "B-option_text", 8: "I-option_text",
            9: "B-context", 10: "I-context"
        }
        
    # 2. Get training sample
    sample = load_local_or_hub_sample()
    if not sample:
        print("Failed to find any training sample. Exiting.")
        return
        
    print(f"\n--- Found Training Sample from metadata: {sample.get('metadata', {})} ---")
    
    # Extract training tokens and ground truth labels
    train_tokens = sample["tokens"]
    train_labels = sample["labels"]
    
    # 3. Load Tokenizer
    print("\nLoading tokenizer from local results directory...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # 4. Load Model
    print("Loading model/adapter...")
    is_lora = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    if is_lora:
        with open(os.path.join(model_dir, "adapter_config.json"), "r") as f:
            config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "jhu-clsp/mmbert-base")
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
    
    # 5. Run inference using the pre-computed input_ids from the training sample
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    
    # Ensure they are tensors and correct shape (1, seq_len)
    input_ids_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        
    preds = torch.argmax(outputs.logits, dim=-1)[0].numpy()
    
    # 6. Compare predictions and ground truths
    print("\nToken-by-Token Comparison (Ground Truth vs Prediction):")
    print(f"{'Token':25} | {'Ground Truth Tag':20} | {'Predicted Tag':20}")
    print("-" * 72)
    
    for t, true_label_id, pred_label_id in zip(train_tokens[:150], train_labels[:150], preds[:150]):  # Limit to 150 for readability
        true_tag = id_to_tag.get(true_label_id, f"IGNORE ({true_label_id})") if true_label_id != -100 else "IGNORE"
        pred_tag = id_to_tag.get(pred_label_id, "UNKNOWN")
        print(f"{t:25} | {true_tag:20} | {pred_tag:20}")
        
    if len(train_tokens) > 150:
        print(f"\n... [Truncated {len(train_tokens) - 150} tokens for display brevity] ...")

if __name__ == "__main__":
    main()
