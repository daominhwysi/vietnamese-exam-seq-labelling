import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("Loading dataset from HF Hub...")
    dataset = load_dataset("daominhwysi/synthetic-seq-labelling-vi-exam-v2", split="train")
    sample = dataset[0]
    
    print("\n--- Metadata ---")
    print(sample["metadata"])
    
    print("\n--- Tokens and Labels (First 100) ---")
    tokens = sample["tokens"]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    
    # Let's decode input_ids using the tokenizer
    tok_sea_lion = AutoTokenizer.from_pretrained('aisingapore/SEA-LION-ModernBERT-300M')
    
    for idx, (t, l, i) in enumerate(zip(tokens[:100], labels[:100], input_ids[:100])):
        t_decoded = tok_sea_lion.decode([i])
        print(f"Index {idx:3} | Token: {t!r:15} | Label ID: {l:4} | ID: {i:6} | Decoded: {t_decoded!r:10}")

if __name__ == "__main__":
    main()
