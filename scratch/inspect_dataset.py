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
    
    # Let's decode input_ids using both tokenizers
    tok_xlmr = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
    tok_mmbert = AutoTokenizer.from_pretrained('jhu-clsp/mmbert-base')
    
    for idx, (t, l, i) in enumerate(zip(tokens[:100], labels[:100], input_ids[:100])):
        t_decoded_xlmr = tok_xlmr.decode([i])
        t_decoded_mmbert = tok_mmbert.decode([i])
        print(f"Index {idx:3} | Token: {t!r:15} | Label ID: {l:4} | ID: {i:6} | Decoded XLM-R: {t_decoded_xlmr!r:10} | Decoded mmBERT: {t_decoded_mmbert!r:10}")

if __name__ == "__main__":
    main()
