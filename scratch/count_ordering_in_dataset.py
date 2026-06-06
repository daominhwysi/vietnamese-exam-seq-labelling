import os
import json
from collections import Counter

def count_types(split):
    jsonl_path = f"output/dataset/{split}.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"File {jsonl_path} does not exist.")
        return
        
    counts = Counter()
    subject_counts = Counter()
    ordering_by_subject = Counter()
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            metadata = data.get("metadata", {})
            subject = metadata.get("subject", "unknown")
            prob_type = metadata.get("problem_type_id", "unknown")
            
            subject_counts[subject] += 1
            counts[prob_type] += 1
            if "order" in prob_type or "reorder" in prob_type:
                ordering_by_subject[subject] += 1
                
    print(f"\n--- {split.upper()} SPLIT DETAILS ---")
    print(f"Total samples: {sum(subject_counts.values())}")
    print("Samples per subject:")
    for sub, cnt in sorted(subject_counts.items()):
        print(f"  {sub}: {cnt}")
        
    print("Samples per problem type:")
    for pt, cnt in sorted(counts.items()):
        print(f"  {pt}: {cnt}")
        
    print("Ordering samples per subject:")
    for sub, cnt in sorted(ordering_by_subject.items()):
        print(f"  {sub}: {cnt}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        count_types(split)
