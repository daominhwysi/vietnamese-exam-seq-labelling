import os
import sys
import json
import glob
from collections import Counter

# Set standard output encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def inspect_exams():
    print("=== INSPECTING EXAMS ===")
    exam_files = glob.glob("output/exams/exam_*.json")
    print(f"Total exam files found: {len(exam_files)}")
    
    subjects = Counter()
    grades = Counter()
    english_exams = []
    
    for f in exam_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) >= 3:
            g_idx = -1
            for idx, part in enumerate(parts):
                if part.startswith("g") and part[1:].isdigit():
                    g_idx = idx
                    break
            if g_idx != -1:
                subject = "_".join(parts[1:g_idx])
                grade = parts[g_idx][1:]
                subjects[subject] += 1
                grades[grade] += 1
                if subject == "english":
                    english_exams.append(f)
                    
    print("\nBreakdown by Subject:")
    for sub, count in sorted(subjects.items()):
        print(f"  {sub}: {count} exams")
        
    print("\nBreakdown by Grade:")
    for grade, count in sorted(grades.items()):
        print(f"  Grade {grade}: {count} exams")
        
    if english_exams:
        print(f"\nFound {len(english_exams)} English exams.")
        # Inspect the latest English exam
        latest_exam = max(english_exams, key=os.path.getmtime)
        print(f"Inspecting latest English exam: {os.path.basename(latest_exam)}")
        with open(latest_exam, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Count sections and questions
        print(f"  Subject: {data.get('subject')}")
        print(f"  Grade: {data.get('grade')}")
        sections = data.get("sections", {})
        print(f"  Number of sections: {len(sections)}")
        
        total_questions = 0
        q_types = Counter()
        problem_types = Counter()
        
        for idx, (sec_name, sec_qs) in enumerate(sections.items()):
            # count questions
            total_questions += len(sec_qs)
            for q in sec_qs:
                q_type = q.get("question_type")
                if q_type:
                    q_types[q_type] += 1
                else:
                    if q.get("is_group"):
                        q_types["group"] += 1
                    else:
                        q_types["standard"] += 1
                        
                prob_type = q.get("problem_type_id")
                if prob_type:
                    problem_types[prob_type] += 1
                    
            print(f"    Section {idx+1} ({sec_name[:60]}...): {len(sec_qs)} questions")
            
        print(f"  Total questions in exam: {total_questions}")
        print("  Question types breakdown:")
        for qt, count in q_types.items():
            print(f"    - {qt}: {count}")
        print("  Problem types breakdown:")
        for pt, count in problem_types.items():
            print(f"    - {pt}: {count}")
            
def inspect_dataset():
    print("\n=== INSPECTING DATASET SPLITS ===")
    dataset_dir = "output/dataset"
    for split in ["train", "val", "test"]:
        stats_file = os.path.join(dataset_dir, f"{split}_stats.json")
        if os.path.exists(stats_file):
            print(f"\nStats for {split} split:")
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        else:
            print(f"Stats file {stats_file} not found.")

    # Load label mapping
    mapping_file = os.path.join(dataset_dir, "label_mapping.json")
    id_to_tag = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            # mapping["id_to_tag"] has string keys
            id_to_tag = {int(k): v for k, v in mapping.get("id_to_tag", {}).items()}
            
    train_jsonl = os.path.join(dataset_dir, "train.jsonl")
    if os.path.exists(train_jsonl):
        print(f"\nChecking first 3 lines of {train_jsonl}...")
        count = 0
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                metadata = data.get("metadata", {})
                tokens = data.get("tokens", [])
                labels = data.get("labels", [])
                print(f"\nSample {count+1}:")
                print(f"  Subject: {metadata.get('subject')}, Grade: {metadata.get('grade')}")
                print(f"  Tokens count: {len(tokens)}")
                print(f"  Labels count: {len(labels)}")
                
                # Check labels distribution
                label_counts = Counter(labels)
                print(f"  Unique labels count:")
                for lid, cnt in sorted(label_counts.items()):
                    tag_name = id_to_tag.get(lid, f"UNKNOWN_{lid}") if lid != -100 else "IGNORE"
                    print(f"    - {lid} ({tag_name}): {cnt}")
                
                # Print sample text sequence with label tags
                sample_pairs = []
                for t, l in zip(tokens[:100], labels[:100]):
                    # replace sentencepiece space symbol with a readable underscore or space
                    t_clean = t.replace("\u2581", " ")
                    tag_name = id_to_tag.get(l, "O") if l != -100 else "IGN"
                    # clean up tags for brevity, e.g. B-stem -> B-st, I-question_label -> I-ql
                    tag_short = tag_name.replace("question_label", "ql").replace("option_label", "ol").replace("option_text", "ot").replace("context", "cx").replace("stem", "st")
                    sample_pairs.append(f"{t_clean}({tag_short})")
                print(f"  First 100 tokens with labels:")
                print(f"    {' '.join(sample_pairs)}")
                
                count += 1
                if count >= 3:
                    break
    else:
        print(f"Dataset file {train_jsonl} not found.")

if __name__ == "__main__":
    inspect_exams()
    inspect_dataset()
