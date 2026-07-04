import json
from pathlib import Path
from typing import Dict, Any

DATASET_DIR = Path("output/dataset")

def load_or_compute_dataset_stats(split: str) -> Dict[str, Any]:
    stats_file = DATASET_DIR / f"{split}_stats.json"
    filepath = DATASET_DIR / f"{split}.jsonl"
    if not filepath.exists():
        return {}
        
    if stats_file.exists() and stats_file.stat().st_mtime > filepath.stat().st_mtime:
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    stats = {
        "split": split,
        "total_samples": 0,
        "subjects": {},
        "grades": {},
        "tag_counts": {},
        "file_size_mb": round(filepath.stat().st_size / (1024 * 1024), 2)
    }
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
            except Exception:
                continue
            stats["total_samples"] += 1
            
            metadata = sample.get("metadata", {})
            subj = metadata.get("subject", "unknown")
            grade = metadata.get("grade", "unknown")
            
            stats["subjects"][subj] = stats["subjects"].get(subj, 0) + 1
            try:
                grade_key = int(grade)
            except Exception:
                grade_key = str(grade)
            stats["grades"][str(grade_key)] = stats["grades"].get(str(grade_key), 0) + 1
            
            tags = sample.get("tags", [])
            for tag in tags:
                if tag != "IGNORE" and tag != "O":
                    stats["tag_counts"][tag] = stats["tag_counts"].get(tag, 0) + 1
                    
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to write stats cache for {split}: {e}")
        
    return stats
