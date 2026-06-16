import json
from pathlib import Path

def inspect_spans():
    json_path = Path("output/real_exams/real_exam_bc27a60a.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    spans = data["spans"]
    raw_text = data["raw_text"]

    q_spans = [s for s in spans if s["label"] == "question_label"]
    print(f"Total Questions detected: {len(q_spans)}")

    output_lines = [f"Total Questions detected: {len(q_spans)}"]

    for idx, q in enumerate(q_spans):
        start = q["start"]
        end = q_spans[idx+1]["start"] if idx + 1 < len(q_spans) else len(raw_text)
        
        sub_spans = [s for s in spans if s["start"] >= start and s["end"] <= end]
        
        span_summary = []
        for s in sub_spans:
            text_preview = s["text"].replace("\n", " ")
            if len(text_preview) > 30:
                text_preview = text_preview[:27] + "..."
            span_summary.append(f"{s['label']}({repr(text_preview)})")
            
        output_lines.append(f"Q {idx+1:02d} (starts at {start:04d}): {' | '.join(span_summary)}")

    output_path = Path("scratch/exam_span_inspection.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Inspection written to {output_path}")

if __name__ == "__main__":
    inspect_spans()
