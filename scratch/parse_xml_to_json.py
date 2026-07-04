import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add workspace directory to path
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent
sys.path.append(str(workspace_dir))

from src.real_data_annotator.annotate_ocr import parse_xml_annotations

def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_xml_to_json.py <xml_file_path> <relative_input_path>")
        sys.exit(1)

    xml_file = Path(sys.argv[1])
    rel_path = sys.argv[2]

    if not xml_file.exists():
        print(f"Error: XML file {xml_file} does not exist.")
        sys.exit(1)

    with open(xml_file, "r", encoding="utf-8") as f:
        tagged_text = f.read()

    raw_text, spans = parse_xml_annotations(tagged_text)

    path_hash = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]

    result_data = {
        "exam_id": f"real_{path_hash}",
        "created_at": datetime.now().isoformat(),
        "is_real": True,
        "raw_text": raw_text,
        "spans": spans,
        "raw_xml": tagged_text,
        "annotated": True
    }

    output_dir = workspace_dir / "output" / "real-exams"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = output_dir / f"real_exam_{path_hash}.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully parsed and wrote JSON to: {out_json_path.name} ({len(spans)} spans)")

if __name__ == "__main__":
    main()
