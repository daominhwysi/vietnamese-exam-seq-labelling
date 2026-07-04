import os
import hashlib
from pathlib import Path

input_path = Path("real_data_annotator/out")
for root, _, files in os.walk(input_path):
    for f in files:
        if f.endswith(".md"):
            fp = Path(root) / f
            rel = str(fp.relative_to(input_path))
            h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
            if h == "90871cd4":
                print(f"Matched file: {fp}")
                print(f"Relative path: {rel}")
