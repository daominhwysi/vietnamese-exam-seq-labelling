import hashlib
from pathlib import Path

file_path = Path("d:/project/doc-layout-analysis/sequence-labelling-data-generator/real_data_annotator/out/toan/de-chon-hsg-toan-thpt-nam-2025-2026-truong-chuyen-luong-van-chanh-dak-lak.md")
input_path = Path("d:/project/doc-layout-analysis/sequence-labelling-data-generator/real_data_annotator/out")

rel_sig = str(file_path.relative_to(input_path))
path_hash = hashlib.md5(rel_sig.encode("utf-8")).hexdigest()[:8]
print("Relative path:", rel_sig)
print("Path hash:", path_hash)
