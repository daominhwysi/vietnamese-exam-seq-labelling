"""
Tool to audit and repair root data sources directly (both real OCR annotations and synthetic exams).
1. Ensures that all source citations (e.g. '(Adapted from ...)', '(Nguồn: ...)', '(Theo ...)', '(Trích ...)')
   are permanently and cleanly enclosed INSIDE the <stimulus>...</stimulus> XML tags and inside JSON stimulus fields.
2. Audits and corrects <section> tags:
   - Exam part divisions (Phần I, Phần II, Part 1, Chủ đề...) and directions are preserved as <section>.
   - Administrative metadata (Sở GD&ĐT, Trường THPT, Tên đề, Mã đề, Thời gian làm bài, v.v.) and footers (--- HẾT ---) are UNTAGGED (nhãn 'O').
   - Question labels misclassified as <section> (## Câu 4, ## Question 15, ### Bài 1) are CONVERTED to <question_label>.
   - Solutions, answer keys, barem tables misclassified as <section> are CONVERTED to <explanation>.
   - Section tags nested inside <stimulus> are unwrapped.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Standard citation / source attribution regex for English & Vietnamese exams
CITATION_REGEX = r'(?:\*{0,2}\s*[\(\[]\s*(?:Adapted from|Nguồn\s*:|Theo\s+[A-Z0-9a-zÀ-ỹ]|Trích\s*(?::|từ|dẫn)?\s*|Source\s*:)[^\n\r<>]+(?:\([^)<>]*\)[^\n\r<>]*)*[\)\]]\s*\*{0,2}|\*{1,2}\s*(?:Adapted from|Nguồn\s*:|Theo\s+[A-Z0-9a-zÀ-ỹ]|Trích\s*(?::|từ|dẫn)?\s*|Source\s*:)[^\n\r<>]+\*{1,2})'
CITATION_RE = re.compile(rf'^\s*{CITATION_REGEX}\s*$', re.IGNORECASE)

def clean_tag_text(t: str) -> str:
    """Strips markdown header hashes, bold/italic asterisks, and whitespace."""
    clean = re.sub(r'^[#\*\s\-_>\div<>/]+', '', t).strip()
    clean = re.sub(r'[\*\s\-_]+$', '', clean).strip()
    return clean

def fix_xml_sections(xml_content: str) -> str:
    """
    Directly audits and corrects <section> XML tags to strictly adhere to the exam section division definition:
    - Preserves true section divisions (PHẦN I, Part 1, Chủ đề, Chuyên đề, Dạng...) and group directions.
    - Untags administrative metadata (School, Dept, Exam Title, Code, Time, Page count, etc.).
    - Untags footers and end-of-exam markers (--- HẾT ---, proctor notes).
    - Converts misclassified question labels (## Câu 4, ## Question 15) to <question_label>.
    - Converts misclassified solution headers/tables (Hướng dẫn giải, Đáp án, Barem) to <explanation>.
    - Unwraps any section tags nested inside <stimulus>.
    """
    if not xml_content or "<section>" not in xml_content:
        return xml_content

    text = xml_content

    # 1. Unwrap any <section> nested inside <stimulus>
    def _unwrap_in_stim(m):
        inner = m.group(1)
        inner_clean = re.sub(r'</?section>', '', inner)
        return f"<stimulus>{inner_clean}</stimulus>"
    text = re.sub(r'<stimulus>(.*?)</stimulus>', _unwrap_in_stim, text, flags=re.DOTALL)

    # 2. Re-classify all <section>...</section> tags
    def _repl_section(m):
        raw_inner = m.group(1)
        t = raw_inner.strip()
        clean = clean_tag_text(t)
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        first_line = clean_tag_text(lines[0]) if lines else ''
        first_raw = lines[0] if lines else ''

        # A. Noise / Short / Artifacts (Untag)
        if not t or len(clean) < 4 or clean.lower() in ['hoặc', '###', '---', 'hết', 'hết.']:
            return raw_inner

        # B. Footers / End-of-exam markers (Untag)
        if (re.search(r'^(?:[-—*_\s]*\s*)?(?:HẾT|THE END|END OF TEST|HẾT PHẦN|Hết chủ đề|Hết Phần|Cán bộ coi thi|Thí sinh không được)', clean, re.I) or 
            re.search(r'[-—*]{3,}\s*HẾT', t, re.I) or
            'không được sử dụng tài liệu' in t.lower() or
            re.match(r'^—+\s*Hết\s+chủ\s+đề', clean, re.I)):
            return raw_inner

        # C. Question Labels (Convert to <question_label>)
        if (re.match(r'^(?:#+\s*|\*{1,2}\s*)?(?:Câu|Bài|Q|Question)\s*[\dIVXLCDM]+', first_raw, re.I) and
            not re.search(r'(?:có\s+\d+\s+câu|gồm\s+\d+\s+câu|từ\s+câu\s+\d+\s+đến|trả\s+lời\s+từ\s+câu)', first_line, re.I)):
            if len(lines) == 1 and len(clean) < 80:
                return f"<question_label>{raw_inner}</question_label>"

        # D. Solutions / Answers / Barems (Convert to <explanation>)
        if ('<table' in t.lower() or 
            re.match(r'^(?:HƯỚNG DẪN GIẢI|LỜI GIẢI|ĐÁP ÁN|BẢNG ĐÁP ÁN|HƯỚNG DẪN CHẤM|THANG ĐIỂM|BIỂU ĐIỂM|Bổ sung lời giải|Kết quả cuối cùng|Lời giải)', first_line, re.I) or
            re.match(r'^\*{0,2}Lời giải\*{0,2}$', clean, re.I) or
            'hướng dẫn giải' in first_line.lower() or
            'bảng đáp án' in first_line.lower()):
            return f"<explanation>{raw_inner}</explanation>"

        # E. Exam Header / Administrative Metadata (Untag)
        if (re.match(r'^(?:SỞ\s+GD|TRƯỜNG|ĐỀ\s+THI|ĐỀ\s+KHẢO\s+SÁT|ĐỀ\s+KIỂM\s+TRA|KỲ\s+THI|KỲ\s+KIỂM\s+TRA|MÃ\s+ĐỀ|NĂM\s+HỌC|Họ\s+và\s+tên|Số\s+báo\s+danh|Thời\s+gian|Hà\s+Nội|TỔ\s*:|MÔN\s*:|ĐỀ\s+GỐC|ĐỀ\s+SỐ|BỘ\s+GIÁO\s+DỤC|HỘI\s+CÁC\s+TRƯỜNG|UBND|CỤM\s+THPT|CỤM\s+CÁC\s+TRƯỜNG|ĐỀ\s+VIP|ĐỀ\s+ÔN\s+TẬP|KHỞI\s+ĐỘNG|Tổng điểm|dành xét tuyển|Lựa chọn \d+ trong|KẾT NỐI TRI THỨC|ĐỀ\s+LUYỆN\s+THI|ĐÁNH\s+GIÁ\s+TƯ\s+DUY|BỘ\s+ĐỀ\s+THI|Bộ\s+Đề|ĐỀ\s+CHÍNH\s+THỨC|ĐỀ\s+THAM\s+KHẢO)', first_line, re.I) or
            re.search(r'(?:SỞ GD|TRƯỜNG THPT|KỲ THI TỐT NGHIỆP|Thời gian làm bài\s*:|MÃ ĐỀ\s*:|Thời gian hoàn thành phần thi|ĐỀ THI THỬ)', t, re.I) or
            re.match(r'^\*?\([^\)]*(?:trang|phút|ngoại ngữ|chủ đề)[^\)]*\)\*?$', clean, re.I)):
            return raw_inner

        # F. True Section Divisions & Directions (Keep as <section>)
        return f"<section>{raw_inner}</section>"

    text = re.sub(r'<section>(.*?)</section>', _repl_section, text, flags=re.DOTALL)
    return text

def fix_xml_stimulus_citations(xml_content: str) -> str:
    """
    Directly corrects XML content so that any source attribution/citation
    following a reading passage / stimulus is permanently enclosed inside <stimulus>...</stimulus>.
    """
    if not xml_content or ("<stimulus" not in xml_content and "<context" not in xml_content):
        return xml_content

    text = xml_content

    # 1. Normalize legacy <context> to <stimulus>
    text = re.sub(r"<\s*context\s*>", "<stimulus>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*context\s*>", "</stimulus>", text, flags=re.IGNORECASE)

    def _repl_merge(m):
        citation = m.group(1).strip()
        return f"\n\n{citation}</stimulus>"

    def _repl_trailing(m):
        citation = m.group(1).strip()
        return f"\n\n{citation}</stimulus>\n\n"

    # Case 1: Citation mistakenly wrapped in <stem> or <section> immediately after </stimulus>
    text = re.sub(
        rf"</stimulus>\s*<(?:stem|section)>\s*({CITATION_REGEX})\s*</(?:stem|section)>",
        _repl_merge,
        text,
        flags=re.IGNORECASE
    )

    # Case 2: Consecutive <stimulus> tags where second tag is just the citation
    text = re.sub(
        rf"</stimulus>\s*<stimulus>\s*({CITATION_REGEX})\s*</stimulus>",
        _repl_merge,
        text,
        flags=re.IGNORECASE
    )

    # Case 3: Citation at start of <stem> or <section> following </stimulus>
    text = re.sub(
        rf"</stimulus>\s*<(stem|section)>\s*({CITATION_REGEX})\s*[\n\r]+(.*?)</\1>",
        lambda m: f"\n\n{m.group(2).strip()}</stimulus>\n\n<{m.group(1)}>{m.group(3).strip()}</{m.group(1)}>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Case 4: Untagged citation immediately following </stimulus> before next XML tag or EOF
    text = re.sub(
        rf"</stimulus>\s*({CITATION_REGEX})\s*(?=(?:<|$))",
        _repl_trailing,
        text,
        flags=re.IGNORECASE
    )

    # Clean whitespace inside </stimulus>
    text = re.sub(r"[\r\n\t ]+</stimulus>", "</stimulus>", text)
    # Clean excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def fix_json_stimulus(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Directly corrects JSON exam structures (both real annotated merged.json and synthetic exam JSONs)
    to ensure stimulus texts cleanly include their citations.
    """
    modified = False

    # Check for real exam JSON format (with questions list)
    if "questions" in data and isinstance(data["questions"], list):
        for q in data["questions"]:
            if isinstance(q, dict) and "stimulus_text" in q and q["stimulus_text"]:
                stim = q["stimulus_text"]
                cleaned_stim = re.sub(r'[\r\n\t ]+$', '', stim)
                if cleaned_stim != stim:
                    q["stimulus_text"] = cleaned_stim
                    modified = True

    # Check for synthetic exam format (with sections)
    if "sections" in data and isinstance(data["sections"], list):
        for sec in data["sections"]:
            if not isinstance(sec, dict):
                continue
            for q in sec.get("questions", []):
                if not isinstance(q, dict):
                    continue
                if q.get("is_group") and "stimulus" in q:
                    stim = q["stimulus"]
                    cleaned_stim = re.sub(r'[\r\n\t ]+$', '', stim)
                    if cleaned_stim != stim:
                        q["stimulus"] = cleaned_stim
                        modified = True

    # Check for synthetic single question format
    if data.get("is_group") and "stimulus" in data:
        stim = data["stimulus"]
        cleaned_stim = re.sub(r'[\r\n\t ]+$', '', stim)
        if cleaned_stim != stim:
            data["stimulus"] = cleaned_stim
            modified = True

    return data, modified

def fix_all_root_data(root_dir: Path, check_only: bool = False) -> Dict[str, int]:
    """
    Walks through root_dir (following symlinks), finding all real annotated XML/JSON files
    and synthetic JSONs, and updates them in-place.
    """
    stats = {
        "xml_scanned": 0,
        "xml_fixed": 0,
        "json_scanned": 0,
        "json_fixed": 0
    }

    print(f"Scanning root data directory: {root_dir.resolve()} (with symlinks followed)...")

    # Collect files using os.walk with followlinks=True
    xml_files = []
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir, followlinks=True):
        dp = Path(dirpath)
        for fn in filenames:
            if fn.endswith(".xml") and "output/dataset/xml" not in str(dp / fn):
                xml_files.append(dp / fn)
            elif fn.endswith(".json") and fn not in ["label_mapping.json", "train_stats.json", "val_stats.json", "test_stats.json"]:
                json_files.append(dp / fn)

    # 1. Process all XML files
    for xml_path in xml_files:
        stats["xml_scanned"] += 1
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            content = fix_xml_stimulus_citations(original_content)
            content = fix_xml_sections(content)

            if content != original_content:
                stats["xml_fixed"] += 1
                if not check_only:
                    with open(xml_path, "w", encoding="utf-8") as f:
                        f.write(content)
                print(f"  [{'WOULD FIX' if check_only else 'FIXED'}] XML: {xml_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {xml_path}: {e}")

    # 2. Process all JSON files
    for json_path in json_files:
        stats["json_scanned"] += 1
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            fixed_data, was_modified = fix_json_stimulus(data)
            if was_modified:
                stats["json_fixed"] += 1
                if not check_only:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
                print(f"  [{'WOULD FIX' if check_only else 'FIXED'}] JSON: {json_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {json_path}: {e}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Repair stimulus citations and section tags in root XML and JSON data.")
    parser.add_argument("--input-dir", type=str, default="output", help="Root data directory to scan and repair.")
    parser.add_argument("--check-only", action="store_true", help="Only check for issues without writing changes.")
    args = parser.parse_args()

    root_path = Path(args.input_dir)
    if not root_path.exists():
        print(f"Directory '{root_path}' does not exist.")
        return

    stats = fix_all_root_data(root_path, check_only=args.check_only)
    print("\n" + "=" * 60)
    print("ROOT DATA FIX SUMMARY:")
    print("=" * 60)
    print(f"  XML files scanned: {stats['xml_scanned']} (Fixed: {stats['xml_fixed']})")
    print(f"  JSON files scanned: {stats['json_scanned']} (Fixed: {stats['json_fixed']})")
    print("=" * 60)

if __name__ == "__main__":
    main()

