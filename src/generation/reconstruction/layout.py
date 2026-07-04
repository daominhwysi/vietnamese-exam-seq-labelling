import random
import itertools
from typing import List, Dict, Any, Optional
from src.generation.reconstruction.config import ReconstructorConfig
from src.generation.reconstruction.augment import apply_formatting_tag_noise

def generate_random_inline_separator(config: ReconstructorConfig, rng: random.Random) -> str:
    """Generates a highly randomized whitespace/tab separator to simulate raw paper-saving layouts."""
    sep_type = rng.choice(["tabs", "spaces", "mixed"])
    
    if sep_type == "tabs":
        num_tabs = rng.randint(config.min_inline_tabs, config.max_inline_tabs)
        return "\t" * num_tabs
    elif sep_type == "spaces":
        num_spaces = rng.randint(config.min_inline_spaces, config.max_inline_spaces)
        return " " * num_spaces
    else:  # mixed
        num_tabs = rng.randint(1, max(1, config.min_inline_tabs))
        num_spaces = rng.randint(5, max(5, config.min_inline_spaces))
        return ("\t" * num_tabs) + (" " * num_spaces)

def paraphrase_section_title(title: str, rng: random.Random) -> str:
    """
    Paraphrases section titles or instructions to prevent model overfitting.
    """
    if not title:
        return title
        
    title_clean = title.strip()
    
    # 1. Literature Sections
    if "ĐỌC HIỂU" in title_clean.upper():
        lit_reading_templates = [
            "I. ĐỌC HIỂU",
            "PHẦN I. ĐỌC HIỂU",
            "PHẦN I: ĐỌC HIỂU",
            "Phần I. Đọc hiểu",
            "I. ĐỌC - HIỂU",
            "PHẦN THỨ NHẤT: ĐỌC HIỂU"
        ]
        return rng.choice(lit_reading_templates)
        
    if "VIẾT" in title_clean.upper() or "LÀM VĂN" in title_clean.upper():
        lit_writing_templates = [
            "II. VIẾT",
            "PHẦN II. LÀM VĂN",
            "PHẦN II: VIẾT",
            "Phần II. Làm văn",
            "PHẦN THỨ HAI: LÀM VĂN",
            "II. TẬP LÀM VĂN",
            "PHẦN II. TẬP LÀM VĂN"
        ]
        return rng.choice(lit_writing_templates)
        
    # 2. Standard Vietnamese sections
    if "PHẦN I" in title_clean.upper() and ("NHIỀU PHƯƠNG ÁN" in title_clean.upper() or "TRẮC NGHIỆM KHÁCH QUAN" in title_clean.upper() or "TRẮC NGHIỆM" in title_clean.upper() and "ĐÚNG" not in title_clean.upper() and "NGẮN" not in title_clean.upper()):
        score = rng.choice(["", " (3,0 điểm)", " (3.0 điểm)", " (3 điểm)"])
        mcq_templates = [
            f"PHẦN I{score}. Câu trắc nghiệm nhiều phương án lựa chọn.",
            f"PHẦN I{score}. TRẮC NGHIỆM NHIỀU PHƯƠNG ÁN LỰA CHỌN",
            f"Phần I{score}. Câu trắc nghiệm nhiều phương án lựa chọn",
            f"PHẦN I{score}: TRẮC NGHIỆM KHÁCH QUAN",
            f"PHẦN I{score}. TRẮC NGHIỆM",
            f"PHẦN I{score}. CÂU HỎI TRẮC NGHIỆM",
            f"Phần 1{score}: Trắc nghiệm khách quan",
            f"Phần thứ nhất{score}: Câu hỏi trắc nghiệm"
        ]
        return rng.choice(mcq_templates)
        
    if "PHẦN II" in title_clean.upper() and ("ĐÚNG-SAI" in title_clean.upper() or "ĐÚNG SAI" in title_clean.upper()):
        score = rng.choice(["", " (4,0 điểm)", " (4.0 điểm)", " (4 điểm)"])
        tf_templates = [
            f"PHẦN II{score}. Trắc nghiệm đúng-sai",
            f"PHẦN II{score}. Câu trắc nghiệm đúng sai",
            f"PHẦN II{score}. TRẮC NGHIỆM ĐÚNG SAI",
            f"Phần II{score}: Câu trắc nghiệm đúng - sai",
            f"PHẦN II{score}. CÂU HỎI ĐÚNG SAI",
            f"Phần 2{score}. Trắc nghiệm Đúng Sai",
            f"Phần thứ hai{score}: Câu trắc nghiệm đúng sai"
        ]
        return rng.choice(tf_templates)
        
    if "PHẦN III" in title_clean.upper() and ("TRẢ LỜI NGẮN" in title_clean.upper() or "TỰ LUẬN NGẮN" in title_clean.upper()):
        score = rng.choice(["", " (1,5 điểm)", " (1.5 điểm)", " (1.5 điểm)"])
        sa_templates = [
            f"PHẦN III{score}. Trắc nghiệm trả lời ngắn",
            f"PHẦN III{score}. Câu trắc nghiệm trả lời ngắn",
            f"PHẦN III{score}: CÂU TRẮC NGHIỆM TRẢ LỜI NGẮN",
            f"Phần III{score}. Trắc nghiệm tự luận ngắn",
            f"PHẦN III{score}. CÂU HỎI TỰ LUẬN TRẢ LỜI NGẮN",
            f"Phần 3{score}. Trắc nghiệm trả lời ngắn",
            f"Phần thứ ba{score}: Câu hỏi trả lời ngắn"
        ]
        return rng.choice(sa_templates)
        
    if "PHẦN IV" in title_clean.upper() and "TỰ LUẬN" in title_clean.upper():
        score = rng.choice(["", " (1,5 điểm)", " (1.5 điểm)", " (1.5 điểm)"])
        essay_templates = [
            f"PHẦN IV{score}. Tự luận",
            f"PHẦN IV{score}. TỰ LUẬN",
            f"Phần IV{score}: Câu hỏi tự luận",
            f"Phần 4{score}. Tự luận",
            f"Phần thứ tư{score}: Tự luận",
            f"PHẦN IV{score}: BÀI TẬP TỰ LUẬN"
        ]
        return rng.choice(essay_templates)
        
    # 3. English Instructions
    if "PRONUNCIATION" in title_clean.upper() or "UNDERLINED PART DIFFERS" in title_clean.upper():
        pron_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the word whose underlined part differs from the other three in pronunciation in each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the word whose underlined part differs from the other three in pronunciation in each of the following questions.",
            "Choose the letter A, B, C, or D to indicate the word whose underlined part differs from the other three in pronunciation.",
            "Choose the word whose underlined part is pronounced differently from that of the others.",
            "Choose the word whose underlined part is pronounced differently from the other three."
        ]
        return rng.choice(pron_templates)
        
    if "STRESS" in title_clean.upper() or "POSITION OF STRESS" in title_clean.upper():
        stress_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the word that differs from the other three in the position of stress in each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the word that differs from the other three in the position of primary stress in each of the following questions.",
            "Choose the letter A, B, C, or D to indicate the word that differs from the other three in the position of primary stress.",
            "Choose the word whose stress pattern is different from that of the others.",
            "Choose the word that differs from the rest in the position of main stress."
        ]
        return rng.choice(stress_templates)
        
    if "CLOSEST" in title_clean.upper() and "MEANING" in title_clean.upper():
        closest_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the word CLOSEST in meaning to the underlined word in each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the word CLOSEST in meaning to the underlined word in each of the following questions.",
            "Choose the letter A, B, C, or D to indicate the word CLOSEST in meaning to the underlined word.",
            "Choose the word or phrase that is closest in meaning to the underlined part."
        ]
        return rng.choice(closest_templates)
        
    if "OPPOSITE" in title_clean.upper() and "MEANING" in title_clean.upper():
        opposite_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the word(s) OPPOSITE in meaning to the underlined word(s) in each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the word(s) OPPOSITE in meaning to the underlined word(s) in each of the following questions.",
            "Choose the letter A, B, C, or D to indicate the word(s) OPPOSITE in meaning to the underlined word(s).",
            "Choose the word or phrase that is opposite in meaning to the underlined part."
        ]
        return rng.choice(opposite_templates)
        
    if "CORRECT ANSWER" in title_clean.upper() and "GRAMMAR" not in title_clean.upper():
        correct_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the correct answer to each of the following questions.",
            "Choose the letter A, B, C, or D to indicate the correct answer to each of the following questions.",
            "Choose the best option to complete each of the following sentences."
        ]
        return rng.choice(correct_templates)
        
    if "PASSAGE" in title_clean.upper() and ("FITS EACH OF THE NUMBERED BLANKS" in title_clean.upper() or "CLOZE" in title_clean.upper() or "WORD OR PHRASE" in title_clean.upper() and "BLANK" in title_clean.upper()):
        cloze_templates = [
            "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct word or phrase that best fits each of the numbered blanks.",
            "Read the following passage and mark the letter A, B, C, or D to indicate the correct word or phrase that best fits each of the numbered blanks.",
            "Read the passage and choose the correct word or phrase that best fits each of the numbered blanks.",
            "Choose the correct word or phrase to fill in each of the numbered blanks in the following passage."
        ]
        return rng.choice(cloze_templates)
        
    if "PASSAGE" in title_clean.upper() and "CORRECT ANSWER TO EACH OF THE QUESTIONS" in title_clean.upper():
        reading_templates = [
            "Read the following passage and mark the letter A, B, C, or D on your answer sheet to indicate the correct answer to each of the questions.",
            "Read the following passage and mark the letter A, B, C, or D to indicate the correct answer to each of the questions.",
            "Read the passage and choose the correct answer to each of the questions.",
            "Read the following passage and choose the best answer for each question."
        ]
        return rng.choice(reading_templates)
        
    if "COMBINES" in title_clean.upper() or "PAIR OF SENTENCES" in title_clean.upper():
        comb_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that best combines each pair of sentences in the following questions.",
            "Mark the letter A, B, C, or D to indicate the sentence that best combines each pair of sentences in the following questions.",
            "Choose the sentence that best combines each pair of sentences in the following questions."
        ]
        return rng.choice(comb_templates)
        
    if "UNDERLINED PART" in title_clean.upper() and ("NEEDS CORRECTION" in title_clean.upper() or "ERROR" in title_clean.upper()):
        error_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the underlined part that needs correction in each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the underlined part that needs correction in each of the following questions.",
            "Choose the underlined part that needs correction in each of the following sentences."
        ]
        return rng.choice(error_templates)
        
    if "TRANSFORMATION" in title_clean.upper() or "CLOSEST IN MEANING" in title_clean.upper() and "WORD" not in title_clean.upper():
        trans_templates = [
            "Mark the letter A, B, C, or D on your answer sheet to indicate the sentence that is closest in meaning to each of the following questions.",
            "Mark the letter A, B, C, or D to indicate the sentence that is closest in meaning to each of the following questions.",
            "Choose the sentence that is closest in meaning to each of the following questions."
        ]
        return rng.choice(trans_templates)
        
    return title

def format_answer_table(exam_records: List[Dict[str, Any]], format_type: str, direction: str, chunk_size: Optional[int] = None, rng: Optional[random.Random] = None) -> str:
    """
    Formats the answer key table using multiple formats (md, html, csv) and directions (horizontal, vertical).
    """
    if not exam_records:
        return ""
        
    if rng is None:
        rng = random.Random()
        
    if direction == "horizontal":
        if chunk_size is None:
            chunk_size = rng.choice([5, 10, 15, len(exam_records)])
        elif chunk_size <= 0:
            chunk_size = len(exam_records)
            
        table_parts = []
        if format_type == "md":
            for i in range(0, len(exam_records), chunk_size):
                chunk = exam_records[i:i+chunk_size]
                header_row = "| Câu | " + " | ".join(str(r["num"]) for r in chunk) + " |"
                sep_row = "|-----|" + "|".join("---" for _ in chunk) + "|"
                ans_row = "| Đáp án | " + " | ".join(r["answer"] for r in chunk) + " |"
                table_parts.append(f"{header_row}\n{sep_row}\n{ans_row}")
            return "\n\n".join(table_parts) + "\n"
            
        elif format_type == "html":
            for i in range(0, len(exam_records), chunk_size):
                chunk = exam_records[i:i+chunk_size]
                html_lines = ["<table>"]
                html_lines.append("  <thead>")
                html_lines.append("    <tr>")
                html_lines.append("      <th>Câu</th>")
                for r in chunk:
                    html_lines.append(f"      <th>{r['num']}</th>")
                html_lines.append("    </tr>")
                html_lines.append("  </thead>")
                html_lines.append("  <tbody>")
                html_lines.append("    <tr>")
                html_lines.append("      <td>Đáp án</td>")
                for r in chunk:
                    html_lines.append(f"      <td>{r['answer']}</td>")
                html_lines.append("    </tr>")
                html_lines.append("  </tbody>")
                html_lines.append("</table>")
                table_parts.append("\n".join(html_lines))
            return "\n\n".join(table_parts) + "\n"
            
        elif format_type == "csv":
            for i in range(0, len(exam_records), chunk_size):
                chunk = exam_records[i:i+chunk_size]
                header_row = "Câu," + ",".join(str(r["num"]) for r in chunk)
                ans_row = "Đáp án," + ",".join(r["answer"] for r in chunk)
                table_parts.append(f"{header_row}\n{ans_row}")
            return "\n\n".join(table_parts) + "\n"
            
    else:  # vertical
        if format_type == "md":
            lines = ["| Câu | Đáp án |", "|-----|--------|"]
            for r in exam_records:
                lines.append(f"| {r['num']} | {r['answer']} |")
            return "\n".join(lines) + "\n"
            
        elif format_type == "html":
            html_lines = ["<table>"]
            html_lines.append("  <thead>")
            html_lines.append("    <tr>")
            html_lines.append("      <th>Câu</th>")
            html_lines.append("      <th>Đáp án</th>")
            html_lines.append("    </tr>")
            html_lines.append("  </thead>")
            html_lines.append("  <tbody>")
            for r in exam_records:
                html_lines.append("    <tr>")
                html_lines.append(f"      <td>{r['num']}</td>")
                html_lines.append(f"      <td>{r['answer']}</td>")
                html_lines.append("    </tr>")
            html_lines.append("  </tbody>")
            html_lines.append("</table>")
            return "\n".join(html_lines) + "\n"
            
        elif format_type == "csv":
            lines = ["Câu,Đáp án"]
            for r in exam_records:
                lines.append(f"{r['num']},{r['answer']}")
            return "\n".join(lines) + "\n"
            
    return ""

def generate_ordering_choices(labels: List[str], separator: str, rng: random.Random) -> List[str]:
    """Generates 4 multiple choice ordering options (1 correct, up to 3 distractors)."""
    correct_seq = separator.join(labels)
    all_perms = list(itertools.permutations(labels))
    all_seqs = [separator.join(p) for p in all_perms]
    distractors = [s for s in all_seqs if s != correct_seq]
    if len(distractors) >= 3:
        selected_distractors = rng.sample(distractors, 3)
    else:
        selected_distractors = distractors
    candidates = [correct_seq] + selected_distractors
    rng.shuffle(candidates)
    return candidates
