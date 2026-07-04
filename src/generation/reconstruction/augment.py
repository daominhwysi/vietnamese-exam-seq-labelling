import random
import re
from typing import Tuple
from src.generation.reconstruction.config import ReconstructorConfig

def inject_vietnamese_typos(text: str, typo_rate: float, rng: random.Random) -> str:
    """
    Injects realistic Vietnamese typos (tone mark slips, swaps, Telex residues, keyboard slips).
    """
    if not text or typo_rate <= 0.0:
        return text
        
    words = text.split(" ")
    new_words = []
    
    for word in words:
        if not word or rng.random() > typo_rate:
            new_words.append(word)
            continue
            
        typo_type = rng.choice(["tone_strip", "swap", "telex_residue", "char_replace"])
        
        if typo_type == "tone_strip":
            tone_map = {
                'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
                'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
                'ề': 'ê', 'ế': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
                'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
                'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
                'ờ': 'ơ', 'ớ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
                'ồ': 'ô', 'ố': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
                'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
                'ừ': 'ư', 'ứ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
                'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
                'À': 'A', 'Á': 'A', 'È': 'E', 'É': 'E', 'Ì': 'I', 'Í': 'I',
                'Ò': 'O', 'Ó': 'O', 'Ù': 'U', 'Ú': 'U', 'Ỳ': 'Y', 'Ý': 'Y',
                'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
                'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e'
            }
            new_word = "".join(tone_map.get(c, c) for c in word)
            new_words.append(new_word)
            
        elif typo_type == "swap" and len(word) >= 3:
            idx = rng.randint(0, len(word) - 2)
            word_list = list(word)
            word_list[idx], word_list[idx+1] = word_list[idx+1], word_list[idx]
            new_words.append("".join(word_list))
            
        elif typo_type == "telex_residue":
            telex_keys = ['s', 'f', 'r', 'x', 'j', 'w']
            new_words.append(word + rng.choice(telex_keys))
            
        elif typo_type == "char_replace" and len(word) >= 2:
            neighbors = {
                'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc',
                'e': 'wsdr', 'g': 'tyfhvb', 'h': 'yugjbn', 'i': 'ujko',
                'k': 'ijlm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm',
                'o': 'iklp', 'p': 'ol', 'r': 'edtf', 's': 'wedxza',
                't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase',
                'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
            }
            indices = [i for i, c in enumerate(word) if c in neighbors]
            if indices:
                idx = rng.choice(indices)
                c = word[idx]
                word_list = list(word)
                word_list[idx] = rng.choice(neighbors[c])
                new_words.append("".join(word_list))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
            
    return " ".join(new_words)

def randomize_blank_tokens(text: str, rng: random.Random) -> str:
    """
    Randomizes English exam blank formats (<blank/>) into underscores, dots, or text brackets.
    """
    if not text or "<blank" not in text:
        return text
        
    blank_pattern = re.compile(r'<\s*blank\s*/?\s*>')
    
    def repl(match):
        val = rng.random()
        if val < 0.50:
            return "<blank/>"
        elif val < 0.70:
            return "_" * rng.randint(4, 8)
        elif val < 0.90:
            return "." * rng.randint(4, 8)
        else:
            return rng.choice(["[BLANK]", "(blank)", "[blank]", "______"])
            
    return blank_pattern.sub(repl, text)

def process_latex_variations(text: str, placeholder: str, mask_prob: float, rng: random.Random) -> str:
    """
    Handles math formulas ($...$): masks them with placeholder or keeps raw and varies delimiters.
    """
    if not text or "$" not in text:
        return text
        
    latex_pattern = re.compile(r'\$([^$]+)\$')
    
    def repl(match):
        formula = match.group(1)
        if rng.random() < mask_prob:
            return placeholder
        else:
            del_val = rng.random()
            if del_val < 0.80:
                return f"${formula}$"
            elif del_val < 0.90:
                return rng.choice([f"\\( {formula} \\)", f"\\[ {formula} \\]"])
            else:
                return formula
                
    return latex_pattern.sub(repl, text)

def strip_formatting_wrappers(text: str) -> Tuple[str, str, str]:
    """
    Extracts leading and trailing formatting tags/markers from the text,
    leaving the core text in the middle.
    """
    leading = ""
    trailing = ""
    core = text
    
    while True:
        stripped_any = False
        core_stripped = core.strip()
        
        # Check bold markdown: **...**
        if core_stripped.startswith("**") and core_stripped.endswith("**") and len(core_stripped) >= 5:
            start_idx = core.find("**")
            end_idx = core.rfind("**")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                leading += core[:start_idx + 2]
                trailing = core[end_idx:] + trailing
                core = core[start_idx + 2:end_idx]
                stripped_any = True
                continue
                
        # Check italic markdown: *...*
        if core_stripped.startswith("*") and core_stripped.endswith("*") and len(core_stripped) >= 3:
            if not (core_stripped.startswith("**") and core_stripped.endswith("**")):
                start_idx = core.find("*")
                end_idx = core.rfind("*")
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    leading += core[:start_idx + 1]
                    trailing = core[end_idx:] + trailing
                    core = core[start_idx + 1:end_idx]
                    stripped_any = True
                    continue
                    
        # Check HTML tags: <u>...</u>, <b>...</b>, <i>...</i>
        html_patterns = [
            (r"^<u>", r"</u>$", "<u>", "</u>"),
            (r"^<b>", r"</b>$", "<b>", "</b>"),
            (r"^<i>", r"</i>$", "<i>", "</i>")
        ]
        for start_pat, end_pat, open_tag, close_tag in html_patterns:
            core_stripped = core.strip()
            if re.match(start_pat, core_stripped, re.IGNORECASE) and re.search(end_pat, core_stripped, re.IGNORECASE):
                start_idx = core.lower().find(open_tag)
                end_idx = core.lower().rfind(close_tag)
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    leading += core[:start_idx + len(open_tag)]
                    trailing = core[end_idx:] + trailing
                    core = core[start_idx + len(open_tag):end_idx]
                    stripped_any = True
                    break
        if not stripped_any:
            break
            
    return leading, core, trailing

def apply_formatting_tag_noise(text: str, prob: float, rng: random.Random) -> str:
    """
    Wraps text segment in random Markdown/HTML formatting tags.
    """
    if not text or prob <= 0.0 or rng.random() > prob:
        return text
        
    tag = rng.choice(["**", "*", "<u>", "<b>", "<i>"])
    if tag == "**":
        return f"**{text}**"
    elif tag == "*":
        return f"*{text}*"
    elif tag == "<u>":
        return f"<u>{text}</u>"
    elif tag == "<b>":
        return f"<b>{text}</b>"
    elif tag == "<i>":
        return f"<i>{text}</i>"
    return text

def apply_casing_noise(text: str, prob: float, rng: random.Random) -> str:
    """
    Alters capitalization of a label prefix or text segment.
    """
    if not text or prob <= 0.0 or rng.random() > prob:
        return text
        
    case_type = rng.choice(["lower", "upper", "capitalize"])
    if case_type == "lower":
        return text.lower()
    elif case_type == "upper":
        return text.upper()
    elif case_type == "capitalize":
        return text.capitalize()
    return text

def get_label_synonym(prefix_word: str, prob: float, rng: random.Random, subject: str = "") -> str:
    """
    Swaps prefix keywords (e.g., "Câu" -> "Bài", "Question") dynamically.
    """
    if not prefix_word or prefix_word.strip().lower() != "câu" or prob <= 0.0 or rng.random() > prob:
        return prefix_word
        
    synonyms = ["Bài", "Bài tập", "Câu hỏi"]
    if subject == "english":
        synonyms.append("Question")
        
    return rng.choice(synonyms)

def augment_q_prefix_tpl(q_prefix_tpl: str, config: ReconstructorConfig, rng: random.Random, subject: str) -> str:
    if config.synonym_swap_prob > 0.0 or config.casing_noise_prob > 0.0:
        m = re.match(r'^([a-zA-Z\s\u00C0-\u1EF9]+)\{num\}', q_prefix_tpl)
        if m:
            full_match_prefix = m.group(1)
            prefix_word = full_match_prefix.rstrip()
            synonym = prefix_word
            if config.synonym_swap_prob > 0.0:
                synonym = get_label_synonym(prefix_word, config.synonym_swap_prob, rng, subject)
            if config.casing_noise_prob > 0.0:
                synonym = apply_casing_noise(synonym, config.casing_noise_prob, rng)
            trailing_spaces = full_match_prefix[len(prefix_word):]
            new_prefix = synonym + trailing_spaces
            q_prefix_tpl = q_prefix_tpl.replace(full_match_prefix, new_prefix, 1)
    return q_prefix_tpl

def augment_q_label(q_label: str, config: ReconstructorConfig, rng: random.Random) -> str:
    stripped = q_label.rstrip()
    spaces = q_label[len(stripped):]
    if config.formatting_noise_prob > 0.0:
        stripped = apply_formatting_tag_noise(stripped, config.formatting_noise_prob, rng)
    if config.space_noise_rate > 0.0:
        val = rng.random()
        if val < config.space_noise_rate:
            noise_type = rng.choice(["none", "strip", "extra"])
            if noise_type == "strip":
                spaces = ""
            elif noise_type == "extra":
                spaces = "  "
    return stripped + spaces

def augment_opt_lbl(opt_lbl: str, config: ReconstructorConfig, rng: random.Random) -> str:
    stripped = opt_lbl.rstrip()
    spaces = opt_lbl[len(stripped):]
    if config.casing_noise_prob > 0.0:
        stripped = apply_casing_noise(stripped, config.casing_noise_prob, rng)
    if config.formatting_noise_prob > 0.0:
        stripped = apply_formatting_tag_noise(stripped, config.formatting_noise_prob, rng)
    if config.space_noise_rate > 0.0:
        val = rng.random()
        if val < config.space_noise_rate:
            noise_type = rng.choice(["none", "strip", "extra"])
            if noise_type == "strip":
                spaces = ""
            elif noise_type == "extra":
                spaces = "  "
    return stripped + spaces
