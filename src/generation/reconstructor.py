import random
import re
import itertools
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# Available prefix templates for Question labels
DEFAULT_QUESTION_PREFIXES = [
    "Câu {num}: ",
    "Câu {num}. ",
    "Question {num}: ",
    "Question {num}. ",
    "C{num}: ",
    "C{num}. ",
    "Q{num}: ",
    "Q{num}. ",
    "{num}. ",
    "{num}: ",
    "{num}) "
]

# Available styles for option prefixes
OPTION_PREFIX_STYLES = {
    "capital_dot": ["A. ", "B. ", "C. ", "D. "],
    "lowercase_paren": ["a) ", "b) ", "c) ", "d) "],
    "capital_paren": ["A) ", "B) ", "C) ", "D) "],
    "lowercase_dot": ["a. ", "b. ", "c. ", "d. "],
    "bold_capital_dot": ["**A.** ", "**B.** ", "**C.** ", "**D.** "],
    "bold_lowercase_paren": ["**a)** ", "**b)** ", "**c)** ", "**d)** "]
}

# Styles for ordering items
ORDERING_ITEM_STYLES = {
    "char": {
        "labels": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "prefixes": ["{label}. ", "* {label}. ", "{label}) "]
    },
    "index": {
        "labels": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "prefixes": ["{label}. ", "{label}) ", "{label} "]
    }
}

@dataclass
class ReconstructorConfig:
    question_prefix_template: Optional[str] = None
    option_prefix_style: Optional[str] = None
    ordering_item_label_style: Optional[str] = None  # "char" or "index"
    ordering_item_prefix_template: Optional[str] = None
    separator_stem_options: str = "\n"
    separator_options: str = "\n"
    separator_context_questions: str = "\n\n"
    separator_questions: str = "\n\n"
    ordering_choice_separator: str = " – "
    track_separators: bool = False
    include_span_text: bool = True
    seed: Optional[Any] = None
    randomize_q_num: bool = True
    
    # Advanced Data Augmentations
    typo_rate: float = 0.0
    space_noise_rate: float = 0.0
    latex_mask_prob: float = 0.0
    latex_placeholder: str = "[LATEX]"
    enable_permutations: bool = False
    option_drop_prob: float = 0.0
    casing_noise_prob: float = 0.0
    synonym_swap_prob: float = 0.0
    formatting_noise_prob: float = 0.0
    
    # Inline option layout simulation
    inline_option_prob: float = 0.0
    min_inline_spaces: int = 5
    max_inline_spaces: int = 30
    min_inline_tabs: int = 1
    max_inline_tabs: int = 3

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

def get_stable_random(seed_obj: Any) -> random.Random:
    """Generates a stable random number generator from any seed object."""
    if seed_obj is not None:
        if isinstance(seed_obj, str):
            val = 0
            for char in seed_obj:
                val = (val * 31 + ord(char)) & 0xFFFFFFFF
            return random.Random(val)
        else:
            try:
                return random.Random(hash(seed_obj))
            except TypeError:
                return random.Random(hash(str(seed_obj)))
    return random.Random()

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

def reconstruct_question(q_data: Dict[str, Any], config: Optional[ReconstructorConfig] = None, start_q_num: int = 1) -> Dict[str, Any]:
    """
    Reconstructs the raw text of a question (standard or group) and tracks spans of its components.
    
    Returns a copy of q_data enriched with "raw_text" and "spans" keys.
    """
    if config is None:
        config = ReconstructorConfig()
        
    result = dict(q_data)
    
    stable_seed = config.seed
    if stable_seed is None:
        stable_seed = q_data.get("context", "") or q_data.get("stem", "") or str(q_data)
        
    rng = get_stable_random(stable_seed)
    is_inline = rng.random() < config.inline_option_prob
    
    actual_start_q_num = start_q_num
    if config.randomize_q_num:
        actual_start_q_num = rng.randint(1, 40)
    
    q_prefix_tpl = config.question_prefix_template
    if q_prefix_tpl is None:
        q_prefix_tpl = rng.choice(DEFAULT_QUESTION_PREFIXES)
    q_prefix_tpl = augment_q_prefix_tpl(q_prefix_tpl, config, rng, q_data.get("subject", ""))
        
    opt_style_name = config.option_prefix_style
    if opt_style_name is None:
        opt_style_name = rng.choice(list(OPTION_PREFIX_STYLES.keys()))
    opt_prefixes = OPTION_PREFIX_STYLES.get(opt_style_name, OPTION_PREFIX_STYLES["capital_dot"])
    
    ord_item_style = config.ordering_item_label_style
    if ord_item_style is None:
        ord_item_style = rng.choice(["char", "index"])
    
    ord_prefix_tpl = config.ordering_item_prefix_template
    if ord_prefix_tpl is None:
        ord_prefix_tpl = rng.choice(ORDERING_ITEM_STYLES[ord_item_style]["prefixes"])
        
    ord_labels = ORDERING_ITEM_STYLES[ord_item_style]["labels"]
    
    raw_text = ""
    spans = []
    
    def append_segment(text: str, label: str):
        nonlocal raw_text
        if not text:
            return
            
        # Apply data augmentations before stitching to keep spans 100% correct
        if label in ["stem", "option_text", "context"]:
            if config.space_noise_rate > 0.0:
                text = randomize_blank_tokens(text, rng)
            if config.latex_mask_prob > 0.0:
                text = process_latex_variations(text, config.latex_placeholder, config.latex_mask_prob, rng)
            if config.typo_rate > 0.0:
                text = inject_vietnamese_typos(text, config.typo_rate, rng)
                
        elif label == "separator":
            if config.space_noise_rate > 0.0:
                if "\n" in text and not any(c.isalnum() for c in text):
                    val = rng.random()
                    if val < 0.10:
                        text = " "
                    elif val < 0.15:
                        text = "\n\n"
                        
        start = len(raw_text)
        raw_text += text
        end = len(raw_text)
        
        is_separator = (label == "separator")
        if not is_separator or config.track_separators:
            span_entry = {
                "start": start,
                "end": end,
                "label": label
            }
            if config.include_span_text:
                span_entry["text"] = text
            spans.append(span_entry)

    is_group = q_data.get("is_group", False)
    q_type = q_data.get("question_type", "")
    
    if is_group:
        context = q_data.get("context", "")
        if q_data.get("subject") == "english":
            for i in range(1, 40):
                context = re.sub(rf'\({i}\)\s*(?:_{{2,}}|\.{{2,}}|\[\s*BLANK\s*\]|<\s*blank\s*/?>)', f'({i}) <blank />', context)
        append_segment(context, "context")
        
        sub_questions = q_data.get("questions", [])
        if sub_questions:
            append_segment(config.separator_context_questions, "separator")
            
            for idx, sub_q in enumerate(sub_questions):
                q_num = actual_start_q_num + idx
                q_label = q_prefix_tpl.format(num=q_num)
                q_label = augment_q_label(q_label, config, rng)
                append_segment(q_label, "question_label")
                
                stem = sub_q.get("stem", "")
                if q_data.get("subject") == "english":
                    stem = re.sub(r'^(?:Question|Câu)\s*\d+\s*:\s*', '', stem, flags=re.IGNORECASE)
                    stem = re.sub(r'^Mark the letter [A-Z,\s/]+(?:or [A-Z\s]+)? on your answer sheet to indicate the [^.]+\.\s*', '', stem, flags=re.IGNORECASE)
                    stem = re.sub(r'(?:_{2,}|\.{3,}|\[\s*BLANK\s*\]|<\s*blank\s*/?>)', '<blank/>', stem)
                append_segment(stem, "stem")
                
                options = list(sub_q.get("options", []))
                
                if config.option_drop_prob > 0.0 and rng.random() < config.option_drop_prob:
                    keep_count = rng.randint(0, min(3, len(options)))
                    options = options[:keep_count]
                    
                if config.enable_permutations and options and len(options) >= 2:
                    rng.shuffle(options)
                
                if options:
                    append_segment(config.separator_stem_options, "separator")
                    for opt_idx, opt_text in enumerate(options):
                        opt_lbl = opt_prefixes[opt_idx % len(opt_prefixes)]
                        opt_lbl = augment_opt_lbl(opt_lbl, config, rng)
                        append_segment(opt_lbl, "option_label")
                        append_segment(opt_text, "option_text")
                        if opt_idx < len(options) - 1:
                            if is_inline:
                                sep = generate_random_inline_separator(config, rng)
                                append_segment(sep, "separator")
                            else:
                                append_segment(config.separator_options, "separator")
                            
                if idx < len(sub_questions) - 1:
                    append_segment(config.separator_questions, "separator")
    else:
        q_label = q_prefix_tpl.format(num=actual_start_q_num)
        q_label = augment_q_label(q_label, config, rng)
        append_segment(q_label, "question_label")
        
        stem = q_data.get("stem", "")
        if q_data.get("subject") == "english":
            stem = re.sub(r'^(?:Question|Câu)\s*\d+\s*:\s*', '', stem, flags=re.IGNORECASE)
            stem = re.sub(r'^Mark the letter [A-Z,\s/]+(?:or [A-Z\s]+)? on your answer sheet to indicate the [^.]+\.\s*', '', stem, flags=re.IGNORECASE)
            stem = re.sub(r'(?:_{2,}|\.{3,}|\[\s*BLANK\s*\]|<\s*blank\s*/?>)', '<blank/>', stem)
        append_segment(stem, "stem")
        
        options = list(q_data.get("options", []))
        
        if config.option_drop_prob > 0.0 and q_type in ["multiple_choice", "true_false"] and options:
            if rng.random() < config.option_drop_prob:
                keep_count = rng.randint(0, min(3, len(options)))
                options = options[:keep_count]
                
        if config.enable_permutations and q_type == "multiple_choice" and options and len(options) >= 2:
            orig_ans = q_data.get("answer", "")
            opt_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
            orig_idx = -1
            if orig_ans in opt_letters:
                orig_idx = opt_letters.index(orig_ans)
            if 0 <= orig_idx < len(options):
                correct_val = options[orig_idx]
                rng.shuffle(options)
                if correct_val in options:
                    new_idx = options.index(correct_val)
                    result["answer"] = opt_letters[new_idx % len(opt_letters)]
        
        if options:
            append_segment(config.separator_stem_options, "separator")
            
            if q_type == "ordering":
                item_labels = ord_labels[:len(options)]
                for opt_idx, opt_text in enumerate(options):
                    lbl = ord_prefix_tpl.format(label=item_labels[opt_idx])
                    append_segment(lbl, "stem")
                    append_segment(opt_text, "stem")
                    if opt_idx < len(options) - 1:
                        append_segment(config.separator_options, "separator")
                        
                choices = generate_ordering_choices(item_labels, config.ordering_choice_separator, rng)
                append_segment(config.separator_stem_options, "separator")
                
                correct_seq = config.ordering_choice_separator.join(item_labels)
                correct_idx = 0
                if correct_seq in choices:
                    correct_idx = choices.index(correct_seq)
                opt_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
                result["answer"] = opt_letters[correct_idx % len(opt_letters)]
                
                for choice_idx, choice_text in enumerate(choices):
                    opt_lbl = opt_prefixes[choice_idx % len(opt_prefixes)]
                    opt_lbl = augment_opt_lbl(opt_lbl, config, rng)
                    append_segment(opt_lbl, "option_label")
                    append_segment(choice_text, "option_text")
                    if choice_idx < len(choices) - 1:
                        if is_inline:
                            sep = generate_random_inline_separator(config, rng)
                            append_segment(sep, "separator")
                        else:
                            append_segment(config.separator_options, "separator")
            else:
                current_prefixes = opt_prefixes
                if q_type == "true_false" and config.option_prefix_style is None:
                    tf_style = rng.choice(["lowercase_paren", "lowercase_dot", "bold_lowercase_paren"])
                    current_prefixes = OPTION_PREFIX_STYLES[tf_style]
                    
                for opt_idx, opt_text in enumerate(options):
                    opt_lbl = current_prefixes[opt_idx % len(current_prefixes)]
                    opt_lbl = augment_opt_lbl(opt_lbl, config, rng)
                    append_segment(opt_lbl, "option_label")
                    append_segment(opt_text, "option_text")
                    if opt_idx < len(options) - 1:
                        if is_inline:
                            sep = generate_random_inline_separator(config, rng)
                            append_segment(sep, "separator")
                        else:
                            append_segment(config.separator_options, "separator")
                        
    result["raw_text"] = raw_text
    result["spans"] = spans
    return result

def reconstruct_exam(exam_data: Dict[str, Any], config: Optional[ReconstructorConfig] = None) -> Dict[str, Any]:
    """
    Reconstructs the entire exam document, shifting character-level offsets of questions to global document-level offsets.
    
    Returns a copy of exam_data enriched with "raw_text" and "spans" keys.
    """
    if config is None:
        config = ReconstructorConfig()
        
    result = dict(exam_data)
    
    stable_seed = config.seed
    if stable_seed is None:
        stable_seed = exam_data.get("exam_id", "") or str(exam_data)
    rng = get_stable_random(stable_seed)
    
    full_text = ""
    global_spans = []
    
    def append_document_segment(text: str, label: str):
        nonlocal full_text
        if not text:
            return
            
        if label == "separator" and config.space_noise_rate > 0.0:
            if "\n" in text and not any(c.isalnum() for c in text):
                val = rng.random()
                if val < 0.10:
                    text = " "
                elif val < 0.15:
                    text = "\n\n"
                    
        start = len(full_text)
        full_text += text
        end = len(full_text)
        
        is_separator = (label == "separator")
        if not is_separator or config.track_separators:
            span_entry = {
                "start": start,
                "end": end,
                "label": label
            }
            if config.include_span_text:
                span_entry["text"] = text
            global_spans.append(span_entry)

    subj_name = exam_data.get("subject", "unknown")
    from src.generation.generator import SUBJECT_DISPLAY
    subj_display = SUBJECT_DISPLAY.get(subj_name, subj_name.upper())
    grade = exam_data.get("grade", "")
    
    header_text = f"ĐỀ THI MÔN: {subj_display.upper()} - LỚP {grade}\n"
    if config.casing_noise_prob > 0.0 and rng.random() < config.casing_noise_prob:
        header_text = header_text.lower()
    if config.formatting_noise_prob > 0.0 and rng.random() < config.formatting_noise_prob:
        header_text = apply_formatting_tag_noise(header_text.strip(), 1.0, rng) + "\n"
        
    append_document_segment(header_text, "separator")
    append_document_segment(config.separator_questions, "separator")
    
    sections = exam_data.get("sections", {})
    q_num = 1
    
    for section_title, questions in sections.items():
        sec_title_text = f"{section_title}\n"
        if config.casing_noise_prob > 0.0 and rng.random() < config.casing_noise_prob:
            sec_title_text = sec_title_text.lower()
        if config.formatting_noise_prob > 0.0 and rng.random() < config.formatting_noise_prob:
            sec_title_text = apply_formatting_tag_noise(sec_title_text.strip(), 1.0, rng) + "\n"
            
        append_document_segment(sec_title_text, "separator")
        append_document_segment(config.separator_stem_options, "separator")
        
        questions_list = list(questions)
        if config.enable_permutations:
            rng.shuffle(questions_list)
            
        for q_idx, q in enumerate(questions_list):
            q_seed = q.get("context", "") or q.get("stem", "") or str(q)
            q_config = ReconstructorConfig(
                question_prefix_template=config.question_prefix_template,
                option_prefix_style=config.option_prefix_style,
                ordering_item_label_style=config.ordering_item_label_style,
                ordering_item_prefix_template=config.ordering_item_prefix_template,
                separator_stem_options=config.separator_stem_options,
                separator_options=config.separator_options,
                separator_context_questions=config.separator_context_questions,
                separator_questions=config.separator_questions,
                ordering_choice_separator=config.ordering_choice_separator,
                track_separators=config.track_separators,
                include_span_text=config.include_span_text,
                seed=q_seed,
                randomize_q_num=False,
                typo_rate=config.typo_rate,
                space_noise_rate=config.space_noise_rate,
                latex_mask_prob=config.latex_mask_prob,
                latex_placeholder=config.latex_placeholder,
                enable_permutations=config.enable_permutations,
                option_drop_prob=config.option_drop_prob,
                casing_noise_prob=config.casing_noise_prob,
                synonym_swap_prob=config.synonym_swap_prob,
                formatting_noise_prob=config.formatting_noise_prob
            )
            
            q_reconstructed = reconstruct_question(q, q_config, start_q_num=q_num)
            q_text = q_reconstructed["raw_text"]
            q_spans = q_reconstructed["spans"]
            
            offset = len(full_text)
            for span in q_spans:
                shifted_span = dict(span)
                shifted_span["start"] += offset
                shifted_span["end"] += offset
                global_spans.append(shifted_span)
                
            full_text += q_text
            
            if q_idx < len(questions_list) - 1:
                append_document_segment(config.separator_questions, "separator")
                
            if q.get("is_group"):
                q_num += len(q.get("questions", []))
            else:
                q_num += 1
                
        append_document_segment(config.separator_questions, "separator")
        
    stripped_text = full_text.lstrip()
    leading_stripped = len(full_text) - len(stripped_text)
    raw_text = stripped_text.rstrip()
    
    if leading_stripped > 0:
        for span in global_spans:
            span["start"] -= leading_stripped
            span["end"] -= leading_stripped
            
    result["raw_text"] = raw_text
    result["spans"] = global_spans
    return result

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
