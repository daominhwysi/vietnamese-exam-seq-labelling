import random
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
    "Q{num}. "
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

def get_stable_random(seed_obj: Any) -> random.Random:
    """Generates a stable random number generator from any seed object."""
    if seed_obj is not None:
        if isinstance(seed_obj, str):
            # Compute stable hash for string
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

def reconstruct_question(q_data: Dict[str, Any], config: Optional[ReconstructorConfig] = None, start_q_num: int = 1) -> Dict[str, Any]:
    """
    Reconstructs the raw text of a question (standard or group) and tracks spans of its components.
    
    Returns a copy of q_data enriched with "raw_text" and "spans" keys.
    """
    if config is None:
        config = ReconstructorConfig()
        
    # Create copy of dictionary to avoid mutating input
    result = dict(q_data)
    
    # Establish stable RNG seeded by question content if no explicit seed is set
    stable_seed = config.seed
    if stable_seed is None:
        # Use stem (standard) or context (group) as stable seed
        stable_seed = q_data.get("context", "") or q_data.get("stem", "") or str(q_data)
        
    rng = get_stable_random(stable_seed)
    
    # Determine starting question index (randomized if configured)
    actual_start_q_num = start_q_num
    if config.randomize_q_num:
        actual_start_q_num = rng.randint(1, 40)
    
    # Determine question prefix template
    q_prefix_tpl = config.question_prefix_template
    if q_prefix_tpl is None:
        q_prefix_tpl = rng.choice(DEFAULT_QUESTION_PREFIXES)
        
    # Determine option prefix style
    opt_style_name = config.option_prefix_style
    if opt_style_name is None:
        opt_style_name = rng.choice(list(OPTION_PREFIX_STYLES.keys()))
    opt_prefixes = OPTION_PREFIX_STYLES.get(opt_style_name, OPTION_PREFIX_STYLES["capital_dot"])
    
    # Determine ordering item styles
    ord_item_style = config.ordering_item_label_style
    if ord_item_style is None:
        ord_item_style = rng.choice(["char", "index"])
    
    ord_prefix_tpl = config.ordering_item_prefix_template
    if ord_prefix_tpl is None:
        ord_prefix_tpl = rng.choice(ORDERING_ITEM_STYLES[ord_item_style]["prefixes"])
        
    ord_labels = ORDERING_ITEM_STYLES[ord_item_style]["labels"]
    
    # Initialize variables for building raw text and spans
    raw_text = ""
    spans = []
    
    def append_segment(text: str, label: str):
        nonlocal raw_text
        if not text:
            return
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
        # Group question: context followed by sub-questions
        context = q_data.get("context", "")
        append_segment(context, "context")
        
        # Sub-questions
        sub_questions = q_data.get("questions", [])
        if sub_questions:
            append_segment(config.separator_context_questions, "separator")
            
            for idx, sub_q in enumerate(sub_questions):
                q_num = actual_start_q_num + idx
                
                # Question label (e.g. "Câu 1: ")
                q_label = q_prefix_tpl.format(num=q_num)
                append_segment(q_label, "question_label")
                
                # Stem
                stem = sub_q.get("stem", "")
                append_segment(stem, "stem")
                
                # Options
                options = sub_q.get("options", [])
                if options:
                    append_segment(config.separator_stem_options, "separator")
                    
                    for opt_idx, opt_text in enumerate(options):
                        # Option label (e.g. "A. ")
                        opt_lbl = opt_prefixes[opt_idx % len(opt_prefixes)]
                        append_segment(opt_lbl, "option_label")
                        append_segment(opt_text, "option_text")
                        
                        if opt_idx < len(options) - 1:
                            append_segment(config.separator_options, "separator")
                            
                if idx < len(sub_questions) - 1:
                    append_segment(config.separator_questions, "separator")
    else:
        # Standard question: single question label + stem + options
        q_label = q_prefix_tpl.format(num=actual_start_q_num)
        append_segment(q_label, "question_label")
        
        stem = q_data.get("stem", "")
        append_segment(stem, "stem")
        
        options = q_data.get("options", [])
        if options:
            append_segment(config.separator_stem_options, "separator")
            
            if q_type == "ordering":
                # For ordering questions, formatting is special.
                # 1. Print list of items to order (e.g. a. step1, b. step2)
                item_labels = ord_labels[:len(options)]
                for opt_idx, opt_text in enumerate(options):
                    lbl = ord_prefix_tpl.format(label=item_labels[opt_idx])
                    append_segment(lbl, "ordering_item_label")
                    append_segment(opt_text, "ordering_item_text")
                    
                    if opt_idx < len(options) - 1:
                        append_segment(config.separator_options, "separator")
                        
                # 2. Print choices A, B, C, D representing permutations of ordering
                # Generate 4 choices (1 correct, 3 distractors)
                choices = generate_ordering_choices(item_labels, config.ordering_choice_separator, rng)
                append_segment(config.separator_stem_options, "separator")
                
                for choice_idx, choice_text in enumerate(choices):
                    opt_lbl = opt_prefixes[choice_idx % len(opt_prefixes)]
                    append_segment(opt_lbl, "option_label")
                    append_segment(choice_text, "option_text")
                    
                    if choice_idx < len(choices) - 1:
                        append_segment(config.separator_options, "separator")
            else:
                # Standard multiple choice / true-false option formatting
                # Note: true_false option prefixes can use lowercase options specifically if desired,
                # but to be flexible we default to opt_prefixes unless config overridden.
                current_prefixes = opt_prefixes
                if q_type == "true_false" and config.option_prefix_style is None:
                    # Prefer lowercase prefixes for true-false questions by default
                    tf_style = rng.choice(["lowercase_paren", "lowercase_dot", "bold_lowercase_paren"])
                    current_prefixes = OPTION_PREFIX_STYLES[tf_style]
                    
                for opt_idx, opt_text in enumerate(options):
                    opt_lbl = current_prefixes[opt_idx % len(current_prefixes)]
                    append_segment(opt_lbl, "option_label")
                    append_segment(opt_text, "option_text")
                    
                    if opt_idx < len(options) - 1:
                        append_segment(config.separator_options, "separator")
                        
    result["raw_text"] = raw_text
    result["spans"] = spans
    return result

def generate_ordering_choices(labels: List[str], separator: str, rng: random.Random) -> List[str]:
    """Generates 4 multiple choice ordering options (1 correct, up to 3 distractors)."""
    correct_seq = separator.join(labels)
    
    # Get all permutations
    all_perms = list(itertools.permutations(labels))
    all_seqs = [separator.join(p) for p in all_perms]
    
    # Filter out correct sequence to get distractors
    distractors = [s for s in all_seqs if s != correct_seq]
    
    # Select 3 unique random distractors
    if len(distractors) >= 3:
        selected_distractors = rng.sample(distractors, 3)
    else:
        selected_distractors = distractors
        
    candidates = [correct_seq] + selected_distractors
    # Shuffle candidates so the correct sequence is at a random position
    rng.shuffle(candidates)
    return candidates
