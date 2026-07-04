import random
from dataclasses import dataclass
from typing import Optional, Any

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
    explanation_layout: Optional[str] = None
    answer_table_format: Optional[str] = None       # "md", "html", "csv", "random"
    answer_table_direction: Optional[str] = None    # "horizontal", "vertical", "random"
    answer_table_chunk_size: Optional[int] = None
    paraphrase_section_titles: bool = False

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
