# Re-export key reconstructor classes and functions for backward compatibility

from src.generation.reconstruction.config import (
    ReconstructorConfig,
    DEFAULT_QUESTION_PREFIXES,
    OPTION_PREFIX_STYLES,
    ORDERING_ITEM_STYLES,
    get_stable_random
)
from src.generation.reconstruction.augment import (
    inject_vietnamese_typos,
    randomize_blank_tokens,
    process_latex_variations,
    strip_formatting_wrappers,
    apply_formatting_tag_noise,
    apply_casing_noise,
    get_label_synonym,
    augment_q_prefix_tpl,
    augment_q_label,
    augment_opt_lbl
)
from src.generation.reconstruction.layout import (
    generate_random_inline_separator,
    paraphrase_section_title,
    format_answer_table,
    generate_ordering_choices
)
from src.generation.reconstruction.core import (
    reconstruct_question,
    reconstruct_exam
)
