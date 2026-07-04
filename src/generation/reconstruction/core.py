import random
import re
import copy
from typing import List, Dict, Any, Optional

from src.generation.reconstruction.config import (
    ReconstructorConfig,
    DEFAULT_QUESTION_PREFIXES,
    OPTION_PREFIX_STYLES,
    ORDERING_ITEM_STYLES,
    get_stable_random
)
from src.generation.reconstruction.augment import (
    randomize_blank_tokens,
    process_latex_variations,
    inject_vietnamese_typos,
    strip_formatting_wrappers,
    apply_formatting_tag_noise,
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
    
    def append_raw_segment(text: str, label: str):
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
 
    def append_segment(text: str, label: str):
        nonlocal raw_text
        if not text:
            return
            
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
                        
        if label != "separator":
            if label in ["option_label", "question_label"]:
                # For option_label and question_label, if it has a bold wrapper "**...**", keep the "**" inside the label span.
                # However, we still strip leading/trailing whitespace outside the bold wrapper, and keep other wraps if any.
                text_stripped = text.strip()
                if text_stripped.startswith("**") and text_stripped.endswith("**") and len(text_stripped) >= 5:
                    l_spaces = text[:text.find("**")]
                    r_spaces = text[text.rfind("**")+2:]
                    core_val = text[text.find("**"):text.rfind("**")+2]
                    if l_spaces:
                        append_raw_segment(l_spaces, "separator")
                    append_raw_segment(core_val, label)
                    if r_spaces:
                        append_raw_segment(r_spaces, "separator")
                else:
                    leading, core_val, trailing = strip_formatting_wrappers(text)
                    if leading:
                        append_raw_segment(leading, "separator")
                    if core_val:
                        append_raw_segment(core_val, label)
                    if trailing:
                        append_raw_segment(trailing, "separator")
            else:
                leading, core_val, trailing = strip_formatting_wrappers(text)
                if leading:
                    append_raw_segment(leading, "separator")
                if core_val:
                    append_raw_segment(core_val, label)
                if trailing:
                    append_raw_segment(trailing, "separator")
        else:
            append_raw_segment(text, "separator")
 
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
                        
    # Add explanation support
    if is_group:
        sub_questions = q_data.get("questions", [])
        has_any_expl = any(sub_q.get("explanation") for sub_q in sub_questions)
        if has_any_expl:
            append_segment(config.separator_questions, "separator")
            append_segment("Hướng dẫn giải:\n", "section")
            for sub_idx, sub_q in enumerate(sub_questions):
                sub_ans = sub_q.get("answer", "")
                sub_expl = sub_q.get("explanation", "")
                sub_num = sub_idx + 1
                
                append_segment(f"Câu {sub_num}:", "question_label")
                
                expl_content = ""
                if sub_ans:
                    expl_content += f" {sub_ans}"
                if sub_expl:
                    expl_content += f"\n{sub_expl}"
                
                if expl_content:
                    append_segment(expl_content, "explanation")
                
                if sub_idx < len(sub_questions) - 1:
                    append_segment("\n\n", "separator")
    else:
        explanation = q_data.get("explanation", "")
        if explanation:
            append_segment(config.separator_questions, "separator")
            ans = q_data.get("answer", "")
            if ans:
                append_segment("**Đáp án **", "separator")
                append_segment(f"{ans}. Giải thích:**\n\n{explanation}", "explanation")
            else:
                append_segment("**Giải thích:**\n\n", "separator")
                append_segment(explanation, "explanation")
 
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
    
    expl_layout = config.explanation_layout
    if expl_layout not in ["table_only", "interleaved", "separated"]:
        roll = rng.random()
        if roll < 0.70:
            expl_layout = "table_only"
        elif roll < 0.75:
            expl_layout = "interleaved"
        else:
            expl_layout = "separated"
    
    full_text = ""
    global_spans = []
    exam_records = []
    
    def append_raw_document_segment(text: str, label: str):
        nonlocal full_text
        if not text:
            return
            
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
                    
        if label != "separator":
            leading, core, trailing = strip_formatting_wrappers(text)
            if leading:
                append_raw_document_segment(leading, "separator")
            if core:
                append_raw_document_segment(core, label)
            if trailing:
                append_raw_document_segment(trailing, "separator")
        else:
            append_raw_document_segment(text, "separator")

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
        if config.paraphrase_section_titles:
            section_title = paraphrase_section_title(section_title, rng)
        sec_title_text = f"{section_title}\n"
        if config.casing_noise_prob > 0.0 and rng.random() < config.casing_noise_prob:
            sec_title_text = sec_title_text.lower()
        if config.formatting_noise_prob > 0.0 and rng.random() < config.formatting_noise_prob:
            sec_title_text = apply_formatting_tag_noise(sec_title_text.strip(), 1.0, rng) + "\n"
            
        append_document_segment(sec_title_text, "section")
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
            
            q_copy = copy.deepcopy(q)
            if expl_layout in ["separated", "table_only"]:
                q_copy.pop("explanation", None)
                if q_copy.get("is_group") and "questions" in q_copy:
                    for sub_q in q_copy["questions"]:
                        sub_q.pop("explanation", None)
            
            q_reconstructed = reconstruct_question(q_copy, q_config, start_q_num=q_num)
            q_text = q_reconstructed["raw_text"]
            q_spans = q_reconstructed["spans"]
            
            offset = len(full_text)
            for span in q_spans:
                shifted_span = dict(span)
                shifted_span["start"] += offset
                shifted_span["end"] += offset
                global_spans.append(shifted_span)
                
            full_text += q_text
            
            if q.get("is_group"):
                sub_qs = q.get("questions", [])
                for sub_idx, sub_q in enumerate(sub_qs):
                    exam_records.append({
                        "num": q_num + sub_idx,
                        "answer": sub_q.get("answer", ""),
                        "explanation": sub_q.get("explanation", "")
                    })
                next_q_num_increment = len(sub_qs)
            else:
                exam_records.append({
                    "num": q_num,
                    "answer": q.get("answer", ""),
                    "explanation": q.get("explanation", "")
                })
                next_q_num_increment = 1
                
            if q_idx < len(questions_list) - 1:
                append_document_segment(config.separator_questions, "separator")
                
            q_num += next_q_num_increment
            
        append_document_segment(config.separator_questions, "separator")
        
    # Append Answer Key Section
    if exam_records:
        append_document_segment(config.separator_questions, "separator")
        
        ans_title = "ĐÁP ÁN THAM KHẢO\n"
        if config.casing_noise_prob > 0.0 and rng.random() < config.casing_noise_prob:
            ans_title = ans_title.lower()
        if config.formatting_noise_prob > 0.0 and rng.random() < config.formatting_noise_prob:
            ans_title = apply_formatting_tag_noise(ans_title.strip(), 1.0, rng) + "\n"
            
        append_document_segment(ans_title, "section")
        append_document_segment(config.separator_stem_options, "separator")
        
        # Determine table format and direction
        ans_format = config.answer_table_format
        if ans_format == "random":
            roll = rng.random()
            if roll < 0.60:
                ans_format = "md"
            elif roll < 0.80:
                ans_format = "html"
            else:
                ans_format = "csv"
        elif ans_format not in ["md", "html", "csv"]:
            ans_format = "md"
 
        ans_direction = config.answer_table_direction
        if ans_direction == "random":
            roll = rng.random()
            if roll < 0.70:
                ans_direction = "horizontal"
            else:
                ans_direction = "vertical"
        elif ans_direction not in ["horizontal", "vertical"]:
            ans_direction = "horizontal"
 
        tables_text = format_answer_table(
            exam_records=exam_records,
            format_type=ans_format,
            direction=ans_direction,
            chunk_size=config.answer_table_chunk_size,
            rng=rng
        )
        append_document_segment(tables_text, "separator")
 
    # Append Explanation Section
    has_any_expl = any(r["explanation"] for r in exam_records)
    if exam_records and has_any_expl and expl_layout == "separated":
        append_document_segment(config.separator_questions, "separator")
        
        expl_title = "LỜI GIẢI THAM KHẢO\n"
        if config.casing_noise_prob > 0.0 and rng.random() < config.casing_noise_prob:
            expl_title = expl_title.lower()
        if config.formatting_noise_prob > 0.0 and rng.random() < config.formatting_noise_prob:
            expl_title = apply_formatting_tag_noise(expl_title.strip(), 1.0, rng) + "\n"
            
        append_document_segment(expl_title, "section")
        append_document_segment(config.separator_stem_options, "separator")
        
        for r_idx, r in enumerate(exam_records):
            num = r["num"]
            ans = r["answer"]
            expl_text = r["explanation"]
            
            append_document_segment("**", "separator")
            append_document_segment(f"Câu {num}:", "question_label")
            append_document_segment(" ", "separator")
            
            expl_content = f"{ans}**"
            if expl_text:
                expl_content += f"\n\n{expl_text}"
            
            append_document_segment(expl_content, "explanation")
            
            if r_idx < len(exam_records) - 1:
                append_document_segment("\n\n", "separator")
            
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
