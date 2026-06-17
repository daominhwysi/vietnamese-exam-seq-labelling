import sys
sys.stdout.reconfigure(encoding='utf-8')
from src.generation.reconstructor import reconstruct_question, ReconstructorConfig
from src.training.prepare_dataset import get_tag_mappings, align_tokens_to_spans
from transformers import AutoTokenizer

def main():
    config = ReconstructorConfig(
        typo_rate=0.0,
        space_noise_rate=0.0,
        latex_mask_prob=0.0,
        latex_placeholder='[LATEX]'
    )
    q_data = {
        'is_group': False,
        'stem': 'Tại hai điểm A và B trong không khí cách nhau 10 cm, đặt các điện tích.',
        'options': ['A. 1', 'B. 2'],
        'answer': 'A',
        'question_type': 'multiple_choice',
        'subject': 'physics',
        'grade': 11
    }
    q_rec = reconstruct_question(q_data, config, start_q_num=1)
    raw_text = q_rec['raw_text']
    spans = q_rec['spans']

    tokenizer = AutoTokenizer.from_pretrained('aisingapore/SEA-LION-ModernBERT-300M')
    special_tokens = ['<blank />', '<blank/>', '[BLANK]', '[LATEX]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    tokenized = tokenizer(raw_text, return_offsets_mapping=True)
    tag_to_id, id_to_tag = get_tag_mappings()
    labels = align_tokens_to_spans(tokenized['offset_mapping'], spans, tag_to_id)
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])

    print('Raw Text:', repr(raw_text))
    print('\nSpans:')
    for s in spans:
        print(f"  {s['label']}: ({s['start']}, {s['end']}) -> {repr(raw_text[s['start']:s['end']])}")

    print('\nTokens and Aligned Labels:')
    for t, o, l in zip(tokens, tokenized['offset_mapping'], labels):
        tag = id_to_tag.get(l, 'IGNORE') if l != -100 else 'IGNORE'
        print(f'{t:12} | {o} | label: {tag}')

if __name__ == "__main__":
    main()
