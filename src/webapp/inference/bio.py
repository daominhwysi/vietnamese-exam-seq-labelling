from typing import List, Dict

def resolve_bio_violations(predictions: List[int], id_to_tag: Dict[int, str], tag_to_id: Dict[str, int]) -> List[int]:
    corrected_predictions = list(predictions)
    prev_tag = "O"
    for idx, pred_id in enumerate(corrected_predictions):
        tag = id_to_tag[pred_id]
        if tag.startswith("I-"):
            tag_class = tag[2:]
            if prev_tag != f"B-{tag_class}" and prev_tag != f"I-{tag_class}":
                b_tag = f"B-{tag_class}"
                if b_tag in tag_to_id:
                    corrected_predictions[idx] = tag_to_id[b_tag]
                    tag = b_tag
                else:
                    corrected_predictions[idx] = tag_to_id.get("O", 0)
                    tag = "O"
        prev_tag = tag
    return corrected_predictions
