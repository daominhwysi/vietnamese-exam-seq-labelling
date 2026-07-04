import sys
import numpy as np

def get_compute_metrics_fn(label_list: list):
    """
    Returns a compute_metrics function that uses the provided label_list.
    """
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (-100)
        true_predictions = [
            [label_list[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]

        try:
            from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
                "accuracy": np.mean([p_v == l_v for p_seq, l_seq in zip(true_predictions, true_labels) for p_v, l_v in zip(p_seq, l_seq)])
            }
        except ImportError:
            # Fallback to token-level evaluation if seqeval is not installed
            from sklearn.metrics import f1_score, accuracy_score
            flat_preds = [p_v for p_seq in true_predictions for p_v in p_seq]
            flat_labels = [l_v for l_seq in true_labels for l_v in l_seq]
            return {
                "accuracy": accuracy_score(flat_labels, flat_preds),
                "f1_macro": f1_score(flat_labels, flat_preds, average="macro")
            }
            
    return compute_metrics
