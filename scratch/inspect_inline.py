import os
import json

def inspect_inline_samples():
    train_jsonl = "output/dataset/train.jsonl"
    if not os.path.exists(train_jsonl):
        print("Dataset not found.")
        return
        
    print("Searching for inline formatted options in train.jsonl English samples...")
    count = 0
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            metadata = data.get("metadata", {})
            if metadata.get("subject") != "english" or "20260606" not in metadata.get("source_file", ""):
                continue
                
            tokens = data.get("tokens", [])
            labels = data.get("labels", [])
            
            # Check if there are option labels on the same line
            # Reconstruct the string to print
            text_parts = []
            current_span = []
            for t, l in zip(tokens, labels):
                if l == -100:
                    continue
                t_clean = t.replace("\u2581", " ")
                text_parts.append(t_clean)
                
            reconstructed_text = "".join(text_parts)
            
            # Check if option labels like A. and B. are on the same line with some spaces
            # (e.g. "A. ... B. ...") instead of separate lines.
            # Let's count how many times we see option_label sequences on the same line.
            # Let's look for "A. " and "B. " in close proximity without a newline character in between.
            lines = reconstructed_text.split("\n")
            for line_idx, l_text in enumerate(lines):
                # Search for multiple options on the same line, e.g. "A." and "B." or "C." and "D."
                if ("A." in l_text and "B." in l_text) or ("B." in l_text and "C." in l_text) or ("C." in l_text and "D." in l_text):
                    print(f"\nLine {line_num} (Source: {metadata.get('source_file')}) - Sub-line {line_idx}:")
                    print(f"Content: {repr(l_text)}")
                    
                    # Print tokens and labels for this sub-line segment
                    # Find where this line resides in tokens
                    # Let's just find the corresponding tokens for this line
                    # and print them
                    count += 1
                    if count >= 5:
                        return

if __name__ == "__main__":
    inspect_inline_samples()
