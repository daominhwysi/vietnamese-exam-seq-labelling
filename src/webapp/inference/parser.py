from typing import List, Dict, Any

def build_xml(raw_text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Reconstructs the full raw text with inline XML tags around each labeled span.
    """
    sorted_spans = sorted(spans, key=lambda x: (x["start"], -x["end"]))
    
    result = []
    cursor = 0
    
    for span in sorted_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]
        text = span["text"]
        
        if start < cursor:
            continue
            
        if start > cursor:
            result.append(raw_text[cursor:start])
            
        result.append(f"<{label}>{text}</{label}>")
        cursor = end
        
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])
        
    return "".join(result)

def parse_segments_to_questions(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Heuristic parser that groups sequence labeling spans into question structures.
    """
    questions = []
    current_context = ""
    current_question = None
    
    for span in spans:
        label = span["label"]
        text = span["text"].strip()
        if not text:
            continue
            
        if label == "context":
            current_context = text
            
        elif label == "question_label":
            if current_question:
                questions.append(current_question)
            current_question = {
                "question_label": text,
                "context": current_context,
                "stem": "",
                "options": [],
                "current_option_label": None,
                "explanation": ""
            }
            
        elif label == "stem":
            if current_question is None:
                current_question = {
                    "question_label": "",
                    "context": current_context,
                    "stem": text,
                    "options": [],
                    "current_option_label": None,
                    "explanation": ""
                }
            else:
                if current_question["stem"]:
                    current_question["stem"] += "\n" + text
                else:
                    current_question["stem"] = text
                    
        elif label == "option_label":
            if current_question:
                current_question["current_option_label"] = text
                
        elif label == "option_text":
            if current_question:
                opt_lbl = current_question.get("current_option_label", "")
                prefix = f"{opt_lbl} " if opt_lbl else ""
                current_question["options"].append({
                    "label": opt_lbl,
                    "text": text,
                    "full": prefix + text
                })
                current_question["current_option_label"] = None
                
        elif label == "explanation":
            if current_question:
                if current_question.get("explanation"):
                    current_question["explanation"] += "\n" + text
                else:
                    current_question["explanation"] = text
            
        elif label == "section":
            current_context = ""
            
    if current_question:
        questions.append(current_question)
        
    cleaned_questions = []
    for q in questions:
        cleaned_options = [opt["full"] for opt in q["options"]]
        cleaned_questions.append({
            "label": q.get("question_label", ""),
            "context": q.get("context", ""),
            "stem": q.get("stem", ""),
            "options": cleaned_options,
            "raw_options": q.get("options", []),
            "explanation": q.get("explanation", "")
        })
        
    return cleaned_questions
