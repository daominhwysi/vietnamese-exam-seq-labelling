import os
import json
import re
import argparse

def check_options(options):
    """
    Checks if a list of options is prefixed with index-matching labels (e.g. A., B., C., D.).
    Returns a tuple of (is_prefixed, cleaned_options)
    """
    if not isinstance(options, list) or len(options) < 2:
        return False, options
        
    # Ensure all elements are strings
    if not all(isinstance(opt, str) for opt in options):
        return False, options

    cleaned = []
    matches_all = True
    
    for i, opt in enumerate(options):
        # Expected letters for index i: 'A'/'a' for 0, 'B'/'b' for 1, etc.
        upper_char = chr(65 + i)
        lower_char = chr(97 + i)
        
        # Regex to match prefix: letter, followed by divider (. or ) or : or - or space), followed by optional space
        # We require some divider character so we don't match words starting with A/B/C/D.
        pattern = rf"^\s*([{upper_char}{lower_char}])\s*([\.\)\:\-\s])\s*(.*)"
        match = re.match(pattern, opt, re.DOTALL)
        
        if match:
            # Group 3 is the remaining content
            cleaned.append(match.group(3))
        else:
            matches_all = False
            break
            
    if matches_all:
        return True, cleaned
    else:
        return False, options

def process_file(filepath, dry_run=True):
    """
    Loads a JSON file, detects option prefixes, cleans them, and writes back if not dry_run.
    Returns (num_detected_questions, num_cleaned_questions)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return 0, 0
            
    changed = False
    detected_count = 0
    
    if 'sections' not in data:
        return 0, 0
        
    for section_name, questions in data['sections'].items():
        for q in questions:
            # Handle standard question
            if not q.get('is_group', False):
                if 'options' in q and q['options']:
                    is_prefixed, cleaned_opts = check_options(q['options'])
                    if is_prefixed:
                        detected_count += 1
                        if not dry_run:
                            # Verify if there is actual change
                            if q['options'] != cleaned_opts:
                                q['options'] = cleaned_opts
                                changed = True
            else:
                # Handle group question
                if 'questions' in q:
                    for sub_q in q['questions']:
                        if 'options' in sub_q and sub_q['options']:
                            is_prefixed, cleaned_opts = check_options(sub_q['options'])
                            if is_prefixed:
                                detected_count += 1
                                if not dry_run:
                                    if sub_q['options'] != cleaned_opts:
                                        sub_q['options'] = cleaned_opts
                                        changed = True
                                        
    if changed and not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    return detected_count, (detected_count if changed else 0)

def main():
    parser = argparse.ArgumentParser(description="Detect and clean options prefixes in exam files.")
    parser.add_argument('--dir', default="output/exams", help="Directory containing JSON files.")
    parser.add_argument('--clean', action='store_true', help="Actually modify the files (default is dry-run).")
    args = parser.parse_args()
    
    target_dir = args.dir
    dry_run = not args.clean
    
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return
        
    files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    print(f"Found {len(files)} JSON files in {target_dir}")
    
    total_detected = 0
    total_cleaned = 0
    modified_files_count = 0
    
    for filename in files:
        filepath = os.path.join(target_dir, filename)
        detected, cleaned = process_file(filepath, dry_run=dry_run)
        if detected > 0:
            total_detected += detected
            total_cleaned += cleaned
            action_str = "Cleaned" if not dry_run else "Detected"
            print(f"[{action_str}] {filename}: {detected} question(s) with prefixed options.")
            if cleaned > 0 or (dry_run and detected > 0):
                modified_files_count += 1
                
    print("\n" + "="*50)
    if dry_run:
        print(f"Dry-run finished. Detected {total_detected} questions with prefixed options across {modified_files_count} file(s).")
        print("Run with `--clean` to write changes back to files.")
    else:
        print(f"Cleanup finished. Cleaned {total_cleaned} questions across {modified_files_count} file(s).")
    print("="*50)

if __name__ == "__main__":
    main()
