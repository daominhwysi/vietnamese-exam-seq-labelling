import re
import bisect
from typing import List, Tuple

def is_valid_latex(content: str) -> bool:
    content_stripped = content.strip()
    if not content_stripped:
        return False
    
    if len(content_stripped) == 1:
        return content_stripped.isalnum()
        
    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []
    for char in content_stripped:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            last = stack.pop()
            if brackets[last] != char:
                return False
    if stack:
        return False
        
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
        
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True
        
    return False

def get_latex_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        content = match.group(0)[2:-2]
        if is_valid_latex(content):
            spans.append(match.span())
        
    for match in re.finditer(r"\$(?!\s)[^\$\n]+?(?<!\s)\$", text):
        span = match.span()
        content = match.group(0)[1:-1]
        if not is_valid_latex(content):
            continue
            
        overlap = False
        for d_start, d_end in spans:
            if not (span[1] <= d_start or span[0] >= d_end):
                overlap = True
                break
        if not overlap:
            spans.append(span)
            
    spans.sort(key=lambda x: x[0])
    return spans


class OffsetMapper:
    def __init__(self, latex_spans: List[Tuple[int, int]]):
        self.shifts = []
        mod_pos = 0
        orig_pos = 0
        for o_start, o_end in latex_spans:
            segment_len = o_start - orig_pos
            mod_start = mod_pos + segment_len
            mod_end = mod_start + len("[LATEX]")
            shift = o_end - mod_end
            self.shifts.append((mod_start, mod_end, shift, o_start))
            
            orig_pos = o_end
            mod_pos = mod_end
            
        self.starts = [s[0] for s in self.shifts]

    def map_idx(self, idx: int) -> int:
        if not self.shifts:
            return idx
        pos = bisect.bisect_right(self.starts, idx) - 1
        if pos >= 0:
            mod_start, mod_end, shift, o_start = self.shifts[pos]
            if idx < mod_end:
                return o_start
            return idx + shift
        return idx
