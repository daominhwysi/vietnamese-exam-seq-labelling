# Data Structure Summary

## Overview

This project generates synthetic Vietnamese exam questions for **sequence labelling** model training. Questions are generated via the DeepSeek API, returned in XML, parsed, enriched with metadata, and saved as individual JSON files.

---

## Enums (Domain Model)

### Subject

| Value           | Display Name      |
| --------------- | ----------------- |
| `economics_law` | Kinh tế pháp luật |
| `geography`     | Địa lý            |
| `history`       | Lịch sử           |
| `math_algebra`  | Toán Đại số       |
| `math_geometry` | Toán hình học     |
| `physics`       | Vật lý            |
| `chemistry`     | Hóa học           |

### QuestionType

| Value                   | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `multiple_choice`       | Single question, 4 options (A-D)                                 |
| `true_false`            | Single question, 4 true/false statements (a-d)                   |
| `short_answer`          | Single question, no options — open-ended short answer            |
| `ordering`              | Single question, 4 items to reorder chronologically/sequentially |
| `group_multiple_choice` | Shared context + 3-4 multiple-choice sub-questions               |
| `group_short_answer`    | Shared context + 2-3 short-answer sub-questions                  |

### Difficulty (weighted random selection)

| Value              | Display Name    | Weight |
| ------------------ | --------------- | ------ |
| `recognize`        | Nhận biết       | 30%    |
| `comprehend`       | Thông hiểu      | 30%    |
| `low_application`  | Vận dụng thấp   | 10%    |
| `application`      | Vận dụng thường | 20%    |
| `high_application` | Vận dụng cao    | 10%    |

---

## Output JSON Schemas

Each generated file is named: `question_{subject}_g{grade}_{timestamp}_{uuid6}.json` and saved under `output/`.

### 1. Standard Question (`is_group: false`)

```json
{
  "is_group": false,
  "stem": "string — question body with LaTeX in $...$",
  "options": ["string", "string", "..."],
  "subject": "chemistry | physics | geography | ...",
  "grade": 8 | 9 | 10 | 11 | 12,
  "question_type": "multiple_choice | true_false | short_answer | ordering",
  "difficulty": "recognize | comprehend | low_application | application | high_application"
}
```

- `options` length varies by `question_type`:
  - `multiple_choice` / `true_false` / `ordering`: **4 options**
  - `short_answer`: **0 options** (empty array)

### 2. Group Question (`is_group: true`)

```json
{
  "is_group": true,
  "context": "string — shared context/passage with LaTeX in $...$",
  "questions": [
    {
      "stem": "string — sub-question body",
      "options": ["string", "string", "..."]
    }
  ],
  "subject": "chemistry | physics | geography | ...",
  "grade": 8 | 9 | 10 | 11 | 12,
  "question_type": "group_multiple_choice | group_short_answer",
  "difficulty": "recognize | comprehend | low_application | application | high_application"
}
```

- `group_multiple_choice`: each sub-question has **4 options**
- `group_short_answer`: each sub-question has **0 options** (empty array)

---

## AI Prompting Flow

1. **Randomize parameters** — `Subject`, `grade` (8-12), `Difficulty` (weighted), and `QuestionType`.
2. **5% chance** of generating a group question (`group_multiple_choice` or `group_short_answer`).
3. **Build system + user prompts** with:
   - Vietnamese-language instructions for the specific subject/grade/difficulty.
   - Target stem length (30 words for MC, 100 for true/false, 150 for short answer, 50 for ordering).
   - XML output format specification.
   - Rules: LaTeX in `$...$`, no answer key, no option prefixes ("A.", "a)", etc.).
4. **Call DeepSeek API** via `deepseek_client.chat()`, retry up to 3 times on failure.
5. **Parse XML** with `parse_question_xml()` — regex extracts `<stem>`, `<option>`, `<context>`, and sub-`<question>` elements.
6. **Attach metadata** (subject, grade, question_type, difficulty) and save as JSON.

---

## Key XML Formats (AI Output)

### Standard Question

```xml
<question>
  <stem>Question body with $LaTeX$...</stem>
  <option>Option content (no prefix)</option>
  <option>Option content</option>
  <option>Option content</option>
  <option>Option content</option>
</question>
```

### Group Question

```xml
<group_question>
  <context>Shared passage/context...</context>
  <question>
    <stem>Sub-question 1...</stem>
    <option>Option A</option>
    <option>Option B</option>
    <option>Option C</option>
    <option>Option D</option>
  </question>
  <question>
    <stem>Sub-question 2...</stem>
    ...
  </question>
</group_question>
```

### Example

```json
{
  "is_group": false,
  "stem": "Trong phòng thí nghiệm, một học sinh cho 6,5 gam kẽm ($Zn$) tác dụng vừa đủ với dung dịch axit clohidric ($HCl$) thu được muối kẽm clorua ($ZnCl_2$) và khí hiđro ($H_2$). Học sinh đó đã viết phương trình phản ứng: $Zn + 2HCl \\to ZnCl_2 + H_2$. Dựa vào phản ứng này, hãy xét tính đúng – sai của các nhận định sau:",
  "options": [
    "Phương trình hóa học $Zn + 2HCl \\rightarrow ZnCl_2 + H_2$ đã cân bằng đúng.",
    "$1$ mol $Zn$ phản ứng với $2$ mol $HCl$ tạo ra $1$ mol $ZnCl_2$ và $2$ mol $H_2$.",
    "Khối lượng $ZnCl_2$ thu được bằng tổng khối lượng $Zn$ và $HCl$ đã phản ứng.",
    "Trong phản ứng trên, số nguyên tử $Zn$ và $H$ luôn được bảo toàn."
  ],
  "subject": "chemistry",
  "grade": 8,
  "question_type": "true_false",
  "difficulty": "comprehend"
}
```
