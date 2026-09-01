# Vietnamese Exam Sequence Labelling Pipeline

Pipeline sinh dữ liệu, gán nhãn OCR, chuẩn bị dataset và huấn luyện mô hình Sequence Labeling (mmBERT-base, ModernBERT) để bóc tách cấu trúc đề thi giáo dục Việt Nam và đề thi ngoại ngữ.

---

## 🌟 Key Capabilities

- **Bóc tách cấu trúc đề thi**: Phân đoạn tự động nhãn câu hỏi (`question_label`), thân câu hỏi (`stem`), nhãn phương án (`option_label`), nội dung phương án (`option_text`), bài đọc/ngữ liệu (`stimulus`), tiêu đề phần thi (`section`), lời giải & barem điểm (`explanation`).
- **Hỗ trợ 11 môn học & nhiều format kỳ thi**:
  - THPT Quốc gia (GDPT 2018): Toán (Đại số & Hình học), Vật lí, Hóa học, Sinh học, Lịch sử, Địa lí, GD Kinh tế & Pháp luật, Tiếng Anh.
  - Ngữ văn: Đọc hiểu 5 câu, Nghị luận xã hội 200 chữ, Nghị luận văn học 600 chữ (kèm barem 4 tiêu chí), đề HSGQG (20.0đ).
  - Ngoại ngữ: TOEIC trọn vẹn 7 Parts (Single/Multi Passages, Chat Chains, Bảng biểu).
  - Đánh giá Năng lực (ĐHQG HN/HCM) & Đánh giá Tư duy (ĐHBK HN).
- **Sinh dữ liệu nhân tạo (Synthetic Data)**: Sinh câu hỏi và biên soạn đề thi bám sát chương trình lớp 8–12 qua DeepSeek reasoning models.
- **Xử lý đề thi scan/PDF thực tế (Real OCR Pipeline)**: OCR PDF sang Markdown và tự động gán nhãn XML.
- **Template Bank & Data Augmentation**: Kho 168 mẫu tiêu đề, lời giải, barem điểm và bảng đáp án kết hợp nhiễu bố cục (nén khoảng trắng, casing, markdown styling, LaTeX formulas).
- **Web App**: Giao diện duyệt đề thi (cổng 8000) và ứng dụng phân đoạn trực tiếp (cổng 8001).

---

## ⚡ CLI & Pixi Tasks Reference

Dự án sử dụng **Pixi** để quản lý môi trường. Dưới đây là danh sách đầy đủ các lệnh:

### 1. Sinh đề thi nhân tạo (`generate`)
```bash
# Sinh 5 đề thi ngẫu nhiên
pixi run generate -n 5

# Sinh 10 đề thi môn Vật lí lớp 11
pixi run generate -n 10 --subject physics --grade 11

# Sinh đề thi môn Văn với 8 luồng đồng thời
pixi run generate -n 5 --subject literature --grade 12 --concurrency 8
```
* **Tham số**:
  * `-n, --num-exams`: Số lượng đề thi cần sinh (mặc định: `1`).
  * `-s, --subject`: Môn học (`math_algebra`, `math_geometry`, `physics`, `chemistry`, `biology`, `history`, `geography`, `economics_law`, `literature`, `english`, `toeic`).
  * `-g, --grade`: Khối lớp (`8` đến `12`).
  * `-c, --concurrency`: Số luồng sinh câu hỏi đồng thời (mặc định: `8`).
  * `--output-dir`: Thư mục lưu file JSON (mặc định: `output/exams`).

---

### 2. Sửa dữ liệu gốc trên đĩa (`fix-root-data`)
Quét đệ quy thư mục `output/`, sửa các file XML và JSON để đưa trích dẫn nguồn `(Adapted from...)`, `(Nguồn:...)` vào trong thẻ `<stimulus>`.
```bash
pixi run fix-root-data

# Tùy chỉnh thư mục quét
python src/data/fix_root_data.py --input-dir output/real_annotated
```
* **Tham số**:
  * `--input-dir`: Thư mục cần quét và sửa (mặc định: `output`).

---

### 3. Gom dữ liệu thô (`consolidate-raw`)
Gom toàn bộ đề thi OCR thật và đề synthetic trong `output/` vào file `output/dataset/raw_exams.jsonl`.
```bash
pixi run consolidate-raw
```
* **Tham số**:
  * `--input, -i`: Thư mục đầu vào (mặc định: `output`).
  * `--output, -o`: Đường dẫn file JSONL đầu ra (mặc định: `output/dataset/raw_exams.jsonl`).

---

### 4. Chuẩn bị Dataset Tokenized Offline (`prepare-offline-dataset` / `prepare-dataset`)
Cắt sliding window (512, 1024), căn chỉnh nhãn token và xuất ra `train.jsonl`, `val.jsonl`, `test.jsonl`, `xml/`.
```bash
pixi run prepare-offline-dataset

# Tùy chỉnh tham số
sequence-labelling-generator prepare \
    --model jhu-clsp/mmBERT-base \
    --window-sizes 512,1024 \
    --strides 128,256 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```
* **Tham số**:
  * `--model, -m`: Tên model base để lấy tokenizer (mặc định: `jhu-clsp/mmBERT-base`).
  * `--window-sizes`: Độ dài cửa sổ trượt (mặc định: `512,1024`).
  * `--strides`: Bước trượt (mặc định: `128,256`).
  * `--output-dir, -o`: Thư mục lưu dataset (mặc định: `output/dataset`).

---

### 5. Upload Dataset lên Hugging Face Hub

#### a. Upload cho Online Training (`upload-online-dataset`)
Chỉ upload `raw_exams.jsonl` và `label_mapping.json` (dung lượng ~35MB, hoàn tất trong vài giây).
```bash
pixi run upload-online-dataset

# Tùy chỉnh repo
python src/data/upload.py --mode online-only --repo-id daominhwysi/synthetic-seq-labelling-vi-exam-v2
```

#### b. Upload đầy đủ Offline (`upload-dataset`)
Upload toàn bộ dataset splits (`train.jsonl`, `val.jsonl`, `test.jsonl`) và thư mục `xml/`.
```bash
pixi run upload-dataset
```

---

### 6. Huấn luyện mô hình (`train`)

#### a. Chế độ Online Augmentation (Tự động tải `raw_exams.jsonl` từ Hub)
```bash
torchrun --nproc_per_node=2 src/model/train.py \
    --online-augmentation \
    --repo_id daominhwysi/synthetic-seq-labelling-vi-exam-v2 \
    --model_name jhu-clsp/mmBERT-base \
    --output_dir output/models/mmbert-vi-exam-v5 \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --enhanced-head \
    --fp16
```

#### b. Chế độ Offline Dataset (Từ thư mục local)
```bash
torchrun --nproc_per_node=2 src/model/train.py \
    --data-dir output/dataset \
    --model_name jhu-clsp/mmBERT-base \
    --output_dir output/models/mmbert-vi-exam-v5 \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --enhanced-head \
    --fp16
```
* **Tham số chính**:
  * `--model_name`: Model backbone (`jhu-clsp/mmBERT-base` hoặc `answerdotai/ModernBERT-base`).
  * `--enhanced-head`: Kích hoạt Layer Pooling (4 lớp) + Dense MLP + Multi-Sample Dropout (5 masks) + Focal Loss ($\gamma=2.0$).
  * `--online-augmentation`: Tự động thêm nhiễu bố cục trực tiếp trong RAM khi huấn luyện.
  * `--lora_r`, `--lora_alpha`: Cấu hình rank/alpha cho LoRA (mặc định: `16`, `32`).
  * `--fp16` / `--use_bf16`: Bật huấn luyện Mixed Precision.

---

### 7. Phân đoạn thư mục đề thi thô (`batch-inference`)
Phân đoạn toàn bộ các file `.txt` trong thư mục ra JSON cấu trúc, TXT dự đoán và XML gắn thẻ.
```bash
pixi run batch-inference

# Tùy chỉnh đường dẫn
python src/inference/predict_folder.py \
    --input-dir scratch/test_raw_exams \
    --output-dir scratch/inference_results \
    --model-path output/models/mmbert-vi-exam-v5
```
* **Tham số**:
  * `--input-dir`: Thư mục chứa file text đề thi thô.
  * `--output-dir`: Thư mục lưu kết quả xuất ra.
  * `--model-path`: Đường dẫn model checkpoint / LoRA adapter.

---

### 8. Web App

```bash
# Trình duyệt quản lý & xem nhãn đề thi (Cổng 8000)
pixi run view-exams

# Ứng dụng phân đoạn đề thi trực tiếp (Cổng 8001)
pixi run view-inference
```

---

### 9. Kiểm thử (`test`)
```bash
pixi run test
```

---

### 10. Upload Model lên Hugging Face Hub (`upload-model`)
```bash
pixi run upload-model

# Hoặc chỉ định rõ repo
sequence-labelling-generator upload-model \
    --model-dir output/models/mmbert-vi-exam-v5 \
    --repo-id daominhwysi/mmbert-small-vi-exam-seq-labeling-v5
```

---

## 🚀 Pipeline Overview

Quy trình hoạt động gồm 3 bước:
1. **Chuẩn bị Dữ liệu**: Sinh câu hỏi qua LLM + Gán nhãn đề OCR thật $\rightarrow$ Sửa dữ liệu gốc (`fix-root-data`) $\rightarrow$ Gom vào `raw_exams.jsonl` (`consolidate-raw`).
2. **Huấn luyện Mô hình**: Upload dữ liệu lên Hub (`upload-online-dataset`) $\rightarrow$ Huấn luyện `mmBERT-base` với Enhanced Head và Focal Loss trên multi-GPU (Kaggle/Server).
3. **Phân đoạn & Khai thác**: Chạy bóc tách cấu trúc đề thi qua CLI (`batch-inference`) hoặc Web App (`view-inference`).

---

## 🏷️ Sequence Labeling Tag Set

| Tag | Đối tượng gán nhãn | Ví dụ |
| :--- | :--- | :--- |
| `question_label` | Tiền tố câu hỏi | `Câu 1:`, `Question 5:`, `Bài 2.` |
| `stem` | Thân câu hỏi (văn bản, công thức LaTeX) | `Cho hàm số $y=f(x)$ liên tục trên $\mathbb{R}$...` |
| `option_label` | Ký hiệu phương án | `A.`, `B.`, `C.`, `D.`, `a)`, `b)` |
| `option_text` | Nội dung phương án | `x = 2 hoặc x = 3` |
| `stimulus` | Đoạn văn đọc hiểu / Ngữ liệu dùng chung | Đoạn trích văn học, bài báo, trích dẫn nguồn |
| `section` | Tiêu đề phần thi, chỉ dẫn làm bài | `PHẦN I. Câu trắc nghiệm...`, `PART 7` |
| `explanation` | Lời giải chi tiết, barem điểm, bảng đáp án | `* Lời giải: ...`, `| Câu | Đáp án | Điểm |` |

---

## 📁 Project Structure

```text
.
├── pyproject.toml                  # Cấu hình Pixi tasks và dependencies
├── pixi.lock                       # Lockfile môi trường
├── config.yaml                     # Cấu hình pipeline chung
├── AGENTS.md                       # Tài liệu hướng dẫn agent
├── README.md                       # Tài liệu dự án (file này)
│
├── tests/                          # 82 Unit tests
│   ├── test_template_bank.py       # Test Template Bank và formatting
│   ├── test_reconstructor.py       # Test tái tạo văn bản và spans
│   ├── test_fix_root_data.py       # Test sửa dữ liệu gốc XML/JSON
│   ├── test_prepare_dataset.py     # Test tokenizer alignment và sliding window
│   ├── test_enhanced_head.py       # Test Layer Pooling, MSD và Focal Loss
│   ├── test_iter_logger.py         # Test logger huấn luyện
│   ├── test_exam_compiler.py       # Test compiler đề thi
│   ├── test_curriculum.py          # Test curriculum loader
│   ├── test_parser.py              # Test XML parser
│   └── test_codex_provider.py      # Test LLM client
│
├── logs/                           # Báo cáo kỹ thuật & token usage
│   ├── RUN_3_REPORT.md             # Báo cáo lỗi Run #3
│   ├── RUN_4_REPORT.md             # Báo cáo kết quả Run #4 (Recall 97.2%)
│   ├── RUN_5_PREPARATION_REPORT.md # Báo cáo chuẩn bị Run #5
│   └── token_usage_<DATE>.jsonl    # Log token DeepSeek API
│
├── output/                         # Dữ liệu xuất ra (gitignored)
│   ├── dataset/                    # Dataset chuẩn hóa (raw_exams.jsonl, splits, label_mapping.json, xml/)
│   ├── exams/                      # Đề thi synthetic JSON
│   └── real_annotated/             # Đề thi OCR thật JSON/XML
│
├── scratch/                        # Scripts kiểm thử & đánh giá định dạng
│   ├── test_customer_formats.py    # Test định dạng ĐGNL, ĐGTD, SAT, IELTS
│   ├── test_toeic_deep.py          # Test TOEIC bảng biểu và song ngữ
│   ├── test_toeic_remaining.py     # Test TOEIC chat chains và triple passages
│   ├── generate_bio_lit_50.py      # Script sinh batch đề Văn & Sinh
│   └── generate_toeic_lit_50.py    # Script sinh batch đề Văn & TOEIC
│
└── src/                            # Mã nguồn chính
    ├── cli.py                      # CLI console entry point
    │
    ├── data/                       # Xử lý dữ liệu & OCR
    │   ├── fix_root_data.py        # Sửa dữ liệu gốc XML/JSON
    │   ├── prepare.py              # Tokenize dataset & sliding window
    │   ├── upload.py               # Upload dataset lên HF Hub (online-only / all)
    │   ├── pdf_converter.py        # OCR PDF sang Markdown
    │   └── annotate_ocr.py         # Gán nhãn Markdown OCR sang JSON/XML
    │
    ├── generation/                 # Sinh dữ liệu
    │   ├── template_bank.py        # Template bank (168 mẫu layout)
    │   ├── reconstructor.py        # Tái tạo raw text & offset spans
    │   ├── generator.py            # Prompt AI sinh câu hỏi
    │   ├── exam_compiler.py        # Biên soạn đề thi hoàn chỉnh
    │   ├── curriculum.py           # Quản lý khung chương trình
    │   ├── parser.py               # Parse XML câu hỏi từ LLM
    │   └── deepseek_client.py      # Client DeepSeek API
    │
    ├── model/                      # Huấn luyện & Export
    │   ├── head.py                 # Enhanced Classification Head & Focal Loss
    │   ├── train.py                # Train script (LoRA, DDP, Online/Offline)
    │   ├── upload.py               # Upload model lên HF Hub
    │   └── export.py               # Export model sang ONNX
    │
    ├── inference/                  # Suy diễn
    │   ├── predict.py              # Single inference
    │   └── predict_folder.py       # Batch inference thư mục
    │
    ├── utils/                      # Tiện ích
    │   ├── config.py               # Parse config.yaml
    │   ├── token_tracker.py        # Token tracker logger
    │   └── visualize.py            # Tạo HTML visualizer
    │
    └── webapp/                     # Web Application
        ├── main.py                 # FastAPI exam viewer (Cổng 8000)
        ├── inference_app.py        # FastAPI live inference (Cổng 8001)
        ├── inference_helper.py     # Sliding window & LaTeX masking helper
        └── templates/              # Jinja2 templates
```
