# Vietnamese Exam Sequence Labelling — Data Generation, Training & Inference Pipeline

An end-to-end AI system for automatically segmenting, extracting, and structuring components from raw and OCR-scanned Vietnamese educational exams, entrance tests, and bilingual language exams.

---

## 🌟 Key Features (Dự án Làm được Gì?)

- **Tự động Bóc tách & Phân đoạn Đề thi (Automated Exam Structure Extraction)**:
  Tự động nhận diện và bóc tách chính xác từng thành phần trong đề thi từ văn bản thô hoặc kết quả OCR: Thân câu hỏi (`stem`), Ký hiệu phương án (`option_label`), Nội dung đáp án (`option_text`), Đoạn văn / Bài đọc dài (`stimulus`), Tiêu đề phần thi (`section`), Lời giải chi tiết, Bảng biểu điểm barem (`explanation`) và Nhãn câu hỏi (`question_label`).

- **Bao phủ Toàn diện 11 Môn học & Mọi Format Kỳ thi tại Việt Nam**:
  - **Thi Tốt nghiệp THPT (Chuẩn GDPT 2018)**: Toán Đại số, Toán Hình học, Vật lí, Hóa học, Sinh học, Lịch sử, Địa lí, Giáo dục Kinh tế & Pháp luật, Tiếng Anh.
  - **Môn Ngữ văn Chuyên sâu**: Đọc hiểu 5 câu (4.0đ), Đoạn văn Nghị luận xã hội 200 chữ (2.0đ), Bài văn Nghị luận văn học 600 chữ / So sánh tác phẩm (4.0đ) kèm barem chấm 4 tiêu chí chuẩn Bộ GD&ĐT, và Đề thi Học sinh Giỏi Quốc gia (20.0đ).
  - **Khảo thí Quốc tế (TOEIC Parts 1–7)**: Trọn vẹn 7 phần thi (Photographs, Question-Response, Short Conversations, Short Talks, Incomplete Sentences, Text Completion, Reading Single / Double / Triple Passages, Hóa đơn bảng biểu, Chuỗi tin nhắn Chat).
  - **Đánh giá Năng lực (ĐGNL) & Đánh giá Tư duy (ĐGTD)**: ĐHQG Hà Nội, ĐHQG TP.HCM, ĐHBK Hà Nội (Tư duy định lượng, Tư duy định tính, Khoa học & Giải quyết vấn đề).

- **Sinh Đề thi Nhân tạo Không giới hạn (Synthetic Data Generation)**:
  Tự động sinh ngân hàng câu hỏi bám sát chuẩn kiến thức từng khối lớp (8 đến 12) và biên soạn thành các đề thi hoàn chỉnh với AI reasoning (DeepSeek).

- **Bóc tách & Gán nhãn Đề thi Scan / PDF Thực tế (Real OCR Annotation)**:
  Bộ công cụ chuyển đổi tài liệu PDF scan của các trường chuyên, Sở GD&ĐT thành Markdown và tự động gán nhãn thực thể với độ chính xác cao.

- **Thích ứng với Hơn 10 Triệu Biến thể Bố cục In ấn (Layout Robustness)**:
  Ngân hàng Template Tổ hợp (168 templates gốc) giúp mô hình không bị bất ngờ trước bất kỳ phong cách in ấn nào (bảng ma trận trắc nghiệm ngang/dọc, thụt lề tab, chữ hoa/thường, bảng biểu điểm, hoặc công thức toán học LaTeX phức tạp).

- **Giao diện Trực quan & Web App Phân đoạn Đề thi Trực tiếp**:
  Cung cấp 2 ứng dụng web tương tác: Trình duyệt trực quan hóa đề thi tô màu nhãn thực thể và Website phân đoạn đề thi AI trực tiếp trên trình duyệt.

---

## ⚡ Command Line & Pixi Task Reference (Tra cứu Lệnh Chi tiết)

Dự án sử dụng **Pixi** để quản lý môi trường và các tác vụ thực thi. Dưới đây là bảng hướng dẫn chi tiết cho toàn bộ các lệnh trong hệ thống:

### 1. Sinh Đề thi Nhân tạo (`generate`)
Tự động sinh các đề thi tổng hợp bám sát chương trình học và lưu vào `output/exams/`.
```bash
# Sinh 5 đề thi ngẫu nhiên
pixi run generate -n 5

# Sinh 10 đề thi môn Vật lí lớp 11
pixi run generate -n 10 --subject physics --grade 11

# Sinh đề thi với số luồng câu hỏi đồng thời cao
pixi run generate -n 5 --subject literature --grade 12 --concurrency 8
```
* **Các tham số hỗ trợ**:
  * `-n, --num-exams`: Số lượng đề thi cần sinh (mặc định: `1`).
  * `-s, --subject`: Môn học (`math_algebra`, `math_geometry`, `physics`, `chemistry`, `biology`, `history`, `geography`, `economics_law`, `literature`, `english`, `toeic`).
  * `-g, --grade`: Khối lớp (`8`, `9`, `10`, `11`, `12`).
  * `-c, --concurrency`: Số luồng sinh câu hỏi đồng thời (mặc định: `8`).
  * `--output-dir`: Thư mục lưu file đề thi JSON (mặc định: `output/exams`).

---

### 2. Sửa & Chuẩn hóa Dữ liệu Gốc trên Đĩa (`fix-root-data`)
Quét đệ quy toàn bộ thư mục `output/` (kể cả symlink `real_annotated/`), sửa trực tiếp các file XML và JSON để đảm bảo dòng trích dẫn bài đọc `(Adapted from...)`, `(Nguồn:...)` luôn nằm 100% bên trong thẻ `<stimulus>...</stimulus>`.
```bash
pixi run fix-root-data

# Hoặc chỉ định thư mục cần quét
python src/data/fix_root_data.py --input-dir output/real_annotated
```
* **Các tham số hỗ trợ**:
  * `--input-dir`: Thư mục chứa dữ liệu cần rà soát và sửa chữa (mặc định: `output`).

---

### 3. Gom Dữ liệu Thô Toàn bộ Đề thi (`consolidate-raw`)
Quét toàn bộ đề thi OCR thật và đề thi synthetic trong `output/` để tổng hợp thành một file duy nhất `output/dataset/raw_exams.jsonl`.
```bash
pixi run consolidate-raw
```
* **Các tham số hỗ trợ**:
  * `--input, -i`: Thư mục chứa đề thi đầu vào (mặc định: `output`).
  * `--output, -o`: Đường dẫn file JSONL tổng hợp (mặc định: `output/dataset/raw_exams.jsonl`).

---

### 4. Chuẩn bị Dataset Tokenized Offline (`prepare-offline-dataset` / `prepare-dataset`)
Cắt dữ liệu thô theo cửa sổ trượt đa thang đo (512, 1024 token), căn chỉnh nhãn token với character spans và tạo các tệp `train.jsonl`, `val.jsonl`, `test.jsonl` và thư mục `xml/`.
```bash
pixi run prepare-offline-dataset

# Chuẩn bị dataset cho mô hình ModernBERT với tỷ lệ chia tùy chỉnh
sequence-labelling-generator prepare \
    --model answerdotai/ModernBERT-base \
    --window-sizes 512,1024 \
    --strides 128,256 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```
* **Các tham số hỗ trợ**:
  * `--model, -m`: Tên mô hình Hugging Face dùng tokenizer (mặc định: `jhu-clsp/mmBERT-base`).
  * `--window-sizes`: Danh sách độ dài cửa sổ trượt (mặc định: `512,1024`).
  * `--strides`: Danh sách bước trượt tương ứng (mặc định: `128,256`).
  * `--output-dir, -o`: Thư mục lưu dataset đầu ra (mặc định: `output/dataset`).

---

### 5. Upload Dữ liệu lên Hugging Face Hub

#### a. Chế độ Upload Online Gọn nhẹ (`upload-online-dataset`) — *Khuyên dùng cho Online Training*
Chỉ upload 2 file thô cần thiết (`raw_exams.jsonl` ~35MB và `label_mapping.json` ~1KB) lên Hugging Face Hub trong 3–5 giây.
```bash
pixi run upload-online-dataset

# Tùy chỉnh repo đích
python src/data/upload.py --mode online-only --repo-id daominhwysi/synthetic-seq-labelling-vi-exam-v2
```

#### b. Chế độ Upload Toàn bộ Offline (`upload-dataset`)
Upload toàn bộ các file split tokenized (`train.jsonl`, `val.jsonl`, `test.jsonl`) cùng thư mục `xml/` lên Hugging Face Hub.
```bash
pixi run upload-dataset
```

---

### 6. Huấn luyện Mô hình Token Classification (`train`)

#### a. Huấn luyện Đa GPU (PyTorch DDP trên Kaggle 2x T4 hoặc Server)
```bash
# Huấn luyện chế độ Online Dynamic Augmentation (Tự động tải raw_exams.jsonl từ HF Hub):
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

# Huấn luyện chế độ Offline Dataset từ thư mục local:
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

#### b. Huấn luyện trên máy đơn GPU / Local
```bash
pixi run train
```
* **Các tham số huấn luyện cốt lõi**:
  * `--model_name`: Backbone Hugging Face (`jhu-clsp/mmBERT-base` hoặc `answerdotai/ModernBERT-base`).
  * `--enhanced-head`: Bật đầu phân loại nâng cao (Weighted Layer Pooling 4 lớp + Dense MLP + Multi-Sample Dropout + Focal Loss $\gamma=2.0$).
  * `--online-augmentation`: Bật sinh biến thể bố cục ngẫu nhiên trực tiếp trong RAM ở mỗi batch.
  * `--lora_r`, `--lora_alpha`: Tham số rank và alpha của LoRA adapter (mặc định: `16`, `32`).
  * `--fp16` / `--use_bf16`: Bật chế độ huấn luyện Mixed Precision.

---

### 7. Phân đoạn Hàng loạt File Đề thi Thô (`batch-inference`)
Quét toàn bộ các tệp `.txt` đề thi trong một thư mục và tự động phân đoạn ra JSON cấu trúc, bảng dự đoán token, và tệp XML gắn thẻ.
```bash
pixi run batch-inference

# Chỉ định thư mục đầu vào và đầu ra tùy chỉnh
python src/inference/predict_folder.py \
    --input-dir scratch/test_raw_exams \
    --output-dir scratch/inference_results \
    --model-path output/models/mmbert-vi-exam-v5
```
* **Các tham số hỗ trợ**:
  * `--input-dir`: Thư mục chứa các tệp văn bản đề thi `.txt` thô.
  * `--output-dir`: Thư mục lưu kết quả phân đoạn (`_structured.json`, `_predictions.txt`, `_annotated.xml`).
  * `--model-path`: Đường dẫn tới checkpoint mô hình hoặc LoRA adapter.

---

### 8. Khởi chạy Ứng dụng Web Tương tác

#### a. Trình duyệt Quản lý & Xem Đề thi (`view-exams`)
Khởi chạy giao diện FastAPI duyệt danh sách đề thi, thống kê dataset và xem trước nhãn tô màu trực quan tại cổng `8000`.
```bash
pixi run view-exams
# Truy cập: http://127.0.0.1:8000
```

#### b. Website Phân đoạn Đề thi AI Trực tiếp (`view-inference`)
Khởi chạy ứng dụng web độc lập cho phép người dùng dán văn bản đề thi thô vào và nhận kết quả phân đoạn câu hỏi, phương án, bài đọc theo thời gian thực tại cổng `8001`.
```bash
pixi run view-inference
# Truy cập: http://127.0.0.1:8001
```

---

### 9. Chạy Bộ Kiểm thử Tự động (`test`)
Chạy toàn bộ 82 unit tests kiểm tra tính đúng đắn của parser, reconstructor, template bank, tokenizer alignments, và loss functions.
```bash
pixi run test
```

---

### 10. Upload Mô hình lên Hugging Face Hub (`upload-model`)
Upload trọng số LoRA adapter hoặc model fine-tuned hoàn chỉnh kèm Model Card lên Hugging Face Hub.
```bash
pixi run upload-model

# Hoặc tùy chỉnh đường dẫn
sequence-labelling-generator upload-model \
    --model-dir output/models/mmbert-vi-exam-v5 \
    --repo-id daominhwysi/mmbert-small-vi-exam-seq-labeling-v5
```

---

## 🚀 Pipeline Overview (Quy trình Vận hành Nhanh)

Quy trình toàn diện của hệ thống được vận hành khép kín qua 3 giai đoạn súc tích:

1. **Chuẩn bị Dữ liệu (Data Phase)**: Sinh câu hỏi nhân tạo bám sát chuẩn kiến thức bằng AI Reasoning + Gán nhãn đề thi OCR thực tế $\rightarrow$ Rà soát và sửa dữ liệu gốc (`fix-root-data`) $\rightarrow$ Gom vào tập dữ liệu thô chuẩn hóa (`consolidate-raw`).
2. **Huấn luyện Mô hình (Model Training Phase)**: Tải dữ liệu lên Hub (`upload-online-dataset`) $\rightarrow$ Huấn luyện mô hình `mmBERT-base` với kiến trúc Enhanced Head và Focal Loss trên môi trường đa GPU DDP $\rightarrow$ Đạt chỉ số Recall $\ge 98.5\%$, F1 $\ge 96.0\%$.
3. **Phân đoạn & Ứng dụng (Downstream Inference Phase)**: Bóc tách cấu trúc đề thi thô thành dữ liệu có cấu trúc (JSON/XML) thông qua công cụ dòng lệnh hàng loạt (`batch-inference`) hoặc ứng dụng web thời gian thực (`view-inference`).

---

## 🏷️ Sequence Labeling Tag Set & Data Contracts

Mô hình thực hiện gán nhãn chuỗi token theo 7 nhóm thực thể chuẩn:

| Nhãn Entity | Ý nghĩa & Đối tượng gán nhãn | Ví dụ minh họa |
| :--- | :--- | :--- |
| `question_label` | Tiền tố chỉ số thứ tự câu hỏi | `Câu 1:`, `Câu 12.`, `Question 5:`, `Bài 2:` |
| `stem` | Thân câu hỏi chính (chứa văn bản câu hỏi, công thức LaTeX, bảng số liệu) | `Cho hàm số $y=f(x)$ liên tục trên $\mathbb{R}$...` |
| `option_label` | Ký hiệu chữ cái phương án lựa chọn | `A.`, `B.`, `C.`, `D.`, `a)`, `b)` |
| `option_text` | Nội dung câu chữ của phương án lựa chọn | `x = 2 hoặc x = 3`, `Hàm số đồng biến trên...` |
| `stimulus` | Đoạn văn đọc hiểu dài / Ngữ liệu dùng chung cho nhóm câu hỏi | Đoạn trích văn bản văn học, bài báo, trích dẫn `(Adapted from...)` |
| `section` | Tiêu đề phân chia phần thi, chỉ dẫn làm bài | `PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn`, `PART 7` |
| `explanation` | Lời giải chi tiết, hướng dẫn chấm và bảng biểu điểm barem | `* Lời giải: Ta có...`, `| Câu | Đáp án | Điểm |` |

---

## 📁 Project Structure

```text
.
├── pyproject.toml                  # Cấu hình dự án, phụ thuộc Pixi và đăng ký 12 tasks
├── pixi.lock                       # Lockfile quản lý môi trường độc lập đa nền tảng
├── config.yaml                     # Cấu hình siêu tham số toàn pipeline
├── AGENTS.md                       # Nguyên tắc phát triển và tài liệu hệ thống dành cho Agent
├── README.md                       # Tài liệu hướng dẫn sử dụng chính thức của dự án
│
├── tests/                          # Bộ kiểm thử tự động toàn diện (82 Unit Tests)
│   ├── test_template_bank.py       # Kiểm thử Ngân hàng Template 168 mẫu và các kiểu layout
│   ├── test_reconstructor.py       # Kiểm thử tái tạo văn bản đề thi và gán nhãn character offset
│   ├── test_fix_root_data.py       # Kiểm thử công cụ sửa dữ liệu gốc và gom trích dẫn stimulus
│   ├── test_prepare_dataset.py     # Kiểm thử tokenizer alignment và cắt sliding window
│   ├── test_enhanced_head.py       # Kiểm thử Layer Pooling, Multi-Sample Dropout và Focal Loss
│   ├── test_iter_logger.py         # Kiểm thử logger huấn luyện theo iteration
│   ├── test_exam_compiler.py       # Kiểm thử biên soạn đề thi theo cấu trúc phần
│   ├── test_curriculum.py          # Kiểm thử nạp và phân tích khung chương trình
│   ├── test_parser.py              # Kiểm thử bóc tách XML câu hỏi từ LLM
│   └── test_codex_provider.py      # Kiểm thử OpenAI Codex provider và token tracking
│
├── logs/                           # Thư mục lưu nhật ký token và báo cáo kỹ thuật chuyên sâu
│   ├── RUN_3_REPORT.md             # Báo cáo phân tích nguyên nhân lỗi ở Run #3
│   ├── RUN_4_REPORT.md             # Báo cáo bước nhảy vọt thành công ở Run #4 (Recall 97.2%)
│   ├── RUN_5_PREPARATION_REPORT.md # Báo cáo sẵn sàng và ma trận huấn luyện cho Run #5
│   └── token_usage_<DATE>.jsonl    # Nhật ký ghi nhận token gọi DeepSeek API theo ngày
│
├── output/                         # Thư mục chứa dữ liệu và sản phẩm xuất xưởng (gitignored)
│   ├── dataset/                    # Tập dữ liệu huấn luyện đã chuẩn hóa
│   │   ├── raw_exams.jsonl         # 926 đề thi cấu trúc thô phục vụ Online Augmentation
│   │   ├── train.jsonl / val.jsonl # Tập dữ liệu tokenized cắt sliding window offline
│   │   ├── label_mapping.json      # Từ điển ánh xạ nhãn thực thể
│   │   └── xml/                    # File XML gắn thẻ inline tương ứng từng đề thi
│   ├── exams/                      # Đề thi nhân tạo do DeepSeek sinh (`exam_*.json`)
│   └── real_annotated/             # Đề thi OCR thực tế đã gán nhãn (`merged.json`, `merged.xml`)
│
├── scratch/                        # Thư mục chứa script đánh giá chuyên sâu và thử nghiệm
│   ├── test_customer_formats.py    # Kiểm thử định dạng đề thi ĐGNL, ĐGTD, SAT, IELTS
│   ├── test_toeic_deep.py          # Kiểm thử chuyên sâu TOEIC bảng biểu hóa đơn và song ngữ
│   ├── test_toeic_remaining.py     # Kiểm thử TOEIC chat chains và triple passages
│   ├── generate_bio_lit_50.py      # Script sinh batch 50 đề Ngữ văn và Sinh học
│   └── generate_toeic_lit_50.py    # Script sinh batch 50 đề Ngữ văn và TOEIC
│
└── src/                            # Gói mã nguồn chính của dự án
    ├── cli.py                      # Điểm vào dòng lệnh (CLI Console Entry Point)
    │
    ├── data/                       # Subpackage Xử lý Dữ liệu & Pipeline OCR Đề thi Thật
    │   ├── fix_root_data.py        # Sửa vĩnh viễn dữ liệu gốc XML/JSON trực tiếp trên đĩa
    │   ├── prepare.py              # Chuẩn bị dataset, cắt sliding window và căn chỉnh token
    │   ├── upload.py               # Upload dataset (hỗ trợ cả Online-only và Full Offline)
    │   ├── pdf_converter.py        # Chuyển đổi PDF scan sang Markdown qua Vision LLM
    │   └── annotate_ocr.py         # Tự động gán nhãn Markdown OCR thành cấu trúc JSON/XML
    │
    ├── generation/                 # Subpackage Sinh Dữ liệu Đề thi
    │   ├── template_bank.py        # Ngân hàng Template Tổ hợp (168 mẫu, >10M biến thể layout)
    │   ├── reconstructor.py        # Tái tạo văn bản thô và tính toán character offset spans
    │   ├── generator.py            # Điều phối prompt AI sinh câu hỏi bám sát chương trình
    │   ├── exam_compiler.py        # Biên soạn câu hỏi thành đề thi có cấu trúc phần hoàn chỉnh
    │   ├── curriculum.py           # Quản lý khung chương trình môn học và khối lớp
    │   ├── parser.py               # Bóc tách thẻ XML từ phản hồi của LLM
    │   └── deepseek_client.py      # Client kết nối DeepSeek API reasoning models
    │
    ├── model/                      # Subpackage Huấn luyện & Xuất xưởng Mô hình
    │   ├── head.py                 # Enhanced Head (Layer Pooling, MLP, MSD, Focal Loss)
    │   ├── train.py                # Huấn luyện mô hình (Online/Offline, LoRA, DDP, AMP)
    │   ├── upload.py               # Upload model checkpoint & adapter lên Hugging Face Hub
    │   └── export.py               # Hợp nhất LoRA adapter và xuất mô hình sang ONNX
    │
    ├── inference/                  # Subpackage Suy diễn & Phân đoạn Đề thi
    │   ├── predict.py              # Tiện ích phân đoạn suy diễn đơn lẻ
    │   └── predict_folder.py       # Tiện ích phân đoạn hàng loạt thư mục chứa đề thi
    │
    ├── utils/                      # Subpackage Tiện ích Bổ trợ
    │   ├── config.py               # Trình đọc và nạp cấu hình `config.yaml`
    │   ├── token_tracker.py        # Logger theo dõi token API an toàn đa luồng
    │   └── visualize.py            # Trình tạo trang HTML trực quan hóa nhãn token
    │
    └── webapp/                     # Subpackage Ứng dụng Web
        ├── main.py                 # FastAPI Web App quản lý và duyệt đề thi (Cổng 8000)
        ├── inference_app.py        # FastAPI Web App phân đoạn đề thi AI trực tiếp (Cổng 8001)
        ├── inference_helper.py     # Trợ thủ tổng hợp sliding window, LaTeX masking
        └── templates/              # Giao diện Jinja2 HTML tương tác
```
