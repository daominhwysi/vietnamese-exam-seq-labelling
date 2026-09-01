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

## 💡 Chế độ Huấn luyện: Online vs. Offline

- **Online (`--online-augmentation`)**: Huấn luyện trực tiếp từ file thô `raw_exams.jsonl`. Augmentation và tokenization được thực hiện trong quá trình nạp batch. Upload dữ liệu thô bằng lệnh `pixi run upload-online-dataset`.
- **Offline (`prepare-offline-dataset`)**: Tokenize và cắt sliding window trước thành các file `train.jsonl`, `val.jsonl`, `test.jsonl`. Upload toàn bộ splits bằng lệnh `pixi run upload-dataset`.

---

## ⚡ CLI & Pixi Tasks Reference

Dự án sử dụng **Pixi** để quản lý môi trường. Dưới đây là danh sách đầy đủ các lệnh:

### 1. Sinh đề thi nhân tạo (`generate` / `exam`)
```bash
# Sinh 5 đề thi ngẫu nhiên
pixi run generate -n 5

# Sinh 10 đề thi môn Vật lí lớp 11
pixi run generate -n 10 --subject physics --grade 11

# Sinh đề thi môn Văn với 8 luồng đồng thời
pixi run generate -n 5 --subject literature --grade 12 --concurrency 8
```
* **Danh sách toàn bộ tham số**:
  * `-n, --num-exams` *(int, default: 50)*: Số lượng đề thi cần sinh.
  * `-o, --output-dir` *(str, default: `output/exams`)*: Thư mục lưu file JSON đề thi.
  * `-s, --subject` *(str, default: None)*: Lọc môn học (`math_algebra`, `math_geometry`, `physics`, `chemistry`, `biology`, `history`, `geography`, `economics_law`, `literature`, `english`, `toeic`).
  * `-g, --grade` *(int, default: None)*: Lọc khối lớp (`8`, `9`, `10`, `11`, `12`).
  * `-c, --concurrency` *(int, default: 2)*: Số luồng sinh câu hỏi đồng thời cho mỗi đề.
  * `--model` *(str, default: `gpt-5.6-luna`)*: Tên mô hình LLM sử dụng.
  * `--provider` *(choice: `codex`, `deepseek`, `nvidia`, `vilao`, `xah`, `commandcode`, default: None)*: Nhà cung cấp API LLM.
  * `--thinking` *(choice: `high`, `max`, `low`, `medium`, `minimal`, `none`, `xhigh`, default: `low`)*: Mức độ reasoning effort.

---

### 2. Gom dữ liệu thô (`consolidate-raw`)
Gom toàn bộ đề thi OCR thật và đề synthetic trong `output/` vào một file JSONL duy nhất.
```bash
pixi run consolidate-raw
```
* **Danh sách toàn bộ tham số**:
  * `-i, --input-dir` *(str, default: `output`)*: Thư mục đầu vào chứa các thư mục con `exams/` và `real_annotated/`.
  * `-o, --output-file` *(str, default: `output/dataset/raw_exams.jsonl`)*: Đường dẫn file JSONL đầu ra.
  * `--include-all` *(flag, default: False)*: Bao gồm tất cả tài liệu bất kể trạng thái audit pass/fail.

---

### 3. Chuẩn bị Dataset Tokenized Offline (`prepare-offline-dataset` / `prepare-dataset`)
Cắt sliding window, căn chỉnh nhãn token và xuất ra các file split tokenized offline.
```bash
pixi run prepare-offline-dataset

# Tùy chỉnh chi tiết tham số
sequence-labelling-generator prepare \
    --model jhu-clsp/mmBERT-base \
    --max-len 512,1024 \
    --stride 128,256 \
    --train-ratio 0.85 \
    --val-ratio 0.10 \
    --test-ratio 0.05 \
    --exam-level
```
* **Danh sách toàn bộ tham số**:
  * `-i, --input-dir` *(str, default: `output`)*: Thư mục chứa đề thi nguồn.
  * `-o, --output-dir` *(str, default: `output/dataset`)*: Thư mục lưu dataset đầu ra.
  * `--model` *(str, default: `jhu-clsp/mmBERT-base`)*: Tên base model để lấy tokenizer.
  * `--max-len` *(str, default: `512,768,1024,2048`)*: Danh sách độ dài cửa sổ trượt (ngăn cách bởi dấu phẩy).
  * `--stride` *(str, default: `128,192,256,512`)*: Danh sách bước trượt tương ứng (ngăn cách bởi dấu phẩy).
  * `--train-ratio` *(float, default: 0.85)*: Tỷ lệ chia tập train.
  * `--val-ratio` *(float, default: 0.10)*: Tỷ lệ chia tập validation.
  * `--seed` *(int, default: 42)*: Random seed chia split.
  * `--exam-level` *(flag)*: Ghép toàn bộ đề thi thành chuỗi dài thay vì cắt từng câu hỏi đơn lẻ.
  * `--latex-placeholder` *(str, default: `[LATEX]`)*: Chuỗi thay thế cho công thức toán LaTeX.
  * `--typo-rate` *(float, default: 0.02)*: Tỷ lệ chèn lỗi chính tả.
  * `--space-noise-rate` *(float, default: 0.15)*: Tỷ lệ chèn nhiễu khoảng trắng ngẫu nhiên.
  * `--latex-mask-prob` *(float, default: 0.5)*: Xác suất ẩn công thức LaTeX thành placeholder.
  * `--option-drop-prob` *(float, default: 0.05)*: Xác suất loại bỏ ngẫu nhiên 1 phương án lựa chọn.
  * `--casing-noise-prob` *(float, default: 0.10)*: Xác suất chuyển đổi chữ hoa/thường ngẫu nhiên.
  * `--synonym-swap-prob` *(float, default: 0.10)*: Xác suất đổi tiền tố câu hỏi sang từ đồng nghĩa.
  * `--formatting-noise-prob` *(float, default: 0.10)*: Xác suất chèn các thẻ markdown formatting.
  * `--inline-option-prob` *(float, default: 0.0)*: Xác suất dàn trải phương án trên cùng một dòng.
  * `--min-inline-spaces` *(int, default: 5)*, `--max-inline-spaces` *(int, default: 30)*: Số space giữa các option inline.
  * `--min-inline-tabs` *(int, default: 1)*, `--max-inline-tabs` *(int, default: 3)*: Số tab giữa các option inline.
  * `--grid-2x2-prob` *(float, default: 0.0)*: Xác suất xếp phương án thành dạng lưới 2x2.
  * `--same-line-stem-options-prob` *(float, default: 0.0)*: Xác suất đặt thân câu hỏi và phương án A trên cùng 1 dòng.
  * `--flatten-newlines-prob` *(float, default: 0.0)*: Xác suất chuyển toàn bộ dấu xuống dòng thành space.
  * `--collapse-whitespace-prob` *(float, default: 0.0)*: Xác suất nén khoảng trắng thừa thành space đơn.
  * `--only-passed` *(flag, default: True)*: Chỉ sử dụng các tài liệu đã đạt kiểm tra chất lượng.
  * `--include-all` *(flag)*: Sử dụng toàn bộ tài liệu không phân biệt kết quả kiểm tra.

---

### 4. Upload Dataset lên Hugging Face Hub (`upload` / `upload-dataset` / `upload-online-dataset`)
```bash
# Upload chế độ Online (chỉ raw_exams.jsonl + label_mapping.json)
pixi run upload-online-dataset

# Upload chế độ Offline (toàn bộ train/val/test splits + xml/)
pixi run upload-dataset
```
* **Danh sách toàn bộ tham số**:
  * `--mode` *(choice: `all`, `online-only`, `raw-only`, `online`, `raw`, default: `all`)*: Chế độ upload.
  * `--dataset-repo-id, --dataset_repo_id` *(str, default: `daominhwysi/synthetic-seq-labelling-vi-exam-v2`, alias: `--repo-id`)*: Target repo ID trên Hugging Face.
  * `--dataset-dir, --dataset_dir` *(str, default: `output/dataset`)*: Thư mục chứa dữ liệu trên máy local.
  * `--token` *(str, default: None)*: Token xác thực Hugging Face (hoặc lấy từ biến môi trường `HF_TOKEN`).

---

### 5. Huấn luyện mô hình (`train`)
```bash
# Huấn luyện PyTorch DDP với Online Dynamic Augmentation
torchrun --nproc_per_node=2 src/model/train.py \
    --online-augmentation \
    --dataset_repo_id daominhwysi/synthetic-seq-labelling-vi-exam-v2 \
    --model_name jhu-clsp/mmBERT-base \
    --output_dir output/models/mmbert-vi-exam-v5 \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --enhanced-head \
    --fp16
```
* **Danh sách toàn bộ tham số**:
  * `--model_name` *(str, default: `jhu-clsp/mmBERT-base`)*: Tên base model backbone trên Hugging Face.
  * `--output_dir` *(str, default: `./results`)*: Thư mục lưu checkpoint và adapter weights.
  * `--dataset_repo_id, --dataset-repo-id` *(str, default: `daominhwysi/synthetic-seq-labelling-vi-exam-v2`, alias: `--repo_id`)*: Repo ID trên HF Hub để tải dữ liệu.
  * `--data-dir` *(str, default: None)*: Thư mục local chứa dataset offline splits.
  * `--online-augmentation` *(flag)*: Bật sinh biến thể bố cục và tokenize trực tiếp trong RAM lúc huấn luyện.
  * `--raw-data-dir` *(str, default: `output`)*: Thư mục local chứa file `raw_exams.jsonl` hoặc các đề thi gốc.
  * `--train_ratio` *(float, default: 0.85)*: Tỷ lệ tập train khi chia online.
  * `--val_ratio` *(float, default: 0.10)*: Tỷ lệ tập validation khi chia online.
  * `--test_ratio` *(float, default: 0.05)*: Tỷ lệ tập test khi chia online.
  * `--enhanced-head` *(flag, default: True)*: Kích hoạt Weighted Layer Pooling (4 lớp) + Dense MLP + Multi-Sample Dropout (5 masks) + Focal Loss.
  * `--no-enhanced-head` *(flag)*: Tắt Enhanced Head, dùng linear classification head tiêu chuẩn.
  * `--focal-gamma` *(float, default: 2.0)*: Hệ số focusing $\gamma$ của Focal Loss.
  * `--label-smoothing` *(float, default: 0.05)*: Hệ số label smoothing cho loss.
  * `--no-class-weights` *(flag)*: Tắt trọng số phạt theo tần suất nhãn.
  * `--real-upsample-factor` *(float, default: 1.0)*: Hệ số nhân trọng số lấy mẫu cho đề thi OCR thật.
  * `--epochs` *(int, default: 3)*: Số lượng epoch huấn luyện.
  * `--batch_size` *(int, default: 8)*: Batch size huấn luyện trên mỗi GPU.
  * `--eval_batch_size` *(int, default: 8)*: Batch size đánh giá validation.
  * `--eval_accumulation_steps` *(int, default: 10)*: Số bước tích lũy đánh giá trước khi chuyển tensor sang CPU.
  * `--lr` *(float, default: 5e-4)*: Tốc độ học (Learning rate).
  * `--lr-scheduler-type` *(str, default: `linear`)*: Loại scheduler (`linear`, `cosine`, `constant`, `cosine_with_restarts`).
  * `--warmup-ratio` *(float, default: 0.0)*: Tỷ lệ số bước warmup trên tổng số step.
  * `--warmup-steps` *(int, default: 0)*: Số bước warmup cụ thể.
  * `--weight_decay` *(float, default: 0.01)*: Hệ số suy giảm trọng số (Weight decay).
  * `--lora_r` *(int, default: 16)*: Rank $r$ của LoRA adapter.
  * `--lora_alpha` *(int, default: 32)*: Hệ số scaling $\alpha$ của LoRA.
  * `--lora_dropout` *(float, default: 0.1)*: Tỷ lệ dropout của LoRA.
  * `--no-lora` *(flag)*: Tắt LoRA để fine-tune toàn bộ tham số mô hình (Full fine-tuning).
  * `--fp16` *(flag)*: Bật chế độ huấn luyện Mixed Precision float16.
  * `--use_bf16` *(flag)*: Bật chế độ huấn luyện Mixed Precision bfloat16.
  * `--no_fp16` *(flag)*: Tắt chế độ float16.
  * `--gradient-checkpointing` *(flag)*: Bật gradient checkpointing để tiết kiệm VRAM.
  * `--gradient-accumulation-steps` *(int, default: 1)*: Số bước tích lũy gradient trước khi cập nhật trọng số.
  * `--save_total_limit` *(int, default: 2)*: Số lượng checkpoint tối đa được lưu lại.
  * `--logs_per_epoch` *(int, default: 10)*: Số lần ghi log trong 1 epoch (tự động tính logging steps).
  * `--logging_steps` *(int, default: None)*: Số bước cố định giữa các lần ghi log (ghi đè `logs_per_epoch`).
  * `--report_to` *(choice: `wandb`, `tensorboard`, `none`, default: `none`)*: Nền tảng ghi log theo dõi.
  * `--wandb_project` *(str, default: `vietnamese-exam-seq-labelling`)*: Tên dự án Weights & Biases.
  * `--push_to_hub` *(flag)*: Tự động đẩy model/adapter lên Hugging Face Hub sau khi huấn luyện xong.
  * `--hub_model_id` *(str, default: None)*: Target repo ID trên HF Hub để đẩy model lên.
  * `--hf_token` *(str, default: None)*: Token xác thực Hugging Face.
  * `--seed` *(int, default: 42)*: Random seed cho việc tái lập kết quả huấn luyện.

---

### 6. Phân đoạn thư mục đề thi thô (`batch-inference`)
Phân đoạn toàn bộ các file `.txt` và `.md` trong thư mục ra JSON cấu trúc, TXT dự đoán và XML gắn thẻ.
```bash
pixi run batch-inference

# Tùy chỉnh đường dẫn và mô hình
python src/inference/predict_folder.py \
    -i scratch/test_raw_exams \
    -o scratch/inference_results \
    --model-dir output/models/mmbert-vi-exam-v5
```
* **Danh sách toàn bộ tham số**:
  * `-i, --input-dir` *(str, default: `scratch/test_raw_exams`)*: Thư mục chứa các file văn bản đề thi `.txt`/`.md`.
  * `-o, --output-dir` *(str, default: `inference_output`)*: Thư mục lưu kết quả xuất ra.
  * `--model-dir` *(str, default: `./results`)*: Đường dẫn thư mục model checkpoint/adapter local hoặc ID repo trên HF Hub.
  * `--base-model-name` *(str, default: `jhu-clsp/mmBERT-base`)*: Tên base model backbone.
  * `--max-length` *(int, default: 1024)*: Chiều dài cửa sổ trượt tokenization.
  * `--stride` *(int, default: 256)*: Bước trượt overlapping của sliding window.

---

### 7. Phân đoạn file đơn lẻ (`inference`)
```bash
# Phân đoạn trực tiếp chuỗi văn bản
sequence-labelling-generator inference -m ./results -t "Câu 1: Cho hàm số... A. 1 B. 2"

# Phân đoạn từ file văn bản đơn lẻ
sequence-labelling-generator inference -m ./results -f exam.txt -o result.json
```
* **Danh sách toàn bộ tham số**:
  * `-m, --model_dir` *(str, default: `./results`)*: Thư mục checkpoint hoặc HF repo.
  * `--base_model_name` *(str, default: None)*: Tên base model.
  * `-t, --text` *(str, default: None)*: Chuỗi văn bản đề thi truyền trực tiếp qua CLI.
  * `-f, --file` *(str, default: None)*: Đường dẫn file văn bản đầu vào.
  * `-o, --output` *(str, default: None)*: Đường dẫn file lưu kết quả (`.json`, `.xml`, `.txt`).
  * `--max-length` *(int, default: 1024)*: Chiều dài cửa sổ trượt tokenization.
  * `--stride` *(int, default: 256)*: Bước trượt overlapping của sliding window.

---

### 8. Web App
```bash
# Trình duyệt quản lý & xem nhãn đề thi (Cổng 8000)
pixi run view-exams

# Ứng dụng phân đoạn đề thi trực tiếp (Cổng 8001)
pixi run view-inference
```

---

### 9. Upload Model lên Hugging Face Hub (`upload-model`)
```bash
pixi run upload-model

# Tùy chỉnh chi tiết
sequence-labelling-generator upload-model \
    --model-dir output/models/mmbert-vi-exam-v5 \
    --repo-id daominhwysi/mmbert-small-vi-exam-seq-labeling-v5 \
    --private
```
* **Danh sách toàn bộ tham số**:
  * `--model-dir` *(str, default: `./results`)*: Thư mục chứa trọng số model/adapter đã train.
  * `--repo-id` *(str, default: None)*: ID repo model đích trên HF Hub.
  * `--token` *(str, default: None)*: HF write token.
  * `--private` *(flag, default: False)*: Thiết lập repo ở chế độ private.
  * `--commit-message` *(str, default: None)*: Commit message tùy chỉnh.
  * `--dataset-repo` *(str, default: `daominhwysi/synthetic-seq-labelling-vi-exam-v2`)*: ID repo dataset tham chiếu trong Model Card.

---

### 10. Trực quan hóa nhãn Token HTML (`visualize`)
Tạo trang web HTML độc lập trực quan hóa căn chỉnh nhãn token và character span.
```bash
pixi run visualize
```
* **Danh sách toàn bộ tham số**:
  * `-i, --input-file` *(str, default: `output/dataset/train.jsonl`)*: File JSONL chứa token và nhãn.
  * `-o, --output-html` *(str, default: `output/dataset/sample_visualization.html`)*: Đường dẫn file HTML đầu ra.
  * `--max-samples` *(int, default: 1000)*: Số mẫu tối đa nhúng vào trang HTML.

---

### 11. Tiện ích Bảo trì Dữ liệu Gốc (`fix-root-data`)
Script audit và sửa các file XML/JSON cũ trên đĩa để đưa trích dẫn nguồn `(Adapted from...)`, `(Nguồn:...)` vào trong thẻ `<stimulus>`.
```bash
pixi run fix-root-data
```
* **Danh sách toàn bộ tham số**:
  * `--input-dir` *(str, default: `output`)*: Thư mục chứa các file XML/JSON cần rà soát và sửa.

---

### 12. Kiểm thử tự động (`test`)
```bash
pixi run test
```

---

## 🚀 Pipeline Overview

Quy trình hoạt động gồm 3 bước:
1. **Chuẩn bị Dữ liệu**: Sinh câu hỏi qua LLM + Gán nhãn đề OCR thật $\rightarrow$ Gom vào `raw_exams.jsonl` (`consolidate-raw`).
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
