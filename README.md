# Phân loại Ý định Ngân hàng (Banking Intent Detection) với Unsloth

Dự án tinh chỉnh (fine-tune) mô hình ngôn ngữ lớn cho bài toán phân loại ý định trên bộ dữ liệu **BANKING77** sử dụng thư viện **Unsloth** kết hợp với kỹ thuật LoRA.

## 📋 Tổng quan Dự án

Dự án này triển khai một hệ thống phân loại ý định trong lĩnh vực ngân hàng, có khả năng dự đoán **30 loại ý định** từ tin nhắn của khách hàng. Mô hình được fine-tune bằng thư viện **Unsloth** kết hợp với **LoRA (Low-Rank Adaptation)** giúp huấn luyện cực kỳ hiệu quả trên các hệ thống có giới hạn về phần cứng GPU.

### Các Tính năng chính

- **Mô hình (Model)**: Qwen2.5-1.5B-Instruct (được lượng tử hóa 4-bit)
- **Tập dữ liệu**: BANKING77 (chọn ngẫu nhiên 30 intent, ~5.000 mẫu văn bản)
- **Huấn luyện**: Tinh chỉnh LoRA với Unsloth + SFTTrainer
- **Cách tiếp cận**: Phân loại theo hướng sinh văn bản (Generative classification - LLM sẽ viết ra trực tiếp tên nhãn intent)

## 📁 Cấu trúc Thư mục

```text
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py    # Pipeline tiền xử lý dữ liệu
│   ├── train.py              # Huấn luyện mô hình với Unsloth
│   └── inference.py          # Class độc lập dùng để chạy suy luận (dự đoán)
├── configs/
│   ├── train.yaml            # Cấu hình siêu tham số để huấn luyện
│   └── inference.yaml        # Cấu hình lúc dự đoán
├── sample_data/
│   ├── train.csv             # Dữ liệu huấn luyện (tạo tự động)
│   ├── test.csv              # Dữ liệu kiểm tra (tạo tự động)
│   └── label_map.json        # File ánh xạ ID và tên Intent
├── checkpoints/              # Thư mục lưu mô hình sau khi train
├── train.sh                  # File shell để tự động hóa quá trình train
├── inference.sh              # File shell để chạy thử các chế độ dự đoán
├── requirements.txt          # Các thư viện Python cần thiết
└── README.md                 # File bạn đang đọc
```

## 🚀 Cài đặt & Khởi chạy

### Yêu cầu hệ thống

- **Python**: 3.9+
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (Khuyến nghị dùng Google Colab T4 trở lên)
- **Nền tảng**: Google Colab (được khuyên dùng), Kaggle, hoặc máy tính cá nhân có GPU NVIDIA

### Bước 1: Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/banking-intent-unsloth.git
cd banking-intent-unsloth
```

### Bước 2: Cài đặt Thư viện

```bash
pip install -r requirements.txt
```

> **Dành riêng cho Google Colab**, hãy chạy dòng lệnh sau ở ô code đầu tiên:
> ```python
> !pip install unsloth
> ```

## 📦 Chuẩn bị Dữ liệu

### Tải và Tiền xử lý

```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

Lệnh này sẽ thực hiện các việc:
1. Tải bộ dữ liệu BANKING77 từ HuggingFace
2. Rút gọn lấy ngẫu nhiên 30 ý định (trong tổng số 77) để dễ quản lý thời gian huấn luyện
3. Chuẩn hóa dữ liệu văn bản
4. Chia tập dữ liệu thành 80% train / 20% test
5. Lưu kết quả vào `sample_data/train.csv` và `sample_data/test.csv`

### Thống kê Dữ liệu

| Tập (Split) | Số lượng mẫu | Số lượng Intent |
|-------|---------|---------|
| Train | ~4,000  | 30      |
| Test  | ~1,000  | 30      |

## 🏋️ Huấn luyện Mô hình

### Chạy quá trình Huấn luyện

```bash
# Chạy toàn bộ pipeline tự động (Tiền xử lý + Train)
bash train.sh

# Hoặc chỉ chạy script Train (nếu đã tiền xử lý trước đó)
python scripts/train.py --config configs/train.yaml
```

### Siêu tham số Huấn luyện (Hyperparameters)

| Tham số | Giá trị |
|-----------|-------|
| Base Model | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| Max Sequence Length | 512 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0 |
| Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 32 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Weight Decay | 0.01 |
| Epochs | 3 |
| LR Scheduler | Linear |
| Precision | FP16 |
| Warmup Steps | 10 |

### Phương pháp Huấn luyện

Mô hình được huấn luyện bằng phương pháp **phân loại dạng sinh văn bản**, trong đó bài toán phân loại được mô hình hóa dưới dạng điền từ vào chỗ trống cho một mẫu Prompt cho trước:

```text
### Instruction:
Classify the following banking customer message into the correct intent category.
Only respond with the intent label, nothing else.

### Message:
I am still waiting on my card?

### Intent:
card_arrival
```

## 🔮 Suy luận (Dự đoán)

### Sử dụng Class Python trực tiếp

```python
from scripts.inference import IntentClassification

# Khởi tạo (Hệ thống sẽ nạp model từ thư mục checkpoint)
classifier = IntentClassification("configs/inference.yaml")

# Dự đoán intent cho một tin nhắn
result = classifier("I am still waiting on my card?")
print(result)  # → Output: "card_arrival"
```

### Sử dụng qua Dòng lệnh (Command Line)

```bash
# Dự đoán 1 câu chỉ định
bash inference.sh "I want to change my PIN"

# Chạy đánh giá (Evaluate) toàn bộ trên tập Test
bash inference.sh --evaluate

# Chế độ tương tác trực tiếp (Chatbot)
bash inference.sh --interactive

# Chạy Demo các câu ví dụ
bash inference.sh
```

### Giao diện Class Inference

```python
class IntentClassification:
    def __init__(self, model_path):
        """
        Nạp cấu hình, tokenizer và checkpoint của mô hình LLM.
        """
        pass

    def __call__(self, message):
        """
        Dự đoán nhãn intent cho một câu tin nhắn của khách hàng.

        Args:
            message: Chuỗi tin nhắn ngân hàng
        Returns:
            predicted_label: Chuỗi chứa nhãn intent (Ví dụ: 'activate_my_card')
        """
        return predicted_label
```

## 📊 Kết quả

| Chỉ số | Giá trị |
|--------|-------|
| Độ chính xác (Test Accuracy) | Đang cập nhật (điền sau khi train) |
| Thời gian huấn luyện | ~15-20 phút trên GPU T4 |
| Bộ nhớ GPU tối đa | ~8 GB |

## 🎥 Video Demo

> **Link Video demo nghiệm thu:** [Google Drive Link](YOUR_LINK_HERE)
>
> Video bao gồm:
> - Cách thức chạy mã nguồn inference
> - Các ví dụ về tin nhắn đầu vào và kết quả mô hình dự đoán ra
> - Độ chính xác trên tập dữ liệu Test

## ⚙️ Tùy chỉnh Cấu hình

Toàn bộ các tham số đều có thể được sửa đổi tại:
- `configs/train.yaml` — Cấu hình lúc Huấn luyện
- `configs/inference.yaml` — Cấu hình lúc Dự đoán

## 📚 Tài liệu tham khảo

- [Dữ liệu BANKING77](https://huggingface.co/datasets/PolyAI/banking77) — PolyAI
- [Unsloth Library](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation
- [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) — Hướng dẫn từ Unsloth

## 👤 Tác giả

- **Họ và tên**: [Tên của bạn]
- **MSSV**: [Mã số SV của bạn]
- **Môn học**: Ứng dụng Xử lý ngôn ngữ tự nhiên trong Công nghiệp (Applications of Natural Language Processing in Industry)
- **Giảng viên**: TS. Nguyễn Hồng Bửu Long

---
*Trường Đại học Khoa học Tự nhiên - ĐHQG-HCM*
# finetune-banking77
