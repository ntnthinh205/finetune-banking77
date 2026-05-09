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
├── train.sh                  # File shell để tự động hóa quá trình train
├── inference.sh              # File shell để chạy thử các chế độ dự đoán
├── requirements.txt          # Các thư viện Python cần thiết
└── README.md                 
```

## 🚀 Cài đặt & Khởi chạy

### Yêu cầu hệ thống

- **Python**: 3.9+
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (Khuyến nghị dùng Google Colab T4 trở lên)
- **Nền tảng**: Google Colab (được khuyên dùng), Kaggle, hoặc máy tính cá nhân có GPU NVIDIA

### Bước 1: Clone Repository

```bash
git clone https://github.com/ntnthinh205/finetune-banking77.git
cd finetune-banking77
```

### Bước 2: Cài đặt Thư viện

```bash
pip install -r requirements.txt
```

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

*Lưu ý: Bạn phải đảm bảo đã hoàn thành việc **Cài đặt thư viện** (pip install) và **Tiền xử lý dữ liệu** ở các bước phía trên trước khi chạy huấn luyện.*

### Chạy quá trình Huấn luyện

```bash
# Chạy file bash
bash train.sh

# Hoặc chạy trực tiếp script python
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

#### Ví dụ
```python
from scripts.inference import IntentClassification

# Khởi tạo (Hệ thống sẽ nạp model trực tiếp từ thư mục checkpoint)
classifier = IntentClassification("checkpoints/banking77-intent")

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

**Ví dụ đầu ra (Output Example):**

```text
$ bash inference.sh "I just noticed my wallet is missing and my physical card was inside. Please block my account immediately!"

Predicting intent for: I just noticed my wallet is missing and my physical card was inside. Please block my account immediately!
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
============================================================
BANKING77 Intent Detection - Inference
============================================================
Loading model from: checkpoints/banking77-intent
==((====))==  Unsloth 2026.4.8: Fast Qwen2 patching. Transformers: 5.5.0.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights: 100% 338/338 [00:00<00:00, 454.70it/s]
unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit does not have a padding token! Will use pad_token = <|PAD_TOKEN|>.
Unsloth 2026.4.8 patched 28 layers with 28 QKV layers, 28 O layers and 28 MLP layers.
Model loaded successfully!
Number of intent classes: 30
Device: cuda:0

Input message: I just noticed my wallet is missing and my physical card was inside. Please block my account immediately!
Predicted intent: lost_or_stolen_card
Raw model output: lost_or_stolen_card
Valid label: True
```

## 📊 Kết quả

| Chỉ số | Giá trị |
|--------|-------|
| Độ chính xác (Test Accuracy) | **95.21%** (952/1002) |
| Thời gian huấn luyện | 530.0 giây (~8.8 phút) |
| Lỗi huấn luyện (Training Loss)| 0.6251 |
| Bộ nhớ GPU tối đa (Peak) | 3.738 GB |
| Bộ nhớ GPU cho quá trình Train| 2.543 GB |

## 🎥 Video Demo

> **Link Video demo nghiệm thu:** [Google Drive Link](https://drive.google.com/file/d/1r2mISuQRFgZhvHtbOBf-euawgisEOY8-/view?usp=sharing)
> **Link checkpoints:** [Checkpoints](https://drive.google.com/drive/folders/1W9Zn8tF4EzKTyMe2bvg1Rkoy9CirV9YN?usp=sharing)

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

- **Họ và tên**: Nguyễn Trần Ngọc Thịnh
- **MSSV**: 23120362
- **Môn học**: Ứng dụng Xử lý ngôn ngữ tự nhiên trong doanh nghiệp
- **Giảng viên**: TS. Nguyễn Hồng Bửu Long

---
*Trường Đại học Khoa học Tự nhiên - ĐHQG-HCM*
# finetune-banking77
