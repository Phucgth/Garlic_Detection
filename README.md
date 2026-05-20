# Garlic Detection — Overview

## Mục tiêu

Dự án tập trung vào **phân loại bệnh tỏi** từ ảnh RGB bằng các mô hình CNN, với trọng tâm là kiến trúc **EfficientNetB4** và các biến thể attention để xử lý **mất cân bằng lớp** và **khác biệt texture tinh vi**.

## Bài toán

- **Loại bài toán:** Phân loại ảnh nhiều lớp
- **Miền ứng dụng:** Phát hiện bệnh tỏi
- **Input:** Ảnh RGB (nhiều định dạng)
- **Output:** Nhãn bệnh + xác suất
- **Thách thức chính:** Mất cân bằng lớp, dữ liệu ít, khác biệt texture nhỏ

## Kiến trúc cốt lõi

- **Backbone:** EfficientNetB4 pretrained ImageNet, fine-tune chọn lọc (blocks 3–7; BN luôn frozen).
- **Attention:**
- **Head:** GAP → BN → Dense(256) → Dropout → Softmax.
- **Loss:** Adaptive Class-Balanced Focal Loss (cân bằng theo tần suất + thích nghi theo epoch).
- **Mixed precision:** dùng float16 compute, float32 cho các phần nhạy (FFT, BN, softmax).

## Dataset

- **Dataset-1:** ~2,134 ảnh (balanced hơn), phù hợp ablation study.
- **Dataset-2:** ~2,944 ảnh (mở rộng), mất cân bằng.
- 3 lớp: `Fully_Peeled_Garlic`, `Partially_Peeled_Garlic`, `Spoiled_Garlic`.

## Nội dung chính trong repo

- **Notebooks huấn luyện/so sánh:** nhiều biến thể EfficientNetB4, FSDA, E-FSDA (FreqBand, CrossBranch, ChannelShuffle, DynamicConv, CoordAttention, SEFusion, …).
- **Tài liệu thiết kế/kiến trúc:**
  - Chi tiết kiến trúc và pipeline: [doc/ARCHITECTURE_DESIGN.md](doc/ARCHITECTURE_DESIGN.md)
  - Báo cáo kiến trúc & loss: [doc/ARCHITECTURE_REPORT.md](doc/ARCHITECTURE_REPORT.md)
  - Thiết kế chi tiết từng tensor & tham số: [doc/TECHNICAL_REPORT_FSDA_AdaptiveCBLoss.md](doc/TECHNICAL_REPORT_FSDA_AdaptiveCBLoss.md)
  - Danh sách variants E-FSDA: [doc/PROPOSED_VARIANTS.md](doc/PROPOSED_VARIANTS.md)
  - Yêu cầu đề tài & dataset: [doc/requirement.md](doc/requirement.md)
  - Hướng dẫn chạy & export trên Kaggle: [doc/skill.md](doc/skill.md)

## Cách đọc dự án nhanh

1. Xem tổng quan kiến trúc: [doc/ARCHITECTURE_DESIGN.md](doc/ARCHITECTURE_DESIGN.md)
2. Xem báo cáo chi tiết loss & training: [doc/ARCHITECTURE_REPORT.md](doc/ARCHITECTURE_REPORT.md)
3. Xem danh sách biến thể E-FSDA: [doc/PROPOSED_VARIANTS.md](doc/PROPOSED_VARIANTS.md)
4. Mở notebook tương ứng để chạy thử nghiệm.

---

> README này tập trung vào **overview**; chi tiết triển khai và pipeline đầy đủ nằm trong thư mục doc và các notebook.
