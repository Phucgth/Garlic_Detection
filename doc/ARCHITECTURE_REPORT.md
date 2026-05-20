# Kiến Trúc EfficientNetB4 + FSDA + Adaptive CB Focal Loss

### Báo cáo chi tiết từng giai đoạn — phục vụ viết báo

---

## Tổng quan

Mô hình đề xuất gồm 3 thành phần chính:

1. **EfficientNetB4 Backbone** — trích xuất đặc trưng ảnh với selective fine-tuning
2. **FSDA Block** — Frequency-Spatial Dual Attention (đóng góp mới)
3. **Adaptive Class-Balanced Focal Loss** — hàm mất mát thích nghi theo epoch (đóng góp mới)

**Bài toán:** Phân loại 3 lớp tỏi: `Fully_Peeled_Garlic` | `Partially_Peeled_Garlic` | `Spoiled_Garlic`

---

## GIAI ĐOẠN 0 — Tiền Xử Lý Dữ Liệu (Data Pipeline)

### Input

- Đường dẫn ảnh thô định dạng `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Cấu trúc thư mục: `dataset/train/`, `dataset/val/`, `dataset/test/`

### Xử lý

| Bước | Phép toán      | Chi tiết                                                       |
| ---- | -------------- | -------------------------------------------------------------- |
| 1    | Đọc file       | `tf.io.read_file` → decode JPEG với 3 channels RGB             |
| 2    | Resize         | `tf.image.resize` về `(380, 380)`                              |
| 3    | Cast           | → `float32`                                                    |
| 4    | Preprocessing  | `efficientnet.preprocess_input` → chuẩn hoá về range `[-1, 1]` |
| 5    | One-hot encode | `tf.one_hot(label, depth=3)`                                   |

### Augmentation (chỉ Training set)

| Phép biến đổi      | Tham số                   |
| ------------------ | ------------------------- |
| Random Flip        | `horizontal_and_vertical` |
| Random Rotation    | `0.083` (≈ ±30°)          |
| Random Zoom        | `±20%`                    |
| Random Translation | `±20%` theo cả H và W     |
| Random Brightness  | `factor=0.30` (±30%)      |

### Cấu hình tf.data Pipeline

```
shuffle(n, seed) → map(preprocess) → map(augment) → batch(32, drop_remainder=True) → prefetch(AUTOTUNE)
```

| Split | Augmentation | Drop Remainder |
| ----- | ------------ | -------------- |
| Train | Có           | Có             |
| Val   | Không        | Không          |
| Test  | Không        | Không          |

### Output

- Batch tensor: `(B, 380, 380, 3)` — float32
- Nhãn one-hot: `(B, 3)` — float32

---

## GIAI ĐOẠN 1 — EfficientNetB4 Backbone (Feature Extraction)

### Input

- Tensor: `(B, 380, 380, 3)` — ảnh đã preprocessing

### Kiến trúc

EfficientNetB4 pretrained trên **ImageNet**, `include_top=False`.  
Bao gồm 7 MBConv block chính (B1–B7) với chiến lược **selective fine-tuning**:

| Block                                 | Trạng thái              | Lý do                                                         |
| ------------------------------------- | ----------------------- | ------------------------------------------------------------- |
| B1, B2                                | **Frozen** (toàn bộ)    | Đặc trưng low-level (cạnh, màu sắc) đã ổn định từ ImageNet    |
| B3 → B7                               | **Fine-tuned** (trừ BN) | Cần học đặc trưng high-level domain-specific (tổn thương tỏi) |
| **BatchNormalization** (tất cả block) | **Luôn Frozen**         | Tránh cập nhật running statistics sai do batch size nhỏ       |

### Downsampling

$$380 \xrightarrow{\times 5 \text{ stride-2}} \approx 12 \quad \Rightarrow \quad \text{feature map} = 12 \times 12$$

### Mixed Precision

- **Compute dtype:** `float16` → tăng tốc GPU, giảm memory
- **Variable dtype:** `float32` → đảm bảo độ chính xác gradient

### Output

- Feature map: `(B, 12, 12, 1792)` — dtype `float16` dưới mixed precision

---

## GIAI ĐOẠN 2 — FSDA Block (Frequency-Spatial Dual Attention)

> **Đóng góp mới của đề tài.** Kết hợp phân tích tần số (Frequency) và không gian (Spatial) trong một block đơn.

### Input

- Feature map: `(B, 12, 12, 1792)` — dtype `float16`

FSDA Block gồm **hai nhánh song song** cùng nhận chung một feature map:

---

### Nhánh 1: Frequency Channel Attention (`FrequencyChannelAttention`)

**Mục đích:** Xác định _channel nào_ mang thông tin bệnh liên quan đến phân bố tần số (texture, pattern lặp lại của tổn thương).

| Bước                | Phép toán                                | Input Shape                   | Output Shape   | Dtype     |
| ------------------- | ---------------------------------------- | ----------------------------- | -------------- | --------- |
| Cast                | float16 → float32                        | (B,12,12,1792)                | (B,12,12,1792) | float32   |
| Transpose           | NHWC → NCHW                              | (B,12,12,1792)                | (B,1792,12,12) | float32   |
| Tạo complex         | `tf.complex(x, zeros_like(x))`           | (B,1792,12,12)                | (B,1792,12,12) | complex64 |
| FFT2D               | `tf.signal.fft2d` — 2D Fourier Transform | (B,1792,12,12)                | (B,1792,12,12) | complex64 |
| Log-Magnitude       | `log1p(abs(FFT))`                        | (B,1792,12,12)                | (B,1792,12,12) | float32   |
| Global Avg Pool     | `reduce_mean(axis=[2,3])`                | (B,1792,12,12)                | (B,1792)       | float32   |
| FC1 + ReLU          | Dense(1792 → 112, no bias)               | (B,1792)                      | (B,112)        | float32   |
| FC2 + Sigmoid       | Dense(112 → 1792, no bias)               | (B,112)                       | (B,1792)       | float32   |
| Reshape             | vector → spatial                         | (B,1792)                      | (B,1,1,1792)   | float32   |
| Channel Reweighting | `x_f32 × attn` (broadcast)               | (B,12,12,1792) × (B,1,1,1792) | (B,12,12,1792) | float32   |
| Cast back           | float32 → input dtype                    | (B,12,12,1792)                | (B,12,12,1792) | float16   |

**Reduction ratio:** C/16 = 1792/16 = **112** (bottleneck), với `max(C//16, 8)` để tránh quá nhỏ.

**Output nhánh 1:** `freq_out` — `(B, 12, 12, 1792)` float32

---

### Nhánh 2: Spatial Attention (CBAM-style, trong `FSDABlock`)

**Mục đích:** Xác định _vị trí nào_ trong ảnh có tổn thương/bệnh (tập trung vào vùng không gian quan trọng).

| Bước                | Phép toán                                          | Input Shape                  | Output Shape   | Dtype   |
| ------------------- | -------------------------------------------------- | ---------------------------- | -------------- | ------- |
| Cast                | float16 → float32                                  | (B,12,12,1792)               | (B,12,12,1792) | float32 |
| AvgPool (channel)   | `reduce_mean(axis=-1, keepdims=True)`              | (B,12,12,1792)               | (B,12,12,1)    | float32 |
| MaxPool (channel)   | `reduce_max(axis=-1, keepdims=True)`               | (B,12,12,1792)               | (B,12,12,1)    | float32 |
| Concatenate         | channel concat                                     | 2 × (B,12,12,1)              | (B,12,12,2)    | float32 |
| Conv 7×7            | Conv2D(filters=1, kernel=7, padding=same, no bias) | (B,12,12,2)                  | (B,12,12,1)    | float32 |
| Sigmoid             | attention map                                      | (B,12,12,1)                  | (B,12,12,1)    | float32 |
| Spatial Reweighting | `x_f32 × sp_attn` (broadcast)                      | (B,12,12,1792) × (B,12,12,1) | (B,12,12,1792) | float32 |

**Output nhánh 2:** `spatial_out` — `(B, 12, 12, 1792)` float32

---

### Fusion

| Bước                | Phép toán                             | Output Shape   | Dtype   |
| ------------------- | ------------------------------------- | -------------- | ------- |
| Element-wise Add    | `freq_out + spatial_out`              | (B,12,12,1792) | float32 |
| Batch Normalization | `BatchNormalization(dtype='float32')` | (B,12,12,1792) | float32 |
| Cast back           | float32 → input dtype                 | (B,12,12,1792) | float16 |

**Công thức tổng quát:**

$$\text{FSDA}(x) = \text{BN}\bigl(\text{FreqAttn}(x) + \text{SpatAttn}(x)\bigr)$$

**Output FSDA Block:**

- Attended feature map: `(B, 12, 12, 1792)` — float16
- Spatial attention map: `(B, 12, 12, 1)` — float32 (dùng cho Grad-CAM visualization)

---

## GIAI ĐOẠN 3 — Classification Head

### Input

- Attended feature map: `(B, 12, 12, 1792)`

### Các Layer

| Layer                  | Phép toán                       | Input Shape    | Output Shape | Tham số bổ sung                   |
| ---------------------- | ------------------------------- | -------------- | ------------ | --------------------------------- |
| GlobalAveragePooling2D | `mean(H, W)` — collapse spatial | (B,12,12,1792) | (B,1792)     | —                                 |
| BatchNormalization     | normalize, scale, shift         | (B,1792)       | (B,1792)     | `name='head_bn'`, float32         |
| Dense 256 + ReLU       | Fully-connected                 | (B,1792)       | (B,256)      | `kernel_regularizer=L2(1e-5)`     |
| Dropout 0.5            | Drop 50% neurons randomly       | (B,256)        | (B,256)      | Chỉ active khi `training=True`    |
| Dense 3 + Softmax      | Fully-connected + normalize     | (B,256)        | (B,3)        | `dtype='float32'` (force float32) |

> **Lý do force float32 ở Softmax:** Tránh numerical instability (overflow/underflow) trong phép tính xác suất dưới mixed precision.

### Output

- Probability vector: `(B, 3)` — float32
  - Index 0: `Fully_Peeled_Garlic`
  - Index 1: `Partially_Peeled_Garlic`
  - Index 2: `Spoiled_Garlic`

---

## GIAI ĐOẠN 4 — Adaptive Class-Balanced Focal Loss

> **Đóng góp mới của đề tài.** Mở rộng CB Focal Loss (Cui et al., 2019) bằng cơ chế thích nghi động theo epoch dựa trên per-class validation recall.

### 4a. Static Class-Balanced Weights (baseline — Cui et al., 2019)

Tính **một lần** từ phân phối training set trước khi train:

$$E_{n_c} = 1 - \beta^{n_c}$$

$$w_c^{(\text{static})} = \frac{1 - \beta}{E_{n_c}}$$

$$w_c^{(\text{static})} \leftarrow \frac{w_c^{(\text{static})}}{\sum_c w_c^{(\text{static})}} \times C$$

| Tham số | Giá trị          | Ý nghĩa                             |
| ------- | ---------------- | ----------------------------------- |
| $\beta$ | 0.9999           | Điều chỉnh độ nhạy với tần suất mẫu |
| $n_c$   | [1050, 306, 704] | Số mẫu training mỗi class           |
| $C$     | 3                | Số classes                          |

→ Class thiểu số (`Partially_Peeled`, 306 mẫu) nhận weight **cao hơn**.

---

### 4b. Adaptive Factor — `AdaptiveWeightCallback`

Sau **mỗi epoch**, callback thực hiện:

**Bước 1:** Chạy `model.predict(val_ds)` → lấy `y_pred`

**Bước 2:** Tính per-class recall trên validation set:

$$r_c^{(t)} = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}, \quad c = 0, 1, 2$$

**Bước 3:** Tính adaptation target (class recall thấp → target cao):

$$a_c^{(t)} = \left(1 - r_c^{(t)}\right) + \varepsilon, \quad \varepsilon = 0.1$$

**Bước 4:** EMA (Exponential Moving Average) update:

$$f_c^{(t)} = (1 - \tau) \cdot f_c^{(t-1)} + \tau \cdot a_c^{(t)}$$

**Bước 5:** Normalize để giữ nguyên loss scale:

$$f_c^{(t)} \leftarrow \frac{f_c^{(t)}}{\bar{f}^{(t)}} \quad \left(\text{sao cho } \bar{f} = 1\right)$$

**Bước 6:** Gán vào `loss_fn.adaptive_factor` — `tf.Variable`, `trainable=False`

| Tham số              | Giá trị | Ý nghĩa                                                                                  |
| -------------------- | ------- | ---------------------------------------------------------------------------------------- |
| $\tau$               | 0.3     | EMA smoothing — tốc độ thích nghi; thấp = ổn định hơn, cao = nhanh hơn nhưng dễ dao động |
| $\varepsilon$        | 0.1     | Minimum factor tránh weight về 0 khi class đã perfect                                    |
| Khởi tạo $f_c^{(0)}$ | 1.0     | Không ưu tiên class nào ban đầu                                                          |

---

### 4c. Loss Computation (Forward Pass)

**Bước 1:** Combined weight:

$$w_c^{(\text{combined})} = w_c^{(\text{static})} \cdot f_c^{(t)}$$

$$\tilde{w}_c = \frac{w_c^{(\text{combined})}}{\overline{w^{(\text{combined})}}}$$

**Bước 2:** Per-sample weight dựa trên true class:

$$\text{sw}_i = \sum_c y_{i,c} \cdot \tilde{w}_c$$

**Bước 3:** Focal modulation (hard example mining — Lin et al., 2017):

$$p_t^{(i)} = \sum_c y_{i,c} \cdot \hat{p}_{i,c}$$

$$\text{focal}_i = \left(1 - p_t^{(i)}\right)^\gamma$$

**Bước 4:** Cross-entropy per sample:

$$\text{CE}_i = -\sum_c y_{i,c} \log \hat{p}_{i,c}$$

**Bước 5:** Final loss:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \text{sw}_i \cdot \text{focal}_i \cdot \text{CE}_i$$

| Tham số  | Giá trị | Ý nghĩa                                                          |
| -------- | ------- | ---------------------------------------------------------------- |
| $\gamma$ | 2.0     | Focal factor — hard examples (p_t thấp) được khuếch đại mạnh hơn |
| $\beta$  | 0.9999  | CB effective number beta                                         |
| $\tau$   | 0.3     | EMA smoothing                                                    |

---

## GIAI ĐOẠN 5 — Training Configuration

### Optimizer & Learning Rate

| Thành phần  | Giá trị                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------ |
| Optimizer   | Adam (default $\beta_1=0.9$, $\beta_2=0.999$)                                              |
| Initial LR  | $10^{-4}$                                                                                  |
| LR Schedule | `ExponentialDecay`: $\text{lr}(t) = 10^{-4} \times 0.9^{\lfloor t / (S \times 5) \rfloor}$ |
| Decay steps | `steps_per_epoch × 5` (mỗi 5 epoch giảm một lần)                                           |
| Decay rate  | 0.9                                                                                        |
| Staircase   | True (bậc thang, không liên tục)                                                           |

với $S$ = `n_train // batch_size` = số steps mỗi epoch.

### Hyperparameters

| Tham số              | Giá trị          |
| -------------------- | ---------------- |
| Input shape          | (380, 380, 3)    |
| Batch size           | 32               |
| Max epochs           | 30               |
| Dropout rate         | 0.5              |
| L2 regularization    | 1e-5 (Dense 256) |
| FSDA reduction ratio | 16 (C/16 = 112)  |
| FSDA spatial kernel  | 7×7              |

### Callbacks

| Callback                 | Cấu hình                                                   |
| ------------------------ | ---------------------------------------------------------- |
| `AdaptiveWeightCallback` | Cập nhật adaptive_factor sau mỗi epoch, $\tau = 0.3$       |
| `EarlyStopping`          | monitor=`val_loss`, patience=12, restore_best_weights=True |
| `ModelCheckpoint`        | save_best_only=True, monitor=`val_loss`, format=`.keras`   |
| `CSVLogger`              | Ghi log training vào `training_log.csv` mỗi epoch          |

### System Config

| Thành phần      | Giá trị                                |
| --------------- | -------------------------------------- |
| Mixed Precision | `float16` compute, `float32` variables |
| XLA JIT         | `tf.config.optimizer.set_jit(True)`    |
| GPU Memory      | `set_memory_growth(True)` — tránh OOM  |

### Evaluation Protocol

- **3 independent runs** với seeds [42, 123, 456]
- Mỗi run: khởi tạo model mới, dataset shuffle mới
- Kết quả cuối: **Mean ± Std** qua 3 runs

---

## GIAI ĐOẠN 6 — Evaluation Metrics

### Test Set Evaluation

Model tốt nhất (checkpoint theo `val_loss`) được load và đánh giá trên **test set** (không augmentation):

| Metric                   | Mô tả                             | Công thức                                                                                                   |
| ------------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Accuracy**             | Tỉ lệ dự đoán đúng                | $\text{Acc} = \frac{\text{TP} + \text{TN}}{N}$                                                              |
| **Precision (weighted)** | Độ chính xác, weighted by support | $P_w = \sum_c \frac{n_c}{N} \cdot P_c$                                                                      |
| **Recall (weighted)**    | Độ phủ, weighted by support       | $R_w = \sum_c \frac{n_c}{N} \cdot R_c$                                                                      |
| **F1-Score (weighted)**  | Harmonic mean P & R               | $F1_w = \sum_c \frac{n_c}{N} \cdot F1_c$                                                                    |
| **Balanced Accuracy**    | Mean recall qua các class         | $\text{BalAcc} = \frac{1}{C}\sum_c R_c$                                                                     |
| **Cohen's Kappa**        | Agreement vượt trên chance        | $\kappa = \frac{p_o - p_e}{1 - p_e}$                                                                        |
| **Matthews CC (MCC)**    | Robust với imbalanced data        | $\text{MCC} = \frac{\text{TP}\cdot\text{TN} - \text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})\cdots}}$ |

### Outputs được lưu (mỗi run)

| File                            | Nội dung                                     |
| ------------------------------- | -------------------------------------------- |
| `best_model.keras`              | Model weights tốt nhất                       |
| `training_log.csv`              | Loss, accuracy mỗi epoch                     |
| `adaptive_weight_history.csv`   | Per-class recall + adaptive factor mỗi epoch |
| `learning_curve.png`            | Accuracy, Loss, Adaptive factors qua epoch   |
| `confusion_matrix.png`          | Confusion matrix test set                    |
| `classification_report.txt`     | Precision/Recall/F1 per class                |
| `EXPERIMENT_REPORT.txt`         | Tổng hợp kết quả                             |
| `adaptive_weight_evolution.png` | Visualization đóng góp mới                   |

---

## Tóm Tắt Luồng Dữ Liệu

```
Input Image (380×380×3)
        │
        ▼
[Preprocessing + Augmentation]
        │  (B, 380, 380, 3)
        ▼
[EfficientNetB4 Backbone]
  B1,B2 frozen │ B3-B7 fine-tuned (BN frozen)
        │  (B, 12, 12, 1792)  float16
        ▼
┌──────────────────────────────────────┐
│         FSDA Block (Proposed)        │
│  ┌────────────────┐ ┌──────────────┐ │
│  │  Freq Channel  │ │   Spatial    │ │
│  │   Attention    │ │  Attention   │ │
│  │ FFT→log→FC→sig │ │ avg+max→conv │ │
│  │ (B,12,12,1792) │ │ →sigmoid     │ │
│  └───────┬────────┘ └──────┬───────┘ │
│          │    Element-wise │         │
│          └──────── Add ────┘         │
│                    │                 │
│             BatchNorm (float32)      │
└──────────────────────────────────────┘
        │  (B, 12, 12, 1792)  float16
        ▼
[Classification Head]
  GAP → BN → FC256+ReLU → Dropout(0.5) → Softmax
        │  (B, 3)  float32
        ▼
[Output Probabilities]
  Fully_Peeled | Partially_Peeled | Spoiled

        ↑ Training ↑
[Adaptive CB Focal Loss]
  Static CB weights (Cui 2019)
  × Adaptive factor (EMA per-class recall)
  × Focal modulation (Lin 2017)
  → L = mean(sw × (1-pt)^γ × CE)
```

---

## Tài Liệu Tham Khảo

- **EfficientNet:** Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. _ICML_.
- **Class-Balanced Loss:** Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective Number of Samples. _CVPR_.
- **Focal Loss:** Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. _ICCV_.
- **CBAM Spatial Attention:** Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. _ECCV_.
- **Mixed Precision:** Micikevicius, P., et al. (2018). Mixed Precision Training. _ICLR_.
