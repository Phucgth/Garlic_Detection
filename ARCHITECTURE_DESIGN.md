# Detailed Architecture Design

# EfficientNetB4 + Frequency-Spatial Dual Attention (FSDA)

# Garlic Disease Classification

---

## 1. PROBLEM DEFINITION

| Item          | Detail                                                                  |
| ------------- | ----------------------------------------------------------------------- |
| **Task**      | Multi-class image classification                                        |
| **Domain**    | Plant disease detection (Garlic)                                        |
| **Input**     | RGB image of garlic leaf/bulb                                           |
| **Output**    | Disease class label + confidence score                                  |
| **Challenge** | Class imbalance, fine-grained texture differences between disease types |

---

## 2. INPUT PIPELINE

### 2.1 Raw Input

```
Raw image file  (.jpg / .jpeg / .png / .bmp / .tiff)
    └── any resolution (variable)
```

### 2.2 Preprocessing Steps

```
Step 1: tf.io.read_file(path)
            → raw bytes

Step 2: tf.image.decode_jpeg(raw, channels=3)
            → uint8 tensor  (H_orig, W_orig, 3)

Step 3: tf.image.resize(img, [380, 380])
            → float32 tensor  (380, 380, 3)
            → resize bằng bilinear interpolation (default TF)

Step 4: efficientnet_preprocess(img)
            → tf.keras.applications.efficientnet.preprocess_input
            → scale: [0, 255] → [-1, 1]  (công thức: x/127.5 - 1)
            → float32 tensor  (380, 380, 3)

Step 5: One-hot encoding nhãn
            → label: int  →  (N_classes,) float32
```

### 2.3 Data Augmentation (chỉ áp dụng khi training)

```
Áp dụng tuần tự (Sequential pipeline):

① RandomFlip("horizontal_and_vertical")
       → xác suất 50% lật ngang, 50% lật dọc

② RandomRotation(factor=0.083)
       → xoay ngẫu nhiên trong [-30°, +30°]
       → factor = 0.083 = 30/360

③ RandomZoom(height_factor=0.20)
       → zoom in/out ngẫu nhiên ±20%

④ RandomTranslation(height=0.20, width=0.20)
       → dịch chuyển ngẫu nhiên ±20% theo cả 2 chiều

⑤ RandomBrightness(factor=0.30)
       → thay đổi độ sáng ngẫu nhiên ±30%
```

### 2.4 tf.data Pipeline (GPU-optimised)

```
train: shuffle(N, seed) → map(preprocess) → map(augment) → batch(32, drop_remainder=True)  → prefetch(AUTOTUNE)
val:   map(preprocess)  → batch(32) → prefetch(AUTOTUNE)
test:  map(preprocess)  → batch(32) → prefetch(AUTOTUNE)

AUTOTUNE = tf.data.AUTOTUNE  (tự động chọn số CPU workers)
XLA JIT  = ON  (tf.config.optimizer.set_jit(True))
```

---

## 3. MODEL ARCHITECTURE

### 3.1 Overview

```
INPUT (380, 380, 3)
    │
    ▼
┌─────────────────────────────────┐
│   EfficientNetB4 Backbone       │   Feature Extractor
│   (ImageNet pretrained)         │
│   Blocks 1-2: FROZEN            │
│   Blocks 3-7: Fine-tuned        │
│   BN layers:  ALWAYS FROZEN     │
└────────────────┬────────────────┘
                 │ (B, 12, 12, 1792)  — "top_activation" layer
                 ▼
┌─────────────────────────────────┐
│       FSDABlock (Novel)         │   Dual Attention
│   ┌─────────────────────────┐   │
│   │ FreqChannelAttention    │   │
│   │ (FFT-based, channel)    │   │
│   └──────────┬──────────────┘   │
│              ⊕                  │
│   ┌─────────────────────────┐   │
│   │ SpatialAttention        │   │
│   │ (CBAM-style, spatial)   │   │
│   └──────────┬──────────────┘   │
│         BN(float32)             │
└────────────────┬────────────────┘
                 │ (B, 12, 12, 1792)
                 ▼
┌─────────────────────────────────┐
│     Classification Head         │   Classifier
│  GAP → BN → Dense(256) →        │
│  Dropout → Softmax              │
└────────────────┬────────────────┘
                 │ (B, N_classes)
                 ▼
              OUTPUT
         Class Probabilities
```

---

### 3.2 EfficientNetB4 Backbone

**Nguồn gốc:** Tan & Le, 2019 — EfficientNet: Rethinking Model Scaling for CNNs

| Property         | Value                                         |
| ---------------- | --------------------------------------------- |
| Pretrained on    | ImageNet (1.28M images, 1000 classes)         |
| Input shape      | (380, 380, 3)                                 |
| Output shape     | (12, 12, 1792) — spatial 12×12, channels 1792 |
| Total layers     | ~475 layers                                   |
| Total params     | ~19M                                          |
| Compound scaling | depth=1.8 / width=1.9 / resolution=380        |

**Freeze Strategy:**

```
for layer in base.layers:
    if layer.name.startswith("block1") or layer.name.startswith("block2"):
        layer.trainable = False          # Low-level features (edges, textures)
    elif layer.name.startswith("block3") ... "block7":
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True       # High-level semantics
    BatchNormalization → ALWAYS frozen  (tránh batch stat shift)
```

**Lý do chọn blocks 3-7:**

- Blocks 1-2: học edge/corner cơ bản → không cần fine-tune
- Blocks 3-7: học đặc trưng ngữ nghĩa (texture bệnh, màu sắc) → cần adapt sang domain garlic

---

### 3.3 FSDABlock — Core Contribution

#### 3.3.1 FrequencyChannelAttention

**Mục đích:** Trả lời câu hỏi _"WHICH channels carry disease-relevant frequency information?"_

**Input:** `x` — shape `(B, H, W, C)` = `(B, 12, 12, 1792)`

```
Algorithm: FrequencyChannelAttention.call(x)

1. x_f32 = cast(x, float32)                        # (B, 12, 12, 1792) float32

2. x_t = transpose(x_f32, [0, 3, 1, 2])            # (B, 1792, 12, 12)
   x_complex = complex(x_t, zeros_like(x_t))        # cast to complex64

3. x_fft = tf.signal.fft2d(x_complex)              # 2D FFT per channel
   → shape: (B, 1792, 12, 12) complex64
   → mỗi channel được biến đổi sang frequency domain

4. mag = log1p(|x_fft|)                            # log-magnitude spectrum
   → shape: (B, 1792, 12, 12) float32
   → log1p = log(1 + x) để tránh log(0)

5. freq_desc = reduce_mean(mag, axis=[2, 3])        # (B, 1792)
   → global average pooling trên spatial dims
   → tóm gọt thông tin tần số thành descriptor 1D

6. attn = ReLU(FC1(freq_desc))                      # (B, 1792/16) = (B, 112)
   attn = Sigmoid(FC2(attn))                        # (B, 1792)
   → Bottleneck MLP: C → C/16 → C
   → Tương tự SE-Net nhưng input là frequency descriptor, không phải spatial descriptor

7. attn = reshape(attn, [B, 1, 1, 1792])           # broadcast-ready
   out  = x_f32 * attn                              # channel-wise reweighting
   return cast(out, x.dtype)                        # (B, 12, 12, 1792)
```

**Tham số:**

- `FC1`: Dense(1792 → 112), no bias, float32
- `FC2`: Dense(112 → 1792), no bias, float32
- Tổng params: 1792×112 + 112×1792 = **401,408**

---

#### 3.3.2 SpatialAttention (CBAM-style)

**Mục đích:** Trả lời câu hỏi _"WHERE in the image are the disease lesions?"_

**Input:** `x` — shape `(B, 12, 12, 1792)`

```
Algorithm: SpatialAttention (inside FSDABlock.call)

1. x_f32 = cast(x, float32)                        # (B, 12, 12, 1792)

2. avg_pool = reduce_mean(x_f32, axis=-1, keepdims=True)  # (B, 12, 12, 1)
   max_pool = reduce_max(x_f32,  axis=-1, keepdims=True)  # (B, 12, 12, 1)
   → 2 pooling operations trên channel dimension

3. concat = Concat([avg_pool, max_pool], axis=-1)   # (B, 12, 12, 2)

4. sp_attn = Sigmoid(Conv2D(concat))               # (B, 12, 12, 1)
   → Conv2D: 1 filter, kernel 7×7, padding='same', no bias
   → Large kernel (7×7) để capture spatial context rộng

5. spatial_out = x_f32 * sp_attn                   # (B, 12, 12, 1792)
   → broadcast: (B,12,12,1792) × (B,12,12,1) → (B,12,12,1792)
   return spatial_out, sp_attn
```

**Tham số:**

- `Conv2D`: 1×(7×7×2) = **98 params** (rất nhẹ)

---

#### 3.3.3 FSDA Fusion

```
Algorithm: FSDABlock.call(x)

1. freq_out    = FrequencyChannelAttention(x)       # (B, 12, 12, 1792) float32
2. spatial_out = SpatialAttention(x)                # (B, 12, 12, 1792) float32

3. fused = freq_out + spatial_out                   # Element-wise addition
   → NOT concatenation (giữ nguyên channel count)
   → NOT multiplication (tránh vanishing gradient)

4. fused = BatchNorm(fused)                         # normalize fused features
5. fused = cast(fused, input_dtype)                 # trả về float16 nếu cần

return fused, sp_attn                               # sp_attn dùng để visualize
```

**Công thức toán học:**

```
FSDA(x) = BN( FreqAttn(x) + SpatAltn(x) )

FreqAttn(x)  = x ⊗ σ( W₂ · δ( W₁ · GAP(log|FFT(x)|) ) )
SpatAttn(x)  = x ⊗ σ( Conv₇ₓ₇([AvgPool(x) ⊕ MaxPool(x)]) )

Ký hiệu:
  ⊗  = element-wise multiplication
  ⊕  = concatenation
  σ  = Sigmoid
  δ  = ReLU
  GAP = Global Average Pooling
  FFT = 2D Fast Fourier Transform
  |·| = magnitude (complex modulus)
```

---

### 3.4 Classification Head

```
Layer 1: GlobalAveragePooling2D                     (B, 12, 12, 1792) → (B, 1792)
             → average tất cả spatial positions
             → loại bỏ spatial information, giữ channel info

Layer 2: BatchNormalization                         (B, 1792) → (B, 1792)
             → normalize activations trước dense layer

Layer 3: Dense(256, activation='relu')              (B, 1792) → (B, 256)
             → kernel_regularizer=L2(1e-5)
             → học non-linear combination của features

Layer 4: Dropout(0.5)                               (B, 256) → (B, 256)
             → randomly zero 50% neurons during training
             → tránh overfitting

Layer 5: Dense(N_classes, activation='softmax')     (B, 256) → (B, N_classes)
             → dtype='float32' (bắt buộc cho mixed precision)
             → softmax: output là probability distribution
```

---

## 4. LOSS FUNCTION

### 4.1 Categorical Cross-Entropy với Label Smoothing

```
Loss = CategoricalCrossentropy(label_smoothing=0.15)

Standard CE:
    L = -Σ y_i · log(p_i)

With label smoothing (ε=0.15):
    y_smooth_i = y_i · (1 - ε) + ε / K
    L = -Σ y_smooth_i · log(p_i)

Ví dụ (K=5 classes, true class=0):
    Standard:  y = [1.0, 0.0, 0.0, 0.0, 0.0]
    Smoothed:  y = [0.88, 0.03, 0.03, 0.03, 0.03]
```

**Lý do dùng label smoothing:**

- Ngăn model quá tự tin (overconfident)
- Cải thiện calibration
- Regularization effect

### 4.2 Class Weight Balancing

```
cw = sklearn.utils.class_weight.compute_class_weight(
         'balanced',
         classes=unique_classes,
         y=train_labels
     )
→ w_c = N_total / (N_classes × N_c)

→ Các class ít mẫu được nhân trọng số lớn hơn trong loss
→ Tránh bias về class đa số
```

---

## 5. OPTIMIZER & LEARNING RATE SCHEDULE

```
Optimizer: Adam
    β₁ = 0.9  (default)
    β₂ = 0.999 (default)
    ε  = 1e-7  (default)

Learning Rate: ExponentialDecay
    initial_lr    = 1e-4
    decay_steps   = steps_per_epoch × 5
    decay_rate    = 0.9
    staircase     = True

    → lr(step) = 1e-4 × 0.9^⌊step / decay_steps⌋
    → Ví dụ: epoch 0-4: lr=1e-4
             epoch 5-9: lr=9e-5
             epoch 10-14: lr=8.1e-5
             ...
```

---

## 6. TRAINING STRATEGY

### 6.1 Training Configuration

```
BATCH_SIZE   = 32
EPOCHS       = 30  (max, early stopping thường dừng sớm hơn)
PATIENCE     = 12  (early stopping)
```

### 6.2 Mixed Precision Training

```
Policy: mixed_float16

Compute dtype : float16  → forward + backward pass
Variable dtype: float32  → weight storage + update

Lợi ích:
  - ~2x speedup trên GPU Tensor Core
  - ~50% memory giảm → batch size lớn hơn

Lưu ý trong FSDA:
  - FrequencyChannelAttention: tất cả ops trong float32
    (tf.signal.fft2d không hỗ trợ float16)
  - FSDABlock.bn: dtype='float32'
  - Output Dense: dtype='float32' (bắt buộc cho softmax stability)
```

### 6.3 Callbacks

```
① EarlyStopping
      monitor         = 'val_loss'
      patience        = 12
      restore_best_weights = True
      → Dừng training nếu val_loss không cải thiện sau 12 epoch
      → Tự động load lại weights tốt nhất

② ModelCheckpoint
      filepath        = 'best_model.keras'
      monitor         = 'val_loss'
      save_best_only  = True
      → Lưu model có val_loss thấp nhất

③ CSVLogger
      filename        = 'training_log.csv'
      → Ghi lại loss/accuracy mỗi epoch
```

### 6.4 Multi-Run Strategy (Statistical Robustness)

```
N_RUNS       = 3
RANDOM_SEEDS = [42, 123, 456]

Mỗi run:
  1. Set random.seed(seed) + np.random.seed(seed) + tf.random.set_seed(seed)
  2. Tạo mới datasets (shuffle khác nhau)
  3. Khởi tạo model mới (weights khác nhau)
  4. Train độc lập hoàn toàn
  5. Evaluate trên cùng test set

Báo cáo: mean ± std của Accuracy / Precision / Recall / F1
Mục đích: Loại bỏ may rủi của 1 lần chạy
```

---

## 7. EVALUATION METRICS

```
Primary metrics (weighted average):
  ┌─────────────────────────────────────────────────────────────┐
  │ Accuracy  = (TP+TN) / Total                                 │
  │ Precision = TP / (TP+FP)  — weighted avg across classes     │
  │ Recall    = TP / (TP+FN)  — weighted avg across classes     │
  │ F1-Score  = 2×P×R / (P+R) — harmonic mean                  │
  └─────────────────────────────────────────────────────────────┘

Additional metrics:
  - Confusion Matrix (per class)
  - ROC Curves + AUC (one-vs-rest, per class)
  - t-SNE visualization (feature space separability)
  - Inference speed (ms/image, images/second)
```

---

## 8. VISUALIZATION & INTERPRETABILITY

### 8.1 FSDA Spatial Attention Map

```
Source: sp_attn output từ FSDABlock  (B, 12, 12, 1)
Process:
  1. Upsample 12×12 → 380×380 (bilinear)
  2. Normalize: (attn - min) / (max - min)
  3. Apply colormap (jet)
  4. Overlay: 0.55×original + 0.45×heatmap

Ý nghĩa: vùng sáng = nơi model tập trung khi phán đoán bệnh
Ưu điểm vs Grad-CAM: không cần gradient, direct output
```

### 8.2 Grad-CAM++ (EfficientNetB4 top conv)

```
Target layer: 'top_activation'  (last conv layer before FSDA)
Algorithm: Grad-CAM++ (Chattopadhay et al., 2018)

α_k = (∂²y_c / ∂²A_k) / (2·∂²y_c/∂²A_k + Σ A_k·∂³y_c/∂³A_k)
L_c  = ReLU(Σ_k α_k · A_k)

Mixed precision note:
  - tape.watch(conv_out_orig) — phải watch layer output (float16)
  - cast grads → float32 trước khi tính toán
```

### 8.3 t-SNE Feature Visualization

```
Layer: 'gap' output  (B, 1792) — sau FSDA, trước Dense head
Perplexity: 30
n_iter: 1000
Mục đích: Kiểm tra class separability trong feature space
```

### 8.4 Frequency Spectrum Analysis

```
Mỗi class: vẽ FFT log-magnitude + radial frequency profile
Mục đích: Chứng minh hypothesis rằng các class bệnh có
          đặc trưng tần số khác nhau → justifies FSDA design
```

---

## 9. PARAMETER COUNT SUMMARY

| Component                                       | Params (approx) |
| ----------------------------------------------- | --------------- |
| EfficientNetB4 backbone (frozen blocks 1-2)     | ~6.5M frozen    |
| EfficientNetB4 backbone (fine-tuned blocks 3-7) | ~12M trainable  |
| FrequencyChannelAttention (FC1+FC2)             | ~401K           |
| SpatialAttention (Conv2D 7×7)                   | ~98             |
| FSDABlock BN                                    | ~7,168          |
| Head BN                                         | ~7,168          |
| Head Dense(256)                                 | ~459K           |
| Head Dropout                                    | 0               |
| Output Dense                                    | ~256×N_classes  |
| **TOTAL (approx)**                              | **~19.4M**      |

---

## 10. MODEL FILE STRUCTURE (per run)

```
report_EfficientNetB4/
└── finetune_top5_fsda/
    ├── strategy_summary.csv          ← metrics mỗi run
    ├── overall_metrics_summary.csv   ← mean±std tổng hợp
    ├── per_class_metrics.png         ← bar chart P/R/F1 per class
    ├── agg_confusion_matrix.png      ← confusion matrix tổng hợp
    ├── roc_curves.png                ← ROC + AUC per class
    ├── frequency_spectra.png         ← FFT spectrum per class
    ├── fsda_attention_maps.png       ← spatial attention overlay
    ├── gradcam_pp.png                ← Grad-CAM++ visualization
    ├── tsne.png                      ← t-SNE feature space
    ├── final_report.csv              ← final export
    ├── run_1_seed_42/
    │   ├── best_model.keras          ← saved model weights
    │   ├── training_log.csv          ← epoch-by-epoch metrics
    │   ├── learning_curve.png
    │   └── classification_report.txt
    ├── run_2_seed_123/
    └── run_3_seed_456/
```

---

## 11. DESIGN DECISIONS & RATIONALE

| Decision                     | Why                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------ |
| EfficientNetB4 (not B0/B7)   | Balance: accuracy vs. params; 380px input matches disease texture resolution   |
| FFT-based channel attention  | Disease symptoms = texture anomalies = specific frequency bands                |
| Addition fusion (not concat) | Preserves channel count; complementary signals combine without doubling params |
| 7×7 spatial conv             | Large receptive field captures spread-out lesion regions better than 3×3       |
| reduction=16 in FreqAttn     | Standard SE-Net ratio; C/16=112 sufficient for 1792-channel bottleneck         |
| label_smoothing=0.15         | Empirically effective for noisy agricultural image datasets                    |
| BN always frozen             | Fine-tuning with small batch size (32) can corrupt BN statistics from ImageNet |
| 3 seeds × 30 epochs          | Provides statistical confidence; ~90 total training epochs per experiment      |
| Mixed float16                | 2× speed, 50% memory → enables batch=32 with 380px images on single GPU        |
