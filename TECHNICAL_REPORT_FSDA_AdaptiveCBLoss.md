# TECHNICAL REPORT — DETAILED DESIGN
# EfficientNetB4 + FSDA + Adaptive Class-Balanced Focal Loss
# Garlic Disease Classification

> Tài liệu này mô tả CHI TIẾT từng bước tính toán, từng tensor shape,
> từng giá trị hyperparameter — tất cả đều trích xuất trực tiếp từ source code
> `EfficientNetB4-FSDA-AdaptiveCBLoss.ipynb`

---

## MỤC LỤC

1. [STAGE 1 — System Configuration](#stage-1)
2. [STAGE 2 — Input Pipeline & Preprocessing](#stage-2)
3. [STAGE 3 — Data Augmentation](#stage-3)
4. [STAGE 4 — EfficientNetB4 Backbone](#stage-4)
5. [STAGE 5 — FSDA Block (Novel)](#stage-5)
   - 5A. Frequency Channel Attention
   - 5B. Spatial Attention (CBAM-style)
   - 5C. Fusion
6. [STAGE 6 — Classification Head](#stage-6)
7. [STAGE 7 — Adaptive Class-Balanced Focal Loss (Novel)](#stage-7)
   - 7A. Static CB Weights
   - 7B. Focal Modulation
   - 7C. Adaptive Weight Callback (EMA)
8. [STAGE 8 — Optimizer & LR Schedule](#stage-8)
9. [STAGE 9 — Training Loop](#stage-9)
10. [STAGE 10 — Evaluation & Metrics](#stage-10)
11. [TỔNG HỢP PARAMETER COUNT](#param-count)
12. [DATA FLOW SUMMARY (end-to-end)](#data-flow)

---

## <a id="stage-1"></a>STAGE 1 — SYSTEM CONFIGURATION

### 1.1 GPU & Memory

```
TensorFlow version: 2.x
GPU memory growth: enabled (dynamic allocation)
```

### 1.2 Mixed Precision Policy

```
Policy name    : "mixed_float16"
Compute dtype  : float16   ← forward pass + backward pass
Variable dtype : float32   ← weight storage + optimizer state

Lợi ích:
  - ~2x speedup trên GPU Tensor Core (NVIDIA)
  - ~50% giảm VRAM → cho phép batch_size lớn hơn

Lưu ý quan trọng (trong code):
  - FFT operations (tf.signal.fft2d): bắt buộc float32
  - FSDA BN layer: dtype='float32'
  - Output Dense (softmax): dtype='float32'
  → Tất cả đều explicit cast trong code
```

### 1.3 XLA JIT Compilation

```
tf.config.optimizer.set_jit(True)
→ XLA (Accelerated Linear Algebra) compile TF graph
→ Fuse operations, giảm memory overhead
```

---

## <a id="stage-2"></a>STAGE 2 — INPUT PIPELINE & PREPROCESSING

### 2.1 Dataset Structure

```
dataset_final_2006/
├── train/
│   ├── Class_A/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── Class_B/
│   └── ...
├── val/
│   ├── Class_A/
│   └── ...
└── test/
    ├── Class_A/
    └── ...

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
Class names: sorted alphabetically (os.listdir → sorted)
class_to_idx: {'ClassA': 0, 'ClassB': 1, ...} (alphabetical order)
```

### 2.2 Preprocessing Pipeline (per image)

```
Bước 1: tf.io.read_file(path)
         Input:  file path (string)
         Output: raw bytes (string tensor)

Bước 2: tf.image.decode_jpeg(raw, channels=3)
         Input:  raw bytes
         Output: uint8 tensor, shape (H_orig, W_orig, 3)
         Note:   channels=3 force RGB (nếu grayscale → replicate)

Bước 3: tf.image.resize(img, [380, 380])
         Input:  (H_orig, W_orig, 3) bất kỳ resolution
         Output: (380, 380, 3) float32
         Method: bilinear interpolation (default TF)

Bước 4: tf.cast(img, tf.float32)
         Output: (380, 380, 3) float32, range [0, 255]

Bước 5: efficientnet_preprocess(img)
         = tf.keras.applications.efficientnet.preprocess_input(img)
         Công thức: pixel = pixel / 127.5 - 1.0
         Input range:  [0, 255]
         Output range: [-1, +1]
         Output: (380, 380, 3) float32

Bước 6: tf.one_hot(label, depth=num_classes)
         Input:  int label (ví dụ: 2)
         Output: (num_classes,) float32 (ví dụ: [0, 0, 1, 0, 0])
```

### 2.3 tf.data Pipeline

```
AUTOTUNE = tf.data.AUTOTUNE (tự chọn số CPU workers tối ưu)
BATCH_SIZE = 32

Train pipeline:
  Dataset.from_tensor_slices((paths, labels))
    → .shuffle(buffer_size=N_train, seed=seed, reshuffle_each_iteration=True)
    → .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    → .map(augment, num_parallel_calls=AUTOTUNE)          ← chỉ train
    → .batch(32, drop_remainder=True)                     ← drop_remainder=True
    → .prefetch(AUTOTUNE)

Val pipeline:
  Dataset.from_tensor_slices((paths, labels))
    → .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    → .batch(32)                                          ← drop_remainder=False
    → .prefetch(AUTOTUNE)

Test pipeline:
  (giống Val)

Output tensor shapes per batch:
  images: (32, 380, 380, 3)  float16 (auto-cast by mixed precision)
  labels: (32, num_classes)   float32 (one-hot)
```

---

## <a id="stage-3"></a>STAGE 3 — DATA AUGMENTATION (chỉ Training)

```
tf.keras.Sequential (name='augmentation'), áp dụng tuần tự:

┌───┬──────────────────────────────────────────────────────────────────────┐
│ # │ Layer                              │ Chi tiết                       │
├───┼──────────────────────────────────────────────────────────────────────┤
│ 1 │ RandomFlip("horizontal_and_vertical")                               │
│   │   → p=0.5 lật ngang, p=0.5 lật dọc (độc lập)                      │
│   │   → Input/Output: (380, 380, 3)                                    │
├───┼──────────────────────────────────────────────────────────────────────┤
│ 2 │ RandomRotation(factor=0.083)                                        │
│   │   → factor = 0.083 ≈ 30/360                                        │
│   │   → Xoay ngẫu nhiên trong [-30°, +30°]                             │
│   │   → fill_mode='reflect' (default)                                  │
│   │   → Input/Output: (380, 380, 3)                                    │
├───┼──────────────────────────────────────────────────────────────────────┤
│ 3 │ RandomZoom(height_factor=0.20)                                      │
│   │   → Zoom in/out ngẫu nhiên ±20%                                    │
│   │   → Range: [1-0.20, 1+0.20] = [0.80, 1.20]                        │
│   │   → Input/Output: (380, 380, 3)                                    │
├───┼──────────────────────────────────────────────────────────────────────┤
│ 4 │ RandomTranslation(height=0.20, width=0.20)                         │
│   │   → Dịch chuyển ±20% theo cả 2 chiều                               │
│   │   → Max shift: 0.20 × 380 = ±76 pixels                            │
│   │   → Input/Output: (380, 380, 3)                                    │
├───┼──────────────────────────────────────────────────────────────────────┤
│ 5 │ RandomBrightness(factor=0.30)                                       │
│   │   → Thay đổi brightness ±30%                                       │
│   │   → Input/Output: (380, 380, 3)                                    │
└───┴──────────────────────────────────────────────────────────────────────┘

Tổng augmentation params: 0 (tất cả là random ops, không có trainable weights)
Chỉ áp dụng khi training=True
```

---

## <a id="stage-4"></a>STAGE 4 — EfficientNetB4 BACKBONE (Feature Extractor)

### 4.1 Architecture Overview

```
Source: tf.keras.applications.EfficientNetB4
Pretrained: ImageNet (1.28M images, 1000 classes)
Input:  (B, 380, 380, 3)
Output: (B, 12, 12, 1792)   ← "top_activation" layer

Compound scaling (Tan & Le, 2019):
  depth_coefficient  = 1.8
  width_coefficient  = 1.4
  resolution         = 380
  dropout_rate       = 0.4

include_top = False  → loại bỏ classification head gốc (1000 classes)
Total layers ≈ 475
Total params ≈ 19M
```

### 4.2 Freeze Strategy (chi tiết từ code)

```python
apply_freeze_strategy(base, unfreeze_blocks=[3, 4, 5, 6, 7])

Thuật toán:
  1. base.trainable = False          # Đóng băng TẤT CẢ trước
  2. Duyệt từng layer:
     - Nếu layer.name bắt đầu bằng "block3", "block4", "block5", "block6", "block7":
         - Nếu layer KHÔNG phải BatchNormalization:
             → layer.trainable = True   # Mở đóng băng
         - Nếu layer LÀ BatchNormalization:
             → giữ trainable = False    # BN LUÔN đóng băng
     - Nếu layer.name bắt đầu bằng "block1", "block2":
         → giữ trainable = False        # Low-level features đóng băng

Kết quả:
  ┌───────────────┬────────────┬─────────────────────────────────────────┐
  │ Blocks        │ Trainable? │ Lý do                                   │
  ├───────────────┼────────────┼─────────────────────────────────────────┤
  │ Stem + B1-B2  │ FROZEN     │ Edges, corners, basic textures          │
  │               │            │ → universal, không cần adapt            │
  ├───────────────┼────────────┼─────────────────────────────────────────┤
  │ B3-B7         │ TRAINABLE  │ High-level semantics (disease textures, │
  │ (trừ BN)      │            │ color patterns) → cần adapt cho garlic  │
  ├───────────────┼────────────┼─────────────────────────────────────────┤
  │ ALL BN layers │ FROZEN     │ Batch=32 quá nhỏ → BN stats sẽ noisy   │
  │               │            │ → giữ ImageNet statistics ổn định       │
  └───────────────┴────────────┴─────────────────────────────────────────┘

Tensor shape qua backbone:
  Input:  (B, 380, 380, 3)
  Stem:   (B, 190, 190, 48)
  Block1: (B, 190, 190, 24)
  Block2: (B, 95, 95, 32)
  Block3: (B, 48, 48, 56)
  Block4: (B, 24, 24, 112)
  Block5: (B, 24, 24, 160)
  Block6: (B, 12, 12, 272)
  Block7: (B, 12, 12, 448)
  top:    (B, 12, 12, 1792)   ← "top_activation" (final Swish + Conv1x1)
```

---

## <a id="stage-5"></a>STAGE 5 — FSDA BLOCK (Frequency-Spatial Dual Attention)

```
Input:   feat_map = (B, 12, 12, 1792) float16 (từ backbone)
Output:  fused    = (B, 12, 12, 1792) float16
         sp_attn  = (B, 12, 12, 1)    float32 (để visualization)

Cấu trúc bên trong:
  feat_map ──┬── FrequencyChannelAttention ──→ freq_out  ──┐
             │                                              │ Element-wise ADD
             └── SpatialAttention ───────────→ spatial_out ─┘
                                                            │
                                                      BatchNorm(float32)
                                                            │
                                                      cast → input_dtype
                                                            │
                                                         fused
```

### 5A. FrequencyChannelAttention (chi tiết từng bước)

**Mục đích:** Xác định KÊNH NÀO (trong 1792 channels) mang thông tin tần số liên quan đến bệnh.

**Hyperparameters từ code:**
```
reduction = 16
r = max(1792 // 16, 8) = max(112, 8) = 112
FC1: Dense(112, use_bias=False, dtype='float32')
FC2: Dense(1792, use_bias=False, dtype='float32')
```

**Từng bước tính toán:**

```
BƯỚC 1: Cast to float32
  x_f32 = tf.cast(x, tf.float32)
  Input:  (B, 12, 12, 1792) float16
  Output: (B, 12, 12, 1792) float32
  Lý do:  tf.signal.fft2d KHÔNG hỗ trợ float16

BƯỚC 2: Transpose (NHWC → NCHW)
  x_t = tf.transpose(x_f32, [0, 3, 1, 2])
  Input:  (B, 12, 12, 1792)    [batch, height, width, channels]
  Output: (B, 1792, 12, 12)    [batch, channels, height, width]
  Lý do:  FFT2D cần spatial dims ở cuối (axis[-2], axis[-1])

BƯỚC 3: Cast to complex64
  x_complex = tf.complex(x_t, tf.zeros_like(x_t))
  Input:  (B, 1792, 12, 12) float32
  Output: (B, 1792, 12, 12) complex64
  Giải thích: real part = x_t, imaginary part = 0
               → tạo complex tensor cho FFT

BƯỚC 4: 2D Fast Fourier Transform
  x_fft = tf.signal.fft2d(x_complex)
  Input:  (B, 1792, 12, 12) complex64
  Output: (B, 1792, 12, 12) complex64
  Giải thích: FFT được áp dụng TRÊN TỪNG CHANNEL độc lập
              → mỗi channel (12×12 spatial) → frequency domain (12×12 freq)
              → kết quả complex: X[u,v] = Σ_x Σ_y  f[x,y] · e^{-j2π(ux/12 + vy/12)}

BƯỚC 5: Log-magnitude spectrum
  mag = tf.math.log1p(tf.abs(x_fft))
  Bước 5a: tf.abs(x_fft)
    Input:  (B, 1792, 12, 12) complex64
    Output: (B, 1792, 12, 12) float32
    Công thức: |X[u,v]| = sqrt(Re² + Im²)

  Bước 5b: tf.math.log1p(...)
    Input:  (B, 1792, 12, 12) float32
    Output: (B, 1792, 12, 12) float32
    Công thức: log1p(x) = log(1 + x)
    Lý do:  - Nén dynamic range (magnitude có thể rất lớn)
            - log1p thay vì log để tránh log(0) = -inf
            - +1 đảm bảo output ≥ 0

BƯỚC 6: Global Average Pooling (spatial frequency dims)
  freq_desc = tf.reduce_mean(mag, axis=[2, 3])
  Input:  (B, 1792, 12, 12) float32
  Output: (B, 1792)          float32
  Công thức: freq_desc[b, c] = (1/144) × Σ_{u=0}^{11} Σ_{v=0}^{11} mag[b, c, u, v]
  Giải thích: Tóm tắt thông tin tần số mỗi channel thành 1 scalar
              → "frequency descriptor" cho mỗi channel

BƯỚC 7: Bottleneck MLP — FC1 (squeeze)
  attn = tf.nn.relu(self.fc1(freq_desc))
  Input:  (B, 1792) float32
  Output: (B, 112)  float32

  Chi tiết FC1:
    W1 shape: (1792, 112)    ← 200,704 params
    bias:     None (use_bias=False)
    dtype:    float32
    Công thức: attn = ReLU(freq_desc @ W1)
               ReLU(x) = max(0, x)

BƯỚC 8: Bottleneck MLP — FC2 (excitation)
  attn = tf.nn.sigmoid(self.fc2(attn))
  Input:  (B, 112)  float32
  Output: (B, 1792) float32

  Chi tiết FC2:
    W2 shape: (112, 1792)    ← 200,704 params
    bias:     None (use_bias=False)
    dtype:    float32
    Công thức: attn = σ(attn @ W2)
               σ(x) = 1 / (1 + e^{-x})    → output range (0, 1)

BƯỚC 9: Reshape (broadcast-ready)
  attn = tf.reshape(attn, [B, 1, 1, 1792])
  Input:  (B, 1792)
  Output: (B, 1, 1, 1792)
  Lý do:  Chuẩn bị cho element-wise multiply với spatial tensor

BƯỚC 10: Channel-wise reweighting
  out = x_f32 * attn
  Input:  x_f32 = (B, 12, 12, 1792)
          attn  = (B, 1, 1, 1792)    ← broadcast over H, W
  Output: (B, 12, 12, 1792) float32
  Công thức: out[b, h, w, c] = x_f32[b, h, w, c] × attn[b, 0, 0, c]
  Giải thích: Mỗi channel được nhân với 1 hệ số trong (0,1)
              Channel quan trọng (tần số bệnh) → hệ số ≈ 1
              Channel không quan trọng → hệ số ≈ 0

BƯỚC 11: Cast về input dtype
  return tf.cast(out, x.dtype)
  Output: (B, 12, 12, 1792) float16 (nếu mixed precision)

TỔNG PARAMS FrequencyChannelAttention:
  FC1: 1792 × 112 = 200,704
  FC2: 112 × 1792 = 200,704
  TOTAL = 401,408 params (all float32, all trainable)
```

**Công thức tổng hợp:**
```
FreqAttn(x) = x ⊗ σ( W₂ · ReLU( W₁ · GAP( log(1 + |FFT2D(x)|) ) ) )

Trong đó:
  ⊗ = element-wise multiplication (channel-wise)
  σ = Sigmoid activation
  W₁ ∈ R^{1792 × 112}  (squeeze)
  W₂ ∈ R^{112 × 1792}  (excitation)
  GAP = Global Average Pooling trên spatial frequency dimensions
  FFT2D = 2D Discrete Fourier Transform (per channel)
```

---

### 5B. SpatialAttention (CBAM-style) (chi tiết từng bước)

**Mục đích:** Xác định VỊ TRÍ NÀO trên ảnh có tổn thương bệnh.

**Hyperparameters từ code:**
```
spatial_kernel = 7
Conv2D: 1 filter, kernel 7×7, padding='same', use_bias=False, dtype='float32'
```

**Từng bước tính toán:**

```
BƯỚC 1: Cast to float32
  x_f32 = tf.cast(x, tf.float32)
  Input:  (B, 12, 12, 1792) float16
  Output: (B, 12, 12, 1792) float32

BƯỚC 2: Average Pooling (channel axis)
  avg_pool = tf.reduce_mean(x_f32, axis=-1, keepdims=True)
  Input:  (B, 12, 12, 1792)
  Output: (B, 12, 12, 1)
  Công thức: avg_pool[b, h, w, 0] = (1/1792) × Σ_{c=0}^{1791} x_f32[b, h, w, c]
  Giải thích: Tại mỗi vị trí spatial, tính trung bình tất cả channels

BƯỚC 3: Max Pooling (channel axis)
  max_pool = tf.reduce_max(x_f32, axis=-1, keepdims=True)
  Input:  (B, 12, 12, 1792)
  Output: (B, 12, 12, 1)
  Công thức: max_pool[b, h, w, 0] = max_{c} x_f32[b, h, w, c]
  Giải thích: Tại mỗi vị trí spatial, lấy giá trị lớn nhất qua channels

BƯỚC 4: Concatenate
  concat = tf.concat([avg_pool, max_pool], axis=-1)
  Input:  avg_pool = (B, 12, 12, 1)
          max_pool = (B, 12, 12, 1)
  Output: (B, 12, 12, 2)
  Giải thích: Kết hợp 2 cách tóm tắt channel information

BƯỚC 5: Conv2D 7×7
  conv_out = self.sp_conv(concat)
  Input:  (B, 12, 12, 2)
  Output: (B, 12, 12, 1)

  Chi tiết Conv2D:
    filters         = 1
    kernel_size     = (7, 7)
    padding         = 'same'       → output spatial = input spatial
    use_bias        = False
    kernel_init     = 'glorot_uniform'
    dtype           = 'float32'
    Kernel shape    = (7, 7, 2, 1)  → 7 × 7 × 2 × 1 = 98 params

  Công thức:
    conv_out[b, h, w, 0] = Σ_{i=-3}^{3} Σ_{j=-3}^{3} Σ_{k=0}^{1}
                           W[i+3, j+3, k, 0] × concat[b, h+i, w+j, k]
  (with zero-padding at borders due to padding='same')

  Lý do dùng kernel 7×7 (lớn):
    - Feature map chỉ 12×12 → receptive field 7×7 bao phủ >50% spatial
    - Capture context rộng hơn → phát hiện vùng tổn thương trải rộng

BƯỚC 6: Sigmoid activation
  sp_attn = tf.nn.sigmoid(conv_out)
  Input:  (B, 12, 12, 1)
  Output: (B, 12, 12, 1)    range (0, 1)
  Công thức: sp_attn[b, h, w, 0] = 1 / (1 + e^{-conv_out[b,h,w,0]})
  Giải thích: Mỗi vị trí spatial nhận 1 "attention score" trong (0,1)
              Score cao → vùng quan trọng (lesion)
              Score thấp → vùng background

BƯỚC 7: Spatial-wise reweighting
  spatial_out = x_f32 * sp_attn
  Input:  x_f32   = (B, 12, 12, 1792)
          sp_attn = (B, 12, 12, 1)     ← broadcast over C
  Output: (B, 12, 12, 1792)
  Công thức: spatial_out[b, h, w, c] = x_f32[b, h, w, c] × sp_attn[b, h, w, 0]
  Giải thích: TẤT CẢ channels tại cùng vị trí (h,w) được nhân cùng 1 hệ số

TỔNG PARAMS SpatialAttention:
  Conv2D kernel: 7 × 7 × 2 × 1 = 98 params (float32, trainable)
```

**Công thức tổng hợp:**
```
SpatAttn(x) = x ⊗ σ( Conv_{7×7}( [AvgPool_c(x) ⊕ MaxPool_c(x)] ) )

Trong đó:
  ⊗ = element-wise multiplication (spatial-wise, broadcast over channels)
  ⊕ = concatenation along channel axis
  σ = Sigmoid
  AvgPool_c = reduce_mean over channel axis → (B,H,W,1)
  MaxPool_c = reduce_max over channel axis  → (B,H,W,1)
  Conv_{7×7} = single-filter convolution with 7×7 kernel
```

---

### 5C. FSDA Fusion (chi tiết từng bước)

```
BƯỚC 1: Element-wise Addition
  fused = freq_out + spatial_out
  Input:  freq_out    = (B, 12, 12, 1792) float32  (từ 5A)
          spatial_out = (B, 12, 12, 1792) float32  (từ 5B)
  Output: (B, 12, 12, 1792) float32
  Công thức: fused[b,h,w,c] = freq_out[b,h,w,c] + spatial_out[b,h,w,c]

  Lý do chọn Addition (không phải Concatenation hay Multiplication):
    - Concatenation: sẽ double channels (1792 → 3584) → tăng params downstream
    - Multiplication: vanishing gradient risk (freq ≈ 0 → spatial bị kill)
    - Addition: complementary signals cộng lại, giữ nguyên channel count

BƯỚC 2: BatchNormalization
  fused = self.bn(fused, training=training)
  Input:  (B, 12, 12, 1792) float32
  Output: (B, 12, 12, 1792) float32

  Chi tiết BN:
    dtype = 'float32' (explicit)
    gamma: (1792,) → scale
    beta:  (1792,) → shift
    moving_mean: (1792,)
    moving_var:  (1792,)
    Params: 1792 × 4 = 7,168

  Công thức (inference):
    y = gamma × (x - moving_mean) / sqrt(moving_var + epsilon) + beta

  Công thức (training):
    batch_mean = mean(x, axis=[0,1,2])
    batch_var  = var(x, axis=[0,1,2])
    y = gamma × (x - batch_mean) / sqrt(batch_var + epsilon) + beta
    moving_mean = momentum × moving_mean + (1 - momentum) × batch_mean
    moving_var  = momentum × moving_var  + (1 - momentum) × batch_var

BƯỚC 3: Cast về input dtype
  fused = tf.cast(fused, input_dtype)
  Output: (B, 12, 12, 1792) float16 (nếu mixed precision)

RETURN: (fused, sp_attn)
  fused:   (B, 12, 12, 1792) float16  → tiếp tục vào Classification Head
  sp_attn: (B, 12, 12, 1)    float32  → dùng để visualization (attention map overlay)
```

**Công thức FSDA hoàn chỉnh:**
```
FSDA(x) = BN( FreqAttn(x) + SpatAttn(x) )

FSDA(x) = BN( x ⊗ σ(W₂·δ(W₁·GAP(log(1+|FFT₂D(x)|))))
             + x ⊗ σ(Conv₇ₓ₇([AvgPool(x) ⊕ MaxPool(x)])) )

TỔNG PARAMS FSDA:
  FreqChannelAttn FC1:  200,704
  FreqChannelAttn FC2:  200,704
  SpatialAttn Conv2D:        98
  BatchNorm:              7,168
  TOTAL FSDA:           408,674 params
```

---

## <a id="stage-6"></a>STAGE 6 — CLASSIFICATION HEAD

```
Input: fused = (B, 12, 12, 1792) float16 (từ FSDA output)

LAYER 1: GlobalAveragePooling2D (name='gap')
  Input:  (B, 12, 12, 1792)
  Output: (B, 1792)
  Công thức: gap[b, c] = (1/144) × Σ_{h=0}^{11} Σ_{w=0}^{11} fused[b, h, w, c]
  Params: 0
  Giải thích: Loại bỏ spatial information, giữ channel info
              12 × 12 = 144 spatial positions → average

LAYER 2: BatchNormalization (name='head_bn')
  Input:  (B, 1792)
  Output: (B, 1792)
  Params: 1792 × 4 = 7,168 (gamma, beta, moving_mean, moving_var)
  Note:   default dtype (follows mixed precision policy)

LAYER 3: Dense(256, activation='relu', kernel_regularizer=l2(1e-5)) (name='head_dense')
  Input:  (B, 1792)
  Output: (B, 256)
  Kernel: (1792, 256) = 458,752 params
  Bias:   (256,)      = 256 params
  Total:  459,008 params

  Công thức: y = ReLU(x @ W + b)
  L2 regularization: loss += 1e-5 × Σ W²
    → Penalize large weights → prevent overfitting

LAYER 4: Dropout(0.5) (name='head_dropout')
  Input:  (B, 256)
  Output: (B, 256)
  Params: 0
  Behavior:
    Training: randomly zero out 50% neurons, scale remaining by 2
              mask ~ Bernoulli(p=0.5), y = x × mask / 0.5
    Inference: identity (y = x)

LAYER 5: Dense(num_classes, activation='softmax', dtype='float32') (name='predictions')
  Input:  (B, 256)
  Output: (B, num_classes)
  Kernel: (256, N) params
  Bias:   (N,)    params
  Total:  256 × N + N = 257 × N params

  dtype='float32': BẮT BUỘC cho softmax stability
    (float16 softmax có thể overflow khi exp(x) với x lớn)

  Công thức softmax: p_i = exp(z_i) / Σ_j exp(z_j)
    → output range (0, 1)
    → Σ p_i = 1 (probability distribution)

TỔNG PARAMS HEAD:
  GAP:           0
  BN:        7,168
  Dense(256): 459,008
  Dropout:        0
  Dense(N):   257 × N
  TOTAL:     466,176 + 257×N
```

---

## <a id="stage-7"></a>STAGE 7 — ADAPTIVE CLASS-BALANCED FOCAL LOSS (Novel Contribution)

### 7A. Static Class-Balanced Weights (Cui et al., CVPR 2019)

```
Input: samples_per_class = [n₁, n₂, ..., n_K]  (số mẫu training mỗi class)
       beta = 0.9999

Bước 1: Compute effective number
  E_c = 1 - β^{n_c}
  Ví dụ: n_c = 1000, β = 0.9999
         E_c = 1 - 0.9999^1000 = 1 - 0.9048 ≈ 0.0952

Bước 2: Compute raw weights
  w_c = (1 - β) / E_c
      = 0.0001 / E_c

Bước 3: Normalize
  w_c = w_c / Σ_c w_c × K
  → weights sum = K (number of classes)
  → trung bình weight = 1.0

Kết quả: static_weights = tf.constant(weights, dtype=tf.float32)  — shape (K,)

Ý nghĩa:
  - Class ít mẫu → E_c nhỏ → w_c LỚN
  - Class nhiều mẫu → E_c lớn → w_c NHỎ
  - Cân bằng đóng góp của mỗi class vào tổng loss
```

### 7B. Loss Computation (forward pass)

```python
def call(self, y_true, y_pred):
```

```
Input:
  y_true: (B, K) float32   — one-hot encoded labels
  y_pred: (B, K) float32   — softmax probabilities

BƯỚC 1: Cast & Clip
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
  Lý do clip: tránh log(0) = -inf và log(1) numerical issues

BƯỚC 2: Combined weights = static × adaptive
  combined_weights = self.static_weights * self.adaptive_factor
  Input:  static_weights  = (K,) float32   — cố định
          adaptive_factor = (K,) float32   — cập nhật mỗi epoch
  Output: combined_weights = (K,) float32

  Normalize: combined_weights = combined_weights / mean(combined_weights)
  → Giữ loss magnitude ổn định qua các epoch

BƯỚC 3: Per-sample weight
  sample_w = tf.reduce_sum(y_true * combined_weights, axis=-1)
  Input:  y_true           = (B, K)
          combined_weights = (K,)     ← broadcast
  Output: sample_w = (B,)
  Giải thích: Mỗi sample nhận weight của TRUE class
              y_true = [0, 0, 1, 0, 0], weights = [w0, w1, w2, w3, w4]
              → sample_w = w2

BƯỚC 4: Focal modulation (Lin et al., 2017)
  pt    = tf.reduce_sum(y_true * y_pred, axis=-1)    # (B,)
  focal = tf.pow(1.0 - pt, self.gamma)                # (B,)

  Giải thích:
    pt = predicted probability of TRUE class
    gamma = 2.0

    Nếu pt → 1 (easy sample, predicted đúng):
      focal = (1 - 1)^2 = 0  → loss ≈ 0 (bỏ qua)

    Nếu pt → 0 (hard sample, predicted sai):
      focal = (1 - 0)^2 = 1  → loss giữ nguyên (focus vào)

    Nếu pt = 0.5 (uncertain):
      focal = (0.5)^2 = 0.25 → loss giảm 75%

BƯỚC 5: Cross-entropy
  ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)   # (B,)
  Công thức: CE = -Σ_c y_true_c × log(p_c)
             = -log(p_{true_class})    (vì y_true one-hot)

BƯỚC 6: Final loss
  loss = tf.reduce_mean(sample_w * focal * ce)
  Công thức:
    L = (1/B) × Σ_{b=1}^{B} w_{c_b} × (1 - p_{c_b})^γ × (-log(p_{c_b}))

  Trong đó:
    w_{c_b} = combined weight cho true class của sample b
    p_{c_b} = predicted probability cho true class của sample b
    γ = 2.0

TOÀN BỘ CÔNG THỨC:
  L = (1/B) Σ_b [ w_c^{static} × f_c^{adaptive}(t) × (1-p_t)^γ × (-log p_t) ]

  Trong đó:
    w_c^{static}     = (1-β) / (1-β^{n_c}) × normalization
    f_c^{adaptive}   = EMA-updated factor (xem 7C)
    p_t              = predicted prob of true class
    γ                = 2.0 (focal gamma)
    β                = 0.9999 (CB beta)
```

### 7C. Adaptive Weight Callback (EMA Update — Novel)

```
Trigger: on_epoch_end (sau mỗi epoch training)
Hyperparameters:
  tau     = 0.3     (EMA smoothing factor)
  epsilon = 0.1     (minimum adaptation factor)

BƯỚC 1: Predict trên validation set
  y_pred_probs = model.predict(val_ds, verbose=0)      # (N_val, K) float32
  y_pred = np.argmax(y_pred_probs, axis=1)              # (N_val,)  int

BƯỚC 2: Extract true labels
  y_true = concatenate([argmax(y, axis=1) for _, y in val_ds])   # (N_val,) int
  (one-hot → integer labels)

BƯỚC 3: Compute per-class recall
  for c in range(K):
    mask = (y_true == c)
    if mask.sum() > 0:
      recall_c = mean(y_pred[mask] == c)     # TP / (TP + FN) cho class c
    else:
      recall_c = 1.0                          # no samples → assume perfect

  per_class_recall = [recall_0, recall_1, ..., recall_{K-1}]   # (K,)

BƯỚC 4: Compute adaptation target
  adaptation_target_c = (1 - recall_c) + epsilon

  Giải thích:
    recall_c = 1.0 (perfect) → target = 0.0 + 0.1 = 0.1 (giảm weight)
    recall_c = 0.0 (worst)   → target = 1.0 + 0.1 = 1.1 (tăng weight MẠNH)
    recall_c = 0.5 (medium)  → target = 0.5 + 0.1 = 0.6

    epsilon = 0.1 đảm bảo không bao giờ target = 0 (tránh zero weight)

BƯỚC 5: EMA update
  new_factor_c = (1 - tau) × current_factor_c + tau × adaptation_target_c
               = 0.7 × current_factor_c + 0.3 × adaptation_target_c

  Giải thích:
    tau = 0.3 → 70% giữ giá trị cũ, 30% cập nhật mới
    → Smooth, tránh oscillation (nhảy qua nhảy lại)
    → Convergence dần dần

    Ví dụ epoch 1 (initial factor = 1.0):
      recall = [0.9, 0.5, 0.8]
      target = [0.2, 0.6, 0.3]
      new    = 0.7 × [1.0, 1.0, 1.0] + 0.3 × [0.2, 0.6, 0.3]
             = [0.7 + 0.06, 0.7 + 0.18, 0.7 + 0.09]
             = [0.76, 0.88, 0.79]

BƯỚC 6: Normalize
  new_factor = new_factor / mean(new_factor)
  → Mean = 1.0 → tổng loss magnitude không đổi

  Tiếp ví dụ: mean = (0.76 + 0.88 + 0.79) / 3 = 0.81
              new_factor = [0.938, 1.086, 0.975]
              → Class 2 (recall=0.5, thấp nhất) nhận factor CAO NHẤT (1.086)
              → Class 1 (recall=0.9, cao nhất) nhận factor THẤP NHẤT (0.938)

BƯỚC 7: Assign
  self.loss_fn.adaptive_factor.assign(new_factor)
  → tf.Variable được cập nhật → ảnh hưởng ngay epoch tiếp theo

LOG OUTPUT mỗi epoch:
  [AdaptiveCB] Epoch 5 | Recall: {'ClassA': '0.920', 'ClassB': '0.650', ...}
                       | Factors: {'ClassA': '0.871', 'ClassB': '1.234', ...}
```

**Toàn bộ adaptive mechanism (công thức toán):**
```
Tại epoch t:

1. recall_c^(t) = TP_c / (TP_c + FN_c)     trên validation set

2. target_c^(t) = (1 - recall_c^(t)) + ε    với ε = 0.1

3. f_c^(t) = (1 - τ) × f_c^(t-1) + τ × target_c^(t)    với τ = 0.3, f_c^(0) = 1

4. f_c^(t) = f_c^(t) / mean(f^(t))          normalize

5. w_c^(t) = w_c^{static} × f_c^(t)         final weight for loss

Feedback loop:
  Low recall → high target → high factor → high loss weight → model focuses more
  High recall → low target → low factor → low loss weight → model relaxes
```

---

## <a id="stage-8"></a>STAGE 8 — OPTIMIZER & LEARNING RATE SCHEDULE

### 8.1 Optimizer: Adam

```
tf.keras.optimizers.Adam(learning_rate=lr_schedule)

Hyperparameters (all default except lr):
  β₁ = 0.9       (first moment decay)
  β₂ = 0.999     (second moment decay)
  ε  = 1e-7      (numerical stability)

Update rule:
  m_t = β₁ × m_{t-1} + (1 - β₁) × g_t           # first moment
  v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²           # second moment
  m̂_t = m_t / (1 - β₁^t)                          # bias correction
  v̂_t = v_t / (1 - β₂^t)                          # bias correction
  θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)          # parameter update
```

### 8.2 Learning Rate Schedule: ExponentialDecay (Staircase)

```
tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-4,
    decay_steps           = steps_per_epoch × 5,
    decay_rate            = 0.9,
    staircase             = True,
)

Công thức:
  lr(step) = initial_lr × decay_rate ^ floor(step / decay_steps)
           = 1e-4 × 0.9 ^ floor(step / (steps_per_epoch × 5))

Staircase = True: lr chỉ thay đổi mỗi 5 epoch (step function)

Ví dụ (giả sử steps_per_epoch = 50):
  decay_steps = 50 × 5 = 250

  Epoch 0-4   (step 0-249):   lr = 1e-4 × 0.9^0 = 1.0000e-4
  Epoch 5-9   (step 250-499): lr = 1e-4 × 0.9^1 = 9.0000e-5
  Epoch 10-14 (step 500-749): lr = 1e-4 × 0.9^2 = 8.1000e-5
  Epoch 15-19 (step 750-999): lr = 1e-4 × 0.9^3 = 7.2900e-5
  Epoch 20-24:                lr = 1e-4 × 0.9^4 = 6.5610e-5
  Epoch 25-29:                lr = 1e-4 × 0.9^5 = 5.9049e-5
```

---

## <a id="stage-9"></a>STAGE 9 — TRAINING LOOP

### 9.1 Multi-Run Strategy

```
N_RUNS       = 3
RANDOM_SEEDS = [42, 123, 456]

Mỗi run:
  1. Set seeds: random.seed(s), np.random.seed(s), tf.random.set_seed(s)
  2. Tạo mới datasets (shuffle order khác nhau)
  3. Khởi tạo model MỚI (random weight init khác nhau)
  4. Build loss function MỚI (adaptive factor reset to 1.0)
  5. Train hoàn toàn độc lập
  6. Evaluate trên CÙNG test set
  7. tf.keras.backend.clear_session() → giải phóng GPU memory
```

### 9.2 Callbacks

```
Thứ tự callback (quan trọng — thứ tự trong list):

1. AdaptiveWeightCallback        ← CHẠY ĐẦU TIÊN mỗi epoch end
   → Predict trên val_ds
   → Cập nhật adaptive_factor
   → Ảnh hưởng loss epoch tiếp theo

2. EarlyStopping
   monitor              = 'val_loss'
   patience             = 12
   restore_best_weights = True
   → Nếu val_loss không cải thiện sau 12 epoch liên tiếp → DỪNG
   → Tự động load lại weights từ epoch tốt nhất

3. CSVLogger
   filename = '{RESULT_DIR}/training_log.csv'
   append   = False
   → Ghi: epoch, loss, accuracy, val_loss, val_accuracy

4. ModelCheckpoint
   filepath       = '{RESULT_DIR}/best_model.keras'
   save_best_only = True
   monitor        = 'val_loss'
   → Lưu model weights khi val_loss đạt giá trị thấp nhất
```

### 9.3 Training Call

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[adaptive_cb, early_stop, csv_logger, checkpoint],
)
```

```
Mỗi epoch:
  1. Forward pass: images → backbone → FSDA → head → predictions
  2. Loss computation: AdaptiveClassBalancedFocalLoss
  3. Backward pass: gradients via backpropagation
  4. Optimizer update: Adam
  5. Callbacks: adaptive weight update → early stop check → save log → checkpoint

Tổng max: 30 epochs × 3 runs = 90 epochs
Thực tế: early stopping thường dừng sớm hơn (~15-20 epochs)
```

---

## <a id="stage-10"></a>STAGE 10 — EVALUATION & METRICS

### 10.1 Test Set Evaluation

```
best_model = load_model('best_model.keras', custom_objects=CUSTOM_OBJECTS)
pred_probs = best_model.predict(test_ds)     # (N_test, K) float32
y_pred     = argmax(pred_probs, axis=1)       # (N_test,) int
y_true     = meta.test_classes                # (N_test,) int
```

### 10.2 Metrics Computed

```
Primary (weighted average across classes):
  Accuracy  = correct / total
  Precision = Σ_c (w_c × TP_c / (TP_c + FP_c))     weighted
  Recall    = Σ_c (w_c × TP_c / (TP_c + FN_c))     weighted
  F1-Score  = Σ_c (w_c × 2·P_c·R_c / (P_c + R_c))  weighted

Additional:
  Balanced Accuracy = (1/K) × Σ_c recall_c
  Cohen's Kappa     = (p_o - p_e) / (1 - p_e)
  Matthews Corr Coef = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

Per-class: precision, recall, f1-score (via classification_report)
Confusion Matrix: (K × K) integer matrix
```

### 10.3 Aggregation Across Runs

```
For each metric M in {Accuracy, Precision, Recall, F1}:
  mean_M = (1/3) × (M_run1 + M_run2 + M_run3)
  std_M  = sqrt((1/3) × Σ (M_run_i - mean_M)²)

Report format: "Metric: mean ± std"
```

---

## <a id="param-count"></a>TỔNG HỢP PARAMETER COUNT

```
┌──────────────────────────────────────────────┬──────────────┬────────────┐
│ Component                                     │ Params       │ Trainable? │
├──────────────────────────────────────────────┼──────────────┼────────────┤
│ EfficientNetB4 Stem + Block 1-2              │ ~6.5M        │ FROZEN     │
│ EfficientNetB4 Block 3-7 (non-BN layers)     │ ~12M         │ YES        │
│ EfficientNetB4 ALL BN layers                 │ ~0.5M        │ FROZEN     │
├──────────────────────────────────────────────┼──────────────┼────────────┤
│ FSDA — FreqChannelAttn FC1 (1792×112)        │ 200,704      │ YES        │
│ FSDA — FreqChannelAttn FC2 (112×1792)        │ 200,704      │ YES        │
│ FSDA — SpatialAttn Conv2D (7×7×2×1)         │ 98           │ YES        │
│ FSDA — BatchNorm (γ,β,μ,σ² × 1792)          │ 7,168        │ YES*       │
├──────────────────────────────────────────────┼──────────────┼────────────┤
│ Head — GAP                                    │ 0            │ —          │
│ Head — BN (γ,β,μ,σ² × 1792)                 │ 7,168        │ YES*       │
│ Head — Dense(1792→256) + bias                │ 459,008      │ YES        │
│ Head — Dropout(0.5)                          │ 0            │ —          │
│ Head — Dense(256→N) + bias                   │ 256N + N     │ YES        │
├──────────────────────────────────────────────┼──────────────┼────────────┤
│ Loss — adaptive_factor                       │ N            │ NO (var)   │
│ Loss — static_weights                        │ N            │ NO (const) │
├──────────────────────────────────────────────┼──────────────┼────────────┤
│ TOTAL                                         │ ~19.4M       │            │
│ Trainable                                     │ ~12.9M       │            │
│ Non-trainable (frozen)                        │ ~6.5M        │            │
└──────────────────────────────────────────────┴──────────────┴────────────┘

* BN layers có gamma, beta (trainable) và moving_mean, moving_var (non-trainable)
  Nhưng backbone BN layers đều frozen → gamma, beta cũng frozen
```

---

## <a id="data-flow"></a>DATA FLOW SUMMARY (end-to-end)

```
Raw Image File (.jpg/.png)
    │
    ▼ tf.io.read_file
Raw Bytes (string)
    │
    ▼ tf.image.decode_jpeg(channels=3)
(H_orig, W_orig, 3) uint8
    │
    ▼ tf.image.resize([380, 380])
(380, 380, 3) float32, range [0, 255]
    │
    ▼ preprocess_input (÷127.5 - 1)
(380, 380, 3) float32, range [-1, +1]
    │
    ▼ [Training only] Augmentation pipeline
(380, 380, 3) float32
    │
    ▼ Batch
(32, 380, 380, 3) float32 → auto-cast float16 (mixed precision)
    │
    ▼ ════════════════════════════════════════════
    ▼ EfficientNetB4 Backbone
    ▼ ════════════════════════════════════════════
    │
    ▼ Stem Conv + BN + Swish
(32, 190, 190, 48) float16
    │
    ▼ Block 1 (MBConv1, k3, FROZEN)
(32, 190, 190, 24) float16
    │
    ▼ Block 2 (MBConv6, k3, FROZEN)
(32, 95, 95, 32) float16
    │
    ▼ Block 3 (MBConv6, k5, FINE-TUNED)
(32, 48, 48, 56) float16
    │
    ▼ Block 4 (MBConv6, k3, FINE-TUNED)
(32, 24, 24, 112) float16
    │
    ▼ Block 5 (MBConv6, k5, FINE-TUNED)
(32, 24, 24, 160) float16
    │
    ▼ Block 6 (MBConv6, k5, FINE-TUNED)
(32, 12, 12, 272) float16
    │
    ▼ Block 7 (MBConv6, k3, FINE-TUNED)
(32, 12, 12, 448) float16
    │
    ▼ Top Conv1x1 + BN + Swish (top_activation)
(32, 12, 12, 1792) float16
    │
    ▼ ════════════════════════════════════════════
    ▼ FSDA Block
    ▼ ════════════════════════════════════════════
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼ FREQUENCY CHANNEL ATTENTION                  ▼ SPATIAL ATTENTION
    │                                              │
    ▼ cast float32                                 ▼ cast float32
   (32, 12, 12, 1792)                             (32, 12, 12, 1792)
    │                                              │
    ▼ transpose [0,3,1,2]                          ├── AvgPool(axis=-1)
   (32, 1792, 12, 12)                             │   (32, 12, 12, 1)
    │                                              │
    ▼ complex(x, 0)                                ├── MaxPool(axis=-1)
   (32, 1792, 12, 12) complex64                   │   (32, 12, 12, 1)
    │                                              │
    ▼ fft2d                                        ▼ Concat(axis=-1)
   (32, 1792, 12, 12) complex64                  (32, 12, 12, 2)
    │                                              │
    ▼ abs → log1p                                  ▼ Conv2D(7×7, 1 filter)
   (32, 1792, 12, 12) float32                    (32, 12, 12, 1)
    │                                              │
    ▼ GAP(axis=[2,3])                              ▼ Sigmoid
   (32, 1792) float32                             (32, 12, 12, 1) ← sp_attn_map
    │                                              │
    ▼ FC1(1792→112) + ReLU                         ▼ x_f32 × sp_attn (broadcast)
   (32, 112) float32                              (32, 12, 12, 1792) float32
    │                                              │
    ▼ FC2(112→1792) + Sigmoid                      │ = spatial_out
   (32, 1792) float32                              │
    │                                              │
    ▼ reshape [B,1,1,1792]                         │
   (32, 1, 1, 1792)                                │
    │                                              │
    ▼ x_f32 × attn (broadcast)                    │
   (32, 12, 12, 1792) float32                      │
    │ = freq_out                                   │
    │                                              │
    └────────────── ⊕ (ADD) ───────────────────────┘
                    │
                   (32, 12, 12, 1792) float32
                    │
                    ▼ BatchNorm (float32)
                   (32, 12, 12, 1792) float32
                    │
                    ▼ cast float16
                   (32, 12, 12, 1792) float16
                    │
    ▼ ════════════════════════════════════════════
    ▼ Classification Head
    ▼ ════════════════════════════════════════════
                    │
                    ▼ GlobalAveragePooling2D
                   (32, 1792) float16
                    │
                    ▼ BatchNormalization
                   (32, 1792)
                    │
                    ▼ Dense(256, ReLU, L2=1e-5)
                   (32, 256)
                    │
                    ▼ Dropout(0.5)
                   (32, 256)
                    │
                    ▼ Dense(N, softmax, float32)
                   (32, N) float32
                    │
    ▼ ════════════════════════════════════════════
    ▼ Loss Computation
    ▼ ════════════════════════════════════════════
                    │
                    ▼ clip predictions [1e-7, 1-1e-7]
                    ▼ combined_w = static_w × adaptive_factor / mean
                    ▼ sample_w = sum(y_true × combined_w)        per sample
                    ▼ pt = sum(y_true × y_pred)                   per sample
                    ▼ focal = (1 - pt)^2.0                       per sample
                    ▼ ce = -sum(y_true × log(y_pred))             per sample
                    ▼ loss = mean(sample_w × focal × ce)          scalar
                    │
                    ▼ Backpropagation → Adam update
                    │
              [End of epoch]
                    │
                    ▼ AdaptiveWeightCallback
                    ▼ predict(val_ds) → per-class recall
                    ▼ target_c = (1 - recall_c) + 0.1
                    ▼ factor_c = 0.7 × factor_c + 0.3 × target_c
                    ▼ factor = factor / mean(factor)
                    ▼ assign → loss_fn.adaptive_factor
                    │
              [Next epoch → updated weights]
```

---

## OUTPUT FILES (per run)

```
report_EfficientNetB4/finetune_top5_fsda_adaptive_cb/
├── strategy_summary.csv                ← metrics tất cả runs
├── adaptive_weight_evolution.png       ← biểu đồ factor theo epoch
├── EXPERIMENT_REPORT.txt               ← text report tổng hợp
├── run_1_seed_42/
│   ├── best_model.keras                ← saved model (best val_loss)
│   ├── training_log.csv                ← epoch-by-epoch loss/acc
│   ├── learning_curve.png              ← accuracy + loss + factors
│   ├── confusion_matrix.png            ← K×K matrix
│   ├── classification_report.txt       ← per-class P/R/F1
│   └── adaptive_weight_history.csv     ← recall + factor per epoch
├── run_2_seed_123/
│   └── (same structure)
└── run_3_seed_456/
    └── (same structure)
```
