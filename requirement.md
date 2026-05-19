# REQUIREMENT — E-FSDA: Enhanced Frequency-Spatial Dual Attention

## Đề tài: Phân loại tỏi (Garlic Classification) sử dụng EfficientNetB4 + E-FSDA

> **Mục tiêu chính:** Customize và cải tiến module FSDA gốc thành **E-FSDA (Enhanced FSDA)** — chứng minh trước hội đồng rằng có **đóng góp riêng, sáng tạo** so với model baseline.

---

## 1. MÔ TẢ DATASET

### 1.1 Dataset-1 (Balanced — 2,134 ảnh)

| Split     | Fully_Peeled_Garlic | Partially_Peeled_Garlic | Spoiled_Garlic |     Total |
| --------- | ------------------: | ----------------------: | -------------: | --------: |
| **Train** |        482 (32.31%) |            306 (20.51%) |   704 (47.18%) | **1,492** |
| **Val**   |        103 (32.39%) |             65 (20.44%) |   150 (47.17%) |   **318** |
| **Test**  |        105 (32.41%) |             67 (20.68%) |   152 (46.91%) |   **324** |

> **Đặc điểm:** Tỷ lệ phân bố ổn định qua 3 split (~32 / 20 / 47%). Dataset nhỏ hơn, phù hợp cho ablation study và fast iteration.

### 1.2 Dataset-2 (Augmented/Extended — 2,944 ảnh)

| Split     | Fully_Peeled_Garlic | Partially_Peeled_Garlic | Spoiled_Garlic |     Total |
| --------- | ------------------: | ----------------------: | -------------: | --------: |
| **Train** |               1,050 |                     306 |            704 | **2,060** |
| **Val**   |                 225 |                      65 |            150 |   **440** |
| **Test**  |                 225 |                      67 |            152 |   **444** |

> **Đặc điểm:** Lớp `Fully_Peeled_Garlic` được mở rộng đáng kể (482 → 1,050 train, 103 → 225 val/test). Tổng dataset tăng ~38%. Mất cân bằng giảm so với dataset-1.

### 1.3 Phân tích thách thức dataset

| Thách thức                 | Mô tả                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| **Class Imbalance**        | `Partially_Peeled_Garlic` chỉ ~306 train samples ở cả 2 dataset — lớp thiểu số nghiêm trọng |
| **Inter-class Similarity** | `Fully_Peeled` vs `Partially_Peeled` — khác biệt tinh vi, chỉ ở vùng vỏ còn sót             |
| **Texture-dominant**       | Bệnh/hư hỏng thể hiện qua texture (vết thâm, đốm, mốc) → cần phân tích tần số               |
| **Small Dataset**          | < 3,000 ảnh → risk of overfitting, cần regularization mạnh                                  |

---

## 2. BASELINE HIỆN TẠI

### 2.1 Model: EfficientNetB4 + FSDA + AdaptiveCBLoss

| Component     | Chi tiết                                                          |
| ------------- | ----------------------------------------------------------------- |
| **Backbone**  | EfficientNetB4 (ImageNet pretrained), unfreeze blocks [3,4,5,6,7] |
| **Attention** | FSDA Block (Frequency-Spatial Dual Attention) — gốc               |
| **Loss**      | Adaptive Class-Balanced Focal Loss (novel)                        |
| **Input**     | 380 × 380 × 3                                                     |
| **Head**      | GAP → BN → Dense(256) → Dropout(0.5) → Softmax                    |
| **Optimizer** | Adam, lr=1e-4 + ExponentialDecay                                  |
| **Training**  | 3 runs × 3 seeds (42, 123, 456), 30 epochs, patience=12           |

### 2.2 Kết quả Baseline: ~92% Accuracy

### 2.3 FSDA Gốc — Kiến trúc

```
feat_map (B, 12, 12, 1792)
    ├── FrequencyChannelAttention (FFT → log-mag → MLP → sigmoid gate)
    │       → freq_out = x ⊗ σ(W₂·ReLU(W₁·GAP(log(1+|FFT₂D(x)|))))
    │
    └── SpatialAttention (Avg+Max pool → Conv 7×7 → sigmoid)
            → spatial_out = x ⊗ σ(Conv₇ₓ₇([AvgPool(x) ⊕ MaxPool(x)]))

fused = BN(freq_out + spatial_out)    ← Element-wise Addition
```

**Hạn chế của FSDA gốc:**

1. **Frequency branch đối xử tất cả tần số như nhau** — không phân biệt low/mid/high frequency bands
2. **Hai branch hoạt động độc lập** — không có cross-branch information exchange
3. **Fusion quá đơn giản** — chỉ element-wise addition, không có learnable fusion weight
4. **Attention sharpness cố định** — sigmoid luôn có cùng temperature, không adaptive

---

## 3. ĐỀ XUẤT: E-FSDA (Enhanced FSDA)

### 3.1 Các cải tiến đề xuất (Proposed Novel Contributions)

#### Novel 1: Frequency Band Decomposition (Phân giải băng tần)

- **Vấn đề:** FSDA gốc tính FFT rồi average toàn bộ spectrum → mất thông tin cấu trúc tần số
- **Giải pháp:** Chia frequency spectrum thành N bands (Low/High hoặc Low/Mid/High) bằng radial masks
- **Mỗi band có MLP riêng** (fc1, fc2) để học tầm quan trọng riêng
- **Band importance weights** (learnable, softmax-normalized) — model tự học band nào quan trọng nhất cho garlic classification
- **Lý do cho garlic:**
  - Low-freq: overall color/brightness → phân biệt healthy vs spoiled
  - Mid-freq: texture patterns → vết thâm, surface roughness
  - High-freq: sharp edges → ranh giới vùng hư

#### Novel 2: Learnable Temperature Scaling

- **Vấn đề:** Sigmoid trong FSDA gốc luôn có cùng slope → attention sharpness cố định
- **Giải pháp:** Thêm learnable temperature parameter τ: `σ(x/τ)` thay vì `σ(x)`
- **Channel temperature:** τ_channel ∈ ℝ^C — mỗi channel có temperature riêng
- **Spatial temperature:** τ_spatial ∈ ℝ^1 — điều khiển sharpness spatial attention
- **Constraint:** NonNeg() + softplus → đảm bảo τ > 0
- **Lý do:** Cho phép model tự quyết attention nên sharp (focus) hay diffuse (distributed)

#### Novel 3: Cross-Branch Gating (Optional — advanced variant)

- **Vấn đề:** Freq branch và spatial branch trong FSDA gốc hoạt động hoàn toàn độc lập
- **Giải pháp:** Spatial output gate frequency branch và ngược lại
  - `sp_gate = σ(FC(GAP(sp_attn ⊗ x)))` → gate cho freq_out
  - `freq_hint = σ(Conv1x1(freq_out))` → refine spatial_out
- **Lý do:** "Where to look" (spatial) nên ảnh hưởng "what frequencies matter" (channel), và ngược lại

#### Novel 4: Gated Fusion + Residual Connection (Optional)

- **Vấn đề:** Simple addition fusion không cho model control tỷ lệ freq vs spatial
- **Giải pháp:** Learnable fusion gate `g` + residual skip connection
  - `fused = g × freq_out + (1-g) × spatial_out + x`
- **Gate g** learned from global features via Dense → sigmoid

### 3.2 Chiến lược thí nghiệm

```
Experiment Plan:
┌─────────────────────────────────────────────────────────────┐
│  E-FSDA v1: FSDA + Temperature Scaling                     │
│  (Minimum viable enhancement — đã verify chạy được)        │
├─────────────────────────────────────────────────────────────┤
│  E-FSDA v2: FSDA + Frequency Band Decomposition            │
│           + Temperature Scaling                             │
│  (Core novelty — frequency band + temperature)              │
├─────────────────────────────────────────────────────────────┤
│  E-FSDA v3: E-FSDA v2 + Cross-Branch Gating                │
│           + Gated Fusion + Residual                         │
│  (Full model — most parameters, highest complexity)         │
└─────────────────────────────────────────────────────────────┘

So sánh đầy đủ:
  Baseline (A): EfficientNetB4 + FSDA (gốc) + CE Loss
  Baseline (B): EfficientNetB4 + FSDA (gốc) + AdaptiveCBLoss
  Proposed (C): EfficientNetB4 + E-FSDA (best variant) + AdaptiveCBLoss

Mỗi experiment: 3 runs × 3 seeds (42, 123, 456) → mean ± std
Chạy trên CẢ 2 DATASET (dataset-1 và dataset-2)
```

---

## 4. METRICS CẦN THU THẬP

### 4.1 Classification Metrics (per experiment, per dataset)

| Metric                | Mô tả                                     |
| --------------------- | ----------------------------------------- |
| **Accuracy**          | Overall accuracy (mean ± std over 3 runs) |
| **Precision**         | Weighted average precision                |
| **Recall**            | Weighted average recall                   |
| **F1-Score**          | Weighted average F1                       |
| **Balanced Accuracy** | Average per-class recall                  |
| **Cohen's Kappa**     | Agreement beyond chance                   |
| **MCC**               | Matthews Correlation Coefficient          |
| **Per-class P/R/F1**  | Breakdown per garlic class                |
| **AUC-ROC**           | Per-class + macro average                 |

### 4.2 Outputs cần export cho luận văn & bài báo

#### Hình ảnh / Biểu đồ:

- [ ] **Learning curves** (accuracy + loss, train vs val) — per experiment
- [ ] **Confusion matrix** (raw counts + normalized) — aggregate over 3 runs
- [ ] **ROC curves** (per-class + macro average) — best run
- [ ] **Per-class metrics bar chart** (precision, recall, F1 with error bars)
- [ ] **t-SNE feature visualization** (GAP features, colored by class)
- [ ] **FSDA/E-FSDA spatial attention maps** (overlay on original images)
- [ ] **Grad-CAM++ heatmaps** (on EfficientNetB4 top conv features)
- [ ] **Frequency spectrum visualization** (FFT per class, radial profiles)
- [ ] **Comparison bar chart** — Baseline FSDA vs E-FSDA (all metrics side-by-side)
- [ ] **Adaptive weight evolution** (CB loss factors over epochs)
- [ ] **Dataset distribution chart** (class distribution per split)

#### Bảng biểu:

- [ ] **Comparison table** — FSDA vs E-FSDA vs variants (all metrics, both datasets)
- [ ] **Ablation study table** — contribution of each E-FSDA component
- [ ] **Per-class comparison** — which class benefits most from E-FSDA
- [ ] **Hyperparameter table** — all config values
- [ ] **Model complexity comparison** — params, FLOPs, inference speed

#### Text / CSV:

- [ ] `classification_report.txt` — per run
- [ ] `strategy_summary.csv` — all runs aggregated
- [ ] `adaptive_weight_history.csv` — weight evolution per epoch
- [ ] `training_log.csv` — epoch-by-epoch metrics
- [ ] `final_comparison_report.txt` — FSDA vs E-FSDA summary

---

## 5. YÊU CẦU KỸ THUẬT

### 5.1 Môi trường chạy

- **Platform:** Kaggle Notebook (GPU P100 hoặc T4)
- **Framework:** TensorFlow 2.x + Keras 3
- **Mixed Precision:** mixed_float16
- **XLA JIT:** ON

### 5.2 Reproducibility

- 3 independent runs per experiment
- Fixed seeds: [42, 123, 456]
- Deterministic data loading (sorted filenames)

### 5.3 Output Format

- Tất cả plots saved dpi=300 (publication quality)
- Export PNG + PDF nếu cần
- Zip toàn bộ output folder để tải về

---

## 6. KẾ HOẠCH THỰC HIỆN

| Phase | Nội dung                                              | Trạng thái       |
| ----- | ----------------------------------------------------- | ---------------- |
| 1     | Chạy baseline FSDA + AdaptiveCBLoss trên cả 2 dataset | ✅ Đã có ~92%    |
| 2     | Implement E-FSDA (best variant)                       | 🔄 Đang thiết kế |
| 3     | Chạy E-FSDA trên cả 2 dataset                         | ⬜ Chưa bắt đầu  |
| 4     | So sánh & tạo comparison charts                       | ⬜ Chưa bắt đầu  |
| 5     | Export tất cả artifacts cho luận văn                  | ⬜ Chưa bắt đầu  |
| 6     | Viết analysis, kết luận                               | ⬜ Chưa bắt đầu  |

---

## 7. TIÊU CHÍ THÀNH CÔNG

1. **E-FSDA đạt accuracy ≥ baseline FSDA** (hoặc comparable) trên cả 2 dataset
2. **Có evidence rõ ràng về đóng góp riêng:**
   - Ablation study: bỏ từng component → metrics giảm
   - Learnable parameters (temperature, band weights) converge to meaningful values
   - Attention maps cho thấy E-FSDA focus chính xác hơn vào vùng bệnh
3. **Đủ materials cho bài báo khoa học** (tất cả figures, tables, text)
4. **Trình bày mạch lạc trước hội đồng** — giải thích được tại sao mỗi cải tiến có ý nghĩa
