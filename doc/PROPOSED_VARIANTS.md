# ĐỀ XUẤT CÁC BIẾN THỂ E-FSDA MỚI

> Mục tiêu: Tạo ra nhiều variants để thực nghiệm, tìm đóng góp vượt trội nhất so với FSDA gốc.

---

## TỔNG QUAN CÁC VARIANTS

| Variant | Tên gọi               | Ý tưởng chính                               | Độ phức tạp |
| ------- | --------------------- | ------------------------------------------- | ----------- |
| v2      | FreqBand              | Frequency Band Decomposition + Temperature  | Trung bình  |
| v3      | CrossBranch           | Cross-Branch Gating + Gated Fusion          | Cao         |
| **v4**  | **AdaptivePool**      | Multi-Scale Adaptive Pooling Attention      | Trung bình  |
| **v5**  | **ChannelShuffle**    | Channel Shuffle + Group Attention           | Thấp        |
| **v6**  | **DynamicConv**       | Dynamic Convolution Spatial Attention       | Trung bình  |
| **v7**  | **WaveletAttention**  | Wavelet Transform thay FFT                  | Cao         |
| **v8**  | **CoordAttention**    | Coordinate Attention Integration            | Thấp        |
| **v9**  | **FreqSpatialFusion** | Attention Fusion via Squeeze-Excitation     | Trung bình  |
| **v10** | **SpectrumWeighting** | Learnable Spectral Filter + Phase Attention | Cao         |

---

## CHI TIẾT TỪNG VARIANT MỚI

---

### V4: Multi-Scale Adaptive Pooling Attention (AdaptivePool)

**Motivation:**

- FSDA gốc chỉ dùng GAP (Global Average Pooling) → mất thông tin spatial scale
- Garlic images có features ở nhiều scales: vết thâm nhỏ (local) vs màu tổng thể (global)

**Cải tiến:**

```
Spatial Branch mới:
  x → [AdaptiveAvgPool(1×1), AdaptiveAvgPool(3×3), AdaptiveAvgPool(6×6)]
     → Concat → Conv1×1 → Sigmoid → Attention map (multi-scale)

Frequency Branch:
  Giữ nguyên FSDA gốc + Temperature Scaling
```

**Lý do cho garlic:**

- Scale 1×1: global color/brightness (healthy vs spoiled)
- Scale 3×3: medium texture regions (peeling patterns)
- Scale 6×6: local defects (small spots, mold patches)

**Novel point:** Multi-scale spatial pooling pyramid cho attention, thay vì single-scale.

---

### V5: Channel Shuffle Group Attention (ChannelShuffle)

**Motivation:**

- FSDA xử lý tất cả channels cùng nhau → expensive + redundant
- Nhiều channels có patterns tương tự → group processing hiệu quả hơn

**Cải tiến:**

```
Input: x (B, H, W, C)
  → Split into G groups: [x₁, x₂, ..., xG] each (B, H, W, C/G)
  → Per-group frequency attention (smaller MLP per group)
  → Channel Shuffle (cross-group information exchange)
  → Spatial attention (shared across groups)
  → Concat + Conv1×1 projection
```

**Novel point:** Group-wise frequency attention + channel shuffle = lightweight + effective.

- Giảm params đáng kể (MLP nhỏ hơn per group)
- Channel shuffle cho cross-group communication (lấy cảm hứng từ ShuffleNet)

---

### V6: Dynamic Convolution Spatial Attention (DynamicConv)

**Motivation:**

- FSDA spatial branch dùng Conv 7×7 cố định → kernel weights không adaptive theo input
- Mỗi ảnh garlic cần attention pattern khác nhau (vị trí hư hại khác nhau)

**Cải tiến:**

```
Spatial Branch:
  x → GAP → FC → K kernel weights (softmax)  [Input-dependent]
  x → [AvgPool, MaxPool] → concat
     → Σᵢ wᵢ × Convᵢ(7×7)  [Weighted sum of K static kernels]
     → σ(result / τ_spatial)

Frequency Branch:
  Giữ FSDA gốc + Temperature Scaling
```

**Novel point:** Dynamic convolution — kernel weights được generate từ input, mỗi image có attention kernel riêng.

**Lý do cho garlic:**

- Vết hư ở góc trái → cần kernel focus trái
- Vết hư ở center → cần kernel focus center
- Dynamic kernel tự adapt theo từng image

---

### V7: Wavelet Attention (WaveletAttention)

**Motivation:**

- FFT cho global frequency info nhưng mất spatial locality
- Wavelet Transform (DWT) giữ cả frequency VÀ spatial information
- Garlic defects cần biết TẦN SỐ GÌ ở ĐÂU

**Cải tiến:**

```
Frequency Branch thay bằng Wavelet Branch:
  x → Haar DWT → [LL, LH, HL, HH] subbands
     → Per-subband attention:
        LL: low-freq (smooth regions) → MLP₁ → weight₁
        LH: horizontal edges → MLP₂ → weight₂
        HL: vertical edges → MLP₃ → weight₃
        HH: diagonal/texture → MLP₄ → weight₄
     → Weighted combine: Σ wᵢ × subbandᵢ
     → IDWT (reconstruct) → Channel attention gate

Spatial Branch: Giữ nguyên FSDA + Temperature
```

**Novel point:** Wavelet thay FFT → spatial-frequency locality, subband-specific attention.

**Lý do cho garlic:**

- LH subband: horizontal peeling patterns
- HL subband: vertical crack lines
- HH subband: texture roughness (mold, spots)
- LL subband: overall color health indicator

**Lưu ý implementation:** Haar wavelet đơn giản, chỉ cần average/difference operations, không cần thư viện ngoài.

---

### V8: Coordinate Attention Integration (CoordAttention)

**Motivation:**

- FSDA spatial attention tạo 2D map nhưng không encode positional information explicitly
- Coordinate Attention (CVPR 2021) encode cả channel + spatial position efficiently

**Cải tiến:**

```
Thay Spatial Branch bằng Coordinate Attention:
  x → AvgPool along H → (B, 1, W, C) = x_h encoding
  x → AvgPool along W → (B, H, 1, C) = x_w encoding
  → Concat along spatial dim → Conv1×1 + BN + ReLU (shared)
  → Split back → separate Conv1×1 each
  → σ(x_h) ⊗ σ(x_w) → 2D coordinate attention map
  → x_out = x ⊗ coord_attention

Frequency Branch: FSDA gốc + Temperature Scaling
Fusion: Gated fusion (learnable α)
```

**Novel point:** Coordinate attention captures long-range spatial dependencies with positional encoding, kết hợp với frequency attention.

**Ưu điểm:**

- Lightweight (ít params hơn Conv 7×7)
- Encode vị trí chính xác (x,y) → biết defect ở đâu
- Proven effective (CVPR 2021 paper)

---

### V9: Squeeze-Excitation Fusion Attention (FreqSpatialFusion)

**Motivation:**

- FSDA gốc fusion bằng addition → weight cố định 50/50
- Cần adaptive fusion dựa trên content của input

**Cải tiến:**

```
Freq Branch: FSDA frequency (giữ nguyên)
Spatial Branch: FSDA spatial (giữ nguyên)

Novel Fusion Module:
  freq_out, spatial_out → Stack → (B, H, W, 2C)
  → SE Block:
     GAP → FC(2C, 2C/r) → ReLU → FC(2C/r, 2C) → Sigmoid
     → Split sigmoid into [gate_f, gate_s] each (B,1,1,C)
  → fused = gate_f ⊗ freq_out + gate_s ⊗ spatial_out + x (residual)
```

**Novel point:** SE-based adaptive fusion learns per-channel importance of frequency vs spatial attention.

**Ưu điểm:**

- Không thay đổi 2 branches → dễ so sánh ablation
- Chỉ thêm fusion module nhỏ
- Residual connection ổn định training

---

### V10: Learnable Spectral Filter + Phase Attention (SpectrumWeighting)

**Motivation:**

- FSDA gốc chỉ dùng magnitude của FFT, BỎ QUA phase information
- Phase chứa structural/edge information quan trọng
- Magnitude chứa texture/energy information

**Cải tiến:**

```
Frequency Branch (Enhanced):
  x → FFT2D → separate: Magnitude + Phase

  Magnitude Path:
    |FFT| → Learnable Spectral Filter (element-wise weights, freq-domain)
           → Log → GAP → MLP → Channel attention (magnitude-based)

  Phase Path:
    ∠FFT → Conv1×1 (in freq domain) → IFFT → real part
          → Spatial attention map (phase-based structure)

  Combined: mag_attention ⊗ phase_attention → final gate

Spatial Branch: Giữ nguyên FSDA
```

**Novel point:** Dual magnitude-phase frequency attention — khai thác TOÀN BỘ thông tin FFT thay vì chỉ magnitude.

**Lý do cho garlic:**

- Magnitude: texture energy (rough vs smooth surface)
- Phase: edge structure (boundary of peeled regions, shape of spots)
- Combined: both "what texture" + "where structure"

---

## BẢNG SO SÁNH ĐỘ KHẢ THI

| Variant           | Params thêm | Dễ implement | Risk OOM   | Novelty level | Recommend        |
| ----------------- | ----------- | ------------ | ---------- | ------------- | ---------------- |
| v4 AdaptivePool   | Thấp        | ⭐⭐⭐⭐⭐   | Thấp       | ⭐⭐⭐        | ✅ Nên thử       |
| v5 ChannelShuffle | Rất thấp    | ⭐⭐⭐⭐     | Rất thấp   | ⭐⭐⭐        | ✅ Nên thử       |
| v6 DynamicConv    | Trung bình  | ⭐⭐⭐       | Thấp       | ⭐⭐⭐⭐      | ✅ Nên thử       |
| v7 Wavelet        | Trung bình  | ⭐⭐         | Trung bình | ⭐⭐⭐⭐⭐    | ⚠️ Advanced      |
| v8 CoordAttention | Thấp        | ⭐⭐⭐⭐⭐   | Rất thấp   | ⭐⭐⭐⭐      | ✅ Recommend cao |
| v9 SE-Fusion      | Rất thấp    | ⭐⭐⭐⭐⭐   | Rất thấp   | ⭐⭐⭐        | ✅ Nên thử       |
| v10 Phase         | Cao         | ⭐⭐         | Trung bình | ⭐⭐⭐⭐⭐    | ⚠️ Advanced      |

---

## ĐỀ XUẤT THỨ TỰ ƯU TIÊN CHẠY

```
Priority 1 (Quick wins, dễ implement, risk thấp):
  ✅ V8: CoordAttention — proven, lightweight, high novelty
  ✅ V9: SE-Fusion — minimal change, focus on fusion improvement
  ✅ V5: ChannelShuffle — lightweight, novel grouping idea

Priority 2 (Medium effort, high potential):
  ✅ V4: AdaptivePool — multi-scale, intuitive for defect detection
  ✅ V6: DynamicConv — input-adaptive, strong novelty argument

Priority 3 (High effort, highest novelty for thesis defense):
  ⚠️ V7: Wavelet — strongest novelty but complex implementation
  ⚠️ V10: Phase — very novel but may need careful tuning
```

---

## CHIẾN LƯỢC BẢO VỆ TRƯỚC HỘI ĐỒNG

### Nếu 1 variant thắng rõ ràng:

- Present variant đó là "proposed E-FSDA"
- Ablation study: bỏ từng component → chứng minh mỗi phần đều cần thiết
- So sánh với FSDA gốc trên cả 2 datasets

### Nếu nhiều variants comparable:

- Present như "family of E-FSDA enhancements"
- Analysis: variant nào tốt cho scenario nào
- Trade-off analysis: accuracy vs complexity vs inference speed

### Câu hỏi hội đồng có thể hỏi & cách trả lời:

1. **"Tại sao chọn approach này?"** → Motivated by dataset characteristics (texture, small defects, class imbalance)
2. **"So với attention mechanisms khác?"** → Ablation table + complexity comparison
3. **"Có generalize được không?"** → Test trên 2 datasets khác nhau (dataset-1 vs dataset-2)
4. **"Contribution gì so với FSDA gốc?"** → Specific improvement + learned parameters analysis

---

## GHI CHÚ IMPLEMENTATION

### Template chung cho mọi variant:

```python
class EFSDA_VX(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        # ... init params

    def build(self, input_shape):
        C = input_shape[-1]
        # ... build sub-layers
        # QUAN TRỌNG: explicit build tất cả sub-layers
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.cast(inputs, tf.float32)  # Mixed precision safety
        # ... forward pass
        return output

    def compute_output_spec(self, input_spec):
        return input_spec  # Bypass Keras 3 tracing issues

    def get_config(self):
        config = super().get_config()
        config.update({...})
        return config
```

### Checklist mỗi variant:

- [ ] Float32 cast cho FFT/sigmoid/softmax
- [ ] compute_output_spec()
- [ ] get_config() đầy đủ
- [ ] Explicit build() cho sub-layers
- [ ] Temperature parameter (nếu có) dùng softplus constraint
- [ ] Test forward pass với dummy input trước khi train
