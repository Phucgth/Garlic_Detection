# EfficientNetB4 + FSDA — Mermaid Diagrams

# Paste từng block bên dưới vào https://mermaid.live

# ════════════════════════════════════════════════════════════════

# DIAGRAM 1: TỔNG QUAN KIẾN TRÚC (Overall Architecture)

# ════════════════════════════════════════════════════════════════

```mermaid
flowchart TD
    A(["`**Input Image**
    380 × 380 × 3`"]) --> B

    subgraph BACKBONE["🔷 EfficientNetB4 Backbone (ImageNet Pretrained)"]
        direction TB
        B["Block 1 — Frozen ❄️"] --> C["Block 2 — Frozen ❄️"]
        C --> D["Block 3 — Fine-tuned 🔥"]
        D --> E["Block 4 — Fine-tuned 🔥"]
        E --> F["Block 5 — Fine-tuned 🔥"]
        F --> G["Block 6 — Fine-tuned 🔥"]
        G --> H["Block 7 — Fine-tuned 🔥\n(BN always frozen)"]
    end

    H --> FM(["`**Feature Map**
    12 × 12 × 1792`"])

    FM --> FSDA

    subgraph FSDA["🟠 FSDA Block — Frequency-Spatial Dual Attention"]
        direction TB
        SPLIT((" ")) --> FREQ & SPAT

        subgraph FREQ["① Frequency Channel Attention"]
            direction TB
            F1["FFT2D → complex spectrum"] --> F2["log1p(|FFT|)\nmagnitude map"]
            F2 --> F3["Global Mean (H,W) → (B, C)"]
            F3 --> F4["FC: C → C/16, ReLU"]
            F4 --> F5["FC: C/16 → C, Sigmoid"]
            F5 --> F6["Channel Reweighting\nx ← x × attn(1,1,C)"]
        end

        subgraph SPAT["② Spatial Attention (CBAM-style)"]
            direction TB
            S1["AvgPool(C→1) ‖ MaxPool(C→1)"] --> S2["Concat → (B, H, W, 2)"]
            S2 --> S3["Conv2D(1 filter, 7×7, same)"]
            S3 --> S4["Sigmoid"]
            S4 --> S5["Spatial Reweighting\nx ← x × attn(H,W,1)"]
        end

        F6 & S5 --> ADD["⊕  Element-wise Addition"]
        ADD --> BN["BatchNorm (float32)"]
    end

    BN --> HEAD

    subgraph HEAD["🟢 Classification Head"]
        direction TB
        H1["GlobalAveragePooling2D → (B, 1792)"] --> H2
        H2["BatchNormalization"] --> H3
        H3["Dense(256, ReLU) + L2(1e-5)"] --> H4
        H4["Dropout(0.5)"] --> H5
        H5["Dense(N_classes, Softmax)\n— float32 output —"]
    end

    H5 --> OUT(["`**Class Probabilities**
    Softmax Distribution`"])

    style BACKBONE fill:#dbeafe,stroke:#3b82f6,stroke-width:2px
    style FSDA fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    style FREQ fill:#ffedd5,stroke:#ea580c,stroke-width:1.5px
    style SPAT fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px
    style HEAD fill:#dcfce7,stroke:#16a34a,stroke-width:2px
    style A fill:#bbf7d0,stroke:#15803d,stroke-width:2px,color:#000
    style FM fill:#bfdbfe,stroke:#1d4ed8,stroke-width:2px,color:#000
    style OUT fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#000
    style ADD fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px
    style BN fill:#f3e8ff,stroke:#7c3aed,stroke-width:1.5px
    style SPLIT fill:#f59e0b,stroke:#b45309
```

---

# ════════════════════════════════════════════════════════════════

# DIAGRAM 2: FSDA BLOCK — CHI TIẾT (Zoom-in)

# ════════════════════════════════════════════════════════════════

```mermaid
flowchart LR
    IN(["`**x**
    B×12×12×1792
    float16`"])

    IN --> FPATH & SPATH

    subgraph FPATH["BRANCH 1 — Frequency Channel Attention"]
        direction TB
        FA1["`Transpose
        (B,H,W,C) → (B,C,H,W)`"]
        FA2["`FFT2D
        → complex spectrum`"]
        FA3["`log1p( |FFT| )
        log-magnitude`"]
        FA4["`reduce_mean(H,W)
        → descriptor (B, C)`"]
        FA5["`FC₁: C → C/16
        ReLU  (float32)`"]
        FA6["`FC₂: C/16 → C
        Sigmoid  (float32)`"]
        FA7["`reshape → (B,1,1,C)
        x_freq = x × attn`"]
        FA1-->FA2-->FA3-->FA4-->FA5-->FA6-->FA7
    end

    subgraph SPATH["BRANCH 2 — Spatial Attention"]
        direction TB
        SA1["`AvgPool: reduce_mean(C)
        → (B,H,W,1)`"]
        SA2["`MaxPool: reduce_max(C)
        → (B,H,W,1)`"]
        SA3["`Concat
        → (B,H,W,2)`"]
        SA4["`Conv2D
        1 filter, 7×7, same
        (float32)`"]
        SA5["`Sigmoid
        → attn_map (B,H,W,1)`"]
        SA6["`x_spat = x × attn_map`"]
        SA1 & SA2 --> SA3 --> SA4 --> SA5 --> SA6
    end

    FA7 & SA6 --> ADD["`⊕
    x_freq + x_spat`"]
    ADD --> BN["`BatchNorm
    float32`"]
    BN --> CAST["`cast → input dtype
    (float16)`"]

    CAST --> OUT1(["`**fused**
    B×12×12×1792`"])
    SA5 --> OUT2(["`**sp_attn_map**
    B×12×12×1
    (for visualization)`"])

    style FPATH fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style SPATH fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style ADD fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px
    style IN fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#000
    style OUT1 fill:#bbf7d0,stroke:#15803d,stroke-width:2px,color:#000
    style OUT2 fill:#fce7f3,stroke:#be185d,stroke-width:2px,color:#000
```

---

# ════════════════════════════════════════════════════════════════

# DIAGRAM 3: TRAINING STRATEGY

# ════════════════════════════════════════════════════════════════

```mermaid
flowchart TD
    DS[("Dataset\ndataset_final_2006")] --> SPLIT

    subgraph SPLIT["Data Split"]
        TR["Train set"] & VA["Val set"] & TE["Test set"]
    end

    TR --> AUG

    subgraph AUG["Data Augmentation (USE_AUG=True)"]
        direction LR
        A1["RandomFlip\nhorizontal+vertical"]
        A2["RandomRotation\n±30°"]
        A3["RandomZoom\n20%"]
        A4["RandomTranslation\n20%"]
        A5["RandomBrightness\n30%"]
    end

    AUG --> MODEL["EfficientNetB4 + FSDA"]
    VA  --> MODEL

    subgraph TRAIN["Training Loop (30 epochs max)"]
        direction TB
        LOSS["`Loss: CategoricalCrossentropy
        label_smoothing = 0.15`"]
        OPT["`Optimizer: Adam
        ExponentialDecay lr=1e-4
        decay every 5 epochs × 0.9`"]
        REG["`Regularization:
        • Dropout(0.5)
        • L2(1e-5) on Dense
        • Class-weight balancing`"]
        CB["`Callbacks:
        • EarlyStopping(patience=12)
        • ModelCheckpoint (best val_loss)
        • CSVLogger`"]
    end

    MODEL --> TRAIN
    TRAIN --> BEST["best_model.keras"]
    BEST --> EVAL["Evaluate on Test Set"]

    subgraph MULTIRUN["Multi-Run (3 seeds: 42, 123, 456)"]
        R1["Run 1\nseed=42"] & R2["Run 2\nseed=123"] & R3["Run 3\nseed=456"]
    end

    EVAL --> MULTIRUN
    MULTIRUN --> REPORT["`Final Report
    mean ± std
    Accuracy / F1 / Precision / Recall`"]

    style SPLIT fill:#e0f2fe,stroke:#0284c7
    style AUG fill:#fef9c3,stroke:#ca8a04
    style TRAIN fill:#f0fdf4,stroke:#16a34a
    style MULTIRUN fill:#fdf4ff,stroke:#a21caf
    style BEST fill:#dcfce7,stroke:#15803d
    style REPORT fill:#fce7f3,stroke:#be185d
```

---

# ════════════════════════════════════════════════════════════════

# DIAGRAM 4: MIXED PRECISION FLOW

# ════════════════════════════════════════════════════════════════

```mermaid
flowchart LR
    subgraph MP["Mixed Precision: mixed_float16"]
        direction TB
        I["Input\nfloat32"] --> B["EfficientNetB4\ncompute: float16\nvariables: float32"]
        B --> F["FSDABlock\nCompute: float32\n(cast inside)"]
        F --> G["GAP + Dense Head\ncompute: float16"]
        G --> O["Output Dense\nactivation=softmax\ndtype='float32'"]
    end

    style I fill:#e0f2fe,stroke:#0284c7
    style B fill:#dbeafe,stroke:#1d4ed8
    style F fill:#ffedd5,stroke:#ea580c
    style G fill:#dcfce7,stroke:#15803d
    style O fill:#fce7f3,stroke:#be185d
    style MP fill:#f8fafc,stroke:#64748b,stroke-width:2px
```
