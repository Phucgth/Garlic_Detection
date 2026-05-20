# SKILL — E-FSDA Kaggle Training & Export Guide

> Hướng dẫn chi tiết để chạy train, export artifacts, và đóng gói kết quả cho luận văn.

---

## 1. MÔI TRƯỜNG KAGGLE

### 1.1 Setup Notebook

```
Settings:
  - Accelerator: GPU P100 hoặc T4 x2
  - Internet: ON (để tải pretrained weights lần đầu)
  - Persistence: Files (để lưu output)
  - Language: Python
```

### 1.2 Dataset trên Kaggle

```
Kaggle Dataset paths:
  Dataset-1: /kaggle/input/datasets/giaphuc/dataset-garlic-2106/dataset_final_2006
  Dataset-2: /kaggle/input/datasets/giaphuc/dataset-garlic-2944/dataset_final_2944
  (Tùy chỉnh path theo tên dataset đã upload)
```

### 1.3 Lưu ý quan trọng

- Kaggle session timeout: 12h (GPU) → đủ cho 3 runs × 30 epochs
- Output limit: 20GB → zip kết quả trước khi session hết
- Nếu OOM: giảm BATCH_SIZE từ 32 → 16, hoặc dùng batch=4 cho t-SNE/Grad-CAM

---

## 2. WORKFLOW CHẠY THÍ NGHIỆM

### 2.1 Thứ tự chạy notebooks

```
PHASE 1 — Baseline (đã có kết quả ~92%)
  ✅ EfficientNetB4-FSDA-AdaptiveCBLoss.ipynb  → dataset-1
  ✅ EfficientNetB4-FSDA-AdaptiveCBLoss.ipynb  → dataset-2 (đổi DATA_DIR)

PHASE 2 — E-FSDA Experiments
  🔄 EfficientNetB4-EFSDA-v2-FreqBand.ipynb   → dataset-1
  🔄 EfficientNetB4-EFSDA-v2-FreqBand.ipynb   → dataset-2
  🔄 EfficientNetB4-EFSDA-v3-CrossBranch.ipynb → dataset-1
  🔄 EfficientNetB4-EFSDA-v3-CrossBranch.ipynb → dataset-2

PHASE 3 — Comparison & Export
  ⬜ Comparison notebook (tổng hợp kết quả, tạo charts so sánh)
```

### 2.2 Checklist trước khi chạy mỗi notebook

- [ ] Kiểm tra `DATA_DIR` đúng dataset path trên Kaggle
- [ ] Kiểm tra `BASE_RESULT_DIR` — unique cho mỗi experiment
- [ ] Kiểm tra `STRATEGY_KEY` và `STRATEGY_LABEL` — đúng tên experiment
- [ ] GPU enabled + mixed precision ON
- [ ] `N_RUNS = 3`, `RANDOM_SEEDS = [42, 123, 456]`
- [ ] `EPOCHS = 30`, `PATIENCE = 12`

### 2.3 Sau khi chạy xong — Review checklist

- [ ] Kiểm tra 3 runs đều hoàn thành (không crash giữa chừng)
- [ ] Kiểm tra `strategy_summary.csv` — 3 rows, metrics hợp lý
- [ ] Kiểm tra confusion matrix — không có class nào accuracy = 0
- [ ] Kiểm tra learning curves — không diverge, val_loss có giảm
- [ ] So sánh nhanh với baseline — E-FSDA có cải thiện không?

---

## 3. CẤU TRÚC OUTPUT CẦN EXPORT

### 3.1 Per-Experiment Output (tự động tạo bởi notebook)

```
/kaggle/working/report_EfficientNetB4/{STRATEGY_KEY}/
├── run_1_seed_42/
│   ├── best_model.keras              # Model weights
│   ├── training_log.csv              # Epoch-by-epoch metrics
│   ├── classification_report.txt     # sklearn report
│   ├── confusion_matrix.png          # Per-run CM
│   ├── learning_curve.png            # Acc + Loss curves
│   └── adaptive_weight_history.csv   # CB loss factor evolution
├── run_2_seed_123/
│   └── (same structure)
├── run_3_seed_456/
│   └── (same structure)
├── strategy_summary.csv              # All runs summary
├── overall_metrics_summary.csv       # Mean ± std
├── per_class_metrics.png             # Bar chart with error bars
├── agg_confusion_matrix.png          # Aggregate CM (raw + normalized)
├── roc_curves.png                    # Per-class ROC + macro AUC
├── tsne.png                          # t-SNE feature visualization
├── fsda_attention_maps.png           # Spatial attention overlays
├── gradcam_pp.png                    # Grad-CAM++ heatmaps
└── frequency_spectra.png             # FFT spectrum per class
```

### 3.2 Comparison Output (cần tạo thêm)

```
/kaggle/working/comparison/
├── comparison_table.csv              # All experiments side-by-side
├── comparison_bar_chart.png          # Metrics bar chart (FSDA vs E-FSDA)
├── ablation_study.csv                # Component contribution
├── dataset_comparison.csv            # Same model, dataset-1 vs dataset-2
├── model_complexity.csv              # Params, FLOPs, speed
└── final_summary_report.txt          # Text summary for thesis
```

---

## 4. EXPORT ARTIFACTS CHO LUẬN VĂN

### 4.1 Figures cần cho bài báo/luận văn

| Figure # | Nội dung                          | File                        | Dùng cho                    |
| -------- | --------------------------------- | --------------------------- | --------------------------- |
| Fig.1    | Overall architecture diagram      | `architecture_*.png`        | Chapter 3 - Methodology     |
| Fig.2    | E-FSDA block internal diagram     | Cần tạo mới                 | Chapter 3 - Proposed Method |
| Fig.3    | Dataset distribution              | Cần tạo                     | Chapter 4 - Dataset         |
| Fig.4    | Learning curves (best experiment) | `learning_curve.png`        | Chapter 5 - Results         |
| Fig.5    | Confusion matrix (normalized)     | `agg_confusion_matrix.png`  | Chapter 5                   |
| Fig.6    | ROC curves                        | `roc_curves.png`            | Chapter 5                   |
| Fig.7    | Per-class metrics bar chart       | `per_class_metrics.png`     | Chapter 5                   |
| Fig.8    | Comparison: FSDA vs E-FSDA        | `comparison_bar_chart.png`  | Chapter 5                   |
| Fig.9    | Attention map comparison          | Side-by-side FSDA vs E-FSDA | Chapter 5                   |
| Fig.10   | t-SNE visualization               | `tsne.png`                  | Chapter 5                   |
| Fig.11   | Grad-CAM++ heatmaps               | `gradcam_pp.png`            | Chapter 5                   |
| Fig.12   | Frequency spectrum per class      | `frequency_spectra.png`     | Chapter 3 (motivation)      |
| Fig.13   | Adaptive CB weight evolution      | `adaptive_factors.png`      | Chapter 5                   |

### 4.2 Tables cần cho bài báo/luận văn

| Table # | Nội dung                       | Nguồn                       |
| ------- | ------------------------------ | --------------------------- |
| Tab.1   | Dataset statistics             | `requirement.md` Section 1  |
| Tab.2   | Hyperparameter configuration   | Notebook config cell        |
| Tab.3   | Model complexity comparison    | Model summary + timing      |
| Tab.4   | Main results (all experiments) | `comparison_table.csv`      |
| Tab.5   | Ablation study                 | `ablation_study.csv`        |
| Tab.6   | Per-class performance          | `classification_report.txt` |
| Tab.7   | Comparison with related work   | Manual from literature      |

---

## 5. CODE SNIPPETS — COMPARISON & EXPORT

### 5.1 Tạo comparison chart (chạy sau khi có kết quả tất cả experiments)

```python
# ========== COMPARISON: FSDA vs E-FSDA ========== #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results from each experiment
experiments = {
    'FSDA (Baseline)': '/kaggle/working/report_EfficientNetB4/finetune_top5_fsda_adaptive_cb/strategy_summary.csv',
    'E-FSDA v2 (FreqBand)': '/kaggle/working/report_EfficientNetB4/efsda_v2_freqband_temperature/summary.csv',
    'E-FSDA v3 (CrossBranch)': '/kaggle/working/report_EfficientNetB4/efsda_v3_crossbranch_multiscale/summary.csv',
}

results = {}
for name, path in experiments.items():
    df = pd.read_csv(path)
    results[name] = {
        'Accuracy': f"{df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}",
        'Precision': f"{df['precision'].mean():.4f} ± {df['precision'].std():.4f}",
        'Recall': f"{df['recall'].mean():.4f} ± {df['recall'].std():.4f}",
        'F1-Score': f"{df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}",
        'acc_mean': df['accuracy'].mean(),
        'acc_std': df['accuracy'].std(),
    }

# Bar chart comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25
colors = ['#2196F3', '#4CAF50', '#FF9800']

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, data) in enumerate(results.items()):
    means = [float(data[m].split(' ± ')[0]) for m in metrics]
    stds = [float(data[m].split(' ± ')[1]) for m in metrics]
    ax.bar(x + i*width, means, width, yerr=stds, label=name,
           color=colors[i], alpha=0.85, capsize=4, edgecolor='white')

ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Model Comparison: FSDA vs E-FSDA Variants', fontweight='bold', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.85, 1.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/comparison/comparison_bar_chart.png', dpi=300)
plt.show()
```

### 5.2 Tạo dataset distribution chart

```python
# ========== DATASET DISTRIBUTION ========== #
import matplotlib.pyplot as plt
import numpy as np

classes = ['Fully_Peeled\nGarlic', 'Partially_Peeled\nGarlic', 'Spoiled\nGarlic']

# Dataset-1
d1_train = [482, 306, 704]
d1_val = [103, 65, 150]
d1_test = [105, 67, 152]

# Dataset-2
d2_train = [1050, 306, 704]
d2_val = [225, 65, 150]
d2_test = [225, 67, 152]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#66BB6A', '#FFA726', '#EF5350']

for ax, title, train, val, test in [
    (axes[0], 'Dataset-1 (2,134 images)', d1_train, d1_val, d1_test),
    (axes[1], 'Dataset-2 (2,944 images)', d2_train, d2_val, d2_test),
]:
    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, train, w, label='Train', color=colors[0], edgecolor='white')
    ax.bar(x, val, w, label='Val', color=colors[1], edgecolor='white')
    ax.bar(x + w, test, w, label='Test', color=colors[2], edgecolor='white')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel('Number of Images')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars, fontsize=7, padding=2)

plt.tight_layout()
plt.savefig('/kaggle/working/comparison/dataset_distribution.png', dpi=300)
plt.show()
```

### 5.3 ZIP tất cả kết quả

```python
# ========== ZIP ALL RESULTS ========== #
import shutil
import os

# Zip từng experiment
experiments_to_zip = [
    'finetune_top5_fsda_adaptive_cb',
    'efsda_v2_freqband_temperature',
    'efsda_v3_crossbranch_multiscale',
]

for exp in experiments_to_zip:
    src = f'/kaggle/working/report_EfficientNetB4/{exp}'
    if os.path.exists(src):
        shutil.make_archive(f'/kaggle/working/{exp}_results', 'zip', src)
        print(f"✅ Zipped: {exp}_results.zip")

# Zip comparison folder
if os.path.exists('/kaggle/working/comparison'):
    shutil.make_archive('/kaggle/working/comparison_results', 'zip',
                        '/kaggle/working/comparison')
    print("✅ Zipped: comparison_results.zip")

# Zip EVERYTHING into one master archive
shutil.make_archive('/kaggle/working/THESIS_ALL_RESULTS', 'zip',
                    '/kaggle/working/report_EfficientNetB4')
print("✅ Master archive: THESIS_ALL_RESULTS.zip")

# List all zip files
for f in sorted(os.listdir('/kaggle/working')):
    if f.endswith('.zip'):
        size_mb = os.path.getsize(f'/kaggle/working/{f}') / (1024*1024)
        print(f"  📦 {f} ({size_mb:.1f} MB)")
```

---

## 6. QUY TẮC KHI TẠO/UPDATE FILE

### 6.1 Review Checklist (bắt buộc sau mỗi lần tạo/update)

- [ ] **Code syntax:** Không có lỗi cú pháp (chạy thử cell đầu tiên)
- [ ] **Path:** Tất cả paths đúng format Kaggle (`/kaggle/input/...`, `/kaggle/working/...`)
- [ ] **Mixed precision:** Tất cả custom layers có `dtype='float32'` cho sensitive ops
- [ ] **Float32 casting:** Tất cả FFT, sigmoid, softmax operations cast to float32
- [ ] **Build():** Tất cả sub-layers được explicitly built trong parent's build()
- [ ] **compute_output_spec():** Có trong mọi custom layer dùng FFT (bypass Keras 3 tracing)
- [ ] **get_config():** Tất cả custom layers có get_config() đầy đủ
- [ ] **CUSTOM_OBJECTS:** Dict chứa tất cả custom classes (cho load_model)
- [ ] **Seeds:** 3 seeds cố định [42, 123, 456]
- [ ] **DPI:** Tất cả plots saved dpi=300
- [ ] **Memory:** gc.collect() + clear_session() trước mỗi section nặng (t-SNE, Grad-CAM)

### 6.2 Naming Convention

```
Notebook:    EfficientNetB4-{VARIANT}.ipynb
Strategy:    {variant_key}
Result dir:  /kaggle/working/report_EfficientNetB4/{strategy_key}/
Zip:         /kaggle/working/{strategy_key}_results.zip
```

---

## 7. TROUBLESHOOTING

| Vấn đề                          | Giải pháp                                         |
| ------------------------------- | ------------------------------------------------- |
| OOM khi training                | Giảm BATCH_SIZE: 32 → 16                          |
| OOM khi t-SNE/Grad-CAM          | Dùng batch=4 hoặc 2, gc.collect() trước           |
| `InvalidArgumentError: FFT`     | Kiểm tra float32 cast trước fft2d                 |
| `ValueError: weights not built` | Kiểm tra explicit build() trong custom layers     |
| Session timeout 12h             | Ưu tiên chạy training trước, visualization sau    |
| `TypeError: Expected float32`   | Kiểm tra mixed precision casting tại mỗi use-site |
| Keras 3 load_model fail         | Kiểm tra compute_output_spec() + CUSTOM_OBJECTS   |

---

## 8. TIMELINE ĐỀ XUẤT

| Ngày  | Task                                      | Output                 |
| ----- | ----------------------------------------- | ---------------------- |
| Day 1 | Chạy E-FSDA v2 trên dataset-1             | Results zip            |
| Day 2 | Chạy E-FSDA v2 trên dataset-2             | Results zip            |
| Day 3 | Chạy E-FSDA v3 trên dataset-1 + dataset-2 | Results zip            |
| Day 4 | Tạo comparison charts + tables            | comparison/ folder     |
| Day 5 | Review, viết analysis, zip tất cả         | THESIS_ALL_RESULTS.zip |
