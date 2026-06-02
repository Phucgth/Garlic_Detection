# PROMPT TẠO CÁC VERSION GDE-Net DỰA TRÊN NOTEBOOK EfficientNetB4 + CBLoss Only

File gốc tham chiếu:

```text
D:\private\CaoHoc\luanvanLasted\LuanVanTN_Lasted\src_code\Garlic_Detection\final\src_code\efficientnetb4-ablation-cbloss-only.ipynb
```

Mục tiêu: dùng notebook EfficientNetB4 fine-tuned + CBLoss only hiện tại làm baseline mạnh, sau đó tạo các version kiến trúc mới để kiểm chứng từng đóng góp của **GDE-Net: Garlic Dual-Evidence EfficientNet**.

---

## 0. Nguyên tắc bắt buộc khi tạo các version

Khi chỉnh notebook, hãy giữ nguyên tối đa các phần sau từ notebook gốc:

- Data loading.
- Train/validation/test split.
- Image size.
- Augmentation hiện tại.
- Batch size nếu không có lý do thay đổi.
- Optimizer, learning rate schedule, early stopping, checkpoint.
- Fine-tuning strategy tốt nhất hiện tại: **EfficientNetB4 unfreeze Blocks 3-4-5-6-7**.
- Cách chạy 3 runs.
- Cách tính metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC Macro
  - AUC Weighted
  - Confusion matrix
  - Classification report theo từng class
- Format bảng kết quả `Mean +/- Std`, `Run 1`, `Run 2`, `Run 3`.

Không được thay đổi pipeline dữ liệu nếu không có yêu cầu rõ ràng, vì mục tiêu là so sánh công bằng với baseline:

```text
EfficientNetB4 fine-tuned + CBLoss only
F1-score = 0.9362 +/- 0.0052
AUC Macro = 0.9936 +/- 0.0004
```

---

# PROMPT MASTER

Dùng prompt này trước tiên cho AI/Cursor/Claude/ChatGPT khi bạn muốn nó đọc notebook gốc và tạo version mới.

```text
Bạn là chuyên gia deep learning và code TensorFlow/Keras.
Tôi có notebook gốc tên:

efficientnetb4-ablation-cbloss-only.ipynb

Notebook này đã chạy thành công EfficientNetB4 fine-tuned + Class-Balanced Focal Loss only, với kết quả:
Accuracy = 0.9362 +/- 0.0052
Precision = 0.9364 +/- 0.0051
Recall = 0.9362 +/- 0.0052
F1-score = 0.9362 +/- 0.0052
AUC Macro = 0.9936 +/- 0.0004
AUC Weighted = 0.9928 +/- 0.0004

Nhiệm vụ của bạn:
1. Không phá pipeline dữ liệu, training loop, evaluation code và 3-run experiment hiện tại.
2. Tạo các version kiến trúc mới dựa trên EfficientNetB4 fine-tuned + CBLoss only.
3. Mục tiêu là kiểm chứng từng đóng góp của kiến trúc GDE-Net:
   - Global GAP+GMP head.
   - Coverage Evidence Branch.
   - Peak-Defect Evidence Branch.
   - Evidence-Gated Logit Fusion.
   - Evidence auxiliary loss.
   - Diversity loss giữa coverage map và defect map.
   - Consistency loss nếu dễ triển khai.
4. Mỗi version phải có tên model rõ ràng, lưu kết quả rõ ràng, và in bảng metrics giống notebook gốc.
5. Không được claim mô hình tốt hơn nếu kết quả không vượt baseline.
6. Nếu một version không vượt baseline, vẫn giữ lại để làm ablation âm tính.

Hãy tạo code theo từng cell rõ ràng, có thể copy vào notebook Kaggle/Jupyter.
```

---

# VERSION 0 — Baseline Lock: EfficientNetB4 + CBLoss Only

## Mục đích

Khóa baseline mạnh nhất hiện tại để mọi version sau so sánh công bằng.

## Prompt

```text
Hãy đọc notebook gốc EfficientNetB4 + CBLoss only và tạo một bản clean tên:

V0_EfficientNetB4_CBLoss_Baseline.ipynb

Yêu cầu:
1. Giữ nguyên kiến trúc EfficientNetB4 fine-tuned + CBLoss only.
2. Giữ nguyên data pipeline, augmentation, training loop, callbacks và evaluation.
3. Chuẩn hóa lại code thành các cell rõ ràng:
   - Cell 1: imports + config + seed.
   - Cell 2: dataset paths + class names.
   - Cell 3: data loaders.
   - Cell 4: Class-Balanced Focal Loss.
   - Cell 5: build EfficientNetB4 baseline.
   - Cell 6: train one run.
   - Cell 7: run 3 seeds.
   - Cell 8: evaluation + mean/std table.
   - Cell 9: confusion matrix + classification report.
   - Cell 10: save results to CSV.
4. In lại kết quả theo format:
   Metric | Mean +/- Std | Run 1 | Run 2 | Run 3
5. Đặt tên experiment:
   EfficientNetB4_finetuned_CBLoss_only
6. Không thêm kiến trúc mới trong version này.
```

---

# VERSION 1 — Global GAP+GMP Head

## Ý tưởng

Baseline hiện tại nhiều khả năng dùng GAP. Version này kiểm tra xem thêm GMP có giúp giữ activation lỗi nhỏ hay không.

## Kiến trúc

```text
EfficientNetB4 final feature map
→ GAP
→ GMP
→ Concatenate
→ Dense
→ Dropout
→ Softmax
```

## Prompt

```text
Từ notebook gốc EfficientNetB4 + CBLoss only, hãy tạo version:

V1_EfficientNetB4_GAP_GMP_CBLoss.ipynb

Mục tiêu:
Kiểm tra riêng đóng góp của Global Average Pooling + Global Max Pooling.

Yêu cầu kiến trúc:
1. Backbone: EfficientNetB4, giữ fine-tuning strategy giống notebook gốc.
2. Lấy final feature map của EfficientNetB4, không include top.
3. Tạo global head:
   - gap = GlobalAveragePooling2D()(features)
   - gmp = GlobalMaxPooling2D()(features)
   - x = Concatenate()([gap, gmp])
   - x = BatchNormalization()(x)
   - x = Dropout(0.3 hoặc giá trị notebook gốc đang dùng)(x)
   - logits = Dense(num_classes, activation='softmax')(x)
4. Loss: giữ Class-Balanced Focal Loss như notebook gốc.
5. Không thêm coverage branch, defect branch, gate hay attention ở version này.
6. Giữ nguyên training loop 3 runs.
7. In bảng metrics giống notebook gốc.
8. Lưu kết quả với tên:
   V1_EfficientNetB4_GAP_GMP_CBLoss_results.csv

Mục tiêu phân tích:
So sánh V1 với V0 để biết GAP+GMP có cải thiện so với baseline không.
```

---

# VERSION 2 — Coverage Evidence Branch Only

## Ý tưởng

Branch này học bằng chứng diện tích/vùng phủ, phù hợp với phân biệt:

- Fully_Peeled_Garlic
- Partially_Peeled_Garlic

## Kiến trúc

```text
EfficientNetB4 final feature map
→ Conv1x1 reduce channel
→ Coverage evidence map
→ coverage statistics
→ evidence logits
→ final logits = global logits + alpha * evidence logits
```

## Prompt

```text
Từ notebook gốc EfficientNetB4 + CBLoss only, hãy tạo version:

V2_EfficientNetB4_CoverageEvidence_CBLoss.ipynb

Mục tiêu:
Kiểm tra riêng Coverage Evidence Branch cho bài toán garlic classification.

Yêu cầu kiến trúc:
1. Backbone: EfficientNetB4 fine-tuned như notebook gốc.
2. Lấy final feature map.
3. Tạo shared reduced feature:
   F = Conv2D(256, 1, padding='same', activation='swish')(features)
   F = BatchNormalization()(F)
4. Global branch:
   gap = GlobalAveragePooling2D()(F)
   gmp = GlobalMaxPooling2D()(F)
   global_feature = Concatenate()([gap, gmp])
   global_logits = Dense(num_classes)(global_feature)
5. Coverage Evidence Branch:
   M_cov = Conv2D(1, 1, activation='sigmoid', name='coverage_map')(F)
   cov_mean = GlobalAveragePooling2D()(M_cov)
   cov_max = GlobalMaxPooling2D()(M_cov)
   coverage_feature = Concatenate()([cov_mean, cov_max])
   coverage_feature = Dense(64, activation='swish')(coverage_feature)
   evidence_logits = Dense(num_classes, name='coverage_logits')(coverage_feature)
6. Evidence gate:
   alpha = Dense(1, activation='sigmoid', name='coverage_gate')(global_feature)
   final_logits_raw = global_logits + alpha * evidence_logits
   outputs = Softmax(name='final_softmax')(final_logits_raw)
7. Model outputs:
   Ưu tiên dùng output chính final_softmax.
   Nếu triển khai auxiliary loss được thì model output gồm:
   - final_softmax
   - coverage_softmax = Softmax()(evidence_logits)
8. Loss:
   - final loss: CBLoss.
   - Nếu dùng auxiliary output: total loss = 1.0 * CBLoss(final) + 0.3 * CBLoss(coverage).
9. Không thêm peak-defect branch ở version này.
10. Giữ 3 runs và evaluation giống notebook gốc.
11. Lưu coverage maps hoặc ít nhất tạo function visualize coverage_map cho vài ảnh test.
12. Lưu kết quả:
    V2_EfficientNetB4_CoverageEvidence_CBLoss_results.csv

Mục tiêu phân tích:
So sánh V2 với V0/V1 để biết evidence vùng phủ có giúp không.
Đặc biệt in per-class F1/Recall cho Fully_Peeled_Garlic và Partially_Peeled_Garlic.
```

---

# VERSION 3 — Peak-Defect Evidence Branch Only

## Ý tưởng

Branch này dùng top-k pooling để bắt vết hư nhỏ, đốm lỗi, vùng bất thường cục bộ.

## Kiến trúc

```text
EfficientNetB4 final feature map
→ Conv1x1 reduce channel
→ defect evidence map
→ top-k pooling
→ evidence logits
→ gated logit fusion
```

## Prompt

```text
Từ notebook gốc EfficientNetB4 + CBLoss only, hãy tạo version:

V3_EfficientNetB4_PeakDefectEvidence_CBLoss.ipynb

Mục tiêu:
Kiểm tra riêng Peak-Defect Evidence Branch dùng top-k pooling để bắt lỗi nhỏ.

Yêu cầu kiến trúc:
1. Backbone: EfficientNetB4 fine-tuned như notebook gốc.
2. Lấy final feature map.
3. Tạo reduced feature:
   F = Conv2D(256, 1, padding='same', activation='swish')(features)
   F = BatchNormalization()(F)
4. Global branch:
   gap = GlobalAveragePooling2D()(F)
   gmp = GlobalMaxPooling2D()(F)
   global_feature = Concatenate()([gap, gmp])
   global_logits = Dense(num_classes)(global_feature)
5. Peak-Defect Evidence Branch:
   M_def = Conv2D(1, 1, activation='sigmoid', name='defect_map')(F)
6. Tạo custom TopKPooling2D layer:
   - Flatten spatial map H*W.
   - Chọn top k% vị trí có activation cao nhất.
   - k_ratio mặc định = 0.1 hoặc 0.15.
   - Lấy mean top-k và max activation.
7. defect_feature = Concatenate()([topk_mean, max_activation])
8. defect_feature = Dense(64, activation='swish')(defect_feature)
9. evidence_logits = Dense(num_classes, name='defect_logits')(defect_feature)
10. Evidence gate:
    alpha = Dense(1, activation='sigmoid', name='defect_gate')(global_feature)
    final_logits_raw = global_logits + alpha * evidence_logits
    outputs = Softmax(name='final_softmax')(final_logits_raw)
11. Loss:
    - final loss: CBLoss.
    - Nếu dùng auxiliary output: total loss = 1.0 * CBLoss(final) + 0.3 * CBLoss(defect).
12. Không thêm coverage branch ở version này.
13. Giữ 3 runs và evaluation giống notebook gốc.
14. Lưu defect maps hoặc tạo function visualize defect_map cho vài ảnh test.
15. Lưu kết quả:
    V3_EfficientNetB4_PeakDefectEvidence_CBLoss_results.csv

Mục tiêu phân tích:
So sánh V3 với V0/V1 để biết top-k defect evidence có giúp không.
Đặc biệt in per-class F1/Recall cho Spoiled_Garlic.
```

---

# VERSION 4 — Dual Evidence Head Without Diversity Loss

## Ý tưởng

Kết hợp Coverage Evidence và Peak-Defect Evidence, nhưng chưa dùng diversity loss.

## Kiến trúc

```text
Global logits
Coverage logits
Defect logits
→ evidence gate
→ final logits
```

## Prompt

```text
Từ notebook gốc EfficientNetB4 + CBLoss only, hãy tạo version:

V4_EfficientNetB4_DualEvidence_NoDiversity_CBLoss.ipynb

Mục tiêu:
Kiểm tra việc kết hợp Coverage Evidence Branch và Peak-Defect Evidence Branch.

Yêu cầu kiến trúc:
1. Backbone: EfficientNetB4 fine-tuned giống notebook gốc.
2. Reduced feature:
   F = Conv2D(256, 1, padding='same', activation='swish')(features)
   F = BatchNormalization()(F)
3. Global branch:
   GAP + GMP → global_feature → global_logits
4. Coverage branch:
   M_cov = Conv2D(1, 1, activation='sigmoid', name='coverage_map')(F)
   cov_mean = GlobalAveragePooling2D()(M_cov)
   cov_max = GlobalMaxPooling2D()(M_cov)
   coverage_feature = Dense(64, activation='swish')(Concat([cov_mean, cov_max]))
   coverage_logits = Dense(num_classes, name='coverage_logits')(coverage_feature)
5. Defect branch:
   M_def = Conv2D(1, 1, activation='sigmoid', name='defect_map')(F)
   topk_mean, max_activation = TopKPooling2D(k_ratio=0.1)(M_def)
   defect_feature = Dense(64, activation='swish')(Concat([topk_mean, max_activation]))
   defect_logits = Dense(num_classes, name='defect_logits')(defect_feature)
6. Evidence gate:
   evidence_concat = Concatenate()([global_feature, coverage_feature, defect_feature])
   alpha_cov = Dense(1, activation='sigmoid', name='alpha_coverage')(evidence_concat)
   alpha_def = Dense(1, activation='sigmoid', name='alpha_defect')(evidence_concat)
   final_logits_raw = global_logits + alpha_cov * coverage_logits + alpha_def * defect_logits
   outputs = Softmax(name='final_softmax')(final_logits_raw)
7. Loss:
   - final output: CBLoss.
   - auxiliary coverage output: 0.2 hoặc 0.3 * CBLoss.
   - auxiliary defect output: 0.2 hoặc 0.3 * CBLoss.
8. Chưa thêm diversity loss ở version này.
9. Giữ 3 runs và evaluation như notebook gốc.
10. In thêm trung bình alpha_cov và alpha_def trên test set theo từng class nếu làm được.
11. Lưu kết quả:
    V4_EfficientNetB4_DualEvidence_NoDiversity_CBLoss_results.csv

Mục tiêu phân tích:
So sánh V4 với V2 và V3 để xem hai evidence branch có bổ sung nhau không.
```

---

# VERSION 5 — Full GDE-Net With Diversity Loss

## Ý tưởng

Full model với coverage branch, defect branch, evidence gate và diversity loss.

Diversity loss ép coverage map và defect map không học trùng nhau.

## Prompt

```text
Từ notebook gốc EfficientNetB4 + CBLoss only, hãy tạo version:

V5_GDENet_EfficientNetB4_DualEvidence_Diversity_CBLoss.ipynb

Tên mô hình:
GDE-Net: Garlic Dual-Evidence EfficientNet

Mục tiêu:
Triển khai full GDE-Net với:
- Global branch.
- Coverage Evidence Branch.
- Peak-Defect Evidence Branch.
- Evidence-Gated Logit Fusion.
- Auxiliary evidence loss.
- Diversity loss giữa coverage_map và defect_map.

Yêu cầu kiến trúc:
1. Backbone:
   EfficientNetB4, include_top=False, pretrained ImageNet, fine-tuned giống notebook gốc.
2. Shared feature:
   F = Conv2D(256, 1, padding='same', activation='swish')(features)
   F = BatchNormalization()(F)
   F = Dropout nhỏ nếu cần, ví dụ SpatialDropout2D(0.1)
3. Global branch:
   GAP + GMP → global_feature → Dense → global_logits
4. Coverage branch:
   coverage_map = Conv2D(1, 1, activation='sigmoid', name='coverage_map')(F)
   coverage stats = mean + max
   coverage_feature = Dense(64, activation='swish')
   coverage_logits = Dense(num_classes, name='coverage_logits')
5. Defect branch:
   defect_map = Conv2D(1, 1, activation='sigmoid', name='defect_map')(F)
   topk pooling with k_ratio = 0.1
   defect_feature = Dense(64, activation='swish')
   defect_logits = Dense(num_classes, name='defect_logits')
6. Evidence gate:
   gate_input = Concatenate()([global_feature, coverage_feature, defect_feature])
   gate = Dense(2, activation='sigmoid', name='evidence_gate')(gate_input)
   alpha_cov = gate[:, 0:1]
   alpha_def = gate[:, 1:2]
   final_logits_raw = global_logits + alpha_cov * coverage_logits + alpha_def * defect_logits
   final_softmax = Softmax(name='final_softmax')(final_logits_raw)
7. Auxiliary outputs:
   coverage_softmax = Softmax(name='coverage_softmax')(coverage_logits)
   defect_softmax = Softmax(name='defect_softmax')(defect_logits)
8. Loss:
   total loss =
   1.0 * CBLoss(final_softmax)
   + 0.2 hoặc 0.3 * CBLoss(coverage_softmax)
   + 0.2 hoặc 0.3 * CBLoss(defect_softmax)
   + lambda_div * diversity_loss
9. Diversity loss:
   diversity_loss = mean(coverage_map * defect_map)
   Lambda gợi ý: 0.01, 0.03, hoặc 0.05.
   Nếu compile Keras khó thêm custom loss cho intermediate maps, hãy tạo custom training loop hoặc tạo custom layer add_loss().
10. Giữ 3 runs.
11. Evaluation chính chỉ dùng final_softmax.
12. In metrics giống notebook gốc.
13. In thêm:
   - mean alpha_cov theo class
   - mean alpha_def theo class
   - mean coverage activation theo class
   - mean defect activation theo class
14. Lưu:
   - V5_GDENet_results.csv
   - confusion matrix
   - classification report
   - vài hình visualize coverage_map và defect_map.

Mục tiêu phân tích:
Full GDE-Net phải được so sánh với:
- V0 EfficientNetB4 + CBLoss only.
- V1 GAP+GMP.
- V2 Coverage only.
- V3 Defect only.
- V4 Dual evidence no diversity.

Nếu Full GDE-Net không vượt overall F1, hãy kiểm tra per-class F1/Recall.
Đặc biệt chú ý lớp Partially_Peeled_Garlic và Spoiled_Garlic.
```

---

# VERSION 6 — Lightweight GDE-Net Without Auxiliary Loss

## Mục đích

Nếu version full quá phức tạp hoặc bị lỗi compile, dùng bản nhẹ để chạy trước.

## Prompt

```text
Tạo version:

V6_GDENet_Lightweight_NoAuxLoss.ipynb

Dựa trên notebook gốc EfficientNetB4 + CBLoss only.

Mục tiêu:
Tạo bản GDE-Net nhẹ, không dùng auxiliary outputs, không dùng custom training loop.

Yêu cầu:
1. Backbone EfficientNetB4 fine-tuned như notebook gốc.
2. Shared feature Conv1x1 256.
3. Global branch GAP+GMP → global_logits.
4. Coverage map → mean/max → coverage_logits.
5. Defect map → top-k/max → defect_logits.
6. Evidence gate → final_logits.
7. Chỉ output final_softmax.
8. Chỉ dùng CBLoss hoặc CE cho final_softmax.
9. Không dùng diversity loss.
10. Không dùng consistency loss.
11. Giữ 3 runs và evaluation như notebook gốc.
12. Lưu results CSV.

Mục tiêu:
Chạy nhanh để kiểm tra kiến trúc dual-evidence có tiềm năng vượt baseline không trước khi thêm loss phụ.
```

---

# VERSION 7 — GDE-Net With CategoricalCrossentropy

## Mục đích

Vì CBLoss không luôn vượt CE, cần kiểm tra kiến trúc GDE-Net với CE.

## Prompt

```text
Từ version GDE-Net tốt nhất hiện tại, hãy tạo bản:

V7_GDENet_CategoricalCrossentropy.ipynb

Mục tiêu:
Kiểm tra xem GDE-Net hoạt động tốt hơn với CategoricalCrossentropy hay Class-Balanced Focal Loss.

Yêu cầu:
1. Giữ nguyên kiến trúc GDE-Net tốt nhất.
2. Thay loss chính từ CBLoss sang CategoricalCrossentropy.
3. Nếu có auxiliary outputs, dùng CategoricalCrossentropy cho auxiliary outputs.
4. Giữ nguyên dataset, augmentation, training loop, callbacks.
5. Chạy 3 runs.
6. In bảng metrics giống notebook gốc.
7. Lưu kết quả:
   V7_GDENet_CE_results.csv
8. So sánh trực tiếp:
   - EfficientNetB4 + CE
   - EfficientNetB4 + CBLoss
   - GDE-Net + CE
   - GDE-Net + CBLoss
```

---

# VERSION 8 — Evidence Visualization Notebook

## Mục đích

Tạo bằng chứng trực quan để phản biện: model nhìn vào đâu?

## Prompt

```text
Tạo notebook:

V8_GDENet_Evidence_Visualization.ipynb

Dựa trên checkpoint tốt nhất của GDE-Net.

Mục tiêu:
Tạo hình minh họa cho báo cáo/bài báo.

Yêu cầu:
1. Load model checkpoint tốt nhất của GDE-Net.
2. Lấy một số ảnh test đúng/sai theo từng class:
   - Fully_Peeled_Garlic
   - Partially_Peeled_Garlic
   - Spoiled_Garlic
3. Với mỗi ảnh, hiển thị:
   - ảnh gốc
   - coverage_map overlay
   - defect_map overlay
   - Grad-CAM của final prediction
   - predicted class + confidence
   - true label
4. Không cần dùng OCR hoặc tool ngoài.
5. Lưu hình:
   - evidence_correct_examples.png
   - evidence_wrong_examples.png
   - evidence_per_class_examples.png
6. In thêm thống kê:
   - mean coverage activation theo class
   - mean defect activation theo class
   - mean alpha_cov theo class
   - mean alpha_def theo class

Mục tiêu phân tích:
Nếu GDE-Net không tăng F1 nhiều, visualization vẫn giúp chứng minh model học evidence có ý nghĩa.
```

---

# VERSION 9 — Final Ablation Summary Notebook

## Mục đích

Gom tất cả CSV kết quả thành bảng ablation cuối.

## Prompt

```text
Tạo notebook:

V9_GDENet_Final_Ablation_Summary.ipynb

Mục tiêu:
Đọc tất cả file CSV kết quả từ các version:
- V0 EfficientNetB4 + CBLoss only
- V1 GAP+GMP
- V2 Coverage Evidence
- V3 Peak-Defect Evidence
- V4 Dual Evidence no diversity
- V5 Full GDE-Net
- V7 GDE-Net + CE nếu có

Yêu cầu:
1. Tạo bảng tổng hợp:
   Model | Accuracy | Precision | Recall | F1-score | AUC Macro | AUC Weighted
2. Tạo bảng delta so với baseline:
   Model | ΔAccuracy | ΔF1 | ΔAUC Macro | ΔAUC Weighted
3. Tạo bảng per-class:
   Model | Class | Precision | Recall | F1-score | Support
4. Tạo bảng component ablation:
   Model | GAP+GMP | Coverage | Defect | Gate | Aux Loss | Diversity | Loss | F1
5. Highlight model tốt nhất theo:
   - Overall F1
   - AUC Macro
   - Minority-class F1
   - Spoiled_Garlic Recall
   - Partially_Peeled_Garlic Recall
6. Xuất bảng ra:
   - final_ablation_summary.csv
   - final_ablation_summary.xlsx nếu môi trường hỗ trợ
7. Tạo đoạn nhận xét tự động:
   - Nếu GDE-Net vượt baseline: viết kết luận tích cực.
   - Nếu GDE-Net không vượt baseline: viết kết luận trung thực, giải thích rằng baseline EfficientNetB4 + CBLoss đã rất mạnh và dual-evidence không tạo lợi thế rõ.
```

---

# Bảng ablation mục tiêu cho báo cáo

Sau khi chạy, báo cáo nên có bảng như sau:

| Model                   | GAP+GMP | Coverage Evidence | Peak-Defect Evidence | Gate | Aux Loss | Diversity | Loss   | F1-score |
| ----------------------- | ------: | ----------------: | -------------------: | ---: | -------: | --------: | ------ | -------: |
| EfficientNetB4 baseline |       ✗ |                 ✗ |                    ✗ |    ✗ |        ✗ |         ✗ | CE     |      ... |
| EfficientNetB4 + CBLoss |       ✗ |                 ✗ |                    ✗ |    ✗ |        ✗ |         ✗ | CBLoss |   0.9362 |
| V1 GAP+GMP              |       ✓ |                 ✗ |                    ✗ |    ✗ |        ✗ |         ✗ | CBLoss |      ... |
| V2 Coverage only        |       ✓ |                 ✓ |                    ✗ |    ✓ |      ✓/✗ |         ✗ | CBLoss |      ... |
| V3 Defect only          |       ✓ |                 ✗ |                    ✓ |    ✓ |      ✓/✗ |         ✗ | CBLoss |      ... |
| V4 Dual evidence        |       ✓ |                 ✓ |                    ✓ |    ✓ |      ✓/✗ |         ✗ | CBLoss |      ... |
| V5 Full GDE-Net         |       ✓ |                 ✓ |                    ✓ |    ✓ |        ✓ |         ✓ | CBLoss |      ... |
| V7 Full GDE-Net CE      |       ✓ |                 ✓ |                    ✓ |    ✓ |        ✓ |         ✓ | CE     |      ... |

---

# Cách kết luận nếu kết quả tốt

Dùng khi GDE-Net vượt baseline rõ, ví dụ F1 tăng từ 0.9362 lên khoảng 0.943 trở lên, hoặc minority class tăng rõ.

```text
The proposed GDE-Net achieved the best overall performance compared with the strong EfficientNetB4 + CBLoss baseline. The improvement is attributed to the explicit modeling of two complementary types of visual evidence: coverage evidence for peeled/unpeeled regions and peak-defect evidence for localized spoilage patterns. The ablation results show that each component contributes to the final performance, while the evidence visualization further confirms that the model focuses on semantically meaningful garlic regions.
```

---

# Cách kết luận nếu kết quả chỉ tăng nhẹ

Dùng khi F1 chỉ tăng khoảng 0.001 đến 0.004.

```text
GDE-Net produced a marginal improvement over the EfficientNetB4 + CBLoss baseline. Although the overall gain is limited, the model provides better interpretability through coverage and defect evidence maps. The results suggest that the baseline EfficientNetB4 already captures strong discriminative features, while the proposed evidence branches mainly improve model explainability and class-specific behavior.
```

---

# Cách kết luận nếu kết quả không vượt baseline

Dùng khi GDE-Net thấp hơn EfficientNetB4 + CBLoss.

```text
Although GDE-Net introduces garlic-specific evidence modeling, the experimental results show that it does not outperform the strong EfficientNetB4 + CBLoss baseline. This indicates that the fine-tuned EfficientNetB4 already captures sufficient discriminative features for the current dataset. Nevertheless, the negative result is valuable, as it shows that adding evidence branches may introduce additional complexity without clear performance gain when the dataset is relatively small or visually separable.
```

---

# Lưu ý phản biện quan trọng

Không nên viết:

```text
We propose a novel EfficientNetB4 with attention.
```

Câu này yếu vì EfficientNet + attention đã có nhiều.

Nên viết:

```text
We propose a garlic-specific dual-evidence classification head that decomposes visual decision cues into coverage evidence and peak-defect evidence using only image-level supervision.
```

Điểm mới cần bảo vệ là:

1. **Garlic-specific evidence decomposition**:
   - coverage evidence
   - peak-defect evidence

2. **Weakly supervised evidence learning**:
   - không cần mask vùng lỗi
   - vẫn tạo evidence maps từ image-level labels

3. **Evidence-gated logit fusion**:
   - không fuse feature lung tung
   - giữ baseline global classifier mạnh
   - evidence branch chỉ bổ sung logit khi cần

4. **Ablation và visualization rõ**:
   - coverage branch có giúp không
   - defect branch có giúp không
   - gate có hoạt động không
   - diversity loss có tránh map học trùng không

---

# Checklist trước khi đưa vào báo cáo

- [ ] Có baseline EfficientNetB4 + CE.
- [ ] Có baseline EfficientNetB4 + CBLoss.
- [ ] Có GDE-Net + CE.
- [ ] Có GDE-Net + CBLoss.
- [ ] Có ablation coverage only.
- [ ] Có ablation defect only.
- [ ] Có ablation dual evidence.
- [ ] Có ablation diversity loss.
- [ ] Có 3 runs hoặc ít nhất top models chạy 3 runs.
- [ ] Có mean ± std.
- [ ] Có per-class Precision/Recall/F1.
- [ ] Có confusion matrix.
- [ ] Có evidence map visualization.
- [ ] Có phân tích lỗi các ảnh sai.
- [ ] Không claim “significant” nếu chưa có kiểm định thống kê.
- [ ] Nếu muốn dùng “significant”, chạy paired t-test hoặc Wilcoxon trên kết quả từng run/fold.

---

# Prompt ngắn để tạo ngay version full GDE-Net

Dùng prompt này nếu muốn tạo nhanh code full model:

```text
Dựa trên notebook EfficientNetB4 + CBLoss only đang chạy tốt, hãy tạo một notebook mới triển khai GDE-Net: Garlic Dual-Evidence EfficientNet.

Giữ nguyên toàn bộ data pipeline, training loop 3 runs, callbacks, metrics và fine-tuning strategy của notebook gốc. Chỉ thay hàm build_model.

Kiến trúc mới:
1. EfficientNetB4 include_top=False, pretrained ImageNet.
2. Lấy final feature map.
3. Conv2D 1x1 giảm kênh về 256 + BatchNorm + swish.
4. Global branch: GAP + GMP → Dense → global logits.
5. Coverage Evidence Branch:
   - coverage_map = Conv2D(1, 1, sigmoid)
   - lấy mean và max activation
   - Dense 64 → coverage logits.
6. Peak-Defect Evidence Branch:
   - defect_map = Conv2D(1, 1, sigmoid)
   - custom TopKPooling2D lấy top 10% activation + max activation
   - Dense 64 → defect logits.
7. Evidence gate:
   - gate input = concat(global feature, coverage feature, defect feature)
   - Dense(2, sigmoid) tạo alpha_cov và alpha_def
   - final logits = global logits + alpha_cov * coverage_logits + alpha_def * defect_logits
   - Softmax final.
8. Loss chính dùng CBLoss giống notebook gốc.
9. Nếu dễ triển khai, thêm auxiliary outputs coverage_softmax và defect_softmax với weight 0.2.
10. Nếu dễ triển khai, thêm diversity loss mean(coverage_map * defect_map) bằng add_loss với lambda 0.03.
11. Evaluation chỉ dùng final output.
12. In bảng Mean +/- Std giống notebook gốc và lưu CSV.
13. Không thay đổi dataset hoặc augmentation.
```
