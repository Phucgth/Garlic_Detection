# HSP-GarlicNet Ablation Prompts

File baseline gốc:

```text
D:\private\CaoHoc\luanvanLasted\LuanVanTN_Lasted\src_code\Garlic_Detection\final\src_code\efficientnetb4-ablation-cbloss-only.ipynb
```

Baseline hiện tại:

```text
EfficientNetB4 fine-tuned + CBLoss only
Accuracy     = 0.9362 +/- 0.0052
Precision    = 0.9364 +/- 0.0051
Recall       = 0.9362 +/- 0.0052
F1-score     = 0.9362 +/- 0.0052
AUC Macro    = 0.9936 +/- 0.0004
AUC Weighted = 0.9928 +/- 0.0004
```

V1 hiện tại:

```text
EfficientNetB4 + GAP/GMP + CBLoss
F1-score = 0.9372 +/- 0.0076
```

Kiến trúc đề xuất:

# HSP-GarlicNet: Hierarchical Spoilage–Peeling Guided EfficientNet

Thay vì phân loại phẳng 3 lớp:

```text
Fully_Peeled_Garlic
Partially_Peeled_Garlic
Spoiled_Garlic
```

HSP-GarlicNet tách thành 2 quyết định:

```text
Task 1: Spoilage detection
Spoiled_Garlic vs Non-spoiled

Task 2: Peeling-state recognition
Fully_Peeled_Garlic vs Partially_Peeled_Garlic
Chỉ tính peeling loss trên mẫu non-spoiled.
```

Công thức hierarchical probability:

```text
p_spoil   = P(Spoiled)
p_partial = P(Partially | Non-spoiled)

P(Fully_Peeled_Garlic)    = (1 - p_spoil) * (1 - p_partial)
P(Partially_Peeled_Garlic)= (1 - p_spoil) * p_partial
P(Spoiled_Garlic)         = p_spoil
```

---

## 1. Prompt Master

```text
Bạn là chuyên gia deep learning TensorFlow/Keras và nghiên cứu mô hình phân loại ảnh.

Tôi có notebook baseline:
efficientnetb4-ablation-cbloss-only.ipynb

Notebook này chạy tốt EfficientNetB4 fine-tuned + Class-Balanced Focal Loss với kết quả:
Accuracy = 0.9362 +/- 0.0052
Precision = 0.9364 +/- 0.0051
Recall = 0.9362 +/- 0.0052
F1-score = 0.9362 +/- 0.0052
AUC Macro = 0.9936 +/- 0.0004
AUC Weighted = 0.9928 +/- 0.0004

Nhiệm vụ:
1. Không phá data pipeline, augmentation, training loop, callbacks, evaluation và 3-run experiment hiện tại.
2. Giữ EfficientNetB4 fine-tuned strategy giống notebook gốc, đặc biệt unfreeze Blocks 3-4-5-6-7.
3. Chỉ thay build_model và bổ sung nhãn phụ/loss khi cần.
4. Tạo các version ablation cho HSP-GarlicNet:
   - GAP/GMP dual pooling.
   - Spoilage binary auxiliary head.
   - Masked peeling binary auxiliary head.
   - Hierarchical probability composition.
   - Residual hierarchical logit fusion với learnable beta.
   - Hierarchy consistency loss.
5. Mỗi version phải in bảng:
   Metric | Mean +/- Std | Run 1 | Run 2 | Run 3
6. Mỗi version phải lưu CSV kết quả, confusion matrix và classification report.
7. Metrics chính luôn là 3-class metrics trên test set.
8. Không được claim tốt hơn nếu chưa vượt baseline.
9. Code chia cell rõ ràng, có thể copy vào Kaggle/Jupyter.
```

---

## 2. V0 — EfficientNetB4 + CBLoss Baseline

Mục đích: khóa baseline mạnh nhất.

```text
Tạo notebook:
V0_EfficientNetB4_CBLoss_Baseline.ipynb

Dựa trên notebook efficientnetb4-ablation-cbloss-only.ipynb.

Yêu cầu:
1. Giữ nguyên EfficientNetB4 fine-tuned + CBLoss only.
2. Giữ nguyên data pipeline, augmentation, callbacks, optimizer, scheduler, training loop 3 runs.
3. Chuẩn hóa cell:
   Cell 1: imports, config, seed
   Cell 2: dataset paths, class names
   Cell 3: data loaders
   Cell 4: Class-Balanced Focal Loss
   Cell 5: build baseline model
   Cell 6: train one run
   Cell 7: run 3 seeds
   Cell 8: evaluation mean/std
   Cell 9: confusion matrix + classification report
   Cell 10: save CSV
4. Tên experiment:
   V0_EfficientNetB4_CBLoss_Baseline
5. Không thêm head phụ hoặc hierarchical logic.
```

---

## 3. V1 — EfficientNetB4 + GAP/GMP + CBLoss

Mục đích: kiểm tra dual pooling.

```text
Tạo notebook:
V1_EfficientNetB4_GAP_GMP_CBLoss.ipynb

Dựa trên V0.

Yêu cầu kiến trúc:
1. Backbone EfficientNetB4 include_top=False, pretrained ImageNet.
2. Giữ fine-tuning strategy giống V0.
3. Lấy final feature map.
4. Head:
   gap = GlobalAveragePooling2D()(features)
   gmp = GlobalMaxPooling2D()(features)
   x = Concatenate(name="gap_gmp_concat")([gap, gmp])
   x = BatchNormalization()(x)
   x = Dropout(0.3)(x)
   output = Dense(num_classes, activation="softmax", name="flat_softmax")(x)
5. Loss: CBLoss.
6. Không thêm spoilage head, peeling head hoặc hierarchical fusion.
7. Chạy 3 runs.
8. Lưu:
   V1_EfficientNetB4_GAP_GMP_CBLoss_results.csv

Mục tiêu phân tích:
So sánh V1 với V0 để xem GAP/GMP có cải thiện F1, AUC và per-class recall không.
```

---

## 4. H1 — Add Spoilage Binary Auxiliary Head

Mục đích: kiểm tra task phụ spoiled vs non-spoiled.

Nhãn phụ:

```text
Spoiled_Garlic = 1
Fully_Peeled_Garlic = 0
Partially_Peeled_Garlic = 0
```

```text
Tạo notebook:
H1_EfficientNetB4_GAP_GMP_SpoilageAux_CBLoss.ipynb

Dựa trên V1.

Yêu cầu dữ liệu:
1. Tạo y_spoilage từ label gốc:
   Spoiled_Garlic = 1
   Fully_Peeled_Garlic = 0
   Partially_Peeled_Garlic = 0
2. Dataset trả về:
   inputs,
   {
     "flat_softmax": y_3class,
     "spoilage_sigmoid": y_spoilage
   }

Yêu cầu kiến trúc:
1. Backbone + GAP/GMP shared feature giống V1.
2. Flat head:
   flat_output = Dense(num_classes, activation="softmax", name="flat_softmax")(shared)
3. Spoilage head:
   spoil_x = Dense(128, activation="swish")(shared)
   spoil_x = Dropout(0.2)(spoil_x)
   spoilage_output = Dense(1, activation="sigmoid", name="spoilage_sigmoid")(spoil_x)
4. Model outputs:
   [flat_softmax, spoilage_sigmoid]

Loss:
1. flat_softmax: CBLoss, weight 1.0
2. spoilage_sigmoid: BinaryCrossentropy, weight 0.2 hoặc 0.3

Evaluation:
1. Metrics chính dùng flat_softmax.
2. In thêm binary spoilage metrics nếu được:
   accuracy, precision, recall, F1.
3. Lưu:
   H1_EfficientNetB4_GAP_GMP_SpoilageAux_CBLoss_results.csv

Mục tiêu:
So sánh H1 với V1 để xem spoilage auxiliary task có cải thiện Spoiled_Garlic recall/F1 không.
```

---

## 5. H2 — Add Masked Peeling Auxiliary Head

Mục đích: phân biệt Fully vs Partially, nhưng không ép ảnh Spoiled học trạng thái peeling.

Nhãn phụ:

```text
Fully_Peeled_Garlic = 0
Partially_Peeled_Garlic = 1
Spoiled_Garlic = masked, sample_weight = 0
```

```text
Tạo notebook:
H2_EfficientNetB4_GAP_GMP_Spoilage_PeelingAux_CBLoss.ipynb

Dựa trên H1.

Yêu cầu dữ liệu:
1. y_spoilage:
   Spoiled_Garlic = 1
   Fully/Partially = 0
2. y_peeling:
   Fully_Peeled_Garlic = 0
   Partially_Peeled_Garlic = 1
   Spoiled_Garlic = 0 hoặc giá trị bất kỳ
3. peeling_sample_weight:
   Fully_Peeled_Garlic = 1
   Partially_Peeled_Garlic = 1
   Spoiled_Garlic = 0

Nếu fit hỗ trợ sample_weight dict:
   inputs,
   outputs_dict,
   sample_weight_dict

Nếu khó dùng sample_weight:
   tạo y_true_peeling gồm 2 cột:
   cột 0 = peeling label
   cột 1 = mask
   masked_bce = BCE(label, pred) * mask / mean(mask)

Yêu cầu kiến trúc:
1. Backbone + shared feature giống H1.
2. Outputs:
   flat_softmax
   spoilage_sigmoid
   peeling_sigmoid
3. Peeling head:
   peel_x = Dense(128, activation="swish")(shared)
   peel_x = Dropout(0.2)(peel_x)
   peeling_output = Dense(1, activation="sigmoid", name="peeling_sigmoid")(peel_x)

Loss:
1. flat_softmax: CBLoss, weight 1.0
2. spoilage_sigmoid: BinaryCrossentropy, weight 0.2
3. peeling_sigmoid: masked BinaryCrossentropy, weight 0.2

Evaluation:
1. Metrics chính dùng flat_softmax.
2. In binary spoilage metrics.
3. In peeling binary metrics trên subset non-spoiled test.
4. Lưu:
   H2_EfficientNetB4_GAP_GMP_Spoilage_PeelingAux_CBLoss_results.csv

Mục tiêu:
So sánh H2 với H1 để xem masked peeling auxiliary task có cải thiện Fully/Partially không.
```

---

## 6. H3 — Hierarchical Probability Composition

Mục đích: tạo final prediction từ p_spoil và p_partial.

```text
Tạo notebook:
H3_EfficientNetB4_HierarchicalProbability_CBLoss.ipynb

Dựa trên H2.

Yêu cầu kiến trúc:
1. Backbone + GAP/GMP shared feature giống H2.
2. Spoilage head:
   p_spoil = Dense(1, activation="sigmoid", name="spoilage_sigmoid")(spoil_x)
3. Peeling head:
   p_partial = Dense(1, activation="sigmoid", name="peeling_sigmoid")(peel_x)
4. Lambda hierarchical probability:
   p_fully = (1 - p_spoil) * (1 - p_partial)
   p_partially = (1 - p_spoil) * p_partial
   p_spoiled = p_spoil
   hierarchical_probs = Concatenate(name="hierarchical_probs")(
      [p_fully, p_partially, p_spoiled]
   )
5. Thứ tự class phải khớp với class_indices.
6. Outputs:
   hierarchical_probs
   spoilage_sigmoid
   peeling_sigmoid

Loss:
1. hierarchical_probs: CBLoss hoặc CategoricalCrossentropy cho 3 classes.
2. spoilage_sigmoid: BinaryCrossentropy, weight 0.2
3. peeling_sigmoid: masked BinaryCrossentropy, weight 0.2

Evaluation:
1. Metrics chính dùng hierarchical_probs.
2. In 3-class metrics, spoilage metrics, peeling metrics.
3. Lưu:
   H3_EfficientNetB4_HierarchicalProbability_CBLoss_results.csv

Mục tiêu:
Xem hierarchical probability tự nó có đủ mạnh không khi không dùng flat 3-class head.
```

---

## 7. H4 — Residual Hierarchical Logit Fusion

Đây là version quan trọng nhất.

Mục đích: giữ flat classifier mạnh, thêm hierarchical branch như tín hiệu hỗ trợ nhẹ.

Công thức:

```text
hier_logits = log(clip(hier_probs, 1e-7, 1.0))
final_logits = flat_logits + beta * hier_logits
final_softmax = softmax(final_logits)
```

Trong đó `beta` là learnable scalar, khởi tạo 0.0 hoặc 0.1.

```text
Tạo notebook:
H4_HSP_GarlicNet_ResidualFusion_CBLoss.ipynb

Dựa trên H2.

Yêu cầu kiến trúc:
1. Backbone EfficientNetB4 + GAP/GMP shared feature giống V1/H2.
2. Flat head:
   flat_logits = Dense(num_classes, name="flat_logits")(shared)
   flat_softmax = Softmax(name="flat_softmax")(flat_logits)
3. Spoilage head:
   p_spoil = Dense(1, activation="sigmoid", name="spoilage_sigmoid")(spoil_x)
4. Peeling head:
   p_partial = Dense(1, activation="sigmoid", name="peeling_sigmoid")(peel_x)
5. Hierarchical probability:
   p_fully = (1 - p_spoil) * (1 - p_partial)
   p_partially = (1 - p_spoil) * p_partial
   p_spoiled = p_spoil
   hier_probs = Concatenate(name="hierarchical_probs")([p_fully, p_partially, p_spoiled])
6. Convert to logits:
   hier_logits = log(clip(hier_probs, 1e-7, 1.0))
7. Custom LearnableScalar layer:
   beta initialized to 0.0 hoặc 0.1
   final_logits = flat_logits + beta * hier_logits
8. final_softmax = Softmax(name="final_softmax")(final_logits)
9. Outputs:
   final_softmax
   flat_softmax
   spoilage_sigmoid
   peeling_sigmoid

Loss:
1. final_softmax: CBLoss, weight 1.0
2. flat_softmax: CBLoss, weight 0.3
3. spoilage_sigmoid: BinaryCrossentropy, weight 0.2
4. peeling_sigmoid: masked BinaryCrossentropy, weight 0.2

Evaluation:
1. Metrics chính dùng final_softmax.
2. In thêm flat_softmax metrics để so sánh nội bộ.
3. In spoilage binary metrics.
4. In peeling binary metrics trên subset non-spoiled.
5. In giá trị beta sau mỗi run.
6. Lưu:
   H4_HSP_GarlicNet_ResidualFusion_CBLoss_results.csv

Mục tiêu:
Nếu beta học gần 0, hierarchical branch không giúp.
Nếu beta > 0 và final_softmax tốt hơn flat_softmax, hierarchical branch có đóng góp.
```

---

## 8. H5 — Full HSP-GarlicNet With Consistency Loss

Mục đích: thêm consistency loss giữa flat prediction và hierarchical prediction.

```text
Tạo notebook:
H5_HSP_GarlicNet_Full_Consistency_CBLoss.ipynb

Dựa trên H4.

Yêu cầu:
1. Giữ nguyên kiến trúc H4.
2. Thêm hierarchy consistency loss:
   KL(flat_softmax || hierarchical_probs)
   hoặc symmetric KL:
   0.5 * KL(flat_softmax || hierarchical_probs)
   + 0.5 * KL(hierarchical_probs || flat_softmax)
3. Dùng clip probability:
   clip(prob, 1e-7, 1.0)
4. lambda_consistency = 0.05 hoặc 0.1.
5. Nếu model.add_loss khó dùng, thêm hierarchical_probs làm output phụ và dùng custom loss.
6. Outputs chính:
   final_softmax
   flat_softmax
   spoilage_sigmoid
   peeling_sigmoid
   hierarchical_probs nếu cần

Loss:
1. final_softmax: CBLoss, weight 1.0
2. flat_softmax: CBLoss, weight 0.3
3. spoilage_sigmoid: BinaryCrossentropy, weight 0.2
4. peeling_sigmoid: masked BinaryCrossentropy, weight 0.2
5. consistency loss: weight 0.05 hoặc 0.1

Evaluation:
1. Metrics chính dùng final_softmax.
2. In flat_softmax metrics.
3. In hierarchical_probs metrics nếu có.
4. In beta value.
5. In binary spoilage và peeling metrics.
6. Lưu:
   H5_HSP_GarlicNet_Full_Consistency_CBLoss_results.csv

Mục tiêu:
So sánh H5 với H4. Nếu H5 thấp hơn H4 thì chọn H4 làm proposed.
```

---

## 9. H6 — HSP-GarlicNet With CategoricalCrossentropy

Mục đích: kiểm tra proposed có phụ thuộc CBLoss không.

```text
Tạo notebook:
H6_HSP_GarlicNet_CE.ipynb

Dựa trên version HSP tốt nhất giữa H4/H5.

Yêu cầu:
1. Giữ nguyên kiến trúc.
2. Thay CBLoss ở các output 3-class bằng CategoricalCrossentropy.
3. Binary outputs vẫn dùng BinaryCrossentropy.
4. Giữ nguyên pipeline, augmentation, callbacks, training loop 3 runs.
5. Lưu:
   H6_HSP_GarlicNet_CE_results.csv

Mục tiêu:
So sánh:
EfficientNetB4 + CE
EfficientNetB4 + CBLoss
HSP-GarlicNet + CE
HSP-GarlicNet + CBLoss
```

---

## 10. H7 — Lightweight HSP-GarlicNet

Mục đích: bản nhẹ nếu H4/H5 lỗi compile hoặc quá phức tạp.

```text
Tạo notebook:
H7_HSP_GarlicNet_Lightweight.ipynb

Yêu cầu:
1. Backbone EfficientNetB4 + GAP/GMP.
2. Flat logits.
3. Spoilage sigmoid.
4. Peeling sigmoid.
5. Hierarchical probability composition.
6. Residual fusion:
   final_logits = flat_logits + beta * log(hierarchical_probs)
7. Chỉ output final_softmax.
8. Chỉ dùng loss chính CBLoss hoặc CE cho final_softmax.
9. Không dùng auxiliary loss.
10. Không dùng consistency loss.
11. In beta sau khi train.
12. Chạy 3 runs.
13. Lưu:
   H7_HSP_GarlicNet_Lightweight_results.csv
```

---

## 11. H8 — Error Analysis and Diagnostics

Mục đích: phân tích lỗi để đưa vào báo cáo.

```text
Tạo notebook:
H8_HSP_GarlicNet_Error_Analysis.ipynb

Dựa trên checkpoint tốt nhất.

Yêu cầu:
1. Load model tốt nhất.
2. Predict trên test set.
3. Tạo confusion matrix.
4. Tạo classification report.
5. Lưu danh sách ảnh sai:
   true label
   predicted label
   confidence
   p_spoil nếu có
   p_partial nếu có
6. Tách lỗi:
   Fully bị nhầm Partially
   Partially bị nhầm Fully
   Spoiled bị nhầm Non-spoiled
   Non-spoiled bị nhầm Spoiled
7. Nếu có hierarchical heads, in:
   spoilage binary confusion matrix
   peeling binary confusion matrix trên non-spoiled subset
   mean p_spoil theo từng class
   mean p_partial theo Fully/Partially subset
8. Lưu:
   H8_error_cases.csv
   H8_hierarchy_diagnostics.csv
```

---

## 12. H9 — Final Ablation Summary

Mục đích: gộp tất cả kết quả thành bảng cuối.

```text
Tạo notebook:
H9_HSP_GarlicNet_Final_Ablation_Summary.ipynb

Yêu cầu:
1. Đọc CSV kết quả:
   V0
   V1
   H1
   H2
   H3
   H4
   H5
   H6 nếu có
   H7 nếu có
2. Tạo bảng tổng hợp:
   Model | Accuracy | Precision | Recall | F1-score | AUC Macro | AUC Weighted
3. Tạo bảng delta so với V0:
   Model | Delta Accuracy | Delta F1 | Delta AUC Macro | Delta AUC Weighted
4. Tạo bảng delta so với V1.
5. Tạo bảng component ablation:
   Model | GAP/GMP | Spoilage Head | Masked Peeling Head | Hierarchical Composition | Residual Fusion | Consistency Loss | Loss | F1
6. Highlight model tốt nhất theo:
   Overall F1
   AUC Macro
   Partially_Peeled_Garlic Recall/F1
   Spoiled_Garlic Recall/F1
   Std thấp nhất
7. Xuất:
   final_hsp_ablation_summary.csv
   final_hsp_ablation_summary.xlsx nếu hỗ trợ.
```

---

## 13. Bảng ablation mục tiêu

| Model | GAP/GMP | Spoilage Aux | Masked Peeling Aux | Hierarchical Prob | Residual Fusion | Consistency | Loss | F1-score |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| V0 EfficientNetB4 + CBLoss | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | CBLoss | 0.9362 |
| V1 GAP/GMP + CBLoss | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | CBLoss | 0.9372 |
| H1 Spoilage Aux | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | CBLoss | ... |
| H2 Spoilage + Peeling Aux | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | CBLoss | ... |
| H3 Hierarchical Probability | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | CBLoss | ... |
| H4 Residual Fusion | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | CBLoss | ... |
| H5 Full HSP-GarlicNet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CBLoss | ... |
| H6 HSP + CE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CE | ... |

---

## 14. Kết luận nếu kết quả tốt

```text
The proposed HSP-GarlicNet achieved the best overall performance compared with the strong EfficientNetB4 + CBLoss baseline. The improvement indicates that the label hierarchy of garlic quality classification provides useful inductive bias. By decomposing the task into spoilage detection and peeling-state recognition, the model learns more structured decision boundaries than a flat three-class classifier. The masked peeling auxiliary loss further prevents irrelevant supervision on spoiled samples, while the residual hierarchical fusion allows the model to benefit from hierarchical predictions without disrupting the strong flat classifier.
```

---

## 15. Kết luận nếu chỉ tăng nhẹ

```text
HSP-GarlicNet produced a marginal improvement over the EfficientNetB4 + CBLoss baseline. Although the overall gain is limited, the hierarchical heads provide a more interpretable decision structure by separating spoilage detection from peeling-state recognition. The results suggest that the strong EfficientNetB4 baseline already captures highly discriminative features, while the proposed hierarchy-aware design mainly improves the semantic organization and diagnostic capability of the classifier.
```

---

## 16. Kết luận nếu không vượt baseline

```text
Although HSP-GarlicNet introduces a hierarchy-aware classification structure, the experimental results show that it does not outperform the strong EfficientNetB4 + CBLoss baseline. This suggests that the dataset may already be sufficiently separable using flat discriminative features. Nevertheless, the ablation provides useful insight: adding task structure does not necessarily improve performance when the backbone is already strong and the dataset size is limited. Therefore, the final model should prioritize the simpler and more stable EfficientNetB4 + CBLoss or EfficientNetB4 + GAP/GMP + CBLoss configuration.
```

---

## 17. Câu phản biện trước hội đồng

```text
Dạ đúng, EfficientNetB4 không phải đóng góp mới của em. EfficientNetB4 được dùng làm backbone mạnh để trích xuất đặc trưng ảnh. Đóng góp của em nằm ở phần head phân loại có nhận thức cấu trúc nhãn của bài toán tỏi. Thay vì xem ba lớp Fully Peeled, Partially Peeled và Spoiled là ba lớp độc lập, em tách bài toán thành hai quyết định có ý nghĩa thực tế: thứ nhất là phát hiện tỏi hư hay không hư, thứ hai là nếu không hư thì phân biệt mức độ bóc vỏ fully hay partially. Đặc biệt, peeling loss chỉ được tính trên các ảnh không hư, tránh ép ảnh spoiled học một trạng thái bóc vỏ không phù hợp. Cuối cùng, em kết hợp xác suất phân cấp với classifier phẳng bằng residual logit fusion để giữ độ ổn định của baseline nhưng vẫn tận dụng được cấu trúc nhãn.
```

---

## 18. Prompt ngắn tạo ngay H4

```text
Dựa trên notebook EfficientNetB4 + CBLoss only đang chạy tốt, hãy tạo notebook H4_HSP_GarlicNet_ResidualFusion_CBLoss.

Giữ nguyên toàn bộ data pipeline, augmentation, training loop 3 runs, callbacks, metrics và fine-tuning strategy. Chỉ thay build_model và bổ sung nhãn phụ cần thiết.

Kiến trúc:
1. EfficientNetB4 include_top=False, pretrained ImageNet, fine-tuned Blocks 3-4-5-6-7.
2. Lấy final feature map.
3. GAP + GMP → concat → BN → Dropout → shared feature.
4. Flat head:
   flat_logits = Dense(3)
   flat_softmax = Softmax(flat_logits)
5. Spoilage head:
   p_spoil = sigmoid(Dense(1))
   label: Spoiled=1, Fully/Partially=0.
6. Masked peeling head:
   p_partial = sigmoid(Dense(1))
   label: Partially=1, Fully=0, Spoiled bị mask không tính loss.
7. Hierarchical probability:
   P_Fully = (1 - p_spoil) * (1 - p_partial)
   P_Partially = (1 - p_spoil) * p_partial
   P_Spoiled = p_spoil
8. hier_logits = log(clip(hier_probs, 1e-7, 1.0))
9. Learnable beta initialized at 0.0 or 0.1.
10. final_logits = flat_logits + beta * hier_logits
11. final_softmax = Softmax(final_logits)
12. Outputs:
    final_softmax, flat_softmax, spoilage_sigmoid, peeling_sigmoid
13. Loss:
    final_softmax: CBLoss weight 1.0
    flat_softmax: CBLoss weight 0.3
    spoilage_sigmoid: BinaryCrossentropy weight 0.2
    peeling_sigmoid: masked BinaryCrossentropy weight 0.2
14. Evaluation chính dùng final_softmax.
15. In thêm flat_softmax metrics, spoilage metrics, peeling metrics và beta value.
16. Lưu CSV kết quả, confusion matrix và classification report.
```
