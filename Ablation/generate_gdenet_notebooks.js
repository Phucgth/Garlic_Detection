const fs = require("fs");
const path = require("path");

const here = __dirname;
const basePath = path.join(here, "efficientnetb4-ablation-cbloss-only.ipynb");

function sourceLines(text) {
  const normalized = text.replace(/\r\n/g, "\n");
  return normalized.endsWith("\n")
    ? normalized.match(/.*\n/g) || []
    : normalized.split(/(?<=\n)/);
}

function pyBool(value) {
  return value ? "True" : "False";
}

function sanitizeModelName(name) {
  return name.replace(/[^A-Za-z0-9_]+/g, "_");
}

const variants = [
  {
    file: "V0_EfficientNetB4_CBLoss_Baseline.ipynb",
    resultsCsv: "V0_EfficientNetB4_CBLoss_Baseline_results.csv",
    key: "v0_efficientnetb4_b34567_cbloss_baseline",
    label: "V0 EfficientNetB4 fine-tuned + CBLoss baseline",
    short: "V0 EfficientNetB4 CBLoss",
    modelTitle: "EfficientNetB4 baseline classifier",
    lossType: "cbloss",
    useGapGmp: false,
    useCoverage: false,
    useDefect: false,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Features: final EfficientNetB4 semantic map -> GAP",
      "Classifier: BN -> Dense(256) -> Dropout -> Softmax",
      "Loss: Class-Balanced Focal Loss",
    ],
  },
  {
    file: "V1_EfficientNetB4_GAP_GMP_CBLoss.ipynb",
    resultsCsv: "V1_EfficientNetB4_GAP_GMP_CBLoss_results.csv",
    key: "v1_efficientnetb4_gap_gmp_cbloss",
    label: "V1 EfficientNetB4 GAP+GMP + CBLoss",
    short: "V1 GAP+GMP",
    modelTitle: "EfficientNetB4 GAP+GMP classifier",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: false,
    useDefect: false,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Features: final EfficientNetB4 semantic map -> GAP + GMP",
      "Classifier: BN -> Dense(256) -> Dropout -> Softmax",
      "Loss: Class-Balanced Focal Loss",
    ],
  },
  {
    file: "V2_EfficientNetB4_CoverageEvidence_CBLoss.ipynb",
    resultsCsv: "V2_EfficientNetB4_CoverageEvidence_CBLoss_results.csv",
    key: "v2_efficientnetb4_coverage_evidence_cbloss",
    label: "V2 EfficientNetB4 Coverage Evidence + CBLoss",
    short: "V2 Coverage Evidence",
    modelTitle: "EfficientNetB4 coverage evidence classifier",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: true,
    useDefect: false,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Coverage evidence: sigmoid coverage_map -> mean/max -> coverage logits",
      "Fusion: global logits + coverage_gate * coverage logits",
      "Loss: Class-Balanced Focal Loss on final_softmax",
    ],
  },
  {
    file: "V3_EfficientNetB4_PeakDefectEvidence_CBLoss.ipynb",
    resultsCsv: "V3_EfficientNetB4_PeakDefectEvidence_CBLoss_results.csv",
    key: "v3_efficientnetb4_peak_defect_evidence_cbloss",
    label: "V3 EfficientNetB4 Peak-Defect Evidence + CBLoss",
    short: "V3 Peak-Defect Evidence",
    modelTitle: "EfficientNetB4 peak-defect evidence classifier",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: false,
    useDefect: true,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Peak-defect evidence: sigmoid defect_map -> top-k/max -> defect logits",
      "Fusion: global logits + defect_gate * defect logits",
      "Loss: Class-Balanced Focal Loss on final_softmax",
    ],
  },
  {
    file: "V4_EfficientNetB4_DualEvidence_NoDiversity_CBLoss.ipynb",
    resultsCsv: "V4_EfficientNetB4_DualEvidence_NoDiversity_CBLoss_results.csv",
    key: "v4_efficientnetb4_dual_evidence_no_diversity_cbloss",
    label: "V4 EfficientNetB4 Dual Evidence no diversity + CBLoss",
    short: "V4 Dual Evidence",
    modelTitle: "EfficientNetB4 dual evidence classifier",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: true,
    useDefect: true,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Coverage evidence branch + Peak-defect evidence branch",
      "Evidence-gated logit fusion without diversity loss",
      "Loss: Class-Balanced Focal Loss on final_softmax",
    ],
  },
  {
    file: "V5_GDENet_EfficientNetB4_DualEvidence_Diversity_CBLoss.ipynb",
    resultsCsv: "V5_GDENet_results.csv",
    key: "v5_gdenet_dual_evidence_diversity_cbloss",
    label: "V5 GDE-Net dual evidence + diversity + CBLoss",
    short: "V5 GDE-Net",
    modelTitle: "GDE-Net: Garlic Dual-Evidence EfficientNet",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: true,
    useDefect: true,
    useDiversity: true,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Coverage evidence branch + Peak-defect evidence branch",
      "Evidence-gated logit fusion",
      "Diversity loss: lambda_div * mean(coverage_map * defect_map)",
      "Loss: Class-Balanced Focal Loss on final_softmax plus diversity add_loss",
    ],
  },
  {
    file: "V6_GDENet_Lightweight_NoAuxLoss.ipynb",
    resultsCsv: "V6_GDENet_Lightweight_NoAuxLoss_results.csv",
    key: "v6_gdenet_lightweight_no_aux_cbloss",
    label: "V6 GDE-Net lightweight no auxiliary loss + CBLoss",
    short: "V6 GDE-Net Lightweight",
    modelTitle: "Lightweight GDE-Net without auxiliary outputs",
    lossType: "cbloss",
    useGapGmp: true,
    useCoverage: true,
    useDefect: true,
    useDiversity: false,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Coverage evidence branch + Peak-defect evidence branch",
      "Evidence-gated logit fusion",
      "No auxiliary outputs, no custom training loop, no diversity loss",
      "Loss: Class-Balanced Focal Loss on final_softmax",
    ],
  },
  {
    file: "V7_GDENet_CategoricalCrossentropy.ipynb",
    resultsCsv: "V7_GDENet_CE_results.csv",
    key: "v7_gdenet_dual_evidence_diversity_ce",
    label: "V7 GDE-Net dual evidence + diversity + CategoricalCrossentropy",
    short: "V7 GDE-Net CE",
    modelTitle: "GDE-Net with CategoricalCrossentropy",
    lossType: "ce",
    useGapGmp: true,
    useCoverage: true,
    useDefect: true,
    useDiversity: true,
    summary: [
      "Backbone: EfficientNetB4 (unfreeze [3, 4, 5, 6, 7])",
      "Shared feature: Conv1x1(256) + BN",
      "Global branch: GAP + GMP -> global logits",
      "Coverage evidence branch + Peak-defect evidence branch",
      "Evidence-gated logit fusion with diversity add_loss",
      "Loss: CategoricalCrossentropy on final_softmax plus diversity add_loss",
    ],
  },
];

function markdownTitle(v) {
  return `# ${v.label}\n\n` +
    `Generated from \`efficientnetb4-ablation-cbloss-only.ipynb\` using the requirements in \`GDE_Net_prompt_versions.md\`.\n\n` +
    `This notebook keeps the original data pipeline, augmentation, 3-run training loop, callbacks, fine-tuning strategy, TTA inference, and metric reporting. Only the model head/loss variant is changed for ablation.\n`;
}

function configCell(v) {
  const archList = v.summary.map((s) => `    ${JSON.stringify(s)},`).join("\n");
  const lossName = v.lossType === "ce" ? "Categorical Crossentropy" : "Class-Balanced Focal Loss";
  const lossDescription = v.lossType === "ce"
    ? "Categorical Crossentropy (label_smoothing={LABEL_SMOOTHING})"
    : "Class-Balanced Focal Loss (gamma={FOCAL_GAMMA}, beta=0.9999)";

  return `# ============================================================================
# CELL 2: CONFIGURATION & HYPERPARAMETERS
# ============================================================================
# --- Experiment Identification ---
STRATEGY_KEY   = "${v.key}"
STRATEGY_LABEL = "${v.label}"
MODEL_VARIANT  = "${sanitizeModelName(v.key)}"

# --- Data Paths ---
DATA_DIR        = "/kaggle/input/datasets/giaphuc/dataset-0803/dataset_split_0803"
BASE_RESULT_DIR = f"/kaggle/working/report_EfficientNetB4/{STRATEGY_KEY}"
RESULTS_CSV_NAME = "${v.resultsCsv}"
os.makedirs(BASE_RESULT_DIR, exist_ok=True)

# --- Model Architecture ---
INPUT_SHAPE     = (380, 380, 3)       # EfficientNetB4 standard
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 8e-5
UNFREEZE_BLOCKS = [3, 4, 5, 6, 7]
DROPOUT_RATE    = 0.25
PATIENCE        = 12

# --- Optimization and generalization controls ---
WEIGHT_DECAY    = 1e-5
LABEL_SMOOTHING = 0.03
TTA_ROUNDS      = 4
TTA_INFER_BATCH = 8

# --- GDE-Net ablation switches ---
FEAT_DIM             = 256
SE_REDUCTION         = 8       # kept for API compatibility with the baseline builder
EVIDENCE_DIM         = 64
DEFECT_TOPK_RATIO    = 0.10
DIVERSITY_LAMBDA     = 0.03
USE_GAP_GMP          = ${pyBool(v.useGapGmp)}
USE_COVERAGE_BRANCH  = ${pyBool(v.useCoverage)}
USE_DEFECT_BRANCH    = ${pyBool(v.useDefect)}
USE_DIVERSITY_LOSS   = ${pyBool(v.useDiversity)}
USE_AUXILIARY_OUTPUTS = False  # kept False to preserve the baseline tf.data/training loop

# --- Loss ---
LOSS_TYPE       = "${v.lossType}"
FOCAL_GAMMA     = 2.0
LOSS_NAME       = "${lossName}"
LOSS_DESCRIPTION = f"${lossDescription}"

ARCHITECTURE_SUMMARY = [
${archList}
]

# --- Reproducibility ---
N_RUNS       = 3
RANDOM_SEEDS = [42, 123, 456]
AUTOTUNE     = tf.data.AUTOTUNE
tf.config.optimizer.set_jit(True)
all_runs_results = []

# --- Print Summary ---
print("=" * 60)
print("  EXPERIMENT CONFIGURATION")
print("=" * 60)
print(f"  Strategy    : {STRATEGY_LABEL}")
print(f"  Dataset     : {DATA_DIR.split('/')[-1]}")
print(f"  Input Shape : {INPUT_SHAPE}")
print(f"  Batch Size  : {BATCH_SIZE}")
print(f"  Epochs      : {EPOCHS} (patience={PATIENCE})")
print(f"  LR          : {LR} (CosineDecay -> 1e-6)")
print(f"  Optimizer   : AdamW (wd={WEIGHT_DECAY}, clipnorm=1.0)")
print(f"  Label smooth: {LABEL_SMOOTHING}")
print(f"  TTA rounds  : {TTA_ROUNDS}")
print(f"  TTA infer bs: {TTA_INFER_BATCH}")
print(f"  Unfreeze    : blocks {UNFREEZE_BLOCKS}")
print(f"  Runs        : {N_RUNS} x seeds {RANDOM_SEEDS}")
print("-" * 60)
print("  [Ablation] Architecture:")
for item in ARCHITECTURE_SUMMARY:
    print(f"             - {item}")
print(f"  [Loss]    {LOSS_DESCRIPTION}")
print("=" * 60)
`;
}

function architectureCell() {
  return `# ============================================================================
# CELL 3: GDE-NET ABLATION ARCHITECTURE (Keras 3 compatible)
# ============================================================================
# All versions keep the same EfficientNetB4 backbone and training protocol.
# The switches in Cell 2 control whether this becomes V0, V1, V2, V3, V4,
# V5, V6, or V7.
# ============================================================================


@tf.keras.utils.register_keras_serializable(package="GDE")
class CastToFloat32(Layer):
    """Cast tensors to float32 before logits/softmax under mixed precision."""
    def call(self, x):
        return tf.cast(x, tf.float32)


@tf.keras.utils.register_keras_serializable(package="GDE")
class TopKPooling2D(Layer):
    """Return mean(top-k spatial activations) and max activation for a map."""
    def __init__(self, k_ratio=0.10, **kwargs):
        super().__init__(**kwargs)
        self.k_ratio = float(k_ratio)
        self.k = None

    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        if h is None or w is None:
            raise ValueError("TopKPooling2D requires static spatial dimensions.")
        self.k = max(1, int(round(float(h * w) * self.k_ratio)))
        super().build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        flat = tf.reshape(x, [tf.shape(x)[0], -1])
        top_values = tf.math.top_k(flat, k=self.k, sorted=False).values
        topk_mean = tf.reduce_mean(top_values, axis=-1, keepdims=True)
        topk_max = tf.reduce_max(top_values, axis=-1, keepdims=True)
        return [topk_mean, topk_max]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 1), (input_shape[0], 1)]

    def get_config(self):
        cfg = super().get_config()
        cfg["k_ratio"] = self.k_ratio
        return cfg


@tf.keras.utils.register_keras_serializable(package="GDE")
class DiversityRegularizer(Layer):
    """Add lambda * mean(coverage_map * defect_map) as a model loss."""
    def __init__(self, lambda_div=0.03, **kwargs):
        super().__init__(**kwargs)
        self.lambda_div = float(lambda_div)

    def call(self, inputs):
        coverage_map, defect_map = inputs
        coverage_map = tf.cast(coverage_map, tf.float32)
        defect_map = tf.cast(defect_map, tf.float32)
        self.add_loss(self.lambda_div * tf.reduce_mean(coverage_map * defect_map))
        return coverage_map

    def get_config(self):
        cfg = super().get_config()
        cfg["lambda_div"] = self.lambda_div
        return cfg


@tf.keras.utils.register_keras_serializable(package="GDE")
class GatedLogitFusion(Layer):
    """Fuse global logits with one or two evidence-logit branches."""
    def __init__(self, mode="dual", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def call(self, inputs):
        if self.mode == "dual":
            global_logits, coverage_logits, defect_logits, gate = inputs
            global_logits = tf.cast(global_logits, tf.float32)
            coverage_logits = tf.cast(coverage_logits, tf.float32)
            defect_logits = tf.cast(defect_logits, tf.float32)
            gate = tf.cast(gate, tf.float32)
            alpha_cov = gate[:, 0:1]
            alpha_def = gate[:, 1:2]
            return global_logits + alpha_cov * coverage_logits + alpha_def * defect_logits

        global_logits, evidence_logits, alpha = inputs
        return (
            tf.cast(global_logits, tf.float32)
            + tf.cast(alpha, tf.float32) * tf.cast(evidence_logits, tf.float32)
        )

    def get_config(self):
        cfg = super().get_config()
        cfg["mode"] = self.mode
        return cfg


def _global_feature_head(feature_map, feat_dim, dropout_rate):
    """Global classifier feature used by all variants."""
    gap = GlobalAveragePooling2D(name="gap_global")(feature_map)
    if USE_GAP_GMP:
        gmp = GlobalMaxPooling2D(name="gmp_global")(feature_map)
        x = Concatenate(name="global_gap_gmp")([gap, gmp])
    else:
        x = gap

    x = BatchNormalization(name="head_bn")(x)
    x = Dense(feat_dim, activation="relu", kernel_regularizer=l2(1e-4), name="head_dense")(x)
    x = Dropout(dropout_rate, name="head_dropout")(x)
    return CastToFloat32(name="head_feature_f32")(x)


def _coverage_branch(shared_feature, num_classes):
    coverage_map = Conv2D(1, 1, padding="same", activation="sigmoid", name="coverage_map")(shared_feature)
    cov_mean = GlobalAveragePooling2D(name="coverage_mean")(coverage_map)
    cov_max = GlobalMaxPooling2D(name="coverage_max")(coverage_map)
    cov_stats = Concatenate(name="coverage_stats")([cov_mean, cov_max])
    cov_feature = Dense(EVIDENCE_DIM, activation="swish", name="coverage_feature")(cov_stats)
    cov_feature = CastToFloat32(name="coverage_feature_f32")(cov_feature)
    cov_logits = Dense(num_classes, dtype="float32", name="coverage_logits")(cov_feature)
    return coverage_map, cov_feature, cov_logits


def _defect_branch(shared_feature, num_classes):
    defect_map = Conv2D(1, 1, padding="same", activation="sigmoid", name="defect_map")(shared_feature)
    topk_mean, topk_max = TopKPooling2D(k_ratio=DEFECT_TOPK_RATIO, name="defect_topk_pool")(defect_map)
    def_stats = Concatenate(name="defect_stats")([topk_mean, topk_max])
    def_feature = Dense(EVIDENCE_DIM, activation="swish", name="defect_feature")(def_stats)
    def_feature = CastToFloat32(name="defect_feature_f32")(def_feature)
    def_logits = Dense(num_classes, dtype="float32", name="defect_logits")(def_feature)
    return defect_map, def_feature, def_logits


def build_mscaf_classifier(input_shape, num_classes, feat_dim=256,
                           se_reduction=8, dropout_rate=0.4):
    """Build the selected EfficientNetB4/GDE-Net ablation classifier."""
    backbone_base = EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape)
    inputs = Input(shape=input_shape, name="input_image")
    semantic_map = backbone_base(inputs)

    if USE_COVERAGE_BRANCH or USE_DEFECT_BRANCH:
        shared_feature = Conv2D(256, 1, padding="same", activation="swish", name="shared_reduce_conv")(semantic_map)
        shared_feature = BatchNormalization(name="shared_reduce_bn")(shared_feature)
    else:
        shared_feature = semantic_map

    global_feature = _global_feature_head(shared_feature, feat_dim, dropout_rate)
    global_logits = Dense(num_classes, dtype="float32", name="global_logits")(global_feature)

    coverage_map = coverage_feature = coverage_logits = None
    defect_map = defect_feature = defect_logits = None

    if USE_COVERAGE_BRANCH:
        coverage_map, coverage_feature, coverage_logits = _coverage_branch(shared_feature, num_classes)

    if USE_DEFECT_BRANCH:
        defect_map, defect_feature, defect_logits = _defect_branch(shared_feature, num_classes)

    if USE_DIVERSITY_LOSS and USE_COVERAGE_BRANCH and USE_DEFECT_BRANCH:
        coverage_map = DiversityRegularizer(
            lambda_div=DIVERSITY_LAMBDA,
            name="diversity_regularizer",
        )([coverage_map, defect_map])
        # Recompute coverage statistics so the add_loss layer stays connected
        # to the final model graph.
        cov_mean = GlobalAveragePooling2D(name="coverage_mean_div")(coverage_map)
        cov_max = GlobalMaxPooling2D(name="coverage_max_div")(coverage_map)
        cov_stats = Concatenate(name="coverage_stats_div")([cov_mean, cov_max])
        coverage_feature = Dense(EVIDENCE_DIM, activation="swish", name="coverage_feature_div")(cov_stats)
        coverage_feature = CastToFloat32(name="coverage_feature_div_f32")(coverage_feature)
        coverage_logits = Dense(num_classes, dtype="float32", name="coverage_logits_div")(coverage_feature)

    if USE_COVERAGE_BRANCH and USE_DEFECT_BRANCH:
        gate_input = Concatenate(name="evidence_gate_input")([
            global_feature, coverage_feature, defect_feature,
        ])
        gate = Dense(2, activation="sigmoid", dtype="float32", name="evidence_gate")(gate_input)
        final_logits = GatedLogitFusion(mode="dual", name="evidence_logit_fusion")([
            global_logits, coverage_logits, defect_logits, gate,
        ])
    elif USE_COVERAGE_BRANCH:
        alpha = Dense(1, activation="sigmoid", dtype="float32", name="coverage_gate")(global_feature)
        final_logits = GatedLogitFusion(mode="single", name="coverage_logit_fusion")([
            global_logits, coverage_logits, alpha,
        ])
    elif USE_DEFECT_BRANCH:
        alpha = Dense(1, activation="sigmoid", dtype="float32", name="defect_gate")(global_feature)
        final_logits = GatedLogitFusion(mode="single", name="defect_logit_fusion")([
            global_logits, defect_logits, alpha,
        ])
    else:
        final_logits = global_logits

    outputs = tf.keras.layers.Softmax(name="final_softmax", dtype="float32")(final_logits)
    model = Model(inputs=inputs, outputs=outputs, name=MODEL_VARIANT)
    return model, backbone_base


print("[OK] GDE-Net ablation architecture ready")
print(f"   - GAP+GMP: {USE_GAP_GMP}")
print(f"   - Coverage branch: {USE_COVERAGE_BRANCH}")
print(f"   - Defect branch: {USE_DEFECT_BRANCH}")
print(f"   - Diversity loss: {USE_DIVERSITY_LOSS} (lambda={DIVERSITY_LAMBDA})")
print(f"   - Auxiliary outputs: {USE_AUXILIARY_OUTPUTS} (disabled to keep baseline pipeline)")
`;
}

function compileCell() {
  return `# ============================================================================
# CELL 6: MODEL BUILDER (Functional API)
# ============================================================================
# Uses standard model.compile(loss=...) so Keras can track loss consistently.
# The dataset and 3-run loop are unchanged from the baseline notebook.
# ============================================================================


def apply_freeze_strategy(base_model, unfreeze_blocks):
    """Freeze backbone except specified blocks. Keep BN frozen."""
    base_model.trainable = False
    for layer in base_model.layers:
        for block_num in unfreeze_blocks:
            if layer.name.startswith(f"block{block_num}"):
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                break
    trainable = sum(1 for l in base_model.layers if l.trainable)
    total = len(base_model.layers)
    print(f"  Backbone: {trainable}/{total} layers trainable")


def build_and_compile_model(num_classes, samples_per_class, steps_per_epoch):
    """Build, freeze, and compile the selected ablation model."""
    model, backbone_base = build_mscaf_classifier(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        feat_dim=FEAT_DIM,
        se_reduction=SE_REDUCTION,
        dropout_rate=DROPOUT_RATE,
    )

    apply_freeze_strategy(backbone_base, UNFREEZE_BLOCKS)

    if LOSS_TYPE.lower() == "ce":
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
            name="categorical_crossentropy",
        )
    else:
        loss_fn = ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            gamma=FOCAL_GAMMA,
            beta=0.9999,
            label_smoothing=LABEL_SMOOTHING,
        )

    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    )

    auc_metric = tf.keras.metrics.AUC(
        name="auc_ovr",
        curve="ROC",
        multi_label=True,
        num_labels=num_classes,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy", auc_metric],
    )

    print(f"  Model params: {model.count_params():,}")
    if len(model.losses) > 0:
        print(f"  Extra model losses: {len(model.losses)} (e.g. diversity regularization)")
    return model


print("[OK] Model builder ready")
print("   - Keras loss tracking: enabled")
print("   - EarlyStopping on val_auc_ovr")
print("   - CosineDecay LR schedule")
`;
}

function customObjectsCell(original) {
  const markerStart = "custom_objects = {";
  const markerEnd = "model = load_model(";
  const start = original.indexOf(markerStart);
  const end = original.indexOf(markerEnd);
  if (start === -1 || end === -1) return original;
  const replacement = `custom_objects = {
    "CastToFloat32": CastToFloat32,
    "TopKPooling2D": TopKPooling2D,
    "DiversityRegularizer": DiversityRegularizer,
    "GatedLogitFusion": GatedLogitFusion,
    "ClassBalancedFocalLoss": ClassBalancedFocalLoss,
}
`;
  return original.slice(0, start) + replacement + original.slice(end);
}

function gradcamCell(original) {
  return original
    .replace(
      /# Prefer .* semantic\/local attention maps\n    for name in \[[^\n]+\]:/,
      "# Prefer the shared GDE feature map, then the EfficientNetB4 semantic map.\n    for name in ['shared_reduce_bn', 'shared_reduce_conv', 'top_activation', 'efficientnetb4']:"
    )
    .replace(/Grad-CAM .+?\(EfficientNetB4 backbone\)/g, "Grad-CAM - selected ablation model");
}

function evidenceVisualizationCell() {
  return `# ========== EVIDENCE MAP AND GATE VISUALIZATION ========== #
# This cell runs for GDE-Net variants that contain coverage_map and/or defect_map.
# It is skipped automatically for V0/V1.

evidence_layer_names = [
    name for name in ["coverage_map", "defect_map"]
    if name in [layer.name for layer in model.layers]
]
gate_layer_names = [
    name for name in ["coverage_gate", "defect_gate", "evidence_gate"]
    if name in [layer.name for layer in model.layers]
]

if len(evidence_layer_names) == 0:
    print("Evidence maps skipped: this ablation variant has no coverage/defect evidence maps.")
else:
    import matplotlib.cm as cm_lib

    def _normalize_map(m):
        m = np.asarray(m, dtype=np.float32)
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    def _overlay_map(orig_uint8, evidence_map, alpha=0.42):
        h, w = orig_uint8.shape[:2]
        resized = tf.image.resize(evidence_map[..., np.newaxis], [h, w]).numpy()[:, :, 0]
        resized = _normalize_map(resized)
        heat = (cm_lib.jet(resized)[:, :, :3] * 255).astype(np.uint8)
        return (orig_uint8 * (1.0 - alpha) + heat * alpha).astype(np.uint8), resized

    if "sample_paths" not in globals() or len(sample_paths) == 0:
        sample_paths, sample_labels, sample_indices = [], [], []
        for ci, cname in enumerate(class_names):
            idxs = np.where(y_true == ci)[0]
            if len(idxs) == 0:
                continue
            idx = int(idxs[0])
            sample_paths.append(os.path.join(test_dir, test_filenames[idx]))
            sample_labels.append(cname)
            sample_indices.append(idx)

    probe_outputs = [model.get_layer(name).output for name in evidence_layer_names + gate_layer_names]
    evidence_probe = Model(inputs=model.input, outputs=probe_outputs)

    n_samples = min(len(sample_paths), max(3, len(class_names)))
    n_cols = 1 + len(evidence_layer_names)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 3.3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row in range(n_samples):
        img_orig = load_img(sample_paths[row], target_size=INPUT_SHAPE[:2])
        img_arr = img_to_array(img_orig)
        orig_uint8 = np.clip(img_arr, 0, 255).astype(np.uint8)
        img_proc = efficientnet_preprocess(np.expand_dims(img_arr.copy(), 0))

        outputs = evidence_probe(img_proc, training=False)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        map_outputs = outputs[:len(evidence_layer_names)]

        idx = sample_indices[row]
        pred_idx = int(y_pred[idx])
        conf = float(pred_probs[idx][pred_idx])
        axes[row, 0].imshow(orig_uint8)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(
            f"True: {sample_labels[row]}\\nPred: {class_names[pred_idx]} ({conf:.2f})",
            fontsize=8,
            color="green" if y_true[idx] == pred_idx else "red",
        )

        for col, (layer_name, map_tensor) in enumerate(zip(evidence_layer_names, map_outputs), start=1):
            evidence_map = np.asarray(map_tensor[0, :, :, 0], dtype=np.float32)
            overlay, resized = _overlay_map(orig_uint8, evidence_map)
            axes[row, col].imshow(overlay)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{layer_name}\\nmean={resized.mean():.3f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "evidence_maps.png"), dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved evidence map visualization -> {RESULT_DIR}/evidence_maps.png")

    # Aggregate map/gate statistics by true class on the selected test set.
    stat_rows = []
    accum = {
        cname: {
            "coverage_activation": [],
            "defect_activation": [],
            "alpha_cov": [],
            "alpha_def": [],
        }
        for cname in class_names
    }

    for batch_x, batch_y in test_ds:
        outputs = evidence_probe(batch_x, training=False)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        map_outputs = outputs[:len(evidence_layer_names)]
        gate_outputs = outputs[len(evidence_layer_names):]
        y_batch = np.argmax(batch_y.numpy(), axis=1)

        for bi, yi in enumerate(y_batch):
            cname = class_names[int(yi)]
            for layer_name, map_tensor in zip(evidence_layer_names, map_outputs):
                val = float(np.mean(np.asarray(map_tensor[bi], dtype=np.float32)))
                if layer_name == "coverage_map":
                    accum[cname]["coverage_activation"].append(val)
                elif layer_name == "defect_map":
                    accum[cname]["defect_activation"].append(val)

            for gate_name, gate_tensor in zip(gate_layer_names, gate_outputs):
                gate_val = np.asarray(gate_tensor[bi], dtype=np.float32).reshape(-1)
                if gate_name == "coverage_gate":
                    accum[cname]["alpha_cov"].append(float(gate_val[0]))
                elif gate_name == "defect_gate":
                    accum[cname]["alpha_def"].append(float(gate_val[0]))
                elif gate_name == "evidence_gate":
                    if len(gate_val) > 0:
                        accum[cname]["alpha_cov"].append(float(gate_val[0]))
                    if len(gate_val) > 1:
                        accum[cname]["alpha_def"].append(float(gate_val[1]))

    for cname, vals in accum.items():
        row = {"class": cname}
        for metric_name, metric_vals in vals.items():
            row[metric_name] = float(np.mean(metric_vals)) if len(metric_vals) else np.nan
        stat_rows.append(row)

    evidence_stats_df = pd.DataFrame(stat_rows)
    evidence_stats_path = os.path.join(RESULT_DIR, "evidence_activation_by_class.csv")
    evidence_stats_df.to_csv(evidence_stats_path, index=False)
    print("\\nEvidence/gate statistics by class:")
    print(evidence_stats_df.to_string(index=False))
    print(f"Saved -> {evidence_stats_path}")
`;
}

function reportingCell(original) {
  const target = "overall_df.to_csv(os.path.join(BASE_RESULT_DIR, 'overall_metrics_summary.csv'), index=False)";
  const replacement = `${target}

variant_results_path = os.path.join(BASE_RESULT_DIR, RESULTS_CSV_NAME)
overall_df[['Metric', 'Mean +/- Std'] + run_cols].to_csv(variant_results_path, index=False)
print(f"Saved variant results -> {variant_results_path}")`;
  return original.includes(target) ? original.replace(target, replacement) : original;
}

function updateTextEverywhere(nb, v) {
  for (const cell of nb.cells) {
    if (!Array.isArray(cell.source)) continue;
    let src = cell.source.join("");
    src = src
      .replace(/EfficientNetB4 fine-tuned \+ CBLoss only/g, v.label)
      .replace(/MS-CAF \(EfficientNetB4 backbone\) \+ MS-CAF v2/g, v.modelTitle)
      .replace(/MS-CAF \(EfficientNetB4 backbone\)/g, v.label)
      .replace(/MS-CAF v2/g, v.short)
      .replace(/MS-CAF/g, v.short)
      .replace(/mscaf/g, "gdenet_ablation");
    cell.source = sourceLines(src);
  }
}

function makeNotebook(v) {
  const nb = JSON.parse(JSON.stringify(baseNotebook));
  updateTextEverywhere(nb, v);

  nb.cells[0].source = sourceLines(markdownTitle(v));
  nb.cells[2].source = sourceLines(configCell(v));
  nb.cells[3].source = sourceLines(architectureCell(v));
  nb.cells[6].source = sourceLines(compileCell(v));
  nb.cells[9].source = sourceLines(reportingCell(nb.cells[9].source.join("")));
  nb.cells[16].source = sourceLines(customObjectsCell(nb.cells[16].source.join("")));
  nb.cells[17].source = sourceLines(gradcamCell(nb.cells[17].source.join("")));
  nb.cells[19].source = sourceLines(evidenceVisualizationCell());

  for (const cell of nb.cells) {
    if ("execution_count" in cell) cell.execution_count = null;
    if ("outputs" in cell) cell.outputs = [];
  }
  return nb;
}

function summaryNotebook() {
  const nb = {
    cells: [
      {
        cell_type: "markdown",
        metadata: {},
        source: sourceLines(`# V9 GDE-Net Final Ablation Summary\n\nReads available GDE-Net ablation result CSV files from Kaggle working output and builds the final comparison tables.\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`import os\nimport glob\nimport numpy as np\nimport pandas as pd\n\nBASE_REPORT_ROOT = "/kaggle/working/report_EfficientNetB4"\nOUT_DIR = os.path.join(BASE_REPORT_ROOT, "final_gdenet_ablation_summary")\nos.makedirs(OUT_DIR, exist_ok=True)\nprint(f"Reading reports from: {BASE_REPORT_ROOT}")\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`strategy_files = sorted(glob.glob(os.path.join(BASE_REPORT_ROOT, "*", "strategy_summary.csv")))\nif not strategy_files:\n    raise FileNotFoundError("No strategy_summary.csv files found. Run the ablation notebooks first.")\n\nrun_frames = []\nfor fpath in strategy_files:\n    df = pd.read_csv(fpath)\n    df["source_file"] = fpath\n    run_frames.append(df)\n\nruns_df = pd.concat(run_frames, ignore_index=True)\nprint(f"Loaded {len(runs_df)} run rows from {len(strategy_files)} strategy summaries.")\nruns_df.head()\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_macro", "auc_weighted"]\nsummary_rows = []\nfor (strategy_key, strategy_label), g in runs_df.groupby(["strategy_key", "strategy_label"], dropna=False):\n    row = {\n        "Model": strategy_label,\n        "strategy_key": strategy_key,\n        "n_runs": len(g),\n    }\n    for col in metric_cols:\n        vals = g[col].astype(float).values\n        row[col] = float(np.nanmean(vals))\n        row[f"{col}_std"] = float(np.nanstd(vals))\n        row[f"{col}_mean_std"] = f"{np.nanmean(vals):.4f} +/- {np.nanstd(vals):.4f}"\n    summary_rows.append(row)\n\noverall_summary = pd.DataFrame(summary_rows).sort_values("f1_score", ascending=False)\noverall_summary.to_csv(os.path.join(OUT_DIR, "final_ablation_summary.csv"), index=False)\nprint("Overall summary:")\nprint(overall_summary[["Model", "accuracy_mean_std", "precision_mean_std", "recall_mean_std", "f1_score_mean_std", "auc_macro_mean_std", "auc_weighted_mean_std"]].to_string(index=False))\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`baseline_candidates = overall_summary[overall_summary["strategy_key"].str.contains("v0|cbloss_baseline|cbloss_only", case=False, na=False)]\nif len(baseline_candidates) == 0:\n    baseline = overall_summary.iloc[-1]\n    print("Baseline key not found; using the last row as fallback baseline.")\nelse:\n    baseline = baseline_candidates.iloc[0]\n\nbaseline_metrics = {col: baseline[col] for col in metric_cols}\ndelta_df = overall_summary[["Model", "strategy_key"]].copy()\nfor col in metric_cols:\n    delta_df[f"delta_{col}"] = overall_summary[col] - baseline_metrics[col]\n\ndelta_df.to_csv(os.path.join(OUT_DIR, "final_ablation_delta_vs_baseline.csv"), index=False)\nprint(f"Baseline: {baseline['Model']}")\nprint(delta_df.to_string(index=False))\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`def component_flags(strategy_key):\n    key = str(strategy_key).lower()\n    return {\n        "GAP+GMP": "v1" in key or "gap_gmp" in key or "gdenet" in key or "evidence" in key,\n        "Coverage": "coverage" in key or "dual" in key or "gdenet" in key,\n        "Defect": "defect" in key or "dual" in key or "gdenet" in key,\n        "Gate": "coverage" in key or "defect" in key or "dual" in key or "gdenet" in key,\n        "Aux Loss": False,\n        "Diversity": "diversity" in key,\n        "Loss": "CE" if key.endswith("_ce") or "categoricalcrossentropy" in key else "CBLoss",\n    }\n\ncomponent_rows = []\nfor _, row in overall_summary.iterrows():\n    flags = component_flags(row["strategy_key"])\n    flags.update({"Model": row["Model"], "F1": row["f1_score"]})\n    component_rows.append(flags)\n\ncomponent_df = pd.DataFrame(component_rows)\ncomponent_df.to_csv(os.path.join(OUT_DIR, "component_ablation_table.csv"), index=False)\nprint(component_df.to_string(index=False))\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`per_class_files = sorted(glob.glob(os.path.join(BASE_REPORT_ROOT, "*", "per_class_metrics_summary.csv")))\nper_class_frames = []\nfor fpath in per_class_files:\n    strategy_key = os.path.basename(os.path.dirname(fpath))\n    df = pd.read_csv(fpath)\n    df["strategy_key"] = strategy_key\n    per_class_frames.append(df)\n\nif per_class_frames:\n    per_class_df = pd.concat(per_class_frames, ignore_index=True)\n    per_class_df.to_csv(os.path.join(OUT_DIR, "final_per_class_ablation_summary.csv"), index=False)\n    print(f"Saved per-class summary rows: {len(per_class_df)}")\nelse:\n    print("No per_class_metrics_summary.csv files found yet.")\n`),
      },
      {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(`xlsx_path = os.path.join(OUT_DIR, "final_ablation_summary.xlsx")\ntry:\n    with pd.ExcelWriter(xlsx_path) as writer:\n        overall_summary.to_excel(writer, sheet_name="overall", index=False)\n        delta_df.to_excel(writer, sheet_name="delta_vs_baseline", index=False)\n        component_df.to_excel(writer, sheet_name="components", index=False)\n        if "per_class_df" in globals():\n            per_class_df.to_excel(writer, sheet_name="per_class", index=False)\n    print(f"Saved workbook -> {xlsx_path}")\nexcept Exception as exc:\n    print(f"Excel export skipped: {exc}")\n`),
      },
    ],
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.x",
      },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
  return nb;
}

const baseNotebook = JSON.parse(fs.readFileSync(basePath, "utf8"));

for (const variant of variants) {
  const nb = makeNotebook(variant);
  const outPath = path.join(here, variant.file);
  fs.writeFileSync(outPath, JSON.stringify(nb, null, 1), "utf8");
  console.log(`Wrote ${outPath}`);
}

const v9Path = path.join(here, "V9_GDENet_Final_Ablation_Summary.ipynb");
fs.writeFileSync(v9Path, JSON.stringify(summaryNotebook(), null, 1), "utf8");
console.log(`Wrote ${v9Path}`);
