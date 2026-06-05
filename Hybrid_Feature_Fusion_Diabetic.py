# %% [markdown]
# # Hybrid Feature Fusion – Diabetic Retinopathy Dataset
# ## Using Pre-trained DDR CNN Weights (No Re-training)
# 
# This notebook reuses the CNN trained on DDR for feature extraction on Diabetic Retinopathy Dataset.
# Only the path and save locations need to be changed.

# %% [markdown]
# ## Step 1 – Imports

# %%
import os, time, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ── GPU setup ─────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('TF GPUs :', gpus)
print('PyTorch CUDA:', torch.cuda.is_available())

GLOBAL_START = time.time()

# %% [markdown]
# ## Step 2 – Configure Paths

# %%
# ── CHANGE THESE TWO PATHS ONLY ──────────────────────────────────────────
DATASET_ROOT = r'D:/DR Datasets/Diabetic Retinopathy Dataset/'          # grade subfolders 0-4
CNN_WEIGHTS  = r'D:/Implementation/DR Code/cnn_feature_extractor.h5'
SAVE_DIR     = r'D:/Implementation/DR Code/Diabetic Retinopathy Dataset/'

os.makedirs(SAVE_DIR, exist_ok=True)

N_CLASSES  = 5
DR_CLASSES = ['No DR (0)', 'Mild (1)', 'Moderate (2)',
              'Severe (3)', 'Proliferative (4)']

print('Dataset :', DATASET_ROOT)
print('CNN     :', CNN_WEIGHTS)
print('Save dir:', SAVE_DIR)

# %% [markdown]
# ## Step 3 – Load Image Paths from Grade Folders (0–4)

# %%
# path of dataset file
import os
dataset = "D:/DR Datasets/Diabetic Retinopathy Dataset/"
print(os.listdir(dataset))

# %%
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

image_paths, labels = [], []

for grade in range(5):
    grade_dir = os.path.join(DATASET_ROOT, str(grade))
    if not os.path.isdir(grade_dir):
        print(f'WARNING: folder not found → {grade_dir}')
        continue
    files = sorted([
        f for f in os.listdir(grade_dir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])
    for fname in files:
        image_paths.append(os.path.join(grade_dir, fname))
        labels.append(grade)
    print(f'  Grade {grade}: {len(files):5d} images')

df = pd.DataFrame({'image_path': image_paths, 'label': labels})
print(f'\nTotal: {len(df)} images')
print(df['label'].value_counts().sort_index())

# %% [markdown]
# ## Step 4 – Class Distribution

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the dataset directory (assuming 'dataset' is defined elsewhere in your code)
# dataset = 'path/to/dataset'  # Uncomment and update this if needed

# List folder names in the dataset directory
folder_names = [f for f in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, f))]

# Calculate counts for each folder
counts = []
for folder in folder_names:
    folder_path = os.path.join(dataset, folder)
    num_files = len(os.listdir(folder_path))  # Count the number of files in each folder
    counts.append(num_files)

# Create a custom color palette with blue, orange, yellow, and green
custom_palette = sns.color_palette(["#6495ED", "#FF8C00", "#FFD700", "#32CD32", "#228B22"])

# Create a bar plot for class distribution
plt.figure(figsize=(9, 6))

# Use Seaborn for a bar plot, applying the custom palette
ax = sns.barplot(x=np.arange(len(folder_names)), y=counts, palette=custom_palette, width=0.5)

# Show gridlines on the y-axis
# Show gridlines on the y-axis, ensure gridlines are behind bars
ax.grid(True, which='major', axis='y', color='gray', linewidth=0.7, zorder=0)
ax.set_yticks(np.arange(0, max(counts) + 20, step=500))  # Adjust the step size as needed
ax.tick_params(axis='y', which='both', length=0)  # Hides the tick marks without removing the ticks
ax.set_yticklabels([])  # Remove y-axis tick labels
ax.set_facecolor('white')

# Remove x and y axis labels and ticks
ax.set(xticks=[])
ax.set_xlabel('Classes', fontsize=14, fontweight='bold')

# Display the counts on top of the bars
for i, count in enumerate(counts):
    ax.text(i, count + (max(counts) * 0.02), str(count), ha='center', va='center', fontweight='bold', fontsize=12, color='black')  # On top

# Create a legend for the classes below the plot
handles = [plt.Rectangle((0, 0), 1, 1, color=ax.patches[i].get_facecolor()) for i in range(len(folder_names))]
plt.legend(handles, folder_names, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(folder_names), prop={'weight': 'bold', 'size' : 14})

# Set tighter layout to prevent cutoff
plt.tight_layout()

# Display the plot
plt.show()


# %% [markdown]
# ## Step 5 – Stratified Train / Val / Test Split (70 / 10 / 20)

# %%
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.667, stratify=temp_df['label'], random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f'Train : {len(train_df)} images')
print(f'Val   : {len(val_df)} images')
print(f'Test  : {len(test_df)} images')

# %% [markdown]
# ## Step 6 – CLAHE Preprocessing (identical to DDR)

# %%
from concurrent.futures import ThreadPoolExecutor

IMG_SIZE = 224
_clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # local instance per thread
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img


def load_images_parallel(paths, desc='Loading'):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as exe:
        imgs = list(exe.map(preprocess_image, paths))
    arr = np.array(imgs, dtype=np.float32)
    print(f'  {desc}: {arr.shape}  ({time.time()-t0:.1f}s)')
    return arr


print('Loading images...')
X_train_img = load_images_parallel(train_df['image_path'].tolist(), 'Train')
X_val_img   = load_images_parallel(val_df['image_path'].tolist(),   'Val')
X_test_img  = load_images_parallel(test_df['image_path'].tolist(),  'Test')
print('Done.')

# %% [markdown]
# ## Step 7 – Load Pre-trained CNN and Inspect Layers
# 
# The CNN trained on DDR is loaded. We inspect all layer names to identify the correct 256-d Dense feature layer before building the extractor.

# %%
# ── Load saved DDR CNN ────────────────────────────────────────────────────
cnn_model = load_model(CNN_WEIGHTS)
print('CNN loaded successfully\n')

# ── Force-build by running a dummy input ──────────────────────────────────
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = cnn_model(dummy, training=False)
print('Model built. Layer list:\n')

# ── Print all layers (no output_shape — avoids AttributeError) ────────────
for i, layer in enumerate(cnn_model.layers):
    print(f'[{i:2d}]  {type(layer).__name__:25s}  name: {layer.name}')

# %% [markdown]
# ## Step 8 – Build CNN Feature Extractor
# 
# From the layer list printed above, find the **Dense layer with 256 units** — it is the second-to-last layer (before Dropout and the final softmax).
# Set `TARGET_LAYER_NAME` to its exact name.

# %%
# ── Set this to the Dense-256 layer name from the printout above ──────────
# Typically 'dense' or 'dense_1' — NOT the last Dense (which is the 5-class softmax)
TARGET_LAYER_NAME = 'dense_8'   # ← adjust if needed based on layer printout

# ── Rebuild as Functional model by re-wiring layers ──────────────────────
# This approach works reliably for Sequential models saved with .h5
inp      = tf.keras.Input(shape=(224, 224, 3))
x        = inp
feat_out = None

for layer in cnn_model.layers:
    x = layer(x)
    if layer.name == TARGET_LAYER_NAME:
        feat_out = x
        print(f'Feature output captured at layer: {layer.name}')

if feat_out is None:
    raise ValueError(
        f'Layer "{TARGET_LAYER_NAME}" not found. '
        f'Check the name from Step 7 printout.'
    )

cnn_feature_model = tf.keras.Model(inputs=inp, outputs=feat_out)
print('Feature extractor output shape:', cnn_feature_model.output_shape)
# Should print: (None, 256)

# %% [markdown]
# ## Step 9 – Extract CNN Features (Train / Val / Test)

# %%
print('Extracting CNN features...')
t0 = time.time()

train_cnn_features = cnn_feature_model.predict(X_train_img, batch_size=64, verbose=1)
val_cnn_features   = cnn_feature_model.predict(X_val_img,   batch_size=64, verbose=1)
test_cnn_features  = cnn_feature_model.predict(X_test_img,  batch_size=64, verbose=1)

print(f'\nCNN extraction time: {time.time()-t0:.1f}s')
print('Train:', train_cnn_features.shape)   # (N, 256)
print('Val  :', val_cnn_features.shape)
print('Test :', test_cnn_features.shape)

# %% [markdown]
# ## Step 10 – DenseNet121 Feature Extractor (ImageNet weights, frozen)

# %%
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

base         = DenseNet121(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
base.trainable = False
densenet_model = Model(base.input, GlobalAveragePooling2D()(base.output))
print('DenseNet output shape:', densenet_model.output_shape)  # (None, 1024)

print('Extracting DenseNet features...')
t0 = time.time()

train_dense_features = densenet_model.predict(X_train_img, batch_size=64, verbose=1)
val_dense_features   = densenet_model.predict(X_val_img,   batch_size=64, verbose=1)
test_dense_features  = densenet_model.predict(X_test_img,  batch_size=64, verbose=1)

print(f'DenseNet extraction time: {time.time()-t0:.1f}s')
print('Train:', train_dense_features.shape)   # (N, 1024)

# %% [markdown]
# ## Step 11 – ViT Feature Extractor (Batched Inference)

# %%
from transformers import ViTModel, ViTImageProcessor

vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model_hf  = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model_hf.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit_model_hf = vit_model_hf.to(device)
print('ViT device:', device)


def extract_vit_features_batched(img_array, batch_size=64):
    """Extract ViT CLS-token embeddings in batches. Returns (N, 768)."""

    all_feats = []
    n = len(img_array)
    for start in range(0, n, batch_size):
        batch    = img_array[start:start + batch_size]
        pil_imgs = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in batch]
        inputs   = vit_processor(images=pil_imgs, return_tensors='pt')
        inputs   = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = vit_model_hf(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_feats.append(cls)
        if start % (batch_size * 5) == 0:
            print(f'  ViT: {min(start+batch_size, n)}/{n} images done')
    return np.concatenate(all_feats, axis=0)


print('Extracting ViT features...')
t0 = time.time()

train_vit_features = extract_vit_features_batched(X_train_img, batch_size=64)
val_vit_features   = extract_vit_features_batched(X_val_img,   batch_size=64)
test_vit_features  = extract_vit_features_batched(X_test_img,  batch_size=64)

print(f'ViT extraction time: {time.time()-t0:.1f}s')
print('Train:', train_vit_features.shape)   # (N, 768)

# %% [markdown]
# ## Step 12 – Feature Fusion (CNN 256 + DenseNet 1024 + ViT 768 = 2048)

# %%
train_fused = np.concatenate([train_cnn_features, train_dense_features, train_vit_features], axis=1)
val_fused   = np.concatenate([val_cnn_features,   val_dense_features,   val_vit_features],   axis=1)
test_fused  = np.concatenate([test_cnn_features,  test_dense_features,  test_vit_features],  axis=1)

print('Fused shapes:')
print('  Train:', train_fused.shape)   # (N, 2048)
print('  Val  :', val_fused.shape)
print('  Test :', test_fused.shape)

# %%
np.savez(os.path.join(SAVE_DIR, 'fused_features.npz'), X_train=train_fused, X_val=val_fused, X_test=test_fused, y_train=train_df['label'].values, y_val=val_df['label'].values, y_test=test_df['label'].values)

# %% [markdown]
# ## Step 13 – Standardisation and SMOTE

# %%
# ── Scale ─────────────────────────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(train_fused)   # fit on train only
X_val_scaled   = scaler.transform(val_fused)
X_test_scaled  = scaler.transform(test_fused)

# ── SMOTE ─────────────────────────────────────────────────────────────────
# k_neighbors must be < smallest class count to avoid errors on small datasets
min_class_count = train_df['label'].value_counts().min()
k_neighbors     = max(1, min(5, min_class_count - 1))
print(f'SMOTE k_neighbors={k_neighbors}  (min class size in train={min_class_count})')

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, train_df['label'].values
)

print('After SMOTE:', X_train_smote.shape)
print('Class counts after SMOTE:', np.bincount(y_train_smote))

# %% [markdown]
# ## Step 14 – MOEA/D Feature Selection (Random Forest evaluator)
# 
# Uses **Random Forest** (`n_estimators=50`) as the fast proxy classifier inside MOEA/D to reduce features from **2048 → selected subset**.
# 
# Key speed optimisations:
# - **Random Forest** instead of LightGBM — lower per-fit overhead at small feature counts
# - **Evaluation caching** (`mask.tobytes()` key) — avoids re-evaluating duplicate solutions
# - **n_partitions=8** (9 weight vectors) instead of 10 — smaller population
# - **10 generations** instead of 15 — sufficient for convergence on 2048 features
# - **ElementwiseProblem** — cleaner single-solution interface
# 
# Expected runtime on Messidor (~1200 images, CPU): **5–12 minutes**.

# %%
from sklearn.ensemble import RandomForestClassifier
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# ── Cache evaluated masks to avoid re-computing identical solutions ──
_eval_cache = {}

def _eval_mask(mask_bytes, X_tr, y_tr, X_v, y_v, mask):
    if mask_bytes in _eval_cache:
        return _eval_cache[mask_bytes]
    clf_rf = RandomForestClassifier(
        n_estimators=50,       # fast enough for feature selection proxy
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    clf_rf.fit(X_tr[:, mask], y_tr)
    preds  = clf_rf.predict(X_v[:, mask])
    error  = 1.0 - (preds == y_v).mean()
    feat_r = mask.sum() / len(mask)
    result = [error, feat_r]
    _eval_cache[mask_bytes] = result
    return result


class FeatureSelectionProblem(ElementwiseProblem):
    """Element-wise problem: one solution evaluated per call — easier to cache."""
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__(
            n_var=X_train.shape[1],
            n_obj=2,
            n_ieq_constr=0,
            xl=0.0, xu=1.0
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val

    def _evaluate(self, x, out, *args, **kwargs):
        mask = (x > 0.5).astype(bool)
        if mask.sum() == 0:
            out['F'] = [1.0, 1.0]
            return
        key = mask.tobytes()
        out['F'] = _eval_mask(
            key,
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            mask
        )


problem = FeatureSelectionProblem(
    X_train_smote, y_train_smote,
    X_val_scaled,  val_df['label'].values
)

# n_partitions=49 → 50 weight vectors (population size = 50)
ref_dirs  = get_reference_directions('uniform', 2, n_partitions=49)
algorithm = MOEAD(
    ref_dirs=ref_dirs,
    n_neighbors=10,            # ~20% of population for neighbourhood mating
    prob_neighbor_mating=0.7
)

print(f'Population size: {len(ref_dirs)} | Running MOEA/D with Random Forest (50 generations)...')
t0 = time.time()
result = minimize(
    problem, algorithm,
    termination=('n_gen', 50),
    seed=42, verbose=True
)
print(f'MOEA/D time: {time.time()-t0:.1f}s')
print(f'Unique evaluations (cache size): {len(_eval_cache)}')

# ── Extract best feature subset (lowest classification error) ──────────────
solutions  = result.X
objectives = result.F

# Convert continuous → binary mask
best_idx          = np.argmin(objectives[:, 0])
best_mask         = (solutions[best_idx] > 0.5).astype(bool)
selected_features = np.where(best_mask)[0]
print(f'Selected {len(selected_features)} / {train_fused.shape[1]} features')

np.save(os.path.join(SAVE_DIR, 'Messidor_selected_features.npy'), selected_features)

X_train_selected = X_train_smote[:, selected_features]
X_val_selected   = X_val_scaled[:,  selected_features]
X_test_selected  = X_test_scaled[:, selected_features]
print('Selected feature shapes:')
print('  Train:', X_train_selected.shape)
print('  Val  :', X_val_selected.shape)
print('  Test :', X_test_selected.shape)


# %%
np.savez(os.path.join(SAVE_DIR, 'selected_features.npz'), X_train=X_train_selected, X_val=X_val_selected, X_test=X_test_selected, y_train=y_train_smote, y_val=val_df['label'].values, y_test=test_df['label'].values)

# %% [markdown]
# ## Step 15 – LightGBM Classifier Training

# %%
import lightgbm as lgb
clf = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=5,
    boosting_type='gbdt',
    n_estimators=5000,
    learning_rate=0.005,
    max_depth=12,
    num_leaves=128,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print('Training LightGBM...')
t0 = time.time()
clf.fit(X_train_selected, y_train_smote)
print(f'LightGBM training time: {time.time()-t0:.1f}s')

clf.booster_.save_model(os.path.join(SAVE_DIR, 'lightgbm_Messidor.txt'))

preds = clf.predict(X_test_selected)
probs = clf.predict_proba(X_test_selected)
print('Predictions done. Shape:', preds.shape)

# %% [markdown]
# ## Section 12 – Performance Evaluation (Multiclass)
# 
# A comprehensive set of classification metrics is computed on the test set:
# 
# | Metric | Description |
# |--------|-------------|
# | **Accuracy** | Overall correct predictions |
# | **Macro Precision / Recall / F1** | Unweighted average across all 5 DR grades — treats each grade equally |
# | **Weighted F1** | Average weighted by class frequency — reflects real-world performance |
# | **Quadratic Weighted Kappa (QWK)** | Standard metric for ordinal DR grading; penalises predictions far from the true grade more heavily than adjacent misclassifications |
# | **AUC (OvR)** | Area under the multiclass ROC curve (one-vs-rest) |
# | **PR-AUC** | Area under the precision-recall curve — more informative than AUC under class imbalance |

# %%
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    precision_recall_curve,
    roc_curve
)

from sklearn.preprocessing import label_binarize

from sklearn.calibration import calibration_curve

from sklearn.metrics import brier_score_loss

# %% [markdown]
# ### 12a – Scalar Metrics Computation

# %%
accuracy = accuracy_score(
    test_df['label'],
    preds
)

precision = precision_score(
    test_df['label'],
    preds,
    average='macro'
)

recall = recall_score(
    test_df['label'],
    preds,
    average='macro'
)

f1 = f1_score(
    test_df['label'],
    preds,
    average='macro'
)

macro_f1 = f1_score(
    test_df['label'],
    preds,
    average='macro'
)

weighted_f1 = f1_score(
    test_df['label'],
    preds,
    average='weighted'
)

qwk = cohen_kappa_score(
    test_df['label'],
    preds,
    weights='quadratic'
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)
print("Quadratic Weighted Kappa:", qwk)

# %% [markdown]
# ### 12b – Per-Class Recall
# 
# Per-class recall reveals which DR grades are hardest to detect. Clinically, **high recall for Grades 3 and 4** is critical — missing a severe or proliferative DR case can lead to preventable blindness.

# %%
per_class_recall = recall_score(
    test_df['label'],
    preds,
    average=None
)

print("Per-Class Recall:")
print(per_class_recall)

# %% [markdown]
# ### 12c – Confusion Matrix (Raw Counts)

# %%
cm = confusion_matrix(
    test_df['label'],
    preds
)

print(cm)

# %% [markdown]
# ### 12d – Per-Class Sensitivity and Specificity
# 
# Sensitivity (true positive rate) and specificity (true negative rate) are computed for each DR grade using a one-vs-rest decomposition of the confusion matrix. These are the primary clinical performance metrics reported in ophthalmology literature.

# %%
sensitivity = []
specificity = []

for i in range(len(cm)):

    TP = cm[i, i]

    FN = np.sum(cm[i, :]) - TP

    FP = np.sum(cm[:, i]) - TP

    TN = np.sum(cm) - (TP + FN + FP)

    sens = TP / (TP + FN + 1e-10)

    spec = TN / (TN + FP + 1e-10)

    sensitivity.append(sens)
    specificity.append(spec)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

print("Average Sensitivity:",
      np.mean(sensitivity))

print("Average Specificity:",
      np.mean(specificity))

# %% [markdown]
# ### 12e – Binarise Labels for Multiclass AUC/PR-AUC

# %%
y_test_bin = label_binarize(
    test_df['label'],
    classes=[0,1,2,3,4]
)

# %% [markdown]
# ### 12f – Multiclass AUC (One-vs-Rest)

# %%
auc_score = roc_auc_score(
    y_test_bin,
    probs,
    multi_class='ovr'
)

print("AUC:", auc_score)

# %% [markdown]
# ### 12g – Macro PR-AUC

# %%
pr_auc = average_precision_score(
    y_test_bin,
    probs,
    average='macro'
)

print("PR-AUC:", pr_auc)

# %% [markdown]
# ### 12h – Expected Calibration Error (ECE)
# 
# **ECE** measures how well the model's predicted confidence scores match its actual accuracy. A perfectly calibrated model that predicts 80% confidence should be correct 80% of the time.
# 
# ECE is computed by bucketing predictions into 10 confidence bins and taking the weighted average of |accuracy − confidence| per bin. Lower ECE = better calibration.

# %%
def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10
):

    confidences = np.max(y_prob, axis=1)

    predictions = np.argmax(y_prob, axis=1)

    accuracies = predictions == y_true

    bin_boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        mask = (
            (confidences > bin_boundaries[i]) &
            (confidences <= bin_boundaries[i+1])
        )

        if np.sum(mask) > 0:

            bin_acc = np.mean(
                accuracies[mask]
            )

            bin_conf = np.mean(
                confidences[mask]
            )

            ece += (
                np.abs(bin_acc - bin_conf)
                * np.sum(mask)
                / len(y_true)
            )

    return ece

# %%
ece = expected_calibration_error(
    test_df['label'].values,
    probs
)

print("ECE:", ece)

# %% [markdown]
# ### 12i – Bootstrap 95% Confidence Interval for Accuracy
# 
# 1000 bootstrap samples (sampling with replacement from the test set) are used to estimate the 95% confidence interval for test accuracy. This quantifies the uncertainty in the reported performance numbers and is required by most medical imaging journals.

# %%
from sklearn.utils import resample

n_iterations = 1000

scores = []

y_true = test_df['label'].values

for i in range(n_iterations):

    indices = resample(
        np.arange(len(y_true)),
        replace=True
    )

    score = accuracy_score(
        y_true[indices],
        preds[indices]
    )

    scores.append(score)

lower = np.percentile(scores, 2.5)
upper = np.percentile(scores, 97.5)

print("95% CI:", lower, upper)

# %% [markdown]
# ### 12j – Full Classification Report
# 
# Sklearn's classification report provides precision, recall, F1, and support for each of the 5 DR grades in a single formatted table.

# %%
print(
    classification_report(
        test_df['label'],
        preds
    )
)

# %% [markdown]
# ## Section 13 – Computational Resources
# 
# Records the hardware configuration used for experiments (CPU model, core count, RAM, GPU) for reproducibility reporting in the manuscript methods section.

# %%
import tensorflow as tf
import platform
import psutil

print("CPU:",
      platform.processor())

print("CPU Cores:",
      psutil.cpu_count())

print("RAM:",
      round(psutil.virtual_memory().total / 1e9, 2),
      "GB")

print("GPU:",
      tf.config.list_physical_devices('GPU'))

# %% [markdown]
# ### 13b – Model Parameter Counts
# 
# Reports the total trainable parameter count for the CNN and DenseNet backbones. This is required for the complexity analysis section of the paper.

# %%
print(
    "CNN Parameters:",
    cnn_model.count_params()
)

print(
    "DenseNet Parameters:",
    densenet_model.count_params()
)
 

# %% [markdown]
# ### 13c – GPU Memory Usage

# %%
try:

    info = tf.config.experimental.get_memory_info(
        'GPU:0'
    )

    print(info)

except:
    
    print("GPU memory info unavailable")

# %% [markdown]
# ## Section 14 – Binary Classification Evaluation (Referable vs Non-Referable DR)
# 
# In clinical practice, the most important decision is whether a patient needs **urgent referral** to an ophthalmologist. This section collapses the 5-grade problem into a binary task:
# 
# - **Non-referable**: Grade 0 (No DR) and Grade 1 (Mild DR) → label 0  
# - **Referable**: Grades 2, 3, 4 (Moderate, Severe, Proliferative) → label 1
# 
# Binary metrics (sensitivity, specificity, AUC) at the referral threshold are more directly comparable to clinical screening performance benchmarks.

# %%
y_test_binary = np.where(
    test_df['label'] == 0,
    0,
    1
)

preds_binary = np.where(
    preds == 0,
    0,
    1
)

# %% [markdown]
# ### 14a – Compute Referral Probability
# 
# The probability of referrable DR is the sum of class probabilities for Grades 2, 3, and 4.

# %%
binary_probs = np.sum(
    probs[:, 2:],
    axis=1
)

# %% [markdown]
# ### 14b – Binary Classification Metrics

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

binary_accuracy = accuracy_score(
    y_test_binary,
    preds_binary
)

binary_precision = precision_score(
    y_test_binary,
    preds_binary
)

binary_recall = recall_score(
    y_test_binary,
    preds_binary
)

binary_f1 = f1_score(
    y_test_binary,
    preds_binary
)

binary_auc = roc_auc_score(
    y_test_binary,
    binary_probs
)

binary_pr_auc = average_precision_score(
    y_test_binary,
    binary_probs
)

print("Binary Accuracy:",
      binary_accuracy)

print("Binary Precision:",
      binary_precision)

print("Binary Recall:",
      binary_recall)

print("Binary F1:",
      binary_f1)

print("Binary AUC:",
      binary_auc)

print("Binary PR-AUC:",
      binary_pr_auc)

# %% [markdown]
# ### 14c – Binary Confusion Matrix

# %%
cm_binary = confusion_matrix(
    y_test_binary,
    preds_binary
)

print(cm_binary)

# %% [markdown]
# ### 14d – Binary Sensitivity and Specificity

# %%
TN, FP, FN, TP = cm_binary.ravel()

sensitivity = TP / (TP + FN)

specificity = TN / (TN + FP)

print("Sensitivity:",
      sensitivity)

print("Specificity:",
      specificity)

# %% [markdown]
# ### 14e – Sensitivity at 90% Specificity (Clinical Operating Point)
# 
# WHO screening guidelines recommend a minimum sensitivity of 80% at 90% specificity for automated DR screening tools. This cell computes the achieved sensitivity at the 90% specificity operating point on the ROC curve.

# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    binary_probs
)

specificity_values = 1 - fpr

target_specificity = 0.90

idx = np.argmin(
    np.abs(
        specificity_values
        - target_specificity
    )
)

print(
    "Sensitivity at 90% Specificity:",
    tpr[idx]
)

# %% [markdown]
# ### 14f – Specificity at 90% Sensitivity

# %%
target_sensitivity = 0.90

idx = np.argmin(
    np.abs(
        tpr - target_sensitivity
    )
)

print(
    "Specificity at 90% Sensitivity:",
    specificity_values[idx]
)

# %% [markdown]
# ### 14g – Binary ECE (Calibration at the Referral Decision)

# %%
ece_binary = expected_calibration_error(
    y_test_binary,
    np.column_stack([
        1 - binary_probs,
        binary_probs
    ])
)

print("Binary ECE:",
      ece_binary)

# %% [markdown]
# ## Section 15 – Comprehensive Visualisation Suite
# 
# This section generates all publication-quality figures:
# 
# 1. **Confusion Matrix** — heatmap of predicted vs. true DR grades  
# 2. **Multiclass ROC Curves** — one curve per grade (one-vs-rest), with AUC in the legend  
# 3. **Multiclass Precision-Recall Curves** — especially informative for imbalanced grades  
# 4. **Binary ROC Curve** — referable vs. non-referable DR  
# 5. **Binary Precision-Recall Curve**  
# 6. **Training Accuracy Curve** — CNN training vs. validation accuracy per epoch  
# 7. **Training Loss Curve** — CNN training vs. validation loss per epoch  
# 8. **MOEA/D Feature Selection Plot** — which of the 2048 feature dimensions were selected

# %%
# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)

# ============================================================
# GLOBAL PLOT SETTINGS
# ============================================================

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_df['label'],
    preds
)

plt.figure(figsize=(8,6), dpi=600)

ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=14,
    fontweight='bold'
)

ax.set_ylabel(
    "True Label",
    fontsize=14,
    fontweight='bold'
)

ax.set_title(
    "Confusion Matrix",
    fontsize=16,
    fontweight='bold'
)

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha='right',
    fontweight='bold'
)

ax.set_yticklabels(
    class_names,
    rotation=0,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# ============================================================
# MULTICLASS ROC CURVES
# ============================================================

from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(
    test_df['label'],
    classes=[0,1,2,3,4]
)

plt.figure(figsize=(8,6), dpi=600)

for i in range(5):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2
)

plt.xlabel(
    "False Positive Rate",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "True Positive Rate",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "ROC Curves",
    fontsize=16,
    fontweight='bold'
)

legend = plt.legend(
    prop={
        'weight': 'bold',
        'size': 10
    }
)

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# MULTICLASS PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(8,6), dpi=600)

for i in range(5):

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test_bin[:, i],
        probs[:, i]
    )

    plt.plot(
        recall_vals,
        precision_vals,
        linewidth=2,
        label=class_names[i]
    )

plt.xlabel(
    "Recall",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "Precision",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "Precision-Recall Curves",
    fontsize=16,
    fontweight='bold'
)

plt.legend(
    prop={
        'weight': 'bold',
        'size': 10
    }
)

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# CONFUSION MATRIX BINARY
# ================== ==========================================

class_names_binary = [
    "No DR",
    "DR"
]

cm_binary = confusion_matrix(
     y_test_binary,
    preds_binary
)

plt.figure(figsize=(8,6), dpi=600)

ax = sns.heatmap(
    cm_binary,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    annot_kws={
        "fontsize": 14,
        "fontweight": "bold"
    }
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=14,
    fontweight='bold'
)

ax.set_ylabel(
    "True Label",
    fontsize=14,
    fontweight='bold'
)

ax.set_title(
    "Confusion Matrix",
    fontsize=16,
    fontweight='bold'
)

ax.set_xticklabels(
    class_names_binary,
    rotation=45,
    ha='right',
    fontweight='bold'
)

ax.set_yticklabels(
    class_names_binary,
    rotation=0,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# ============================================================
# BINARY ROC CURVE
# ============================================================

# REFERABLE DR
y_test_binary = np.where(
    test_df['label'] >= 2,
    1,
    0
)

preds_binary = np.where(
    preds >= 2,
    1,
    0
)

binary_probs = np.sum(
    probs[:, 2:],
    axis=1
)

fpr, tpr, _ = roc_curve(
    y_test_binary,
    binary_probs
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6), dpi=600)

plt.plot(
    fpr,
    tpr,
    linewidth=3,
    label=f'AUC = {roc_auc:.4f}'
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2
)

plt.xlabel(
    "False Positive Rate",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "True Positive Rate",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "ROC Curve",
    fontsize=16,
    fontweight='bold'
)

plt.legend(
    prop={
        'weight': 'bold',
        'size': 12
    }
)

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# BINARY PRECISION-RECALL CURVE
# ============================================================

precision_vals, recall_vals, _ = precision_recall_curve(
    y_test_binary,
    binary_probs
)

plt.figure(figsize=(8,6), dpi=600)

plt.plot(
    recall_vals,
    precision_vals,
    linewidth=3
)

plt.xlabel(
    "Recall",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "Precision",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "Precision-Recall Curve",
    fontsize=16,
    fontweight='bold'
)

plt.grid(True)

plt.tight_layout()

plt.show()



# ============================================================
# FEATURE SELECTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,4), dpi=600)

selected_mask = np.zeros(
    X_train_scaled.shape[1]
)

selected_mask[selected_features] = 1

plt.plot(
    selected_mask,
    linewidth=2
)

plt.xlabel(
    "Feature Index",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "Selected",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "MOEA/D Selected Features",
    fontsize=16,
    fontweight='bold'
)

plt.grid(True)

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Section 16 – Multiclass Calibration Error (ECE)
# 
# This section redefines and computes the Expected Calibration Error with detailed inline comments explaining each step of the binning procedure. A well-calibrated model is important for clinical deployment — clinicians need to trust that a 90% confidence score actually means the model is correct ~90% of the time.

# %%
import numpy as np

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10
):

    # Maximum confidence
    confidences = np.max(
        y_prob,
        axis=1
    )

    # Predicted class
    predictions = np.argmax(
        y_prob,
        axis=1
    )

    # Correct or incorrect
    accuracies = (
        predictions == y_true
    )

    # Confidence bins
    bin_boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Samples in this bin
        mask = (
            (confidences > lower) &
            (confidences <= upper)
        )

        if np.sum(mask) > 0:

            # Bin accuracy
            bin_accuracy = np.mean(
                accuracies[mask]
            )

            # Bin confidence
            bin_confidence = np.mean(
                confidences[mask]
            )

            # Weighted gap
            ece += (
                np.abs(
                    bin_accuracy -
                    bin_confidence
                )
                * np.sum(mask)
                / len(y_true)
            )

    return ece

# %%
ece = expected_calibration_error(
    test_df['label'].values,
    probs,
    n_bins=10
)

print("ECE:", ece)

# %% [markdown]
# ### 16a – Multiclass Reliability Diagram
# 
# The reliability diagram plots model confidence (x-axis) against observed accuracy (y-axis) per confidence bin. The dashed diagonal represents perfect calibration. Deviations above the diagonal indicate **underconfidence**; deviations below indicate **overconfidence**.

# %%
import matplotlib.pyplot as plt

confidences = np.max(
    probs,
    axis=1
)

predictions = np.argmax(
    probs,
    axis=1
)

accuracies = (
    predictions ==
    test_df['label'].values
)

n_bins = 10

bin_boundaries = np.linspace(
    0,
    1,
    n_bins + 1
)

bin_centers = []
bin_accuracies = []

for i in range(n_bins):

    lower = bin_boundaries[i]
    upper = bin_boundaries[i + 1]

    mask = (
        (confidences > lower) &
        (confidences <= upper)
    )

    if np.sum(mask) > 0:

        bin_acc = np.mean(
            accuracies[mask]
        )

        bin_conf = np.mean(
            confidences[mask]
        )

        bin_accuracies.append(
            bin_acc
        )

        bin_centers.append(
            bin_conf
        )

plt.figure(figsize=(7,6), dpi=600)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2,
    label='Perfect Calibration'
)

plt.plot(
    bin_centers,
    bin_accuracies,
    marker='o',
    linewidth=3,
    label=f'ECE = {ece:.4f}'
)

plt.xlabel(
    "Confidence",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "Accuracy",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "Reliability Diagram",
    fontsize=16,
    fontweight='bold'
)

plt.legend(
    prop={
        'weight':'bold',
        'size':12
    }
)

plt.grid(True)

plt.tight_layout()

plt.show()

# %% [markdown]
# ### 16b – Binary Reliability Diagram (Referral Decision Calibration)
# 
# Calibration is assessed separately for the binary referral task (Grades 0–1 vs. Grades 2–4). This is the operationally critical calibration check — it determines whether the model's referral confidence scores are trustworthy for clinical triage.

# %%
# ============================================================
# BINARY RELIABILITY DIAGRAM / CALIBRATION PLOT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# REFERABLE DR BINARY LABELS
# ============================================================

# 0,1 = Non-referable
# 2,3,4 = Referable

y_test_binary = np.where(
    test_df['label'] >= 2,
    1,
    0
)

# ============================================================
# BINARY PROBABILITIES
# ============================================================

binary_probs = np.sum(
    probs[:, 2:],
    axis=1
)

# ============================================================
# ECE FUNCTION
# ============================================================

def expected_calibration_error_binary(
    y_true,
    y_prob,
    n_bins=10
):

    predictions = (
        y_prob >= 0.5
    ).astype(int)

    accuracies = (
        predictions == y_true
    )

    bin_boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    bin_confidences = []
    bin_accuracies = []

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        mask = (
            (y_prob > lower) &
            (y_prob <= upper)
        )

        if np.sum(mask) > 0:

            bin_accuracy = np.mean(
                accuracies[mask]
            )

            bin_confidence = np.mean(
                y_prob[mask]
            )

            bin_accuracies.append(
                bin_accuracy
            )

            bin_confidences.append(
                bin_confidence
            )

            ece += (
                np.abs(
                    bin_accuracy -
                    bin_confidence
                )
                * np.sum(mask)
                / len(y_true)
            )

    return (
        ece,
        bin_confidences,
        bin_accuracies
    )

# ============================================================
# COMPUTE ECE
# ============================================================

ece_binary, bin_confidences, bin_accuracies = \
    expected_calibration_error_binary(
        y_test_binary,
        binary_probs,
        n_bins=10
)

print("Binary ECE:", ece_binary)

# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

plt.figure(figsize=(7,6), dpi=600)

# Perfect calibration line
plt.plot(
    [0,1],
    [0,1],
    linestyle='--',
    linewidth=2,
    label='Perfect Calibration'
)

# Model calibration
plt.plot(
    bin_confidences,
    bin_accuracies,
    marker='o',
    linewidth=3,
    markersize=8,
    label=f'Model (ECE = {ece_binary:.4f})'
)

plt.xlabel(
    "Predicted Probability",
    fontsize=14,
    fontweight='bold'
)

plt.ylabel(
    "Observed Accuracy",
    fontsize=14,
    fontweight='bold'
)

plt.title(
    "Reliability Diagram",
    fontsize=16,
    fontweight='bold'
)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.legend(
    prop={
        'weight':'bold',
        'size':11
    }
)

plt.grid(True)

plt.tight_layout()

plt.show()

# %% [markdown]
# ## Section 11 – Explainability and Visual Evidence
# 
# This section adds clinically-grounded interpretability analysis to the Hybrid Feature Fusion model. We implement:
# 1. **Grad-CAM** – gradient-weighted class activation maps on the Custom CNN  
# 2. **Grad-CAM++** – improved localization on the Custom CNN  
# 3. **Integrated Gradients** – attribution maps on DenseNet121  
# 4. **Attention Rollout** – ViT self-attention visualisation  
# 5. **SHAP Feature Importance + Insertion-Deletion** – LightGBM interpretability  
# 
# All visualisations are overlaid on fundus images and annotated against clinically meaningful retinal lesions (microaneurysms, haemorrhages, hard exudates, cotton-wool spots, neovascularisation).

# %%
import sys
!{sys.executable} -m pip install grad-cam shap lime -q

# %%
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import shap
import torch
import tensorflow as tf

from tensorflow.keras.models import Model
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image as pgc_preprocess
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import ViTForImageClassification

warnings.filterwarnings('ignore')

# DR class labels
DR_CLASSES = ['No DR (Grade 0)', 'Mild DR (Grade 1)', 'Moderate DR (Grade 2)',
              'Severe DR (Grade 3)', 'Proliferative DR (Grade 4)']

# Retinal lesion reference per grade
LESION_REFERENCE = {
    0: 'No visible lesions',
    1: 'Microaneurysms only',
    2: 'Microaneurysms + Haemorrhages + Hard Exudates',
    3: 'Haemorrhages + Hard Exudates + Cotton-Wool Spots',
    4: 'Neovascularisation + Vitreous Haemorrhage + Fibrous Proliferation'
}

# %%
# ── Sample Selection ──────────────────────────────────────────────────────
# Pick one representative image per DR grade from the test set
sample_indices = {}
for grade in range(5):
    idx_list = test_df.index[test_df['label'] == grade].tolist()
    if idx_list:
        sample_indices[grade] = idx_list[0]

print('Sample image indices per grade:', sample_indices)

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.1  GRAD-CAM – Custom CNN (TensorFlow/Keras)
# Target layer: last Conv2D (Block 4, second conv)
# ═══════════════════════════════════════════════════════════════════════════

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAM using explicit layer indexing — works for all Sequential models."""
    
    # Find the target layer index
    layer_names = [l.name for l in model.layers]
    conv_idx = layer_names.index(last_conv_layer_name)
    
    # Rebuild as Functional model by re-wiring layers
    inp = tf.keras.Input(shape=img_array.shape[1:])
    x = inp
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_out = x  # capture intermediate output

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Resize heatmap and overlay on original image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    original_uint8 = np.uint8(original_img * 255)
    superimposed = cv2.addWeighted(original_uint8, 1 - alpha, colored, alpha, 0)
    return superimposed


# Find the last Conv2D layer name in the custom CNN
last_conv_name = None
for layer in reversed(cnn_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_name = layer.name
        break
print('Target Conv layer:', last_conv_name)

# ── Visualise Grad-CAM for each DR grade ──────────────────────────────────
n_grades = len(sample_indices)
fig, axes = plt.subplots(n_grades, 3, figsize=(15, 4 * n_grades), dpi=150)

for row, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img = preprocess_image(img_path)               # (224,224,3) float32 [0,1]
    img_tensor = np.expand_dims(orig_img, axis=0)       # (1,224,224,3)

    # Prediction
    preds = cnn_model.predict(img_tensor, verbose=0)
    pred_class = int(np.argmax(preds[0]))

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_tensor, cnn_model, last_conv_name, pred_index=pred_class)
    cam_img = overlay_heatmap(heatmap, orig_img)

    # Column 0 – Original
    axes[row, 0].imshow(orig_img)
    axes[row, 0].set_title(f'Original\nTrue: {DR_CLASSES[grade]}', fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')

    # Column 1 – Heatmap only
    axes[row, 1].imshow(heatmap, cmap='jet')
    axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=9, fontweight='bold')
    axes[row, 1].axis('off')

    # Column 2 – Overlay
    axes[row, 2].imshow(cam_img)
    axes[row, 2].set_title(f'Overlay | Pred: {DR_CLASSES[pred_class]}\nExpected lesions: {LESION_REFERENCE[grade]}',
                            fontsize=8, fontweight='bold')
    axes[row, 2].axis('off')

plt.suptitle('Grad-CAM: Custom CNN Attention per DR Grade', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('gradcam_custom_cnn.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: gradcam_custom_cnn.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.2  GRAD-CAM++ – Custom CNN (TensorFlow/Keras)
# Provides better multi-instance localisation than standard Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════

def make_gradcam_plus_plus(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM++ heatmap — works for Sequential models."""

    # Rebuild as Functional model by re-wiring layers
    inp = tf.keras.Input(shape=img_array.shape[1:])
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_out = x

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, x])

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            with tf.GradientTape() as t0:
                conv_outputs, preds = grad_model(img_array, training=False)
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                score = preds[:, pred_index]
            first_grads  = t0.gradient(score, conv_outputs)
        second_grads = t1.gradient(first_grads, conv_outputs)
    third_grads  = t2.gradient(second_grads, conv_outputs)

    global_sum = tf.reduce_sum(conv_outputs, axis=[0, 1, 2])
    alpha_num  = second_grads[0]
    alpha_den  = 2.0 * second_grads[0] + global_sum * third_grads[0] + 1e-7
    alpha      = alpha_num / alpha_den
    weights    = tf.reduce_sum(tf.nn.relu(first_grads[0]) * alpha, axis=[0, 1])

    heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


fig, axes = plt.subplots(n_grades, 3, figsize=(15, 4 * n_grades), dpi=150)

for row, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img = preprocess_image(img_path)
    img_tensor = np.expand_dims(orig_img, axis=0)

    preds = cnn_model.predict(img_tensor, verbose=0)
    pred_class = int(np.argmax(preds[0]))

    heatmap = make_gradcam_plus_plus(img_tensor, cnn_model, last_conv_name, pred_index=pred_class)
    cam_img  = overlay_heatmap(heatmap, orig_img)

    axes[row, 0].imshow(orig_img)
    axes[row, 0].set_title(f'Original\nTrue: {DR_CLASSES[grade]}', fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(heatmap, cmap='jet')
    axes[row, 1].set_title('Grad-CAM++ Heatmap', fontsize=9, fontweight='bold')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(cam_img)
    axes[row, 2].set_title(f'Overlay | Pred: {DR_CLASSES[pred_class]}\nExpected lesions: {LESION_REFERENCE[grade]}',
                            fontsize=8, fontweight='bold')
    axes[row, 2].axis('off')

plt.suptitle('Grad-CAM++: Custom CNN – Improved Multi-lesion Localisation', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('gradcam_pp_custom_cnn.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: gradcam_pp_custom_cnn.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.3  INTEGRATED GRADIENTS – DenseNet121 (TensorFlow/Keras)
# Attributes prediction to input pixels by integrating gradients along
# the straight-line path from a black baseline to the actual image.
# Clinically highlights microaneurysms, haemorrhages, and exudates.
# ═══════════════════════════════════════════════════════════════════════════

# Build a DenseNet model that outputs class logits (5 classes)
def build_densenet_classifier(num_classes=5):
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(base.input, out)

densenet_clf = build_densenet_classifier()

# NOTE: If you have saved DenseNet weights, load them here:
# densenet_clf.load_weights('densenet_finetuned.h5')


@tf.function
def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images, training=False)
        loss  = preds[:, target_class_idx]
    return tape.gradient(loss, images)


def integrated_gradients(img_array, model, target_class_idx, baseline=None, m_steps=50):
    """Compute Integrated Gradients attribution map."""
    if baseline is None:
        baseline = np.zeros_like(img_array)   # black baseline
    baseline   = tf.cast(baseline, tf.float32)
    img_tensor = tf.cast(img_array, tf.float32)

    # Interpolate between baseline and image
    alphas     = tf.linspace(0.0, 1.0, m_steps + 1)
    interp_imgs = baseline + alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * (img_tensor - baseline)

    # Accumulate gradients across interpolated inputs
    grads = compute_gradients(interp_imgs, target_class_idx, model)

    # Average gradients using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(avg_grads, axis=0)

    # Scale to input delta
    ig = (img_tensor[0] - baseline[0]) * avg_grads
    return ig.numpy()


def attribution_to_heatmap(ig_attr):
    """Convert IG attribution (H,W,3) → single-channel heatmap."""
    # Sum absolute attributions across colour channels
    attr_map = np.sum(np.abs(ig_attr), axis=-1)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    return attr_map


fig, axes = plt.subplots(n_grades, 4, figsize=(20, 4 * n_grades), dpi=150)

for row, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img  = preprocess_image(img_path)            # (224,224,3) float32
    img_tensor = np.expand_dims(orig_img, axis=0)

    preds = densenet_clf.predict(img_tensor, verbose=0)
    pred_class = int(np.argmax(preds[0]))

    ig_attr   = integrated_gradients(img_tensor, densenet_clf, pred_class)
    attr_map  = attribution_to_heatmap(ig_attr)
    cam_img   = overlay_heatmap(attr_map, orig_img, alpha=0.5, colormap=cv2.COLORMAP_PLASMA)

    # Positive-only attribution (what the model looks AT)
    pos_attr  = np.clip(ig_attr, 0, None)
    pos_map   = np.sum(pos_attr, axis=-1)
    pos_map   = (pos_map - pos_map.min()) / (pos_map.max() - pos_map.min() + 1e-8)

    axes[row, 0].imshow(orig_img)
    axes[row, 0].set_title(f'Original\nTrue: {DR_CLASSES[grade]}', fontsize=8, fontweight='bold')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(attr_map, cmap='plasma')
    axes[row, 1].set_title('IG Attribution Map\n(absolute)', fontsize=8, fontweight='bold')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(pos_map, cmap='hot')
    axes[row, 2].set_title('IG Positive Attribution\n(supports prediction)', fontsize=8, fontweight='bold')
    axes[row, 2].axis('off')

    axes[row, 3].imshow(cam_img)
    axes[row, 3].set_title(f'IG Overlay | Pred: {DR_CLASSES[pred_class]}\nLesions: {LESION_REFERENCE[grade]}',
                            fontsize=7, fontweight='bold')
    axes[row, 3].axis('off')

plt.suptitle('Integrated Gradients – DenseNet121: Pixel-level Attribution per DR Grade',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('integrated_gradients_densenet.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: integrated_gradients_densenet.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.4  ATTENTION ROLLOUT – Vision Transformer (ViT)
# Recursively multiplies attention matrices across all ViT layers to
# compute how much each patch attends to the [CLS] token, revealing
# which retinal regions influence the overall classification.
# ═══════════════════════════════════════════════════════════════════════════

import torch
from transformers import ViTModel, ViTImageProcessor

# Re-use the already-loaded vit_model and vit_processor from the main pipeline
# vit_model, vit_processor must be defined earlier in the notebook.

import torch
from transformers import ViTModel, ViTImageProcessor

# ============================================================
# DEVICE
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
vit_model_hf = vit_model_hf.to(device)
vit_model_hf.eval()


def attention_rollout(model,
                      processor,
                      img_np,
                      head_fusion='mean',
                      discard_ratio=0.9):

    """
    Compute attention rollout for a HuggingFace ViTModel.
    Returns a 2-D heatmap of shape (patch_grid, patch_grid).
    """

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================
    inputs = processor(
        images=img_np,
        return_tensors='pt'
    )

    # ========================================================
    # MOVE INPUTS TO SAME DEVICE AS MODEL
    # ========================================================
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    # ========================================================
    # FORWARD PASS
    # ========================================================
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True
        )

    # ========================================================
    # ATTENTION MATRICES
    # ========================================================
    attentions = outputs.attentions

    # ========================================================
    # INITIALIZE ROLLOUT MATRIX
    # ========================================================
    result = torch.eye(
        attentions[0].size(-1),
        device=device
    )

    # ========================================================
    # RECURSIVE ATTENTION ROLLOUT
    # ========================================================
    for attn in attentions:

        # Fuse attention heads
        if head_fusion == 'mean':
            attn_fused = attn.mean(dim=1).squeeze(0)

        elif head_fusion == 'max':
            attn_fused = attn.max(dim=1).values.squeeze(0)

        elif head_fusion == 'min':
            attn_fused = attn.min(dim=1).values.squeeze(0)

        else:
            raise ValueError(
                f'Unknown head_fusion: {head_fusion}'
            )

        # ====================================================
        # DISCARD LOW ATTENTION VALUES
        # ====================================================
        flat = attn_fused.view(-1)

        threshold = torch.quantile(
            flat,
            discard_ratio
        )

        attn_fused = attn_fused.clone()

        attn_fused[attn_fused < threshold] = 0

        # ====================================================
        # ADD RESIDUAL CONNECTION
        # ====================================================
        attn_fused += torch.eye(
            attn_fused.size(-1),
            device=device
        )

        # ====================================================
        # NORMALIZE
        # ====================================================
        attn_fused /= attn_fused.sum(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-8)

        # ====================================================
        # MATRIX MULTIPLICATION
        # ====================================================
        result = torch.matmul(
            attn_fused,
            result
        )

    # ========================================================
    # CLS TOKEN → PATCH ATTENTION
    # ========================================================
    mask = result[0, 1:]

    mask = (
        mask - mask.min()
    ) / (
        mask.max() - mask.min() + 1e-8
    )

    # ========================================================
    # RESHAPE TO PATCH GRID
    # ========================================================
    n_patches = mask.shape[0]

    grid_size = int(n_patches ** 0.5)

    mask = mask[
        :grid_size * grid_size
    ].reshape(
        grid_size,
        grid_size
    )

    # ========================================================
    # MOVE BACK TO CPU FOR NUMPY
    # ========================================================
    return mask.cpu().numpy()

    # Extract CLS → patch attention (skip CLS token at index 0)
    mask = result[0, 1:]   # (n_patches,)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Reshape to square patch grid
    n_patches = mask.shape[0]
    grid_size = int(n_patches ** 0.5)
    mask = mask[:grid_size * grid_size].reshape(grid_size, grid_size).numpy()
    return mask


fig, axes = plt.subplots(n_grades, 3, figsize=(15, 4 * n_grades), dpi=150)

for row, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img  = preprocess_image(img_path)   # (224,224,3) float32 [0,1]

    rollout_map = attention_rollout(vit_model_hf, vit_processor, orig_img)

    # Upsample patch grid → image resolution
    rollout_up  = cv2.resize(rollout_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    rollout_up  = (rollout_up - rollout_up.min()) / (rollout_up.max() - rollout_up.min() + 1e-8)
    cam_img     = overlay_heatmap(rollout_up, orig_img, alpha=0.5, colormap=cv2.COLORMAP_HOT)

    axes[row, 0].imshow(orig_img)
    axes[row, 0].set_title(f'Original\nTrue: {DR_CLASSES[grade]}', fontsize=9, fontweight='bold')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(rollout_up, cmap='hot')
    axes[row, 1].set_title('Attention Rollout Map\n(patch saliency)', fontsize=9, fontweight='bold')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(cam_img)
    axes[row, 2].set_title(f'ViT Attention Overlay\nExpected lesions: {LESION_REFERENCE[grade]}',
                            fontsize=8, fontweight='bold')
    axes[row, 2].axis('off')

plt.suptitle('Attention Rollout – ViT: Global Context Attention per DR Grade',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('attention_rollout_vit.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: attention_rollout_vit.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.5  SHAP via LightGBM built-in — no shap library needed
# LightGBM has native SHAP support through predict() with pred_contrib=True
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import numpy as np

n_explain = min(300, X_test_selected.shape[0])
X_explain  = X_test_selected[:n_explain]
n_features = X_explain.shape[1]

# LightGBM pred_contrib returns SHAP values natively
# Shape: (n_samples, (n_features + 1) * n_classes)
# The +1 column per class is the bias term — we drop it
contrib_raw = clf.predict(X_explain, pred_contrib=True)
print("Raw contrib shape:", contrib_raw.shape)

n_classes   = 5
cols_per_cls = n_features + 1   # features + bias

# Reshape to (n_samples, n_classes, n_features+1) then drop bias
contrib_3d  = contrib_raw.reshape(n_explain, n_classes, cols_per_cls)
shap_values = contrib_3d[:, :, :n_features]   # (n_samples, n_classes, n_features)
print("SHAP values shape:", shap_values.shape)

# ── Global Mean |SHAP| bar chart ──────────────────────────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))   # (n_features,)
top_k  = 30
top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]

plt.figure(figsize=(12, 7), dpi=150)
plt.barh(
    range(top_k),
    mean_abs_shap[top_idx][::-1],
    color='steelblue'
)
plt.yticks(
    range(top_k),
    [f'Feature {i}' for i in top_idx[::-1]],
    fontsize=9
)
plt.xlabel('Mean |SHAP value| across all DR grades', fontsize=12, fontweight='bold')
plt.title(
    'LightGBM – Top-30 Most Important Fused Features\n(CNN + DenseNet + ViT)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('shap_global_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: shap_global_importance.png")

# %%
# ── Improved Per-class SHAP Beeswarm Layout ────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DR_CLASSES = [
    'No DR (G0)',
    'Mild (G1)',
    'Moderate (G2)',
    'Severe (G3)',
    'Proliferative (G4)'
]

# =========================================================
# Better Figure Layout
# 3 plots in first row
# 2 centered plots in second row
# =========================================================

fig = plt.figure(figsize=(24, 14), dpi=300)

gs = gridspec.GridSpec(
    2, 6,
    height_ratios=[1, 1],
    hspace=0.35,
    wspace=0.35
)

# First row (3 plots)
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),

    # Second row (centered 2 plots)
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

# =========================================================
# Create Beeswarm Plots
# =========================================================

for cls in range(5):

    ax = axes[cls]

    sv_cls = shap_values[:, cls, :]   # (n_samples, n_features)

    # Top 15 important features
    top_feats = np.argsort(np.abs(sv_cls).mean(0))[::-1][:15]

    for fi, feat in enumerate(top_feats):

        feat_vals = X_explain[:, feat]
        sv_col = sv_cls[:, feat]

        # Normalize feature values for color mapping
        feat_range = feat_vals.max() - feat_vals.min()
        norm_v = (feat_vals - feat_vals.min()) / (feat_range + 1e-8)

        # Random jitter for beeswarm effect
        jitter = np.random.uniform(-0.22, 0.22, size=len(sv_col))

        sc = ax.scatter(
            sv_col,
            fi + jitter,
            c=norm_v,
            cmap='coolwarm',
            s=14,
            alpha=0.7,
            edgecolors='none'
        )

    # Formatting
    ax.set_yticks(range(15))
    ax.set_yticklabels(
        [f'F{f}' for f in top_feats],
        fontsize=9,
        fontweight='bold'
    )

    ax.axvline(
        0,
        color='black',
        linewidth=1,
        linestyle='--'
    )

    ax.set_title(
        DR_CLASSES[cls],
        fontsize=13,
        fontweight='bold'
    )

    ax.set_xlabel(
        'SHAP Value',
        fontsize=10,
        fontweight='bold'
    )

    ax.tick_params(axis='x', labelsize=9)

    # Invert y-axis (most important feature on top)
    ax.invert_yaxis()

# =========================================================
# Shared Colorbar
# =========================================================

cbar = fig.colorbar(
    sc,
    ax=axes,
    fraction=0.02,
    pad=0.02
)

cbar.set_label(
    'Normalized Feature Value',
    fontsize=11,
    fontweight='bold'
)

# =========================================================
# Main Title
# =========================================================

plt.suptitle(
    'SHAP Beeswarm Analysis for Diabetic Retinopathy Grades\n'
    '(Red = High Feature Value, Blue = Low Feature Value)',
    fontsize=18,
    fontweight='bold',
    y=0.98
)

# =========================================================
# Save Figure
# =========================================================

plt.savefig(
    'shap_beeswarm_per_class.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("Saved: shap_beeswarm_per_class.png")

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.6  INSERTION-DELETION ANALYSIS – CNN (pixel-level faithfulness)
# Insertion: gradually reveal most-salient pixels → AUC should rise fast.
# Deletion : gradually remove most-salient pixels → AUC should drop fast.
# Higher insertion-AUC and lower deletion-AUC = more faithful saliency.
# ═══════════════════════════════════════════════════════════════════════════

def insertion_deletion_curves(
    model,
    img,
    saliency_map,
    n_steps=100,
    batch_size=10
):
    """
    Compute insertion and deletion curves for a single image.
    Returns (steps, insertion_scores, deletion_scores).
    """
    H, W, C = img.shape
    n_pixels = H * W

    # Flatten saliency and sort pixel indices by importance (descending)
    flat_sal  = saliency_map.flatten()
    order     = np.argsort(flat_sal)[::-1]   # most important first

    baseline_blur = cv2.GaussianBlur(img, (51, 51), 0)   # blurred baseline

    step_sizes   = np.linspace(0, 1, n_steps + 1)
    ins_scores   = []
    del_scores   = []

    for frac in step_sizes:
        n_reveal = int(frac * n_pixels)

        # INSERTION: start from blur, reveal top-k pixels
        ins_img = baseline_blur.copy()
        if n_reveal > 0:
            top_k = order[:n_reveal]
            ys, xs = np.unravel_index(top_k, (H, W))
            ins_img[ys, xs] = img[ys, xs]

        # DELETION: start from original, blank top-k pixels
        del_img = img.copy()
        if n_reveal > 0:
            del_img[ys, xs] = baseline_blur[ys, xs]

        ins_tensor = np.expand_dims(ins_img, 0)
        del_tensor = np.expand_dims(del_img, 0)

        ins_p = model.predict(ins_tensor, verbose=0)
        del_p = model.predict(del_tensor, verbose=0)

        ins_scores.append(float(np.max(ins_p)))
        del_scores.append(float(np.max(del_p)))

    return step_sizes, np.array(ins_scores), np.array(del_scores)


fig, axes = plt.subplots(2, n_grades, figsize=(4 * n_grades, 8), dpi=150)

auc_results = []

for col, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img  = preprocess_image(img_path)
    img_tensor = np.expand_dims(orig_img, 0)

    # Get saliency map from Grad-CAM
    preds = cnn_model.predict(img_tensor, verbose=0)
    pred_class = int(np.argmax(preds[0]))
    heatmap = make_gradcam_heatmap(img_tensor, cnn_model, last_conv_name, pred_index=pred_class)
    saliency = cv2.resize(heatmap, (224, 224))

    steps, ins, dels = insertion_deletion_curves(cnn_model, orig_img, saliency, n_steps=50)

    ins_auc  = np.trapz(ins,  steps)
    del_auc  = np.trapz(dels, steps)
    auc_results.append({'grade': grade, 'ins_auc': ins_auc, 'del_auc': del_auc})

    # Overlay
    cam_img = overlay_heatmap(saliency, orig_img)
    axes[0, col].imshow(cam_img)
    axes[0, col].set_title(f'{DR_CLASSES[grade]}\nGrad-CAM Overlay', fontsize=7, fontweight='bold')
    axes[0, col].axis('off')

    # Curve
    axes[1, col].plot(steps, ins,  label=f'Insertion (AUC={ins_auc:.3f})',  color='green')
    axes[1, col].plot(steps, dels, label=f'Deletion  (AUC={del_auc:.3f})',  color='red', linestyle='--')
    axes[1, col].set_xlabel('Fraction of pixels revealed/removed', fontsize=7)
    axes[1, col].set_ylabel('Max predicted confidence', fontsize=7)
    axes[1, col].legend(fontsize=6)
    axes[1, col].set_title(f'Grade {grade}', fontsize=8, fontweight='bold')
    axes[1, col].grid(True)

plt.suptitle('Insertion-Deletion Analysis – CNN Saliency Faithfulness per DR Grade',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('insertion_deletion_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Print AUC table
print('\n── Insertion-Deletion AUC Summary ──────────────────────────────────────')
print(f"{'Grade':<30} {'Insertion AUC':>15} {'Deletion AUC':>15}")
for r in auc_results:
    print(f"{DR_CLASSES[r['grade']]:<30} {r['ins_auc']:>15.4f} {r['del_auc']:>15.4f}")
print('Saved: insertion_deletion_curves.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.7  COMPARATIVE SALIENCY SUMMARY
# Side-by-side: Grad-CAM | Grad-CAM++ | Integrated Gradients | Attn Rollout
# for one representative image per DR grade.
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(n_grades, 5, figsize=(25, 4.5 * n_grades), dpi=150)
col_titles = ['Original', 'Grad-CAM\n(CNN)', 'Grad-CAM++\n(CNN)',
              'Integrated Gradients\n(DenseNet)', 'Attention Rollout\n(ViT)']

for row, (grade, idx) in enumerate(sample_indices.items()):
    img_path = test_df.loc[idx, 'image_path']
    orig_img  = preprocess_image(img_path)
    img_tensor = np.expand_dims(orig_img, 0)

    # CNN prediction
    preds = cnn_model.predict(img_tensor, verbose=0)
    pred_class = int(np.argmax(preds[0]))

    # --- Grad-CAM
    gc_heatmap = make_gradcam_heatmap(img_tensor, cnn_model, last_conv_name, pred_index=pred_class)
    gc_overlay  = overlay_heatmap(cv2.resize(gc_heatmap, (224, 224)), orig_img)

    # --- Grad-CAM++
    gcpp_heatmap = make_gradcam_plus_plus(img_tensor, cnn_model, last_conv_name, pred_index=pred_class)
    gcpp_overlay  = overlay_heatmap(cv2.resize(gcpp_heatmap, (224, 224)), orig_img)

    # --- Integrated Gradients (DenseNet)
    ig_preds = densenet_clf.predict(img_tensor, verbose=0)
    ig_pred_class = int(np.argmax(ig_preds[0]))
    ig_attr   = integrated_gradients(img_tensor, densenet_clf, ig_pred_class)
    ig_map    = attribution_to_heatmap(ig_attr)
    ig_overlay = overlay_heatmap(ig_map, orig_img, alpha=0.5, colormap=cv2.COLORMAP_PLASMA)

    # --- Attention Rollout (ViT)
    ar_map    = attention_rollout(vit_model_hf, vit_processor, orig_img)
    ar_up     = cv2.resize(ar_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    ar_up     = (ar_up - ar_up.min()) / (ar_up.max() - ar_up.min() + 1e-8)
    ar_overlay = overlay_heatmap(ar_up, orig_img, alpha=0.5, colormap=cv2.COLORMAP_HOT)

    visuals = [orig_img, gc_overlay, gcpp_overlay, ig_overlay, ar_overlay]
    for col, (vis, title) in enumerate(zip(visuals, col_titles)):
        axes[row, col].imshow(vis)
        if row == 0:
            axes[row, col].set_title(title, fontsize=9, fontweight='bold')
        if col == 0:
            axes[row, col].set_ylabel(
                f'{DR_CLASSES[grade]}\n{LESION_REFERENCE[grade]}',
                fontsize=7, fontweight='bold', rotation=0, labelpad=80, va='center'
            )
        axes[row, col].axis('off')

plt.suptitle(
    'Explainability Summary: Grad-CAM | Grad-CAM++ | Integrated Gradients | Attention Rollout\n'
    'Showing clinically relevant retinal regions per DR severity grade',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('explainability_summary_all_methods.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: explainability_summary_all_methods.png')

# %%
# ═══════════════════════════════════════════════════════════════════════════
# 11.8  QUANTITATIVE EXPLAINABILITY SUMMARY TABLE
# Reports per-grade insertion/deletion AUCs + predicted class agreement
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd

summary_rows = []
for r in auc_results:
    grade = r['grade']
    summary_rows.append({
        'Grade': grade,
        'DR Severity': DR_CLASSES[grade],
        'Expected Lesions': LESION_REFERENCE[grade],
        'Insertion AUC (↑)': round(r['ins_auc'], 4),
        'Deletion AUC  (↓)': round(r['del_auc'], 4),
        'Faithfulness Score': round(r['ins_auc'] - r['del_auc'], 4)
    })

summary_df = pd.DataFrame(summary_rows)
print('\n── Insertion-Deletion Faithfulness Summary ─────────────────────────────')
print(summary_df.to_string(index=False))
summary_df.to_csv('explainability_quantitative_summary.csv', index=False)
print('\nSaved: explainability_quantitative_summary.csv')

# Plot faithfulness score bar chart
plt.figure(figsize=(8, 5), dpi=150)
bars = plt.bar(
    DR_CLASSES,
    summary_df['Faithfulness Score'],
    color=['#2ecc71','#3498db','#f39c12','#e67e22','#e74c3c'],
    width=0.55
)
for bar, val in zip(bars, summary_df['Faithfulness Score']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.xticks(fontsize=8, rotation=20, ha='right')
plt.ylabel('Faithfulness Score (Insertion AUC − Deletion AUC)', fontsize=10, fontweight='bold')
plt.title('Saliency Faithfulness per DR Grade\n(higher = more clinically meaningful activations)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('faithfulness_scores.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: faithfulness_scores.png')

# %% [markdown]
# ## Section 17 –  Ablation Study 
# 
# ### Speed optimisations over the original ablation
# 
# | Bottleneck | Original | Fast version |
# |---|---|---|
# | SMOTE | Runs inside every variant | Run **once**, reused for all variants |
# | LightGBM trees | 5000 | **300** (sufficient for comparison, not final reporting) |
# | NSGA-II generations | 30 | **10** |
# | RFE step | 50 | **200** (larger steps = fewer iterations) |
# | Per-class F1 heatmap | Retrains 11× | Reuses cached predictions |
# | McNemar | Retrains 10× | Reuses cached predictions |
# 
# > **Note:** The reduced LightGBM trees (300) are for ablation comparison only.  
# > The final proposed model metrics (A11) use the full 5000-tree model from Section 13.

# %% [markdown]
# ### 17.0 – Pre-compute SMOTE Once and Define Fast Evaluator

# %%
import time, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, cohen_kappa_score,
    recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
from scipy.stats import binom

warnings.filterwarnings('ignore')
ABLATION_START = time.time()

DR_CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# ── Reduced LightGBM params for ABLATION ONLY ─────────────────────────────
# 300 trees instead of 5000 — ~16x faster, sufficient for ranking comparison
FAST_LGBM = dict(
    objective='multiclass', num_class=5,
    boosting_type='gbdt',
    n_estimators=300,          # fast ablation — NOT final reporting
    learning_rate=0.05,        # higher LR to compensate fewer trees
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# ── Pre-compute SMOTE ONCE on the full 2048-d fused features ──────────────
# All variants that start from original (unselected) features will use this
print('Pre-computing SMOTE on fused features (runs once)...')
t0 = time.time()
_scaler_abl = StandardScaler()
X_fused_tr_s = _scaler_abl.fit_transform(train_fused)
X_fused_te_s = _scaler_abl.transform(test_fused)

min_cls = train_df['label'].value_counts().min()
k_nb    = max(1, min(5, min_cls - 1))
_sm_abl = SMOTE(random_state=42, k_neighbors=k_nb)
X_sm_abl, y_sm_abl = _sm_abl.fit_resample(X_fused_tr_s, train_df['label'].values)
print(f'  SMOTE done: {X_sm_abl.shape}  ({time.time()-t0:.1f}s)')

y_te_abl = test_df['label'].values
y_bin_abl = label_binarize(y_te_abl, classes=[0,1,2,3,4])


# ── Fast evaluator — no internal SMOTE (uses pre-computed) ────────────────
def fast_eval(X_tr, y_tr, X_te, label=''):
    """Scale → train LightGBM → return metrics dict + predictions."""

    sc   = StandardScaler()
    Xtr  = sc.fit_transform(X_tr)
    Xte  = sc.transform(X_te)
    m    = lgb.LGBMClassifier(**FAST_LGBM)
    m.fit(Xtr, y_tr)
    p    = m.predict(Xte)
    pb   = m.predict_proba(Xte)
    acc  = accuracy_score(y_te_abl, p)
    mf1  = f1_score(y_te_abl, p, average='macro')
    wf1  = f1_score(y_te_abl, p, average='weighted')
    qwk  = cohen_kappa_score(y_te_abl, p, weights='quadratic')
    auc  = roc_auc_score(y_bin_abl, pb, multi_class='ovr')
    prauc= average_precision_score(y_bin_abl, pb, average='macro')
    prec = precision_score(y_te_abl, p, average='macro', zero_division=0)
    rec  = recall_score(y_te_abl, p, average='macro', zero_division=0)
    pcf1 = f1_score(y_te_abl, p, average=None, labels=[0,1,2,3,4], zero_division=0)
    print(f'  [{label}]  Acc={acc:.4f}  F1={mf1:.4f}  QWK={qwk:.4f}  AUC={auc:.4f}')
    return {
        'Configuration': label,
        'N_Features':  X_tr.shape[1],
        'Accuracy':    round(acc, 4),
        'Precision':   round(prec, 4),
        'Recall':      round(rec, 4),
        'Macro_F1':    round(mf1, 4),
        'Weighted_F1': round(wf1, 4),
        'QWK':         round(qwk, 4),
        'AUC':         round(auc, 4),
        'PR_AUC':      round(prauc, 4),
        'Per_Class_F1': pcf1,
    }, p

ablation_results = []
ablation_preds   = {}   # cache predictions for McNemar (no retraining needed)
print('Setup complete.')

# %% [markdown]
# ### 17.1 – Single-Backbone Baselines (A1, A2, A3)

# %%
# ── Pre-scale single-backbone features (SMOTE applied internally) ─────────
# These are small arrays — SMOTE is fast on 256/1024/768-d vectors
from imblearn.over_sampling import SMOTE as _SMOTE

def smote_features(X_tr, y_tr):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    sm = _SMOTE(random_state=42, k_neighbors=k_nb)
    Xs, ys = sm.fit_resample(Xs, y_tr)
    return Xs, ys, sc

print('Running A1/A2/A3 single-backbone baselines...')
t0 = time.time()

for name, Xtr, Xte in [
    ('A1: CNN only',       train_cnn_features,   test_cnn_features),
    ('A2: DenseNet121 only', train_dense_features, test_dense_features),
    ('A3: ViT only',       train_vit_features,   test_vit_features),
]:
    Xtr_s, ytr_s, sc = smote_features(Xtr, train_df['label'].values)
    Xte_s = sc.transform(Xte)
    r, p  = fast_eval(Xtr_s, ytr_s, Xte_s, name)
    ablation_results.append(r)
    ablation_preds[name] = p

print(f'A1/A2/A3 done in {time.time()-t0:.1f}s')

# %% [markdown]
# ### 17.2 – Pairwise Fusion Ablation (A4, A5, A6)

# %%
print('Running A4/A5/A6 pairwise fusion...')
t0 = time.time()

for name, Xtr, Xte in [
    ('A4: CNN + DenseNet',
     np.hstack([train_cnn_features, train_dense_features]),
     np.hstack([test_cnn_features,  test_dense_features])),
    ('A5: CNN + ViT',
     np.hstack([train_cnn_features, train_vit_features]),
     np.hstack([test_cnn_features,  test_vit_features])),
    ('A6: DenseNet + ViT',
     np.hstack([train_dense_features, train_vit_features]),
     np.hstack([test_dense_features,  test_vit_features])),
]:
    Xtr_s, ytr_s, sc = smote_features(Xtr, train_df['label'].values)
    Xte_s = sc.transform(Xte)
    r, p  = fast_eval(Xtr_s, ytr_s, Xte_s, name)
    ablation_results.append(r)
    ablation_preds[name] = p

print(f'A4/A5/A6 done in {time.time()-t0:.1f}s')

# %% [markdown]
# ### 17.3 – Full Fusion Without Feature Selection (A7)

# %%
print('Running A7: Full fusion, no selection...')
t0  = time.time()
# Reuse pre-computed SMOTE arrays (X_sm_abl, y_sm_abl) — no re-SMOTE needed
r, p = fast_eval(X_sm_abl, y_sm_abl, X_fused_te_s, 'A7: Full Fusion (no selection)')
ablation_results.append(r)
ablation_preds['A7: Full Fusion (no selection)'] = p
print(f'A7 done in {time.time()-t0:.1f}s')

# %% [markdown]
# ### 17.4 – PCA Dimensionality Reduction (A8)

# %%
from sklearn.decomposition import PCA

print('Running A8: PCA (95% variance)...')
t0  = time.time()
pca = PCA(n_components=0.95, random_state=42)
# Fit PCA on SMOTE-balanced scaled features
Xtr_pca = pca.fit_transform(X_sm_abl)
Xte_pca = pca.transform(X_fused_te_s)
print(f'  PCA: 2048 → {Xtr_pca.shape[1]} components')
r, p = fast_eval(Xtr_pca, y_sm_abl, Xte_pca, 'A8: Full Fusion + PCA')
ablation_results.append(r)
ablation_preds['A8: Full Fusion + PCA'] = p
print(f'A8 done in {time.time()-t0:.1f}s')

# %% [markdown]
# ### 17.5 – Alternative Feature Selection: LASSO, RFE, mRMR (A9)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

n_target = len(selected_features)   # match MOEA/D feature count for fair comparison

# ── A9a: LASSO ────────────────────────────────────────────────────────────
print('Running A9a: LASSO...')
t0 = time.time()
lasso = LogisticRegression(
    penalty='l1', C=0.01, solver='saga',
    multi_class='multinomial', max_iter=500,
    random_state=42, n_jobs=-1
)
lasso.fit(X_sm_abl, y_sm_abl)
lasso_sel    = SelectFromModel(lasso, prefit=True)
X_tr_lasso   = lasso_sel.transform(X_sm_abl)
X_te_lasso   = lasso_sel.transform(X_fused_te_s)
print(f'  LASSO: {X_tr_lasso.shape[1]} features selected')
r, p = fast_eval(X_tr_lasso, y_sm_abl, X_te_lasso, 'A9a: LASSO')
ablation_results.append(r)
ablation_preds['A9a: LASSO'] = p
print(f'  A9a done in {time.time()-t0:.1f}s')

# ── A9b: RFE (step=200 for speed) ─────────────────────────────────────────
print('Running A9b: RFE (step=200)...')
t0 = time.time()
rfe = RFE(
    estimator=RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1),
    n_features_to_select=n_target,
    step=200    # large step = far fewer iterations = much faster
)
rfe.fit(X_sm_abl, y_sm_abl)
X_tr_rfe = X_sm_abl[:, rfe.support_]
X_te_rfe = X_fused_te_s[:, rfe.support_]
print(f'  RFE: {X_tr_rfe.shape[1]} features selected')
r, p = fast_eval(X_tr_rfe, y_sm_abl, X_te_rfe, 'A9b: RFE')
ablation_results.append(r)
ablation_preds['A9b: RFE'] = p
print(f'  A9b done in {time.time()-t0:.1f}s')



# %% [markdown]
# ### 17.6 – NSGA-II Feature Selection (A10) — 10 Generations

# %%
from sklearn.ensemble import RandomForestClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_min
from joblib import Parallel, delayed

# ── Evaluation cache ──────────────────────────────────────────────────────────
_nsga2_cache = {}

def _eval_single(xi, X_tr, y_tr, X_v, y_v):
    mask = xi.astype(bool)
    if mask.sum() == 0:
        return [1.0, 1.0]
    key = mask.tobytes()
    if key in _nsga2_cache:
        return _nsga2_cache[key]
    clf = RandomForestClassifier(
        n_estimators=30, max_depth=8,
        n_jobs=1, random_state=42,        # n_jobs=1 here; parallelism is at batch level
        class_weight='balanced'
    )
    clf.fit(X_tr[:, mask], y_tr)
    err    = 1.0 - (clf.predict(X_v[:, mask]) == y_v).mean()
    feat_r = mask.sum() / len(mask)
    result = [err, feat_r]
    _nsga2_cache[key] = result
    return result

class BoolSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.random.randint(0, 2, (n_samples, problem.n_var)).astype(bool)

class FastFSProblem(Problem):
    def __init__(self, X_tr, y_tr, X_val, y_val):
        super().__init__(n_var=X_tr.shape[1], n_obj=2,
                         n_ieq_constr=0, xl=0, xu=1, vtype=bool)
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_val, self.y_val = X_val, y_val

    def _evaluate(self, x, out, *args, **kwargs):
        # Parallel batch evaluation — all solutions in one generation run simultaneously
        results = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_eval_single)(xi, self.X_tr, self.y_tr, self.X_val, self.y_val)
            for xi in x
        )
        out['F'] = np.array(results)

print('Running A10: NSGA-II (10 gen, pop=50, RF + cache)...')
t0 = time.time()

_X_val_s = _scaler_abl.transform(val_fused)

nsga2_prob = FastFSProblem(X_sm_abl, y_sm_abl, _X_val_s, val_df['label'].values)
nsga2_algo = NSGA2(
    pop_size=50,
    sampling=BoolSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(prob=1.0 / X_sm_abl.shape[1]),
    eliminate_duplicates=True
)
nsga2_res = pymoo_min(
    nsga2_prob, nsga2_algo,
    termination=('n_gen', 10),
    seed=42, verbose=False
)
nsga2_feats = np.where(nsga2_res.X[np.argmin(nsga2_res.F[:, 0])])[0]
print(f'  NSGA-II selected {len(nsga2_feats)} features in {time.time()-t0:.1f}s')
print(f'  Cache size (unique masks): {len(_nsga2_cache)}')

X_tr_nsga2 = X_sm_abl[:, nsga2_feats]
X_te_nsga2 = X_fused_te_s[:, nsga2_feats]
r, p = fast_eval(X_tr_nsga2, y_sm_abl, X_te_nsga2, 'A10: NSGA-II + Full Fusion')
ablation_results.append(r)
ablation_preds['A10: NSGA-II + Full Fusion'] = p
print('A10 done.')

# %% [markdown]
# ### 17.7 – Proposed Model (A11) — Uses Full 5000-tree Model Results

# %%
# A11 uses the full 5000-tree model already trained in Section 15
# No retraining — just record the existing preds and probs
print('Recording A11: Proposed model (full pipeline)...')

_p11 = preds          # from Section 15
_pb11 = probs         # from Section 15

r_proposed = {
    'Configuration': 'A11: Proposed (MOEA/D)',
    'N_Features':    len(selected_features),
    'Accuracy':      round(accuracy_score(y_te_abl, _p11), 4),
    'Precision':     round(precision_score(y_te_abl, _p11, average='macro', zero_division=0), 4),
    'Recall':        round(recall_score(y_te_abl, _p11, average='macro', zero_division=0), 4),
    'Macro_F1':      round(f1_score(y_te_abl, _p11, average='macro'), 4),
    'Weighted_F1':   round(f1_score(y_te_abl, _p11, average='weighted'), 4),
    'QWK':           round(cohen_kappa_score(y_te_abl, _p11, weights='quadratic'), 4),
    'AUC':           round(roc_auc_score(y_bin_abl, _pb11, multi_class='ovr'), 4),
    'PR_AUC':        round(average_precision_score(y_bin_abl, _pb11, average='macro'), 4),
    'Per_Class_F1':  f1_score(y_te_abl, _p11, average=None, labels=[0,1,2,3,4], zero_division=0),
}
ablation_results.append(r_proposed)
ablation_preds['A11: Proposed (MOEA/D)'] = _p11
print(f'  A11: Acc={r_proposed["Accuracy"]}  F1={r_proposed["Macro_F1"]}  AUC={r_proposed["AUC"]}')

# %% [markdown]
# ### 17.8 – Results Table

# %%
# Drop Per_Class_F1 column for clean display
display_cols = ['Configuration','N_Features','Accuracy','Precision',
                'Recall','Macro_F1','Weighted_F1','QWK','AUC','PR_AUC']
ablation_df  = pd.DataFrame([{k: r[k] for k in display_cols} for r in ablation_results])
ablation_df  = ablation_df.sort_values('Macro_F1', ascending=False).reset_index(drop=True)

print('\n' + '='*110)
print('ABLATION STUDY – COMPLETE RESULTS TABLE')
print('='*110)
print(ablation_df.to_string(index=False))
print('='*110)
ablation_df.to_csv('ablation_study_results.csv', index=False)
print('Saved: ablation_study_results.csv')

# %% [markdown]
# ### 17.9 – Grouped Bar Chart

# %%
metrics_to_plot = ['Accuracy', 'Macro_F1', 'QWK', 'AUC']
metric_labels   = ['Accuracy', 'Macro F1', 'QWK', 'AUC (OvR)']
bar_colors      = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

configs  = ablation_df['Configuration'].tolist()
x        = np.arange(len(configs))
bar_w    = 0.18

fig, ax = plt.subplots(figsize=(max(16, len(configs)*1.6), 7), dpi=150)

for i, (metric, mlabel, color) in enumerate(zip(metrics_to_plot, metric_labels, bar_colors)):
    vals = ablation_df[metric].values
    bars = ax.bar(x + (i - 2) * bar_w, vals, width=bar_w,
                  label=mlabel, color=color, alpha=0.87,
                  edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=90)

# Highlight proposed model
prop_idx = ablation_df.index[ablation_df['Configuration'].str.contains('A11')].tolist()
if prop_idx:
    ax.axvspan(prop_idx[0]-0.45, prop_idx[0]+0.45, alpha=0.1, color='gold', label='Proposed')

ax.set_xticks(x)
ax.set_xticklabels([c.replace(' ','\n') for c in configs], fontsize=8, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Ablation Study – Key Metrics per Configuration\n(proposed highlighted in gold)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('ablation_grouped_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: ablation_grouped_bar.png')

# %% [markdown]
# ### 17.10 – Radar Chart

# %%
radar_metrics = ['Accuracy','Macro_F1','Weighted_F1','QWK','AUC','PR_AUC']
radar_labels  = ['Accuracy','Macro F1','Weighted F1','QWK','AUC','PR-AUC']
N      = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
cmap   = plt.cm.get_cmap('tab10', len(ablation_df))

fig, ax = plt.subplots(figsize=(10, 10), dpi=150, subplot_kw=dict(polar=True))
for i, (_, row) in enumerate(ablation_df.iterrows()):
    vals   = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
    is_p   = 'A11' in row['Configuration']
    ax.plot(angles, vals, color=cmap(i),
            linewidth=3.0 if is_p else 1.2,
            linestyle='-' if is_p else '--',
            alpha=0.9 if is_p else 0.55,
            label=row['Configuration'])
    ax.fill(angles, vals, color=cmap(i), alpha=0.04)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3)
ax.set_title('Ablation – Multi-Metric Radar\n(bold = proposed A11)',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.15), fontsize=8, framealpha=0.7)
plt.tight_layout()
plt.savefig('ablation_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: ablation_radar_chart.png')

# %% [markdown]
# ### 17.11 – Per-Class F1 Heatmap (from Cached Predictions — No Retraining)

# %%
# Compute per-class F1 from cached predictions — zero additional training
heatmap_data = {}
for name, p in ablation_preds.items():
    heatmap_data[name] = f1_score(
        y_te_abl, p, average=None,
        labels=[0,1,2,3,4], zero_division=0
    )

heatmap_df = pd.DataFrame(heatmap_data, index=DR_CLASSES).T
# Sort rows to match ablation_df order
ordered_names = [r['Configuration'] for r in ablation_results if r['Configuration'] in heatmap_data]
heatmap_df = heatmap_df.reindex(ordered_names)

fig, ax = plt.subplots(figsize=(12, 0.55 * len(heatmap_df) + 2), dpi=150)
sns.heatmap(
    heatmap_df, annot=True, fmt='.3f',
    cmap='RdYlGn', vmin=0.0, vmax=1.0,
    linewidths=0.5,
    annot_kws={'fontsize': 9, 'fontweight': 'bold'},
    ax=ax
)
ax.set_xlabel('DR Grade', fontsize=12, fontweight='bold')
ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1 Heatmap – All Ablation Configurations\n(green=high, red=low)',
             fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, fontweight='bold', rotation=0)
plt.tight_layout()
plt.savefig('ablation_per_class_f1_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: ablation_per_class_f1_heatmap.png')

# %% [markdown]
# ### 17.12 – Feature Count vs. Accuracy Scatter

# %%
sel_variants = ['A7: Full Fusion (no selection)','A8: Full Fusion + PCA',
                'A9a: LASSO','A9b: RFE','A9c: mRMR',
                'A10: NSGA-II + Full Fusion','A11: Proposed (MOEA/D)']
sel_df = ablation_df[ablation_df['Configuration'].isin(sel_variants)].copy()

colors_scatter = {
    'A7: Full Fusion (no selection)': '#95a5a6',
    'A8: Full Fusion + PCA':          '#3498db',
    'A9a: LASSO':                     '#e67e22',
    'A9b: RFE':                       '#e74c3c',
    'A9c: mRMR':                      '#f39c12',
    'A10: NSGA-II + Full Fusion':     '#9b59b6',
    'A11: Proposed (MOEA/D)':         '#2ecc71'
}

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
for _, row in sel_df.iterrows():
    is_p  = 'A11' in row['Configuration']
    color = colors_scatter.get(row['Configuration'], '#333')
    ax.scatter(row['N_Features'], row['Accuracy'],
               s=220 if is_p else 120, color=color, zorder=5,
               edgecolors='black', linewidths=1.5 if is_p else 0.7,
               marker='*' if is_p else 'o')
    ax.annotate(row['Configuration'],
                (row['N_Features'], row['Accuracy']),
                textcoords='offset points', xytext=(8, 4),
                fontsize=8, fontweight='bold')
ax.set_xlabel('Number of Selected Features', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Feature Count vs. Accuracy\n(upper-left = ideal: fewer features, higher accuracy)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('ablation_feature_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: ablation_feature_vs_accuracy.png')

# %% [markdown]
# ### 17.13 – Incremental Fusion Gain

# %%
fusion_order = ['A1: CNN only','A4: CNN + DenseNet',
                'A7: Full Fusion (no selection)','A11: Proposed (MOEA/D)']
fusion_df = ablation_df[ablation_df['Configuration'].isin(fusion_order)].copy()
fusion_df = fusion_df.set_index('Configuration').reindex(fusion_order).reset_index()

acc_vals  = fusion_df['Accuracy'].values
gains     = np.diff(acc_vals, prepend=0)
gains[0]  = acc_vals[0]

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
bars = ax.bar(fusion_order, gains,
              color=['#3498db','#2ecc71','#e67e22','#e74c3c'],
              alpha=0.85, edgecolor='white', linewidth=0.8, width=0.5)
for bar, val, abs_val in zip(bars, gains, acc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'+{val:.4f}\n(Total: {abs_val:.4f})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Incremental Accuracy Gain', fontsize=12, fontweight='bold')
ax.set_title('Incremental Fusion Gain per Pipeline Stage', fontsize=12, fontweight='bold')
ax.set_xticklabels([c.replace(': ',':\n') for c in fusion_order], fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('ablation_incremental_gain.png', dpi=300, bbox_inches='tight')
plt.show()
print('Saved: ablation_incremental_gain.png')

# %% [markdown]
# ### 17.14 – McNemar's Test (from Cached Predictions — No Retraining)

# %%
# All predictions already cached in ablation_preds — zero additional training
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    p = binom.cdf(min(b, c), n, 0.5) * 2
    return min(p, 1.0)

p_proposed = ablation_preds['A11: Proposed (MOEA/D)']
correct_p  = (p_proposed == y_te_abl)

print('McNemar Test: Proposed Model (A11) vs All Baselines')
print('='*65)

mcnemar_results = []
for name, p_base in ablation_preds.items():
    if 'A11' in name:
        continue
    correct_b = (p_base == y_te_abl)
    b     = int(np.sum( correct_b & ~correct_p))
    c     = int(np.sum(~correct_b &  correct_p))
    p_val = mcnemar_exact(b, c)
    sig   = '***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns'))
    print(f'{name:<42}  b={b:4d}  c={c:4d}  p={p_val:.4f}  {sig}')
    mcnemar_results.append({
        'vs_Baseline': name,
        'b': b, 'c': c,
        'p_value': round(p_val, 5),
        'Significant': sig
    })

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv('ablation_mcnemar_test.csv', index=False)
print('\nSaved: ablation_mcnemar_test.csv')
print('Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant')

# %% [markdown]
# ### 17.15 – Summary and Total Ablation Runtime

# %%
proposed_row = ablation_df[ablation_df['Configuration'].str.contains('A11')].iloc[0]
a7_row       = ablation_df[ablation_df['Configuration'].str.contains('A7')].iloc[0]
best_single  = ablation_df[ablation_df['Configuration'].str.contains('A1|A2|A3')]['Macro_F1'].max()

fusion_gain    = proposed_row['Macro_F1'] - best_single
selection_gain = proposed_row['Macro_F1'] - a7_row['Macro_F1']

print('\n' + '='*70)
print('ABLATION STUDY – KEY FINDINGS')
print('='*70)
print(f'\n1. REPRESENTATION COMPLEMENTARITY')
print(f'   Best single-backbone Macro F1 : {best_single:.4f}')
print(f'   Proposed model Macro F1        : {proposed_row["Macro_F1"]:.4f}')
print(f'   Gain from multi-backbone fusion: +{fusion_gain:.4f} (+{fusion_gain*100:.2f}%)')
print(f'\n2. MOEA/D FEATURE SELECTION CONTRIBUTION')
print(f'   Full fusion (no selection) F1  : {a7_row["Macro_F1"]:.4f} ({int(a7_row["N_Features"])} features)')
print(f'   Proposed model (MOEA/D) F1     : {proposed_row["Macro_F1"]:.4f} ({int(proposed_row["N_Features"])} features)')
print(f'   Gain from MOEA/D selection     : +{selection_gain:.4f} with {int(a7_row["N_Features"])-int(proposed_row["N_Features"])} fewer features')
print(f'\n3. FULL RANKING (by Macro F1)')
for i, (_, row) in enumerate(ablation_df.iterrows()):
    marker = ' ← PROPOSED' if 'A11' in row['Configuration'] else ''
    print(f'   {i+1:2}. {row["Configuration"]:<42} F1={row["Macro_F1"]:.4f}  AUC={row["AUC"]:.4f}{marker}')

total_min = (time.time() - ABLATION_START) / 60
print(f'\nTotal ablation runtime: {total_min:.1f} minutes')
print('='*70)


