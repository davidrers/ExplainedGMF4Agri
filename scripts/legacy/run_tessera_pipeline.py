"""
Tessera Embedding Extraction + SVM/XGBoost Classification Pipeline
Agent 1 — Tessera
"""

import os, sys, json, time, datetime, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path(r"c:/Users/darey/OneDrive - University of Twente/Documents/ML-Embeddings")
GPKG = BASE / "commercial_crops_ml_ready.gpkg"
OUT  = BASE / "results" / "tessera"
OUT.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    log_lines.append(line)
    print(line, flush=True)

def save_log():
    (OUT / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Extract Tessera embeddings
# ═══════════════════════════════════════════════════════════════════════
log("Loading GeoPackage...")
gdf = gpd.read_file(GPKG)
log(f"Loaded {len(gdf)} fields, columns: {list(gdf.columns)}")
log(f"CRS: {gdf.crs}")
log(f"Split distribution: {gdf['split'].value_counts().to_dict()}")

# Compute centroids in EPSG:4326 for point sampling
centroids = gdf.geometry.centroid
lons = centroids.x.values
lats = centroids.y.values
points = list(zip(lons.tolist(), lats.tolist()))
log(f"Computed {len(points)} centroids for point sampling")
log(f"Lon range: [{lons.min():.4f}, {lons.max():.4f}], Lat range: [{lats.min():.4f}, {lats.max():.4f}]")

# Check if embeddings already exist
emb_path = OUT / "embeddings.csv"
if emb_path.exists():
    log("embeddings.csv already exists, loading from cache...")
    emb_df = pd.read_csv(emb_path)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    log(f"Loaded cached embeddings: {len(emb_df)} rows, {len(emb_cols)} dims")
else:
    log("Extracting Tessera embeddings via sample_embeddings_at_points (year=2023)...")
    from geotessera import GeoTessera
    gt = GeoTessera()

    t0 = time.time()
    # Sample embeddings — the library accepts list of (lon, lat) tuples
    embeddings = gt.sample_embeddings_at_points(points, year=2023)
    elapsed = time.time() - t0
    log(f"Embedding extraction took {elapsed:.1f}s, shape: {embeddings.shape}")

    # Check for NaN rows (missing tiles)
    nan_mask = np.isnan(embeddings).any(axis=1)
    n_nan = nan_mask.sum()
    log(f"NaN rows (missing coverage): {n_nan} / {len(embeddings)}")

    # Build DataFrame
    n_dims = embeddings.shape[1]
    emb_col_names = [f"emb_{i}" for i in range(n_dims)]
    emb_df = pd.DataFrame(embeddings, columns=emb_col_names)
    emb_df.insert(0, "cropfield", gdf["cropfield"].values)
    emb_df.insert(1, "class_id", gdf["class_id"].values)
    emb_df.insert(2, "split", gdf["split"].values)

    # Save
    emb_df.to_csv(emb_path, index=False)
    log(f"Saved embeddings to {emb_path}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Classification
# ═══════════════════════════════════════════════════════════════════════
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]

# Drop rows with NaN embeddings
valid_mask = emb_df[emb_cols].notna().all(axis=1)
n_dropped = (~valid_mask).sum()
if n_dropped > 0:
    log(f"Dropping {n_dropped} rows with NaN embeddings")
    emb_df = emb_df[valid_mask].reset_index(drop=True)

# Split data
train_val_mask = emb_df["split"].isin(["train", "validation"])
test_mask = emb_df["split"] == "test"

X_trainval = emb_df.loc[train_val_mask, emb_cols].values
y_trainval = emb_df.loc[train_val_mask, "class_id"].values
X_test = emb_df.loc[test_mask, emb_cols].values
y_test = emb_df.loc[test_mask, "class_id"].values

log(f"Train+Val: {len(X_trainval)}, Test: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_trainval_s = scaler.fit_transform(X_trainval)
X_test_s = scaler.transform(X_test)
log("Standardized embeddings (fit on train+val)")

# Class names in class_id order 1-9
class_order = sorted(gdf["class_id"].unique())
class_name_map = gdf.drop_duplicates("class_id").set_index("class_id")["class_name"].to_dict()
class_names = [class_name_map[cid] for cid in class_order]
log(f"Class order: {list(zip(class_order, class_names))}")

# ── SVM ────────────────────────────────────────────────────────────────
log("Training SVM with GridSearchCV (RBF kernel)...")
t0 = time.time()
svm_param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}
svm_cv = GridSearchCV(
    SVC(kernel="rbf", class_weight="balanced", random_state=42),
    svm_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
svm_cv.fit(X_trainval_s, y_trainval)
svm_elapsed = time.time() - t0
log(f"SVM GridSearch done in {svm_elapsed:.1f}s, best params: {svm_cv.best_params_}, best CV f1_macro: {svm_cv.best_score_:.4f}")

svm_model = svm_cv.best_estimator_
y_pred_svm = svm_model.predict(X_test_s)

svm_report = classification_report(y_test, y_pred_svm, labels=class_order,
                                    target_names=class_names, output_dict=True)
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_f1_macro = f1_score(y_test, y_pred_svm, average="macro")
svm_f1_weighted = f1_score(y_test, y_pred_svm, average="weighted")
log(f"SVM Test — Accuracy: {svm_acc:.4f}, F1-macro: {svm_f1_macro:.4f}, F1-weighted: {svm_f1_weighted:.4f}")

# ── XGBoost ────────────────────────────────────────────────────────────
log("Training XGBoost with GridSearchCV...")
t0 = time.time()
xgb_param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 400],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
xgb_cv = GridSearchCV(
    xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_order),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    ),
    xgb_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
# XGBoost expects labels 0..K-1
y_trainval_xgb = y_trainval - 1
y_test_xgb = y_test - 1
xgb_cv.fit(X_trainval_s, y_trainval_xgb)
xgb_elapsed = time.time() - t0
log(f"XGBoost GridSearch done in {xgb_elapsed:.1f}s, best params: {xgb_cv.best_params_}, best CV f1_macro: {xgb_cv.best_score_:.4f}")

xgb_model = xgb_cv.best_estimator_
y_pred_xgb_raw = xgb_model.predict(X_test_s)
y_pred_xgb = y_pred_xgb_raw + 1  # back to 1-based

xgb_report = classification_report(y_test, y_pred_xgb, labels=class_order,
                                    target_names=class_names, output_dict=True)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_f1_macro = f1_score(y_test, y_pred_xgb, average="macro")
xgb_f1_weighted = f1_score(y_test, y_pred_xgb, average="weighted")
log(f"XGBoost Test — Accuracy: {xgb_acc:.4f}, F1-macro: {xgb_f1_macro:.4f}, F1-weighted: {xgb_f1_weighted:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Save outputs
# ═══════════════════════════════════════════════════════════════════════

# Classification reports
with open(OUT / "classification_report_svm.json", "w") as f:
    json.dump(svm_report, f, indent=2)
log("Saved classification_report_svm.json")

with open(OUT / "classification_report_xgb.json", "w") as f:
    json.dump(xgb_report, f, indent=2)
log("Saved classification_report_xgb.json")

# Metrics summary
metrics_summary = {
    "svm": {"accuracy": round(svm_acc, 4), "f1_macro": round(svm_f1_macro, 4), "f1_weighted": round(svm_f1_weighted, 4)},
    "xgb": {"accuracy": round(xgb_acc, 4), "f1_macro": round(xgb_f1_macro, 4), "f1_weighted": round(xgb_f1_weighted, 4)},
}
with open(OUT / "metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
log("Saved metrics_summary.json")

# Confusion matrices
def plot_cm(y_true, y_pred, labels, class_names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Saved {path.name}")

plot_cm(y_test, y_pred_svm, class_order, class_names,
        "SVM Confusion Matrix (Test Set)", OUT / "confusion_matrix_svm.png")
plot_cm(y_test, y_pred_xgb, class_order, class_names,
        "XGBoost Confusion Matrix (Test Set)", OUT / "confusion_matrix_xgb.png")

log("Pipeline complete.")
save_log()
print("\nDone. All outputs in:", OUT)
