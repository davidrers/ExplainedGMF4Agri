"""
AlphaEarth Embedding Extraction + SVM/XGBoost Classification Pipeline
=====================================================================
Agent 2 — AlphaEarth
Extracts 64-dim satellite embeddings from Google AlphaEarth Foundations,
then trains SVM and XGBoost classifiers for crop classification.
"""

import os
import sys
import json
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import ee

# Output directory
BASE_DIR = r"c:\Users\darey\OneDrive - University of Twente\Documents\ML-Embeddings"
OUT_DIR = os.path.join(BASE_DIR, "results", "alphaearth")
os.makedirs(OUT_DIR, exist_ok=True)

# Logging
log_lines = []
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    with open(os.path.join(OUT_DIR, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))

# ============================================================
# STEP 1: Extract AlphaEarth embeddings
# ============================================================
def extract_embeddings():
    log("=== STEP 1: Extracting AlphaEarth Embeddings ===")

    embeddings_path = os.path.join(OUT_DIR, "embeddings.csv")
    if os.path.exists(embeddings_path):
        log(f"Embeddings file already exists at {embeddings_path}, loading...")
        df = pd.read_csv(embeddings_path)
        band_cols = [c for c in df.columns if c.startswith("emb_")]
        if len(band_cols) == 64 and len(df) == 4088:
            log(f"Loaded existing embeddings: {df.shape}")
            return df
        else:
            log(f"Existing file has {len(band_cols)} bands and {len(df)} rows, re-extracting...")

    t0 = time.time()

    # Initialize Earth Engine
    log("Initializing Earth Engine...")
    ee.Initialize()
    log("Earth Engine initialized.")

    # Load dataset
    log("Loading GeoPackage...")
    gdf = gpd.read_file(os.path.join(BASE_DIR, "commercial_crops_ml_ready.gpkg"))
    log(f"Loaded {len(gdf)} fields.")

    # Get the AlphaEarth embeddings for 2023
    log("Loading AlphaEarth embedding collection (2023)...")
    embeddings_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

    # Get bounding box of our data
    bounds = gdf.total_bounds  # [xmin, ymin, xmax, ymax]
    bbox = ee.Geometry.BBox(float(bounds[0]) - 0.01, float(bounds[1]) - 0.01,
                            float(bounds[2]) + 0.01, float(bounds[3]) + 0.01)

    emb_image = (embeddings_col
        .filterDate('2023-01-01', '2024-01-01')
        .filterBounds(bbox)
        .mosaic())

    band_names = [f'A{i:02d}' for i in range(64)]
    emb_image = emb_image.select(band_names)
    log("Embedding image prepared (64 bands).")

    # Process in batches to avoid EE limits
    # reduceRegions can handle ~5000 features at once but may timeout for complex polygons
    # We'll batch by ~200 features
    BATCH_SIZE = 200
    all_results = []
    n_batches = (len(gdf) + BATCH_SIZE - 1) // BATCH_SIZE

    log(f"Extracting embeddings in {n_batches} batches of ~{BATCH_SIZE}...")

    for batch_idx in range(n_batches):
        start_i = batch_idx * BATCH_SIZE
        end_i = min((batch_idx + 1) * BATCH_SIZE, len(gdf))
        batch_gdf = gdf.iloc[start_i:end_i]

        bt0 = time.time()

        # Convert to EE FeatureCollection
        features = []
        for idx, row in batch_gdf.iterrows():
            geom = row.geometry
            # Convert to GeoJSON
            geojson = geom.__geo_interface__
            ee_geom = ee.Geometry(geojson)
            props = {
                'cropfield': str(row['cropfield']),
                'class_id': int(row['class_id']),
                'split': str(row['split'])
            }
            features.append(ee.Feature(ee_geom, props))

        fc = ee.FeatureCollection(features)

        # Reduce regions - mean embedding per polygon
        reduced = emb_image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=10
        )

        # Get results
        try:
            result_list = reduced.getInfo()
            batch_data = []
            for feat in result_list['features']:
                props = feat['properties']
                row_data = {
                    'cropfield': props['cropfield'],
                    'class_id': props['class_id'],
                    'split': props['split']
                }
                for i, band in enumerate(band_names):
                    row_data[f'emb_{i}'] = props.get(band, np.nan)
                batch_data.append(row_data)

            all_results.extend(batch_data)
            elapsed = time.time() - bt0
            log(f"  Batch {batch_idx+1}/{n_batches}: {len(batch_data)} fields extracted in {elapsed:.1f}s")
        except Exception as e:
            log(f"  ERROR in batch {batch_idx+1}: {e}")
            # Try smaller sub-batches
            log(f"  Retrying batch {batch_idx+1} with smaller sub-batches...")
            SUB_BATCH = 50
            for sub_start in range(start_i, end_i, SUB_BATCH):
                sub_end = min(sub_start + SUB_BATCH, end_i)
                sub_gdf = gdf.iloc[sub_start:sub_end]
                try:
                    sub_features = []
                    for idx2, row2 in sub_gdf.iterrows():
                        geojson2 = row2.geometry.__geo_interface__
                        ee_geom2 = ee.Geometry(geojson2)
                        props2 = {
                            'cropfield': str(row2['cropfield']),
                            'class_id': int(row2['class_id']),
                            'split': str(row2['split'])
                        }
                        sub_features.append(ee.Feature(ee_geom2, props2))

                    sub_fc = ee.FeatureCollection(sub_features)
                    sub_reduced = emb_image.reduceRegions(
                        collection=sub_fc,
                        reducer=ee.Reducer.mean(),
                        scale=10
                    )
                    sub_result = sub_reduced.getInfo()
                    for feat in sub_result['features']:
                        props = feat['properties']
                        row_data = {
                            'cropfield': props['cropfield'],
                            'class_id': props['class_id'],
                            'split': props['split']
                        }
                        for i, band in enumerate(band_names):
                            row_data[f'emb_{i}'] = props.get(band, np.nan)
                        all_results.append(row_data)
                    log(f"    Sub-batch {sub_start}-{sub_end}: OK")
                except Exception as e2:
                    log(f"    Sub-batch {sub_start}-{sub_end} FAILED: {e2}")

        # Save intermediate progress every 5 batches
        if (batch_idx + 1) % 5 == 0:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(os.path.join(OUT_DIR, "embeddings_partial.csv"), index=False)
            save_log()

    # Create final DataFrame
    df = pd.DataFrame(all_results)

    # Check for NaN embeddings
    emb_cols = [f'emb_{i}' for i in range(64)]
    n_nan = df[emb_cols].isna().any(axis=1).sum()
    log(f"Total extracted: {len(df)} fields, {n_nan} with NaN embeddings")

    if n_nan > 0:
        # Fill NaN with 0 (fields too small for 10m resolution)
        log(f"Filling {n_nan} NaN embedding rows with 0")
        df[emb_cols] = df[emb_cols].fillna(0)

    # Save embeddings
    df.to_csv(embeddings_path, index=False)
    elapsed = time.time() - t0
    log(f"Embeddings saved to {embeddings_path} ({elapsed:.1f}s total)")

    # Clean up partial file
    partial_path = os.path.join(OUT_DIR, "embeddings_partial.csv")
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return df


# ============================================================
# STEP 2 & 3: Train classifiers and produce outputs
# ============================================================
def train_and_evaluate(df):
    log("=== STEP 2: Training Classifiers ===")
    t0 = time.time()

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure xgboost is installed
    try:
        from xgboost import XGBClassifier
    except ImportError:
        log("Installing xgboost...")
        os.system(f"{sys.executable} -m pip install xgboost -q")
        from xgboost import XGBClassifier

    # Class names in order of class_id 1-9
    class_names = [
        "Cereals & Grains",
        "Potatoes & Root Vegetables",
        "Vegetables",
        "Legumes",
        "Onions & Bulbs",
        "Fruit Trees",
        "Herbs & Spices",
        "Oil & Fiber Crops",
        "Seeds & Nursery"
    ]

    # Prepare data
    emb_cols = [f'emb_{i}' for i in range(64)]

    # Split
    train_val_mask = df['split'].isin(['train', 'validation'])
    test_mask = df['split'] == 'test'

    X_trainval = df.loc[train_val_mask, emb_cols].values
    y_trainval = df.loc[train_val_mask, 'class_id'].values
    X_test = df.loc[test_mask, emb_cols].values
    y_test = df.loc[test_mask, 'class_id'].values

    log(f"Train+Val: {len(X_trainval)}, Test: {len(X_test)}")
    log(f"Train+Val class distribution: {np.bincount(y_trainval, minlength=10)[1:]}")
    log(f"Test class distribution: {np.bincount(y_test, minlength=10)[1:]}")

    # Standardize
    log("Standardizing embeddings (fit on train+val)...")
    scaler = StandardScaler()
    X_trainval_s = scaler.fit_transform(X_trainval)
    X_test_s = scaler.transform(X_test)

    # --- SVM with GridSearchCV ---
    log("Training SVM with RBF kernel (GridSearchCV)...")
    t_svm = time.time()

    svm_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_grid = GridSearchCV(
        SVC(kernel='rbf', random_state=42, class_weight='balanced'),
        svm_params,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    svm_grid.fit(X_trainval_s, y_trainval)

    svm_best = svm_grid.best_estimator_
    svm_time = time.time() - t_svm
    log(f"SVM best params: {svm_grid.best_params_}")
    log(f"SVM best CV f1_macro: {svm_grid.best_score_:.4f}")
    log(f"SVM training time: {svm_time:.1f}s")

    # SVM predictions on test
    y_pred_svm = svm_best.predict(X_test_s)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1_macro = f1_score(y_test, y_pred_svm, average='macro')
    svm_f1_weighted = f1_score(y_test, y_pred_svm, average='weighted')
    log(f"SVM Test — Accuracy: {svm_acc:.4f}, F1 macro: {svm_f1_macro:.4f}, F1 weighted: {svm_f1_weighted:.4f}")

    # SVM classification report
    svm_report = classification_report(y_test, y_pred_svm,
                                        labels=list(range(1, 10)),
                                        target_names=class_names,
                                        output_dict=True)
    with open(os.path.join(OUT_DIR, "classification_report_svm.json"), "w") as f:
        json.dump(svm_report, f, indent=2)
    log("SVM classification report saved.")

    # SVM confusion matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=list(range(1, 10)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('SVM Confusion Matrix (AlphaEarth Embeddings)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_svm.png"), dpi=150)
    plt.close()
    log("SVM confusion matrix saved.")

    # --- XGBoost ---
    log("Training XGBoost classifier...")
    t_xgb = time.time()

    # XGBoost with reasonable hyperparameter search
    xgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    xgb_base = XGBClassifier(
        objective='multi:softprob',
        num_class=9,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist'
    )

    xgb_grid = GridSearchCV(
        xgb_base,
        xgb_params,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )

    # XGBoost expects labels 0-indexed
    y_trainval_xgb = y_trainval - 1
    y_test_xgb = y_test - 1

    xgb_grid.fit(X_trainval_s, y_trainval_xgb)

    xgb_best = xgb_grid.best_estimator_
    xgb_time = time.time() - t_xgb
    log(f"XGBoost best params: {xgb_grid.best_params_}")
    log(f"XGBoost best CV f1_macro: {xgb_grid.best_score_:.4f}")
    log(f"XGBoost training time: {xgb_time:.1f}s")

    # XGBoost predictions on test (convert back to 1-indexed)
    y_pred_xgb = xgb_best.predict(X_test_s) + 1

    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_f1_macro = f1_score(y_test, y_pred_xgb, average='macro')
    xgb_f1_weighted = f1_score(y_test, y_pred_xgb, average='weighted')
    log(f"XGBoost Test — Accuracy: {xgb_acc:.4f}, F1 macro: {xgb_f1_macro:.4f}, F1 weighted: {xgb_f1_weighted:.4f}")

    # XGBoost classification report
    xgb_report = classification_report(y_test, y_pred_xgb,
                                        labels=list(range(1, 10)),
                                        target_names=class_names,
                                        output_dict=True)
    with open(os.path.join(OUT_DIR, "classification_report_xgb.json"), "w") as f:
        json.dump(xgb_report, f, indent=2)
    log("XGBoost classification report saved.")

    # XGBoost confusion matrix
    cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=list(range(1, 10)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('XGBoost Confusion Matrix (AlphaEarth Embeddings)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_xgb.png"), dpi=150)
    plt.close()
    log("XGBoost confusion matrix saved.")

    # Metrics summary
    metrics_summary = {
        "svm": {
            "accuracy": round(svm_acc, 4),
            "f1_macro": round(svm_f1_macro, 4),
            "f1_weighted": round(svm_f1_weighted, 4)
        },
        "xgb": {
            "accuracy": round(xgb_acc, 4),
            "f1_macro": round(xgb_f1_macro, 4),
            "f1_weighted": round(xgb_f1_weighted, 4)
        }
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    log(f"Metrics summary saved: {json.dumps(metrics_summary)}")

    total_time = time.time() - t0
    log(f"Total classification time: {total_time:.1f}s")

    return metrics_summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log("Pipeline started.")
    overall_t0 = time.time()

    try:
        df = extract_embeddings()
        metrics = train_and_evaluate(df)
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        log(traceback.format_exc())

    overall_time = time.time() - overall_t0
    log(f"Pipeline completed in {overall_time:.1f}s")
    save_log()
    print("\nDone! Check results/alphaearth/ for outputs.")
