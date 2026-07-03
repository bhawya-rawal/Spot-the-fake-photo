import os
import time
import json
import pickle
import logging
import hashlib
import warnings
import csv
import argparse
from pathlib import Path
import numpy as np
import cv2

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import xgboost as xgb

# Reuse the existing Phase 1 feature extraction pipeline
from features import extract_features, _to_gray, LBP_P
from patches import extract_patches

warnings.filterwarnings("ignore")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("train")

from config import (
    RANDOM_SEED, SUPPORTED_EXTENSIONS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE
)
from utils import get_file_hash

def discover_and_validate_dataset(base_path: Path) -> tuple[list[Path], list[int], dict]:
    """
    Scans real and fake folders, validates images, identifies MD5 duplicates,
    detects corrupted files, warns on class imbalances, and returns validated data.
    """
    real_dir = base_path / "real"
    fake_dir = base_path / "fake"
    
    raw_files = []
    labels = []
    
    for label, directory in [(0, real_dir), (1, fake_dir)]:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory.absolute()}")
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                raw_files.append(entry)
                labels.append(label)
                
    hash_map = {}
    duplicates = []
    corrupted_count = 0
    valid_files = []
    valid_labels = []
    
    for path, label in zip(raw_files, labels):
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Corrupted or unreadable image skipped: {path}")
            corrupted_count += 1
            continue
            
        try:
            file_hash = get_file_hash(path)
            if file_hash in hash_map:
                duplicates.append((path, hash_map[file_hash]))
                logger.warning(f"Duplicate image skipped: {path.name} is identical to {hash_map[file_hash].name}")
                continue
            hash_map[file_hash] = path
        except Exception as e:
            logger.warning(f"Failed to check hash for {path}: {e}. Skipping.")
            continue
            
        valid_files.append(path)
        valid_labels.append(label)
        
    num_real = sum(1 for l in valid_labels if l == 0)
    num_fake = sum(1 for l in valid_labels if l == 1)
    total = len(valid_labels)
    
    imbalance_deviation = 0.0
    if total > 0:
        imbalance_deviation = abs(num_real - num_fake) / total
        if imbalance_deviation > 0.10:
            logger.warning(
                f"Class imbalance warning! Deviation exceeds 10% (current: {imbalance_deviation*100:.2f}%). "
                f"Real: {num_real} ({num_real/total*100:.1f}%), Fake: {num_fake} ({num_fake/total*100:.1f}%)"
            )
            
    summary_data = {
        "raw_discovered": len(raw_files),
        "total_validated": total,
        "real_count": num_real,
        "fake_count": num_fake,
        "duplicates_detected": len(duplicates),
        "corrupted_detected": corrupted_count,
        "imbalance_deviation": imbalance_deviation
    }
    
    return valid_files, valid_labels, summary_data

def compute_dataset_stats_and_plots(files: list[Path], labels: list[int]) -> dict:
    """
    Computes image resolutions, aspect ratios, and average brightness stats.
    Generates and saves the brightness distribution histogram.
    """
    widths = []
    heights = []
    aspect_ratios = []
    brightnesses = []
    
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(np.mean(gray)))
        
    stats = {
        "width": {
            "mean": float(np.mean(widths)), "std": float(np.std(widths)),
            "min": int(np.min(widths)), "max": int(np.max(widths))
        },
        "height": {
            "mean": float(np.mean(heights)), "std": float(np.std(heights)),
            "min": int(np.min(heights)), "max": int(np.max(heights))
        },
        "aspect_ratio": {
            "mean": float(np.mean(aspect_ratios)), "std": float(np.std(aspect_ratios)),
            "min": float(np.min(aspect_ratios)), "max": float(np.max(aspect_ratios))
        },
        "brightness": {
            "mean": float(np.mean(brightnesses)), "std": float(np.std(brightnesses)),
            "min": float(np.min(brightnesses)), "max": float(np.max(brightnesses))
        }
    }
    
    os.makedirs("analysis", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(brightnesses, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Average Image Brightness (Grayscale Mean)")
    plt.ylabel("Frequency")
    plt.title("Image Brightness Distribution")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("analysis/brightness_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats

def split_dataset(
    files: list[Path], 
    labels: list[int]
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    """Performs a stratified split to divide the files into Train (70%), Val (15%), Test (15%)."""
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        files, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    val_fraction = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_fraction,
        random_state=RANDOM_SEED,
        stratify=train_val_labels
    )
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

def extract_features_for_split(files: list[Path], labels: list[int]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Sequentially extracts feature vectors. Filters and returns successful records."""
    X = []
    y = []
    valid_paths = []
    
    for path, label in zip(files, labels):
        try:
            img = cv2.imread(str(path))
            if img is None:
                continue
            feats = extract_features(img)
            X.append(feats)
            y.append(label)
            valid_paths.append(path)
        except Exception as e:
            logger.warning(f"Error extracting features from {path}: {e}")
            continue
            
    if len(X) == 0:
        raise ValueError("No features extracted. Ensure images are present and extraction logic works.")
        
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), valid_paths

def get_feature_names() -> list[str]:
    """Generates the 87 descriptive labels mapping back to the extracted features."""
    base_names = [
        # FFT Features (12)
        "fft_peak_strength",
        "fft_peak_ratio",
        "fft_radial_energy_low",
        "fft_radial_energy_mid",
        "fft_radial_energy_high",
        "fft_spectral_entropy",
        "fft_high_freq_ratio",
        "fft_residual_spike_strength",
        "fft_peak_count",
        "fft_peak_sharpness",
        "fft_anisotropy",
        "fft_off_axis_peak_strength",
        
        # LBP Features (10)
        * [f"lbp_bin_{i}" for i in range(10)],
        
        # Laplacian Features (2)
        "laplacian_mean",
        "laplacian_var",
        
        # Canny Features (1)
        "canny_density",
        
        # HSV Features (4)
        "hsv_s_mean",
        "hsv_s_var",
        "hsv_v_mean",
        "hsv_v_var"
    ]
    
    # Aggregation prefixes (Mean, Var, Max)
    names = []
    names.extend([f"mean_{n}" for n in base_names])
    names.extend([f"var_{n}" for n in base_names])
    names.extend([f"max_{n}" for n in base_names])
    return names

def optimize_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Searches thresholds from 0.30 to 0.70 with 0.001 steps to maximize Validation Accuracy."""
    thresholds = np.arange(0.30, 0.701, 0.001)
    best_th = 0.50
    best_acc = -1.0
    
    for th in thresholds:
        preds = (probs >= th).astype(int)
        score = accuracy_score(y, preds)
        if score > best_acc:
            best_acc = score
            best_th = th
        elif abs(score - best_acc) < 1e-7:
            if abs(th - 0.50) < abs(best_th - 0.50):
                best_th = th
                
    return float(best_th)

def train_and_select_model(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray
) -> tuple[str, object]:
    """
    Combines train and val sets, performs Stratified 5-Fold Cross-Validation,
    searches hyperparameters for LightGBM and XGBoost, optimizes ensemble weights,
    and returns a VotingClassifier trained on the full combined dataset.
    """
    X_comb = np.vstack([X_tr, X_va])
    y_comb = np.concatenate([y_tr, y_va])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # 1. Hyperparameter grid definition
    lgb_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100},
    ]
    xgb_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 50},
        {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100},
    ]
    
    logger.info("Executing 5-Fold Cross-Validation Hyperparameter Search...")
    
    # Grid search for LightGBM
    best_lgb_params = None
    best_lgb_acc = -1.0
    for params in lgb_grid:
        fold_accs = []
        for train_idx, val_idx in cv.split(X_comb, y_comb):
            clf = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1, **params)
            clf.fit(X_comb[train_idx], y_comb[train_idx])
            preds = clf.predict(X_comb[val_idx])
            fold_accs.append(accuracy_score(y_comb[val_idx], preds))
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_lgb_acc:
            best_lgb_acc = mean_acc
            best_lgb_params = params
            
    # Grid search for XGBoost
    best_xgb_params = None
    best_xgb_acc = -1.0
    for params in xgb_grid:
        fold_accs = []
        for train_idx, val_idx in cv.split(X_comb, y_comb):
            clf = xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", **params)
            clf.fit(X_comb[train_idx], y_comb[train_idx])
            preds = clf.predict(X_comb[val_idx])
            fold_accs.append(accuracy_score(y_comb[val_idx], preds))
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_xgb_acc:
            best_xgb_acc = mean_acc
            best_xgb_params = params
            
    logger.info(f"Optimal LightGBM Params: {best_lgb_params} (CV Accuracy: {best_lgb_acc:.4f})")
    logger.info(f"Optimal XGBoost Params: {best_xgb_params} (CV Accuracy: {best_xgb_acc:.4f})")
    
    # 2. Ensemble weight optimization
    logger.info("Optimizing soft-voting ensemble weights...")
    best_w = 0.5
    best_ens_acc = -1.0
    weights_to_search = np.linspace(0.0, 1.0, 11)
    
    for w in weights_to_search:
        fold_accs = []
        for train_idx, val_idx in cv.split(X_comb, y_comb):
            lgb_clf = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1, **best_lgb_params)
            xgb_clf = xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", **best_xgb_params)
            
            ensemble = VotingClassifier(
                estimators=[('lgbm', lgb_clf), ('xgb', xgb_clf)],
                voting='soft',
                weights=[w, 1.0 - w]
            )
            ensemble.fit(X_comb[train_idx], y_comb[train_idx])
            preds = ensemble.predict(X_comb[val_idx])
            fold_accs.append(accuracy_score(y_comb[val_idx], preds))
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_ens_acc:
            best_ens_acc = mean_acc
            best_w = w
            
    logger.info(f"Optimal Ensemble Weight (LGBM vs XGB): [{best_w:.2f}, {1.0 - best_w:.2f}] (CV Accuracy: {best_ens_acc:.4f})")
    
    # 3. Fit final production VotingClassifier on the full combined dataset
    final_lgb = lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1, **best_lgb_params)
    final_xgb = xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", **best_xgb_params)
    
    final_ensemble = VotingClassifier(
        estimators=[
            ('lgbm', final_lgb),
            ('xgb', final_xgb)
        ],
        voting='soft',
        weights=[best_w, 1.0 - best_w]
    )
    final_ensemble.fit(X_comb, y_comb)
    
    return "VotingEnsemble(LGBM+XGB)", final_ensemble

def plot_curves_and_confusion_matrix(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    model_name: str
):
    """Generates and saves test ROC curve, PR curve, and Confusion Matrix."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    plt.figure()
    plt.plot(recall, precision, color='green', lw=2, label=f'PR (area = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("pr_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    conf = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=["Real", "Fake"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance_top20(importances: np.ndarray, feature_names: list[str]):
    """Saves a ranked horizontal bar chart of the top 20 features."""
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:20]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), top_importances[::-1], align='center', color='indigo')
    plt.yticks(range(20), top_names[::-1])
    plt.xlabel('Importance Value')
    plt.title('Top 20 Most Important Handcrafted Features')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_correlation_heatmap(X_sel: np.ndarray, feature_names_sel: list[str]):
    """Saves the feature correlation heatmap for active features."""
    num_to_plot = min(25, X_sel.shape[1])
    corr = np.corrcoef(X_sel[:, :num_to_plot], rowvar=False)
    
    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(range(num_to_plot), feature_names_sel[:num_to_plot], rotation=90, fontsize=8)
    plt.yticks(range(num_to_plot), feature_names_sel[:num_to_plot], fontsize=8)
    plt.title(f"Feature Correlation Heatmap (First {num_to_plot} Active Features)")
    plt.savefig("analysis/feature_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_misclassified_gallery(misclassified_list: list[dict]):
    """Generates a grid layout of the misclassified test images."""
    if not misclassified_list:
        return
    num_imgs = min(16, len(misclassified_list))
    cols = 4
    rows = (num_imgs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    for i in range(len(axes)):
        ax = axes[i]
        if i < num_imgs:
            record = misclassified_list[i]
            img = cv2.imread(record["image_path"])
            if img is not None:
                patches = extract_patches(img)
                center_patch = patches[4]
                patch_rgb = cv2.cvtColor(center_patch, cv2.COLOR_BGR2RGB)
                ax.imshow(patch_rgb)
                
                true_lbl = "Fake" if record["true_label"] == 1 else "Real"
                pred_lbl = "Fake" if record["predicted_label"] == 1 else "Real"
                ax.set_title(f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {record['confidence']*100:.1f}%\n{Path(record['image_path']).name}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("analysis/misclassified_gallery.png", dpi=150, bbox_inches='tight')
    plt.close()

def generate_fft_diagnostics_plot(test_paths: list[Path], y_test: np.ndarray, test_probs: np.ndarray, test_preds: np.ndarray):
    """Generates FFT spectra diagnostic plot with overlaid detected frequency peak markers."""
    idx_cr, idx_fp, idx_cf, idx_fn = None, None, None, None
    for idx, (true, pred) in enumerate(zip(y_test, test_preds)):
        if true == 0 and pred == 0 and idx_cr is None: idx_cr = idx
        elif true == 0 and pred == 1 and idx_fp is None: idx_fp = idx
        elif true == 1 and pred == 1 and idx_cf is None: idx_cf = idx
        elif true == 1 and pred == 0 and idx_fn is None: idx_fn = idx
        
    categories = [
        ("Correct Real", idx_cr),
        ("Incorrect Real (FP)", idx_fp),
        ("Correct Fake", idx_cf),
        ("Incorrect Fake (FN)", idx_fn)
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for col_idx, (title, idx) in enumerate(categories):
        ax_img = axes[0, col_idx]
        ax_fft = axes[1, col_idx]
        
        if idx is None:
            ax_img.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax_fft.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax_img.set_title(title)
            continue
            
        path = test_paths[idx]
        img = cv2.imread(str(path))
        patches = extract_patches(img)
        center_patch = patches[4]
        
        patch_rgb = cv2.cvtColor(center_patch, cv2.COLOR_BGR2RGB)
        ax_img.imshow(patch_rgb)
        ax_img.set_title(f"{title}\n{Path(path).name}")
        ax_img.axis('off')
        
        gray = _to_gray(center_patch)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        window = np.outer(np.hanning(h), np.hanning(w))
        windowed = gray.astype(np.float64) * window
        fft_shifted = np.fft.fftshift(np.fft.fft2(windowed))
        magnitude = np.abs(fft_shifted)
        log_mag = np.log10(magnitude + 1e-10)
        
        im = ax_fft.imshow(log_mag, cmap='viridis')
        ax_fft.set_title("FFT Spectrum (Log)")
        ax_fft.axis('off')
        
        magnitude_no_dc = magnitude.copy()
        magnitude_no_dc[cy, cx] = 0.0
        adaptive_threshold = np.mean(magnitude_no_dc) + 3.0 * np.std(magnitude_no_dc)
        local_max = cv2.dilate(magnitude, np.ones((3, 3))) == magnitude
        y_grid, x_grid = np.ogrid[:h, :w]
        r = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        peaks = local_max & (magnitude > adaptive_threshold) & (r >= 5)
        py_indices, px_indices = np.where(peaks)
        
        if len(px_indices) > 0:
            ax_fft.scatter(px_indices, py_indices, color='red', s=20, marker='x', label='Forensic Peaks' if col_idx == 0 else "")
            if col_idx == 0:
                ax_fft.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=8)
                
        fig.colorbar(im, ax=ax_fft, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig("analysis/fft_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def run_error_analysis(
    test_files: list[Path],
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray
) -> list[dict]:
    """Identifies and exports misclassified items sorted by confidence."""
    misclassified = []
    for path, true, pred, prob in zip(test_files, y_true, preds, probs):
        if true != pred:
            conf = prob if pred == 1 else 1.0 - prob
            misclassified.append({
                "image_path": str(path),
                "true_label": int(true),
                "predicted_label": int(pred),
                "confidence": float(conf)
            })
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    with open("misclassified.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_label", "predicted_label", "confidence"])
        writer.writeheader()
        writer.writerows(misclassified)
    return misclassified

def run_ablation_experiments(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_names):
    """
    Executes Phase 3 Ablation Experiments:
      - Model A: All features (87 features)
      - Model B: Without HSV mean features (81 features)
      - Model C: Frequency/Texture only (75 features)
    """
    logger.info("Executing Ablation Study...")
    
    indices_a = list(range(len(feature_names)))
    indices_b = [i for i, name in enumerate(feature_names) if not ("hsv_s_mean" in name or "hsv_v_mean" in name)]
    indices_c = [i for i, name in enumerate(feature_names) if "hsv" not in name]
    
    experiments = [
        ("Model A (All Features)", indices_a),
        ("Model B (Without HSV Mean)", indices_b),
        ("Model C (Freq/Texture Only)", indices_c)
    ]
    
    ablation_results = {}
    
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    
    print("\n" + "="*70)
    print("                      ABLATION STUDY RESULTS")
    print("="*70)
    
    for title, indices in experiments:
        X_tr_cfg = X_tr[:, indices]
        X_va_cfg = X_va[:, indices]
        X_te_cfg = X_te[:, indices]
        
        best_name, best_model = train_and_select_model(X_tr_cfg, y_tr, X_va_cfg, y_va)
        
        val_probs = best_model.predict_proba(X_va_cfg)[:, 1]
        optimal_th = optimize_threshold(val_probs, y_va)
        
        test_probs = best_model.predict_proba(X_te_cfg)[:, 1]
        test_preds = (test_probs >= optimal_th).astype(int)
        
        val_auc = roc_auc_score(y_va, val_probs)
        val_f1 = f1_score(y_va, (val_probs >= optimal_th).astype(int), zero_division=0)
        
        test_auc = roc_auc_score(y_te, test_probs)
        test_f1 = f1_score(y_te, test_preds, zero_division=0)
        
        ablation_results[title] = {
            "val_roc_auc": val_auc, "val_f1": val_f1,
            "test_roc_auc": test_auc, "test_f1": test_f1,
            "winner_model": best_name
        }
        
        print(f"{title:<28} | Winner: {best_name:<20}")
        print(f"  └─ Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Test AUC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
        
        fpr, tpr, _ = roc_curve(y_te, test_probs)
        plt.plot(fpr, tpr, lw=2, label=f'{title} (AUC = {test_auc:.4f})')
        
    print("="*70 + "\n")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison - Ablation Study')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("analysis/ablation_roc_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Photo Detection Training and Diagnostics")
    parser.add_argument("--ablation", action="store_true", help="Run feature ablation experiments")
    args = parser.parse_args()

    dataset_path = Path("dataset")
    if not dataset_path.exists():
        logger.error("Dataset folder 'dataset/' does not exist in workspace.")
        return

    logger.info("Scanning dataset folders...")
    files, labels, validation_summary = discover_and_validate_dataset(dataset_path)
    
    if not files:
        logger.error("No valid image files found.")
        return
        
    logger.info("Compiling image resolution and brightness statistics...")
    resolution_stats = compute_dataset_stats_and_plots(files, labels)
    
    dataset_summary = {
        **validation_summary,
        "resolution_and_brightness_statistics": resolution_stats
    }
    with open("dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=4)
    logger.info("Saved dataset_summary.json")

    logger.info("Performing stratified splits (70% Train, 15% Val, 15% Test)...")
    train_files, train_labels, val_files, val_labels, test_files, test_labels = split_dataset(files, labels)
    
    logger.info("Extracting features for splits...")
    X_train, y_train, train_paths = extract_features_for_split(train_files, train_labels)
    X_val, y_val, val_paths = extract_features_for_split(val_files, val_labels)
    X_test, y_test, test_paths = extract_features_for_split(test_files, test_labels)
    
    num_features = X_train.shape[1]
    feature_names = get_feature_names()
    
    active_indices = [i for i, name in enumerate(feature_names) if not ("hsv_s_mean" in name or "hsv_v_mean" in name)]
    feature_names_sel = [feature_names[i] for i in active_indices]
    
    if args.ablation:
        run_ablation_experiments(X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        
    logger.info("Training production model (Model B - Without HSV Mean)...")
    X_train_sel = X_train[:, active_indices]
    X_val_sel = X_val[:, active_indices]
    X_test_sel = X_test[:, active_indices]
    
    t_start = time.perf_counter()
    best_model_name, best_model = train_and_select_model(X_train_sel, y_train, X_val_sel, y_val)
    training_time_seconds = time.perf_counter() - t_start
    
    val_best_probs = best_model.predict_proba(X_val_sel)[:, 1]
    optimal_threshold = optimize_threshold(val_best_probs, y_val)
    logger.info(f"Selected Winner Model: {best_model_name}")
    logger.info(f"Optimized Decision Threshold: {optimal_threshold:.2f}")
    
    test_probs = best_model.predict_proba(X_test_sel)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_pr_auc = average_precision_score(y_test, test_probs)
    
    conf = confusion_matrix(y_test, test_preds)
    report = classification_report(y_test, test_preds, zero_division=0)
    
    plot_curves_and_confusion_matrix(y_test, test_probs, test_preds, best_model_name)
    
    preprocessor = ColumnTransformer(
        transformers=[('select', 'passthrough', active_indices)],
        remainder='drop'
    )
    preprocessor.fit(X_train)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    logger.info("Measuring sequential inference latency over the test set...")
    latencies = []
    for test_path in test_paths:
        try:
            t0 = time.perf_counter()
            img = cv2.imread(str(test_path))
            if img is None:
                continue
            feats = extract_features(img)
            pipeline.predict_proba(feats.reshape(1, -1))
            t_lat = time.perf_counter() - t0
            latencies.append(t_lat * 1000.0)
        except Exception:
            continue
    mean_latency_ms = np.mean(latencies) if latencies else 0.0
    
    plot_feature_correlation_heatmap(X_train_sel, feature_names_sel)
    
    has_importance = hasattr(best_model, 'feature_importances_')
    if has_importance:
        importances = best_model.feature_importances_
        with open("feature_importance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Feature_Index", "Feature_Name", "Importance"])
            for idx, imp in enumerate(importances):
                writer.writerow([active_indices[idx], feature_names_sel[idx], float(imp)])
                
        plot_feature_importance_top20(importances, feature_names_sel)
        
        indices = np.argsort(importances)[::-1]
        top_20_str = ""
        for rank, idx in enumerate(indices[:20], 1):
            top_20_str += f"{rank:<5} {feature_names_sel[idx]:<35} {importances[idx]:.4f}\n"
    else:
        top_20_str = "Selected model does not support feature importance natively."
        
    misclassified = run_error_analysis(test_paths, y_test, test_probs, test_preds)
    plot_misclassified_gallery(misclassified)
    
    generate_fft_diagnostics_plot(test_paths, y_test, test_probs, test_preds)
    
    model_export = {
        "model": pipeline,
        "model_name": best_model_name,
        "feature_dimension": num_features,
        "best_threshold": optimal_threshold,
        "feature_names": feature_names_sel,
        "version": "1.1"
    }
    model_pkl_path = "model.pkl"
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_export, f)
    model_size_kb = os.path.getsize(model_pkl_path) / 1024.0
    
    metrics_export = {
        "selected_model": best_model_name,
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1": float(test_f1),
        "roc_auc": float(test_roc_auc),
        "pr_auc": float(test_pr_auc),
        "best_threshold": float(optimal_threshold),
        "training_time_seconds": float(training_time_seconds),
        "inference_latency_ms": float(mean_latency_ms),
        "model_size_kb": float(model_size_kb),
        "number_of_features": int(len(active_indices)),
        "number_of_training_samples": int(len(y_train))
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=4)
        
    latency_ok = mean_latency_ms < 100.0
    metrics_ok = test_f1 >= 0.85 and test_roc_auc >= 0.90
    imbalance_ok = validation_summary["imbalance_deviation"] <= 0.10
    
    if latency_ok and metrics_ok and imbalance_ok:
        deployability = "READY FOR DEPLOYMENT"
        reason = "All performance, latency, and balance metrics meet target thresholds."
    else:
        reasons = []
        if not latency_ok: reasons.append("high latency")
        if not metrics_ok: reasons.append("sub-target classification score")
        if not imbalance_ok: reasons.append("significant class imbalance")
        deployability = "NOT READY"
        reason = f"Flags raised due to: {', '.join(reasons)}."

    print("\n" + "="*50)
    print("                 DATASET VALIDATION")
    print("="*50)
    print(f"Real images found:      {validation_summary['real_count']}")
    print(f"Fake images found:      {validation_summary['fake_count']}")
    print(f"MD5 exact duplicates:   {validation_summary['duplicates_detected']} (skipped)")
    print(f"Corrupted/Unreadable:   {validation_summary['corrupted_detected']} (skipped)")
    print(f"Imbalance deviation:    {validation_summary['imbalance_deviation']*100:.1f}%")
    print(f"Final training:         {len(y_train)} samples")
    print(f"Final validation:       {len(y_val)} samples")
    print(f"Final testing:          {len(y_test)} samples")
    
    print("\n" + "="*50)
    print("                 TRAINING & SELECTION")
    print("="*50)
    print(f"Winner Model:           {best_model_name}")
    print(f"Optimal F1 Threshold:   {optimal_threshold:.2f}")
    
    print("\n" + "="*50)
    print("                 TEST SET PERFORMANCE")
    print("="*50)
    print(f"Accuracy:            {test_acc:.4f}")
    print(f"Precision:           {test_prec:.4f}")
    print(f"Recall:              {test_rec:.4f}")
    print(f"F1 Score:            {test_f1:.4f}")
    print(f"ROC-AUC:             {test_roc_auc:.4f}")
    print(f"PR-AUC (Avg Prec):   {test_pr_auc:.4f}")
    print(f"Inference Latency:   {mean_latency_ms:.2f} ms/image")
    print(f"Model Size on Disk:  {model_size_kb:.2f} KB")
    print("-"*50)
    print("Confusion Matrix:")
    print(conf)
    print("-"*50)
    print("Classification Report:")
    print(report)
    
    if has_importance:
        print("\n" + "="*50)
        print("             FEATURE IMPORTANCE (TOP 20)")
        print("="*50)
        print(top_20_str)
        print("Handcrafted feature contribution summary:")
        print("- Directional FFT anisotropy checks for structural axis alignment.")
        print("- Residual FFT peak detection measures specific LCD/OLED dot pitch spikes.")
        print("- Brightness values are normalized out (only HSV variances are retained).")
        
    print("\n" + "="*50)
    print("                 ERROR ANALYSIS")
    print("="*50)
    print(f"Misclassified test samples: {len(misclassified)}")
    print(f"Report exported to:        misclassified.csv")
    print(f"Gallery visual saved to:   analysis/misclassified_gallery.png")
    if misclassified:
        worst_error = misclassified[0]
        lbl_str = "Fake" if worst_error['true_label'] == 1 else "Real"
        pred_str = "Fake" if worst_error['predicted_label'] == 1 else "Real"
        print(f"Highest confidence error:  {Path(worst_error['image_path']).name}")
        print(f"  └─ True: {lbl_str} | Pred: {pred_str} | Confidence: {worst_error['confidence']*100:.1f}%")
        
    print("\n" + "="*50)
    print("           DEPLOYABILITY RECOMMENDATION")
    print("="*50)
    print(f"Status:  {deployability}")
    print(f"Reason:  {reason}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
