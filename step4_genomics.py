# =============================================================================
# step4_genomics.py
# Purpose: Train a Random Forest classifier on 100 gene expression values
#          to predict patient risk label (0=low, 1=high).
#          Save model + predictions + feature importance to CSV and disk.
# =============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("D:/HISTOPATHOLOGY")
INFO_PATH = BASE_DIR / "dataset_info.json"

with open(INFO_PATH) as f:
    INFO = json.load(f)

GENOME_CSV = Path(INFO["genomics_csv"])
MODEL_DIR  = Path(INFO["model_dir"])
GENE_NAMES = INFO["gene_names"]
REPORT_DIR = BASE_DIR / "reports"

for d in [MODEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# =============================================================================
# PART A — Load data
# =============================================================================

def load_data():
    print(f"\n📋 Loading {GENOME_CSV.name} ...")
    df = pd.read_csv(GENOME_CSV)

    X = df[GENE_NAMES].values.astype(np.float32)
    y = df["risk_label"].values.astype(int)

    print(f"   Patients      : {len(df)}")
    print(f"   Features      : {X.shape[1]} genes")
    print(f"   High-risk (1) : {(y == 1).sum()}")
    print(f"   Low-risk  (0) : {(y == 0).sum()}")

    return df, X, y


# =============================================================================
# PART B — Train Random Forest with cross-validation
# =============================================================================

def train_random_forest(X, y):
    """
    Train Random Forest on all 50 patients.
    Use 5-fold stratified cross-validation to get reliable accuracy estimate.
    Then refit on full data for the final model.
    """
    print("\n🌲 Training Random Forest ...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest — tuned for small datasets
    rf = RandomForestClassifier(
        n_estimators   = 500,
        max_depth      = 4,        # shallow trees to avoid overfitting
        min_samples_split = 4,
        min_samples_leaf  = 2,
        max_features   = "sqrt",
        class_weight   = "balanced",
        random_state   = SEED,
        n_jobs         = -1,
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")

    print(f"\n📊 Cross-validation results (5-fold):")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score*100:.1f}%")
    print(f"   Mean  : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Refit on full dataset
    rf.fit(X_scaled, y)
    train_preds = rf.predict(X_scaled)
    train_acc   = accuracy_score(y, train_preds)
    print(f"\n   Train accuracy (full data) : {train_acc*100:.1f}%")

    return rf, scaler, X_scaled, cv_scores


# =============================================================================
# PART C — Feature importance
# =============================================================================

def plot_feature_importance(rf, top_n=20):
    """Plot and save top N most important genes."""
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    top_genes  = [GENE_NAMES[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#D85A30" if s > top_scores.mean() else "#378ADD"
               for s in top_scores]

    ax.barh(range(top_n), top_scores[::-1], color=colors[::-1],
            edgecolor="white", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=10)
    ax.set_title(f"Top {top_n} Genes by Random Forest Importance",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = REPORT_DIR / "rf_feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Feature importance plot saved → {out_path}")

    return top_genes, top_scores


# =============================================================================
# PART D — Confusion matrix
# =============================================================================

def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low risk", "High risk"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Random Forest — Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    out_path = REPORT_DIR / "rf_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Confusion matrix saved → {out_path}")


# =============================================================================
# PART E — Save predictions to CSV
# =============================================================================

def save_predictions(df, rf, scaler, X_scaled, y):
    probs       = rf.predict_proba(X_scaled)
    preds       = rf.predict(X_scaled)
    pred_labels = ["high-risk" if p == 1 else "low-risk" for p in preds]

    df["rf_prediction"]  = preds
    df["rf_label"]       = pred_labels
    df["rf_confidence"]  = [round(float(probs[i, p]), 4)
                             for i, p in enumerate(preds)]

    df.to_csv(GENOME_CSV, index=False)
    print(f"\n✅ Predictions saved → {GENOME_CSV}")

    print(f"\n📋 Sample predictions:")
    print(df[["patient_id", "risk_label", "rf_label",
              "rf_confidence", "cnn_label"]].head(10).to_string(index=False))


# =============================================================================
# PART F — Save model + importance JSON
# =============================================================================

def save_model(rf, scaler, top_genes, top_scores, cv_scores):
    # Save model + scaler
    joblib.dump(rf,     MODEL_DIR / "random_forest.pkl")
    joblib.dump(scaler, MODEL_DIR / "rf_scaler.pkl")

    # Save importance as JSON for step 5 + app
    importance_dict = {
        gene: round(float(score), 6)
        for gene, score in zip(top_genes, top_scores)
    }
    with open(MODEL_DIR / "gene_importance.json", "w") as f:
        json.dump(importance_dict, f, indent=2)

    # Save CV summary
    cv_summary = {
        "cv_scores"     : [round(float(s), 4) for s in cv_scores],
        "cv_mean"       : round(float(cv_scores.mean()), 4),
        "cv_std"        : round(float(cv_scores.std()),  4),
        "n_estimators"  : 500,
        "max_depth"     : 4,
        "n_genes"       : len(GENE_NAMES),
    }
    with open(MODEL_DIR / "rf_cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print(f"\n✅ Model saved       → {MODEL_DIR / 'random_forest.pkl'}")
    print(f"✅ Scaler saved      → {MODEL_DIR / 'rf_scaler.pkl'}")
    print(f"✅ Importance saved  → {MODEL_DIR / 'gene_importance.json'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 4 — Random Forest Genomics Classifier")
    print("  100 Genes → Risk Prediction")
    print("=" * 60)

    # A: load data
    df, X, y = load_data()

    # B: train Random Forest
    rf, scaler, X_scaled, cv_scores = train_random_forest(X, y)

    # C: feature importance plot
    print("\n📈 Plotting feature importance ...")
    top_genes, top_scores = plot_feature_importance(rf, top_n=20)

    print(f"\n🏆 Top 10 genes by importance:")
    for gene, score in zip(top_genes[:10], top_scores[:10]):
        print(f"   {gene:10s} : {score:.4f}")

    # D: confusion matrix
    print("\n📊 Plotting confusion matrix ...")
    y_pred = rf.predict(X_scaled)
    plot_confusion_matrix(y, y_pred)

    print("\n📋 Classification report:")
    print(classification_report(y, y_pred,
                                 target_names=["Low risk", "High risk"]))

    # E: save predictions to CSV
    save_predictions(df, rf, scaler, X_scaled, y)

    # F: save model + importance
    save_model(rf, scaler, top_genes, top_scores, cv_scores)

    print("\n" + "=" * 60)
    print("  Step 4 complete. Run step5_shap.py next.")
    print("=" * 60)