# =============================================================================
# step5_shap.py
# Purpose: Run SHAP TreeExplainer on the trained Random Forest to explain
#          which genes drive risk predictions for each patient.
#          Save global + per-patient SHAP plots and values to disk.
# =============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("D:/HISTOPATHOLOGY")
INFO_PATH = BASE_DIR / "dataset_info.json"

with open(INFO_PATH) as f:
    INFO = json.load(f)

GENOME_CSV = Path(INFO["genomics_csv"])
MODEL_DIR  = Path(INFO["model_dir"])
GENE_NAMES = INFO["gene_names"]
REPORT_DIR = BASE_DIR / "reports"
SHAP_DIR   = BASE_DIR / "shap"

for d in [REPORT_DIR, SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)


# =============================================================================
# PART A — Load data + model
# =============================================================================

def load_data_and_model():
    print(f"\n📋 Loading data and model ...")

    df     = pd.read_csv(GENOME_CSV)
    rf     = joblib.load(MODEL_DIR / "random_forest.pkl")
    scaler = joblib.load(MODEL_DIR / "rf_scaler.pkl")

    X        = df[GENE_NAMES].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    print(f"   Patients  : {len(df)}")
    print(f"   Genes     : {len(GENE_NAMES)}")
    print(f"   Model     : Random Forest ({rf.n_estimators} trees)")

    return df, rf, scaler, X, X_scaled


# =============================================================================
# PART B — Compute SHAP values
# =============================================================================

def compute_shap_values(rf, X_scaled):
    """
    SHAP TreeExplainer — fastest and exact for tree-based models.
    Returns shap_values for class 1 (high-risk).
    """
    print("\n🧮 Computing SHAP values ...")

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # Handle different SHAP output formats:
    # list of 2 arrays [class0, class1] → take index 1
    # 3D array (50, 100, 2)             → take [:, :, 1]
    # 2D array (50, 100)                → use directly
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]           # (50, 100)
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]     # (50, 100) high-risk class
    else:
        shap_vals = shap_values              # (50, 100)

    print(f"   SHAP values shape : {shap_vals.shape}")
    print(f"   Mean |SHAP|       : {np.abs(shap_vals).mean():.4f}")

    return explainer, shap_vals


# =============================================================================
# PART C — Global SHAP summary (bar chart)
# =============================================================================

def plot_global_shap_bar(shap_vals, top_n=20):
    """
    Global feature importance: mean |SHAP| per gene across all patients.
    Shows which genes matter most overall.
    """
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)   # (100,)
    indices       = np.argsort(mean_abs_shap)[::-1][:top_n]

    top_genes  = [GENE_NAMES[i] for i in indices]
    top_scores = mean_abs_shap[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#D85A30" if s > top_scores.mean() else "#378ADD"
               for s in top_scores]

    ax.barh(range(top_n), top_scores[::-1], color=colors[::-1],
            edgecolor="white", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (impact on high-risk prediction)", fontsize=10)
    ax.set_title(f"Top {top_n} Genes — Global SHAP Importance",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    high_patch = mpatches.Patch(color="#D85A30", label="Above average impact")
    low_patch  = mpatches.Patch(color="#378ADD", label="Below average impact")
    ax.legend(handles=[high_patch, low_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = REPORT_DIR / "shap_global_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Global SHAP bar chart saved → {out_path}")

    return top_genes, top_scores


# =============================================================================
# PART D — SHAP beeswarm plot
# =============================================================================

def plot_shap_beeswarm(explainer, X_scaled, top_n=20):
    """
    Beeswarm plot — shows direction and magnitude of each gene's effect.
    Red = high expression pushes toward high-risk.
    Blue = low expression pushes toward high-risk.
    """
    print("\n🐝 Generating SHAP beeswarm plot ...")

    sv = explainer.shap_values(X_scaled)
    sv = sv[:, :, 1] if (hasattr(sv, 'ndim') and sv.ndim == 3) else (sv[1] if isinstance(sv, list) else sv)
    ev = explainer.expected_value
    ev = ev[1] if hasattr(ev, '__len__') else ev

    shap_exp = shap.Explanation(
        values          = sv,
        base_values     = ev,
        data            = X_scaled,
        feature_names   = GENE_NAMES,
    )

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_exp, max_display=top_n, show=False)
    plt.title("SHAP Beeswarm — Gene Impact on High-Risk Prediction",
              fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = REPORT_DIR / "shap_beeswarm.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Beeswarm plot saved → {out_path}")


# =============================================================================
# PART E — Per-patient SHAP waterfall (top 5 patients)
# =============================================================================

def plot_patient_waterfall(explainer, X_scaled, df, n_patients=5):
    """
    Waterfall plot for individual patients — shows exactly which genes
    pushed the model toward high-risk or low-risk for that patient.
    """
    print(f"\n💧 Generating per-patient waterfall plots ({n_patients} patients) ...")

    # Pick top n_patients — mix of high and low risk
    high_risk = df[df["risk_label"] == 1].head(n_patients // 2 + 1)
    low_risk  = df[df["risk_label"] == 0].head(n_patients // 2)
    sample_df = pd.concat([high_risk, low_risk]).head(n_patients)

    sv_all = explainer.shap_values(X_scaled)
    sv_all = sv_all[:, :, 1] if (hasattr(sv_all, 'ndim') and sv_all.ndim == 3) else (sv_all[1] if isinstance(sv_all, list) else sv_all)
    ev = explainer.expected_value
    ev = ev[1] if hasattr(ev, '__len__') else ev

    for _, row in sample_df.iterrows():
        idx        = df[df["patient_id"] == row["patient_id"]].index[0]
        patient_id = row["patient_id"]
        risk_str   = "High-risk" if row["risk_label"] == 1 else "Low-risk"

        shap_exp = shap.Explanation(
            values        = sv_all[idx],
            base_values   = ev,
            data          = X_scaled[idx],
            feature_names = GENE_NAMES,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, max_display=15, show=False)
        plt.title(
            f"{patient_id} ({risk_str}) — Gene Contributions to Risk Prediction",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()

        out_path = SHAP_DIR / f"shap_waterfall_{patient_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"   ✅ Waterfall plots saved → {SHAP_DIR}")


# =============================================================================
# PART F — Save SHAP values + top genes per patient to CSV
# =============================================================================

def save_shap_to_csv(shap_vals, df, top_n=5):
    """
    For each patient save their top N genes by |SHAP| value.
    Adds shap_top_genes and shap_top_values columns to genomics_clean.csv.
    Also saves full SHAP matrix as shap_values.csv.
    """
    print("\n💾 Saving SHAP values ...")

    # Full SHAP matrix
    shap_df = pd.DataFrame(shap_vals, columns=GENE_NAMES)
    shap_df.insert(0, "patient_id", df["patient_id"].values)
    shap_df.to_csv(SHAP_DIR / "shap_values.csv", index=False)

    # Top N genes per patient
    top_genes_list  = []
    top_values_list = []

    for i in range(len(df)):
        abs_vals = np.abs(shap_vals[i])
        top_idx  = np.argsort(abs_vals)[::-1][:top_n]
        genes    = [GENE_NAMES[j] for j in top_idx]
        vals     = [round(float(shap_vals[i][j]), 4) for j in top_idx]
        top_genes_list.append(", ".join(genes))
        top_values_list.append(", ".join(map(str, vals)))

    df["shap_top_genes"]  = top_genes_list
    df["shap_top_values"] = top_values_list
    df.to_csv(GENOME_CSV, index=False)

    print(f"   ✅ SHAP matrix saved  → {SHAP_DIR / 'shap_values.csv'}")
    print(f"   ✅ Top genes saved    → {GENOME_CSV}")

    print(f"\n📋 Sample patient SHAP top genes:")
    print(df[["patient_id", "risk_label", "rf_label",
              "shap_top_genes"]].head(10).to_string(index=False))


# =============================================================================
# PART G — Print global top genes summary
# =============================================================================

def print_top_genes_summary(top_genes, top_scores, top_n=10):
    print(f"\n🏆 Top {top_n} genes driving risk prediction (SHAP):")
    print(f"   {'Gene':<12} {'Mean |SHAP|':>12}")
    print(f"   {'-'*12} {'-'*12}")
    for gene, score in zip(top_genes[:top_n], top_scores[:top_n]):
        bar = "█" * int(score / top_scores[0] * 20)
        print(f"   {gene:<12} {score:>12.4f}  {bar}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 5 — SHAP Gene Importance Analysis")
    print("  Random Forest → Explainable AI")
    print("=" * 60)

    # A: load data + model
    df, rf, scaler, X, X_scaled = load_data_and_model()

    # B: compute SHAP values
    explainer, shap_vals = compute_shap_values(rf, X_scaled)

    # C: global bar chart
    print("\n📊 Plotting global SHAP bar chart ...")
    top_genes, top_scores = plot_global_shap_bar(shap_vals, top_n=20)

    # D: beeswarm plot
    plot_shap_beeswarm(explainer, X_scaled, top_n=20)

    # E: per-patient waterfall plots
    plot_patient_waterfall(explainer, X_scaled, df, n_patients=5)

    # F: save SHAP to CSV
    save_shap_to_csv(shap_vals, df, top_n=5)

    # G: print summary
    print_top_genes_summary(top_genes, top_scores, top_n=10)

    print("\n" + "=" * 60)
    print("  Step 5 complete. Run step6_report.py next.")
    print("=" * 60)