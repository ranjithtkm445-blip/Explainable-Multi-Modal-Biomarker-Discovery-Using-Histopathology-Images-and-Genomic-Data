# =============================================================================
# step2_load_data.py
# Purpose: Load genomics_clean.csv, verify all 50 patient patch PNGs exist,
#          display a summary of the linked image + genomics data,
#          and save a verification report.
# =============================================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

# ── Load dataset_info.json ────────────────────────────────────────────────────
BASE_DIR  = Path("D:/HISTOPATHOLOGY")
INFO_PATH = BASE_DIR / "dataset_info.json"

with open(INFO_PATH) as f:
    INFO = json.load(f)

PATCH_DIR  = Path(INFO["patch_dir"])
GENOME_CSV = Path(INFO["genomics_csv"])
GENE_NAMES = INFO["gene_names"]


# =============================================================================
# PART A — Load and validate genomics CSV
# =============================================================================

def load_and_validate() -> pd.DataFrame:
    print(f"\n📋 Loading {GENOME_CSV.name} ...")

    df = pd.read_csv(GENOME_CSV)

    # ── Check required columns ────────────────────────────────────────────────
    required = {"patient_id", "image_index", "image_filename",
                "pcam_label", "risk_label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # ── Verify all patch PNGs exist ───────────────────────────────────────────
    print(f"\n🔍 Verifying patch files ...")
    missing_files = []
    for _, row in df.iterrows():
        patch_path = PATCH_DIR / row["image_filename"]
        if not patch_path.exists():
            missing_files.append(row["image_filename"])

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} patch files:\n"
            + "\n".join(missing_files[:5])
        )

    print(f"   ✅ All {len(df)} patch files found in {PATCH_DIR}")

    return df


# =============================================================================
# PART B — Print data summary
# =============================================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  Patient Data Summary")
    print("=" * 60)

    print(f"\n📊 Dataset overview:")
    print(f"   Total patients     : {len(df)}")
    print(f"   High-risk (1)      : {(df['risk_label'] == 1).sum()}")
    print(f"   Low-risk  (0)      : {(df['risk_label'] == 0).sum()}")
    print(f"   Tumor patches      : {(df['pcam_label'] == 1).sum()}")
    print(f"   Normal patches     : {(df['pcam_label'] == 0).sum()}")
    print(f"   Gene columns       : {len(GENE_NAMES)}")

    print(f"\n📋 Patient-image-genomics link (first 10 rows):")
    print(df[["patient_id", "image_index", "image_filename",
              "pcam_label", "risk_label"]].head(10).to_string(index=False))

    print(f"\n🧬 Gene expression stats (first 5 genes):")
    gene_stats = df[GENE_NAMES[:5]].describe().round(3)
    print(gene_stats.to_string())

    print(f"\n🔗 Risk label vs PCam label cross-tab:")
    cross = pd.crosstab(df["risk_label"], df["pcam_label"],
                        rownames=["risk_label"],
                        colnames=["pcam_label (0=normal, 1=tumor)"])
    print(cross.to_string())


# =============================================================================
# PART C — Visualize sample patients (image + top genes)
# =============================================================================

def visualize_patients(df: pd.DataFrame, n_samples: int = 6):
    """
    Show n_samples patients side by side:
    each patient shows their patch + top 5 gene expression values.
    Saves to D:/HISTOPATHOLOGY/patient_overview.png
    """
    print(f"\n🖼️  Generating patient visualization ({n_samples} samples) ...")

    # Pick 3 high-risk + 3 low-risk
    high = df[df["risk_label"] == 1].head(n_samples // 2)
    low  = df[df["risk_label"] == 0].head(n_samples // 2)
    sample_df = pd.concat([high, low]).reset_index(drop=True)

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("Patient-Linked Image + Genomic Data Overview",
                 fontsize=14, fontweight="bold", y=1.01)

    top_genes = GENE_NAMES[:5]

    for i, (_, row) in enumerate(sample_df.iterrows()):
        # ── Patch image ───────────────────────────────────────────────────────
        ax_img = fig.add_subplot(2, n_samples, i + 1)
        img    = Image.open(PATCH_DIR / row["image_filename"]).convert("RGB")
        ax_img.imshow(img)
        ax_img.axis("off")

        risk_str  = "HIGH RISK" if row["risk_label"] == 1 else "LOW RISK"
        pcam_str  = "Tumor"     if row["pcam_label"]  == 1 else "Normal"
        color     = "#D85A30"   if row["risk_label"]  == 1 else "#1D9E75"

        ax_img.set_title(
            f"{row['patient_id']}\n{pcam_str} | {risk_str}",
            fontsize=9, color=color, fontweight="bold"
        )

        # ── Gene bar chart ────────────────────────────────────────────────────
        ax_gene = fig.add_subplot(2, n_samples, n_samples + i + 1)
        values  = [row[g] for g in top_genes]
        colors  = ["#D85A30" if v > np.mean(values) else "#378ADD"
                   for v in values]

        ax_gene.barh(top_genes, values, color=colors, edgecolor="white",
                     height=0.6)
        ax_gene.set_xlim(0, 15)
        ax_gene.set_xlabel("Expression", fontsize=8)
        ax_gene.tick_params(axis="y", labelsize=7)
        ax_gene.tick_params(axis="x", labelsize=7)
        ax_gene.spines[["top", "right"]].set_visible(False)

        if i == 0:
            ax_gene.set_ylabel("Gene", fontsize=8)

    plt.tight_layout()
    out_path = BASE_DIR / "patient_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Visualization saved → {out_path}")


# =============================================================================
# PART D — Save verification report
# =============================================================================

def save_verification_report(df: pd.DataFrame):
    """Save a simple text report confirming data is ready for training."""

    report_lines = [
        "=" * 60,
        "  Step 2 — Data Verification Report",
        "=" * 60,
        f"  Total patients    : {len(df)}",
        f"  High-risk (1)     : {(df['risk_label'] == 1).sum()}",
        f"  Low-risk  (0)     : {(df['risk_label'] == 0).sum()}",
        f"  Tumor patches     : {(df['pcam_label'] == 1).sum()}",
        f"  Normal patches    : {(df['pcam_label'] == 0).sum()}",
        f"  Gene columns      : {len(GENE_NAMES)}",
        f"  All patches found : YES",
        f"  CSV path          : {GENOME_CSV}",
        f"  Patch dir         : {PATCH_DIR}",
        "=" * 60,
        "  Data is ready for step 3 (CNN training + Grad-CAM)",
        "=" * 60,
    ]

    out_path = BASE_DIR / "step2_verification.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Verification report saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 2 — Load and Verify Patient Data")
    print("  Genomics CSV + Patient Patch PNGs")
    print("=" * 60)

    # A: load and validate
    df = load_and_validate()

    # B: print summary
    print_summary(df)

    # C: visualize sample patients
    visualize_patients(df, n_samples=6)

    # D: save verification report
    save_verification_report(df)

    print("\n" + "=" * 60)
    print("  Step 2 complete. Run step3_model.py next.")
    print("=" * 60)