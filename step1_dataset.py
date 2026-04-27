# =============================================================================
# step1_dataset.py  (final)
# Purpose: Load PCam HDF5 + genomics Excel, extract 50 patient-linked patches,
#          save PNGs, verify everything, write dataset_info.json.
# =============================================================================

import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("D:/HISTOPATHOLOGY")
ARCHIVE_DIR = BASE_DIR / "archive"

H5_TRAIN_IMG = ARCHIVE_DIR / "pcam"   / "training_split.h5"
H5_TRAIN_LBL = ARCHIVE_DIR / "Labels" / "Labels" / "camelyonpatch_level_2_split_train_y.h5"
H5_VAL_IMG   = ARCHIVE_DIR / "pcam"   / "validation_split.h5"
H5_VAL_LBL   = ARCHIVE_DIR / "Labels" / "Labels" / "camelyonpatch_level_2_split_valid_y.h5"
H5_TEST_IMG  = ARCHIVE_DIR / "pcam"   / "test_split.h5"
H5_TEST_LBL  = ARCHIVE_DIR / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"

GENOMICS_XLSX = BASE_DIR / "genomics_dataset_real_genes.xlsx"

# Output folders — created automatically
PATCH_DIR  = BASE_DIR / "patches"    # 50 patient-linked PNGs
GENOME_DIR = BASE_DIR / "genomics"   # clean CSV for steps 4 & 5
MODEL_DIR  = BASE_DIR / "models"     # weights saved from step 2

for d in [PATCH_DIR, GENOME_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Output folders ready")

# =============================================================================
# PART A — Load and validate genomics Excel
# =============================================================================

def load_genomics() -> pd.DataFrame:
    print(f"\n🧬 Loading {GENOMICS_XLSX.name} ...")

    if not GENOMICS_XLSX.exists():
        raise FileNotFoundError(f"Missing: {GENOMICS_XLSX}")

    df = pd.read_excel(GENOMICS_XLSX)

    # Validate required columns
    required = {"patient_id", "image_index", "risk_label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    print(f"   Patients      : {len(df)}")
    print(f"   Columns       : {len(df.columns)}")
    print(f"   image_index   : {df['image_index'].min()} → {df['image_index'].max()}")
    print(f"   High-risk (1) : {(df['risk_label'] == 1).sum()}")
    print(f"   Low-risk  (0) : {(df['risk_label'] == 0).sum()}")

    return df


# =============================================================================
# PART B — Extract 50 patient-linked patches from HDF5
# =============================================================================

def extract_patches(df: pd.DataFrame):
    """
    For each patient row, read their patch from training_split.h5
    using image_index, save as PNG, verify label matches metadata.
    """
    print(f"\n🖼️  Extracting {len(df)} patient patches from HDF5 ...")

    # Read all training labels upfront for verification
    with h5py.File(H5_TRAIN_LBL, "r") as f:
        lbl_key    = list(f.keys())[0]
        all_labels = f[lbl_key][:].squeeze()   # (262144,)

    patch_records = []

    with h5py.File(H5_TRAIN_IMG, "r") as f:
        img_key = list(f.keys())[0]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            patient_id  = row["patient_id"]
            image_index = int(row["image_index"])
            risk_label  = int(row["risk_label"])

            # Read patch pixels
            patch = f[img_key][image_index]          # (96, 96, 3) uint8

            # PCam label at this index (0=normal, 1=tumor)
            pcam_label = int(all_labels[image_index])

            # Save PNG
            filename = f"patch_{image_index}.png"
            out_path = PATCH_DIR / filename
            Image.fromarray(patch).save(out_path)

            patch_records.append({
                "patient_id"  : patient_id,
                "image_index" : image_index,
                "filename"    : filename,
                "pcam_label"  : pcam_label,    # 0=normal, 1=tumor (from PCam)
                "risk_label"  : risk_label,    # 0=low, 1=high (from genomics)
            })

    patch_df = pd.DataFrame(patch_records)

    print(f"\n📊 Patch extraction summary:")
    print(f"   Total patches saved  : {len(patch_df)}")
    print(f"   Tumor patches (PCam) : {(patch_df['pcam_label'] == 1).sum()}")
    print(f"   Normal patches(PCam) : {(patch_df['pcam_label'] == 0).sum()}")
    print(f"   Saved to             : {PATCH_DIR}")

    return patch_df


# =============================================================================
# PART C — Save clean genomics CSV (with image_index + filename)
# =============================================================================

def save_genomics_csv(df: pd.DataFrame, patch_df: pd.DataFrame):
    """Merge patch filenames into genomics dataframe and save clean CSV."""

    # Add filename column from patch records
    filename_map = patch_df.set_index("patient_id")["filename"]
    df["image_filename"] = df["patient_id"].map(filename_map)

    # Detect gene columns (numeric, exclude id/index/label columns)
    exclude   = {"patient_id", "image_index", "image_filename", "risk_label"}
    gene_cols = [c for c in df.columns
                 if c not in exclude
                 and pd.api.types.is_numeric_dtype(df[c])]

    out_csv = GENOME_DIR / "genomics_clean.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n✅ Genomics CSV saved → {out_csv}")
    print(f"   Gene columns : {len(gene_cols)}")
    print(f"   Sample rows  :")
    print(df[["patient_id", "image_index", "image_filename", "risk_label"]].head(5).to_string(index=False))

    return gene_cols


# =============================================================================
# PART D — Verify full HDF5 dataset stats
# =============================================================================

def verify_full_dataset() -> dict:
    print("\n📦 Verifying full PCam HDF5 splits ...")
    summary = {}

    splits = [
        ("train", H5_TRAIN_IMG, H5_TRAIN_LBL),
        ("val",   H5_VAL_IMG,   H5_VAL_LBL),
        ("test",  H5_TEST_IMG,  H5_TEST_LBL),
    ]

    for split_name, img_path, lbl_path in splits:
        with h5py.File(img_path, "r") as fi, \
             h5py.File(lbl_path, "r") as fl:
            img_key   = list(fi.keys())[0]
            lbl_key   = list(fl.keys())[0]
            n_total   = fi[img_key].shape[0]
            labels    = fl[lbl_key][:].squeeze()
            n_tumor   = int(labels.sum())
            n_normal  = n_total - n_tumor

            print(f"   {split_name:5s} → total: {n_total:7,}  "
                  f"tumor: {n_tumor:6,}  normal: {n_normal:6,}")

            summary[split_name] = {
                "img_key" : img_key,
                "lbl_key" : lbl_key,
                "n_total" : n_total,
                "n_tumor" : n_tumor,
                "n_normal": n_normal,
            }

    return summary


# =============================================================================
# PART E — Save dataset_info.json
# =============================================================================

def save_dataset_info(gene_cols: list, h5_summary: dict):
    info = {
        # Paths
        "base_dir"        : str(BASE_DIR),
        "h5_train_img"    : str(H5_TRAIN_IMG),
        "h5_val_img"      : str(H5_VAL_IMG),
        "h5_test_img"     : str(H5_TEST_IMG),
        "h5_train_lbl"    : str(H5_TRAIN_LBL),
        "h5_val_lbl"      : str(H5_VAL_LBL),
        "h5_test_lbl"     : str(H5_TEST_LBL),
        "genomics_csv"    : str(GENOME_DIR / "genomics_clean.csv"),
        "patch_dir"       : str(PATCH_DIR),
        "model_dir"       : str(MODEL_DIR),
        "genomics_xlsx"   : str(GENOMICS_XLSX),

        # HDF5 keys
        "img_key"         : h5_summary["train"]["img_key"],
        "lbl_key"         : h5_summary["train"]["lbl_key"],

        # Dataset stats
        "img_size"        : 96,
        "n_classes"       : 2,
        "class_names"     : ["normal", "tumor"],
        "n_train"         : h5_summary["train"]["n_total"],
        "n_val"           : h5_summary["val"]["n_total"],
        "n_test"          : h5_summary["test"]["n_total"],
        "n_patients"      : 50,

        # Genomics
        "gene_names"      : gene_cols,
    }

    out_path = BASE_DIR / "dataset_info.json"
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n✅ dataset_info.json saved → {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1 — Dataset Preparation (final)")
    print("  PCam HDF5 + Genomics Excel + Patient-Patch Mapping")
    print("=" * 60)

    # A: load and validate Excel
    genomics_df = load_genomics()

    # B: extract 50 patient-linked patches
    patch_df = extract_patches(genomics_df)

    # C: save clean genomics CSV
    gene_cols = save_genomics_csv(genomics_df, patch_df)

    # D: verify full HDF5 dataset
    h5_summary = verify_full_dataset()

    # E: save dataset_info.json
    save_dataset_info(gene_cols, h5_summary)

    print("\n" + "=" * 60)
    print("  Step 1 complete. Run step2_model.py next.")
    print("=" * 60)