# =============================================================================
# app.py
# Purpose: Streamlit app for Explainable Multi-Modal Biomarker Discovery.
#          Shows full genomics data (all 100 genes) — SHAP chart + table.
#          CNN + Grad-CAM image analysis. Downloadable PDF report.
# =============================================================================

import json
import os
import io
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import streamlit as st
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm as rl_cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Biomarker Discovery",
    page_icon="🔬",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(".")
INFO_PATH  = BASE_DIR / "dataset_info.json"

@st.cache_resource
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)

INFO       = load_info()
PATCH_DIR  = Path(INFO["patch_dir"])
MODEL_DIR  = Path(INFO["model_dir"])
GENE_NAMES = INFO["gene_names"]
GENOME_CSV = Path(INFO["genomics_csv"])
EXCEL_PATH = Path(INFO["genomics_xlsx"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── PDF colours ───────────────────────────────────────────────────────────────
C_HIGH   = colors.HexColor("#D85A30")
C_LOW    = colors.HexColor("#1D9E75")
C_HEADER = colors.HexColor("#2C2C2A")
C_MUTED  = colors.HexColor("#5F5E5A")
C_BORDER = colors.HexColor("#D3D1C7")
C_BG     = colors.HexColor("#F1EFE8")


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_cnn():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 2),
    )
    ckpt = torch.load(
        MODEL_DIR / "efficientnet_b0_best.pth",
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    return model


@st.cache_resource
def load_rf_models():
    rf     = joblib.load(MODEL_DIR / "random_forest.pkl")
    scaler = joblib.load(MODEL_DIR / "rf_scaler.pkl")
    return rf, scaler


@st.cache_data
def load_genomics():
    return pd.read_csv(GENOME_CSV)


@st.cache_data
def load_excel():
    """Load raw gene values from original Excel sheet."""
    return pd.read_excel(EXCEL_PATH)


# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        model.features[-1].register_forward_hook(self._save_act)
        model.features[-1].register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):    self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients   = go[0].detach()

    def generate(self, tensor):
        tensor = tensor.unsqueeze(0).to(DEVICE)
        tensor.requires_grad_()
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        cls    = int(probs.argmax().item())
        conf   = float(probs[cls].item())

        self.model.zero_grad()
        logits[0, cls].backward()

        grads   = self.gradients[0]
        acts    = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam     = sum(w * a for w, a in zip(weights, acts))
        cam     = torch.clamp(cam, min=0).cpu().numpy()
        cam     = cv2.resize(cam, (96, 96))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cls, conf, cam


def make_overlay(orig: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    colormap = matplotlib.colormaps["jet"]
    heat_c   = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay  = 0.55 * orig + 0.45 * heat_c
    return np.clip(overlay, 0, 255).astype(np.uint8)


# =============================================================================
# INFERENCE
# =============================================================================

def run_cnn(model, gradcam, img_array: np.ndarray):
    tensor = val_transform(Image.fromarray(img_array))
    cls, conf, heatmap = gradcam.generate(tensor)
    overlay = make_overlay(img_array, heatmap)
    return {
        "pred_class" : cls,
        "pred_label" : "Tumor" if cls == 1 else "Normal",
        "confidence" : conf,
        "overlay"    : overlay,
    }


def run_genomics(rf, scaler, patient_row):
    """Run RF + SHAP for ALL 100 genes."""
    X        = np.array([[patient_row[g] for g in GENE_NAMES]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    pred  = int(rf.predict(X_scaled)[0])
    probs = rf.predict_proba(X_scaled)[0]
    conf  = float(probs[pred])

    # SHAP for all genes
    explainer = shap.TreeExplainer(rf)
    sv        = explainer.shap_values(X_scaled)
    sv = sv[:, :, 1] if (hasattr(sv, 'ndim') and sv.ndim == 3) \
         else (sv[1] if isinstance(sv, list) else sv)

    shap_vals = sv[0]   # (100,) — one value per gene

    # Build full gene dataframe (all 100)
    gene_df = pd.DataFrame({
        "Gene"        : GENE_NAMES,
        "Value"       : [round(float(patient_row[g]), 4) for g in GENE_NAMES],
        "SHAP Value"  : [round(float(shap_vals[i]), 4) for i in range(len(GENE_NAMES))],
        "Direction"   : ["↑ Increases risk" if shap_vals[i] > 0
                         else "↓ Decreases risk"
                         for i in range(len(GENE_NAMES))],
        "|SHAP|"      : [abs(float(shap_vals[i])) for i in range(len(GENE_NAMES))],
    }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)

    gene_df["Rank"] = range(1, len(gene_df) + 1)

    top5 = [(row["Gene"], row["SHAP Value"])
            for _, row in gene_df.head(5).iterrows()]

    return {
        "pred"       : pred,
        "pred_label" : "High-risk" if pred == 1 else "Low-risk",
        "confidence" : conf,
        "gene_df"    : gene_df,
        "shap_vals"  : shap_vals,
        "top5"       : top5,
    }


# =============================================================================
# PLOT — ALL 100 GENES SHAP BAR CHART
# =============================================================================

def plot_all_genes_shap(gene_df: pd.DataFrame, patient_id: str) -> plt.Figure:
    """Horizontal bar chart — all 100 genes ranked by |SHAP|."""
    n     = len(gene_df)
    genes = gene_df["Gene"].tolist()
    vals  = gene_df["SHAP Value"].tolist()
    cols  = ["#D85A30" if v > 0 else "#378ADD" for v in vals]

    fig_h = max(8, n * 0.22)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    ax.barh(range(n), vals[::-1] if False else vals,
            color=cols, edgecolor="none", height=0.75)

    # Rank labels on bars
    for i, (v, g) in enumerate(zip(vals, genes)):
        ax.text(v + (0.001 if v >= 0 else -0.001), i,
                g, va="center",
                ha="left" if v >= 0 else "right",
                fontsize=6.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}" for i in range(n)], fontsize=6)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on high-risk prediction)", fontsize=9)
    ax.set_title(
        f"All 100 Genes — SHAP Importance Ranking\nPatient {patient_id}",
        fontsize=10, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()   # rank 1 at top
    fig.tight_layout()
    return fig


def plot_top20_shap(gene_df: pd.DataFrame, patient_id: str) -> plt.Figure:
    """Clean bar chart for top 20 genes — used in PDF."""
    top = gene_df.head(20)
    genes = top["Gene"].tolist()
    vals  = top["SHAP Value"].tolist()
    cols  = ["#D85A30" if v > 0 else "#378ADD" for v in vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(genes[::-1], vals[::-1], color=cols[::-1],
            edgecolor="white", height=0.65)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(f"Top 20 genes — {patient_id}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# =============================================================================
# PDF GENERATION
# =============================================================================

def generate_pdf(patient_id, risk_label, cnn_result, rf_result,
                 orig_arr, tmp_dir) -> bytes:

    pdf_path = os.path.join(tmp_dir, f"report_{patient_id}.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=1.8*rl_cm, rightMargin=1.8*rl_cm,
        topMargin=1.8*rl_cm,  bottomMargin=1.8*rl_cm,
    )
    W = A4[0] - 3.6*rl_cm

    s_title  = ParagraphStyle("t",   fontSize=18, fontName="Helvetica-Bold",
                               textColor=C_HEADER, alignment=TA_CENTER, spaceAfter=6)
    s_sub    = ParagraphStyle("s",   fontSize=9,  fontName="Helvetica",
                               textColor=C_MUTED,  alignment=TA_CENTER, spaceAfter=16)
    s_sec    = ParagraphStyle("sec", fontSize=10, fontName="Helvetica-Bold",
                               textColor=C_MUTED,  spaceAfter=4)
    s_body   = ParagraphStyle("b",   fontSize=9,  fontName="Helvetica",
                               textColor=C_HEADER, spaceAfter=4)
    s_interp = ParagraphStyle("i",   fontSize=9,  fontName="Helvetica-Oblique",
                               textColor=C_HEADER, spaceAfter=4,
                               leftIndent=8, rightIndent=8)

    story = []
    tmp_files = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    risk_str = "HIGH RISK" if risk_label == 1 else "LOW RISK"
    story.append(Paragraph("Multi-Modal Biomarker Discovery", s_title))
    story.append(Paragraph(f"Patient Report — {patient_id} — {risk_str}", s_sub))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=C_BORDER, spaceAfter=12))

    # ── Images ────────────────────────────────────────────────────────────────
    orig_tmp = os.path.join(tmp_dir, "orig.png")
    gcam_tmp = os.path.join(tmp_dir, "gcam.png")
    Image.fromarray(orig_arr).save(orig_tmp)
    Image.fromarray(cnn_result["overlay"]).save(gcam_tmp)
    tmp_files += [orig_tmp, gcam_tmp]

    img_w = W * 0.38
    img_table = Table(
        [[RLImage(orig_tmp, width=img_w, height=img_w),
          RLImage(gcam_tmp, width=img_w, height=img_w)]],
        colWidths=[img_w + 0.3*rl_cm, img_w + 0.3*rl_cm],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(img_table)

    lbl_table = Table(
        [["Original Patch", "Grad-CAM Overlay"]],
        colWidths=[img_w+0.3*rl_cm, img_w+0.3*rl_cm],
    )
    lbl_table.setStyle(TableStyle([
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE",  (0,0), (-1,-1), 8),
        ("TEXTCOLOR", (0,0), (-1,-1), C_MUTED),
    ]))
    story.append(lbl_table)
    story.append(Spacer(1, 10))

    # ── CNN ───────────────────────────────────────────────────────────────────
    story.append(Paragraph("Image Analysis (CNN — EfficientNet-B0)", s_sec))
    story.append(Paragraph(
        f'Prediction: <b>{cnn_result["pred_label"]}</b> '
        f'({cnn_result["confidence"]*100:.1f}% confidence)', s_body))
    story.append(Spacer(1, 8))

    # ── RF ────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Genomic Risk Analysis (Random Forest)", s_sec))
    story.append(Paragraph(
        f'Risk: <b>{rf_result["pred_label"]}</b> '
        f'({rf_result["confidence"]*100:.1f}% confidence)', s_body))
    story.append(Spacer(1, 6))

    # Top 20 gene chart
    gene_fig = plot_top20_shap(rf_result["gene_df"], patient_id)
    gene_tmp = os.path.join(tmp_dir, "genes.png")
    gene_fig.savefig(gene_tmp, dpi=130, bbox_inches="tight")
    plt.close(gene_fig)
    tmp_files.append(gene_tmp)
    story.append(RLImage(gene_tmp, width=W*0.72, height=3.5*rl_cm))
    story.append(Spacer(1, 8))

    # Full 100-gene table
    story.append(Paragraph("Complete Genomics Data — All 100 Genes", s_sec))
    gene_df = rf_result["gene_df"]
    rows    = [["Rank", "Gene", "Value", "SHAP Value", "Direction"]]
    for _, row in gene_df.iterrows():
        rows.append([
            str(int(row["Rank"])),
            row["Gene"],
            f"{row['Value']:+.4f}",
            f"{row['SHAP Value']:+.4f}",
            row["Direction"],
        ])

    full_table = Table(
        rows,
        colWidths=[W*0.08, W*0.18, W*0.18, W*0.18, W*0.35],
    )
    full_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), C_BG),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 7),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ALIGN",      (1,0), (1,-1), "LEFT"),
        ("ALIGN",      (4,0), (4,-1), "LEFT"),
        ("GRID",       (0,0), (-1,-1), 0.25, C_BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#F9F8F4")]),
    ]))
    story.append(full_table)
    story.append(Spacer(1, 10))

    # ── Interpretation ────────────────────────────────────────────────────────
    story.append(Paragraph("Combined Interpretation", s_sec))
    top_gene = rf_result["top5"][0][0]
    top_val  = rf_result["top5"][0][1]
    interp = (
        f"For patient {patient_id}, image analysis detects "
        f"{'tumor' if cnn_result['pred_class']==1 else 'normal'} tissue morphology "
        f"({cnn_result['confidence']*100:.1f}% confidence). "
        f"Genomic model predicts {rf_result['pred_label']} "
        f"({rf_result['confidence']*100:.1f}% confidence) from 100 gene features. "
        f"Most influential gene: {top_gene} "
        f"({'drives' if top_val>0 else 'suppresses'} high-risk signal). "
        f"Ground truth: {'HIGH RISK' if risk_label==1 else 'LOW RISK'}."
    )
    story.append(Paragraph(interp, s_interp))

    doc.build(story)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("🔬 Biomarker Discovery")
    st.sidebar.markdown(
        "Multi-modal cancer analysis combining "
        "histopathology images with genomic data."
    )
    st.sidebar.markdown("---")

    df = load_genomics()
    patient_ids = df["patient_id"].tolist()

    st.sidebar.subheader("Select Patient")
    selected_id = st.sidebar.selectbox("Patient ID", patient_ids, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Or Upload Custom Patch")
    uploaded = st.sidebar.file_uploader(
        "Upload 96×96 histopathology patch",
        type=["png", "jpg", "jpeg"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Models used:**\n"
        "- EfficientNet-B0 (image CNN)\n"
        "- Grad-CAM (image explainability)\n"
        "- Random Forest (genomics)\n"
        "- SHAP (gene importance)\n\n"
        "**Genes:** 100 cancer-relevant genes\n\n"
        "**Patients:** 50 (P1–P50)"
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 Explainable Multi-Modal Biomarker Discovery")
    st.markdown(
        "Combining **histopathology image analysis** (EfficientNet-B0 + Grad-CAM) "
        "with **genomic risk prediction** (Random Forest + SHAP) across 100 cancer genes."
    )
    st.markdown("---")

    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("Loading models ..."):
        cnn_model  = load_cnn()
        gradcam    = GradCAM(cnn_model)
        rf, scaler = load_rf_models()

    # ── Patient data ──────────────────────────────────────────────────────────
    patient_row = df[df["patient_id"] == selected_id].iloc[0]
    risk_label  = int(patient_row["risk_label"])

    if uploaded is not None:
        orig_arr = np.array(
            Image.open(uploaded).convert("RGB").resize((96, 96))
        )
        st.info("Custom patch used for image analysis. "
                "Genomics from selected patient.")
    else:
        patch_path = PATCH_DIR / patient_row["image_filename"]
        orig_arr   = np.array(Image.open(patch_path).convert("RGB"))

    # ── Inference ─────────────────────────────────────────────────────────────
    with st.spinner("Running inference ..."):
        cnn_result = run_cnn(cnn_model, gradcam, orig_arr)
        rf_result  = run_genomics(rf, scaler, patient_row)

    # ── Patient metrics ───────────────────────────────────────────────────────
    risk_str   = "HIGH RISK" if risk_label == 1 else "LOW RISK"
    risk_color = "🔴" if risk_label == 1 else "🟢"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patient",      selected_id)
    col2.metric("Ground Truth", f"{risk_color} {risk_str}")
    col3.metric("PCam Label",
                "Tumor" if patient_row["pcam_label"] == 1 else "Normal")
    col4.metric("Genes Analysed", "100")

    st.markdown("---")

    # ── Section 1: Image Analysis ─────────────────────────────────────────────
    st.subheader("🖼️ Section 1 — Image Analysis")
    c1, c2, c3 = st.columns([1, 1, 1.2])

    with c1:
        st.image(orig_arr, caption="Original Patch (96×96)",
                 use_column_width=True)
    with c2:
        st.image(cnn_result["overlay"], caption="Grad-CAM Heatmap",
                 use_column_width=True)
    with c3:
        icon = "🔴" if cnn_result["pred_class"] == 1 else "🟢"
        st.markdown(f"### {icon} {cnn_result['pred_label']}")
        st.metric("CNN Confidence", f"{cnn_result['confidence']*100:.1f}%")
        st.markdown(
            "Grad-CAM highlights tissue regions most influential "
            "in the tumor/normal prediction."
        )

    st.markdown("---")

    # ── Section 2: Genomics Analysis ──────────────────────────────────────────
    st.subheader("🧬 Section 2 — Genomic Risk Analysis (All 100 Genes)")

    rf_icon = "🔴" if rf_result["pred"] == 1 else "🟢"
    st.markdown(
        f"**RF Prediction:** {rf_icon} {rf_result['pred_label']} "
        f"| **Confidence:** {rf_result['confidence']*100:.1f}%"
    )

    # Full 100-gene SHAP chart
    st.markdown("#### SHAP Importance — All 100 Genes (ranked)")
    fig_all = plot_all_genes_shap(rf_result["gene_df"], selected_id)
    st.pyplot(fig_all, use_container_width=True)
    plt.close(fig_all)

    st.markdown("#### Complete Genomics Data Table — All 100 Genes")
    display_df = rf_result["gene_df"][
        ["Rank", "Gene", "Value", "SHAP Value", "Direction", "|SHAP|"]
    ].copy()
    display_df["|SHAP|"] = display_df["|SHAP|"].round(4)

    # Colour-code direction column
    def highlight_direction(val):
        if "Increases" in str(val):
            return "background-color: #FAECE7; color: #993C1D"
        elif "Decreases" in str(val):
            return "background-color: #E1F5EE; color: #0F6E56"
        return ""

    styled_df = display_df.style.map(
        highlight_direction, subset=["Direction"]
    ).format({
        "Value"      : "{:+.4f}",
        "SHAP Value" : "{:+.4f}",
        "|SHAP|"     : "{:.4f}",
    })

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Top 5 genes summary
    st.markdown("#### Top 5 Most Influential Genes")
    t_cols = st.columns(5)
    for i, (gene, val) in enumerate(rf_result["top5"]):
        direction = "↑" if val > 0 else "↓"
        color     = "#D85A30" if val > 0 else "#1D9E75"
        t_cols[i].markdown(
            f"<div style='text-align:center; padding:10px; "
            f"border:1px solid {color}; border-radius:8px;'>"
            f"<b style='color:{color}'>{direction} {gene}</b><br/>"
            f"<small>{val:+.4f}</small></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Section 3: Combined Interpretation ───────────────────────────────────
    st.subheader("📋 Section 3 — Combined Interpretation")
    top_gene = rf_result["top5"][0][0]
    top_val  = rf_result["top5"][0][1]

    st.info(
        f"**{selected_id}** — "
        f"Image CNN detects **{'tumor' if cnn_result['pred_class']==1 else 'normal'} morphology** "
        f"({cnn_result['confidence']*100:.1f}% confidence via Grad-CAM). "
        f"Random Forest on **100 genes** predicts **{rf_result['pred_label']}** "
        f"({rf_result['confidence']*100:.1f}% confidence). "
        f"Most influential gene: **{top_gene}** "
        f"({'drives' if top_val>0 else 'suppresses'} high-risk signal). "
        f"Ground truth: **{risk_str}**."
    )

    st.markdown("---")

    # ── Section 4: PDF Report ─────────────────────────────────────────────────
    st.subheader("📄 Section 4 — Download Full Report")
    st.markdown(
        "The PDF includes: patch image, Grad-CAM heatmap, CNN result, "
        "top-20 SHAP chart, complete 100-gene table, and interpretation."
    )

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF ..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_bytes = generate_pdf(
                    selected_id, risk_label,
                    cnn_result, rf_result,
                    orig_arr, tmp_dir,
                )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"report_{selected_id}.pdf",
            mime="application/pdf",
        )
        st.success("PDF ready!")


if __name__ == "__main__":
    main()