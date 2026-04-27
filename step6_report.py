# =============================================================================
# step6_report.py
# Purpose: Run full inference (CNN + Random Forest + SHAP) on 10 preloaded
#          patients (5 high-risk + 5 low-risk) and generate one PDF report.
#          Each patient page: patch image, Grad-CAM, CNN result,
#          RF risk prediction, top SHAP genes, combined interpretation.
# =============================================================================

import json
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
import io
import tempfile
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("D:/HISTOPATHOLOGY")
INFO_PATH = BASE_DIR / "dataset_info.json"

with open(INFO_PATH) as f:
    INFO = json.load(f)

GENOME_CSV  = Path(INFO["genomics_csv"])
PATCH_DIR   = Path(INFO["patch_dir"])
GRADCAM_DIR = BASE_DIR / "gradcam"
MODEL_DIR   = Path(INFO["model_dir"])
GENE_NAMES  = INFO["gene_names"]
REPORT_DIR  = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_OUT = REPORT_DIR / "multimodal_patient_report.pdf"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN   = [0.485, 0.456, 0.406]
STD    = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Colours ───────────────────────────────────────────────────────────────────
C_HIGH   = colors.HexColor("#D85A30")
C_LOW    = colors.HexColor("#1D9E75")
C_HEADER = colors.HexColor("#2C2C2A")
C_MUTED  = colors.HexColor("#5F5E5A")
C_BORDER = colors.HexColor("#D3D1C7")
C_BG     = colors.HexColor("#F1EFE8")


# =============================================================================
# PART A — Load models
# =============================================================================

def load_cnn() -> nn.Module:
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


def load_rf():
    rf     = joblib.load(MODEL_DIR / "random_forest.pkl")
    scaler = joblib.load(MODEL_DIR / "rf_scaler.pkl")
    return rf, scaler


# =============================================================================
# PART B — Select 10 patients (5 high-risk + 5 low-risk)
# =============================================================================

def select_patients() -> pd.DataFrame:
    df   = pd.read_csv(GENOME_CSV)
    high = df[df["risk_label"] == 1].head(5)
    low  = df[df["risk_label"] == 0].head(5)
    selected = pd.concat([high, low]).reset_index(drop=True)
    print(f"✅ Selected 10 patients  (5 high-risk + 5 low-risk)")
    return selected, df


# =============================================================================
# PART C — CNN inference + Grad-CAM per patient
# =============================================================================

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        model.features[-1].register_forward_hook(self._save_act)
        model.features[-1].register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):  self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients  = go[0].detach()

    def generate(self, tensor):
        self.model.eval()
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


def make_gradcam_overlay(orig: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    colormap  = matplotlib.colormaps["jet"]
    heat_c    = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay   = (0.55 * orig + 0.45 * heat_c)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_cnn_inference(model, gradcam, patient_row):
    img_path  = PATCH_DIR / patient_row["image_filename"]
    orig_arr  = np.array(Image.open(img_path).convert("RGB"))
    tensor    = val_transform(Image.fromarray(orig_arr))

    cls, conf, heatmap = gradcam.generate(tensor)
    overlay = make_gradcam_overlay(orig_arr, heatmap)

    return {
        "orig"       : orig_arr,
        "overlay"    : overlay,
        "pred_class" : cls,
        "pred_label" : "Tumor" if cls == 1 else "Normal",
        "confidence" : conf,
    }


# =============================================================================
# PART D — RF + SHAP inference per patient
# =============================================================================

def run_genomics_inference(rf, scaler, patient_row, top_n=5):
    X        = np.array([[patient_row[g] for g in GENE_NAMES]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    pred    = int(rf.predict(X_scaled)[0])
    probs   = rf.predict_proba(X_scaled)[0]
    conf    = float(probs[pred])

    # SHAP for this patient
    explainer   = shap.TreeExplainer(rf)
    shap_vals   = explainer.shap_values(X_scaled)
    sv = shap_vals[:, :, 1] if (hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3) \
         else (shap_vals[1] if isinstance(shap_vals, list) else shap_vals)

    abs_vals = np.abs(sv[0])
    top_idx  = np.argsort(abs_vals)[::-1][:top_n]
    top_genes  = [(GENE_NAMES[i], round(float(sv[0][i]), 4)) for i in top_idx]

    return {
        "pred"       : pred,
        "pred_label" : "High-risk" if pred == 1 else "Low-risk",
        "confidence" : conf,
        "top_genes"  : top_genes,
    }


# =============================================================================
# PART E — Save temp images for PDF
# =============================================================================

def arr_to_tmp(arr: np.ndarray, suffix=".png") -> str:
    """Save numpy array as temp PNG, return path string."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    Image.fromarray(arr).save(tmp.name)
    tmp.close()
    return tmp.name


def make_gene_bar_image(top_genes, patient_id) -> str:
    """Create a horizontal bar chart of top SHAP genes, return temp path."""
    genes  = [g for g, _ in top_genes]
    values = [v for _, v in top_genes]
    cols   = ["#D85A30" if v > 0 else "#378ADD" for v in values]

    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.barh(genes[::-1], values[::-1], color=cols[::-1],
            edgecolor="white", height=0.6)
    ax.axvline(0, color="#888780", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=8)
    ax.set_title(f"Top genes — {patient_id}", fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight")
    plt.close()
    tmp.close()
    return tmp.name


# =============================================================================
# PART F — Build PDF
# =============================================================================

def build_pdf(patients_data: list):
    """
    patients_data: list of dicts, one per patient, containing all results.
    Generates one PDF with:
      - Cover page
      - One page per patient
    """
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=A4,
        leftMargin=1.8*rl_cm, rightMargin=1.8*rl_cm,
        topMargin=1.8*rl_cm,  bottomMargin=1.8*rl_cm,
    )

    W = A4[0] - 3.6*rl_cm   # usable width

    # ── Styles ────────────────────────────────────────────────────────────────
    s_title = ParagraphStyle("t", fontSize=22, fontName="Helvetica-Bold",
                             textColor=C_HEADER, alignment=TA_CENTER, spaceAfter=6)
    s_sub   = ParagraphStyle("s", fontSize=10, fontName="Helvetica",
                             textColor=C_MUTED,  alignment=TA_CENTER, spaceAfter=20)
    s_pid   = ParagraphStyle("p", fontSize=15, fontName="Helvetica-Bold",
                             textColor=C_HEADER, spaceAfter=6)
    s_sec   = ParagraphStyle("sec", fontSize=10, fontName="Helvetica-Bold",
                             textColor=C_MUTED,  spaceAfter=4)
    s_body  = ParagraphStyle("b", fontSize=9,  fontName="Helvetica",
                             textColor=C_HEADER, spaceAfter=4)
    s_interp= ParagraphStyle("i", fontSize=9, fontName="Helvetica-Oblique",
                             textColor=C_HEADER, spaceAfter=4,
                             leftIndent=8, rightIndent=8)

    story = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*rl_cm))
    story.append(Paragraph("Explainable Multi-Modal", s_title))
    story.append(Paragraph("Biomarker Discovery Report", s_title))
    story.append(Spacer(1, 0.5*rl_cm))
    story.append(Paragraph(
        "Histopathology Image Analysis + Genomic Risk Prediction", s_sub))
    story.append(Paragraph(
        "EfficientNet-B0 · Grad-CAM · Random Forest · SHAP", s_sub))
    story.append(Spacer(1, 1*rl_cm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=C_BORDER, spaceAfter=20))
    story.append(Paragraph(f"Patients analysed : 10  (5 high-risk + 5 low-risk)",
                           s_body))
    story.append(Paragraph(f"Genes analysed    : {len(GENE_NAMES)}", s_body))
    story.append(Paragraph(
        "Dataset : PatchCamelyon (PCam) histopathology + synthetic genomics",
        s_body))
    story.append(PageBreak())

    # ── One page per patient ──────────────────────────────────────────────────
    tmp_files = []

    for pd_row in patients_data:
        patient_id = pd_row["patient_id"]
        risk_label = pd_row["risk_label"]
        cnn        = pd_row["cnn"]
        rf         = pd_row["rf"]

        risk_color = C_HIGH if risk_label == 1 else C_LOW
        risk_str   = "HIGH RISK" if risk_label == 1 else "LOW RISK"

        # Patient header
        story.append(Paragraph(
            f'<font color="#{risk_color.hexval()[2:]}">■</font>  '
            f'{patient_id} — {risk_str}',
            s_pid
        ))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=C_BORDER, spaceAfter=10))

        # ── Images row: patch | grad-cam ─────────────────────────────────────
        orig_tmp = arr_to_tmp(cnn["orig"])
        gcam_tmp = arr_to_tmp(cnn["overlay"])
        tmp_files += [orig_tmp, gcam_tmp]

        img_w = W * 0.35
        img_h = img_w

        img_table = Table(
            [[
                RLImage(orig_tmp, width=img_w, height=img_h),
                RLImage(gcam_tmp, width=img_w, height=img_h),
            ]],
            colWidths=[img_w + 0.3*rl_cm, img_w + 0.3*rl_cm],
        )
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(img_table)

        # Image labels
        label_table = Table(
            [["Original Patch", "Grad-CAM Overlay"]],
            colWidths=[img_w + 0.3*rl_cm, img_w + 0.3*rl_cm],
        )
        label_table.setStyle(TableStyle([
            ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE",  (0, 0), (-1, -1), 8),
            ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_MUTED),
        ]))
        story.append(label_table)
        story.append(Spacer(1, 8))

        # ── CNN results ───────────────────────────────────────────────────────
        story.append(Paragraph("Image Analysis (CNN)", s_sec))
        cnn_color = C_HIGH if cnn["pred_class"] == 1 else C_LOW
        story.append(Paragraph(
            f'Prediction: <b>{cnn["pred_label"]}</b> '
            f'({cnn["confidence"]*100:.1f}% confidence)',
            s_body
        ))
        story.append(Spacer(1, 6))

        # ── RF + SHAP results ─────────────────────────────────────────────────
        story.append(Paragraph("Genomic Risk Analysis (Random Forest)", s_sec))
        story.append(Paragraph(
            f'Risk prediction : <b>{rf["pred_label"]}</b> '
            f'({rf["confidence"]*100:.1f}% confidence)',
            s_body
        ))

        # Gene bar chart
        gene_tmp = make_gene_bar_image(rf["top_genes"], patient_id)
        tmp_files.append(gene_tmp)
        story.append(RLImage(gene_tmp, width=W * 0.62, height=2.2*rl_cm))
        story.append(Spacer(1, 6))

        # Top genes table
        story.append(Paragraph("Top SHAP genes:", s_sec))
        gene_rows = [["Gene", "SHAP value", "Direction"]]
        for gene, val in rf["top_genes"]:
            direction = "Increases risk" if val > 0 else "Decreases risk"
            gene_rows.append([gene, f"{val:+.4f}", direction])

        gene_table = Table(gene_rows,
                           colWidths=[W*0.3, W*0.2, W*0.4])
        gene_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_BG),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("ALIGN",      (1, 0), (1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.3, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F9F8F4")]),
        ]))
        story.append(gene_table)
        story.append(Spacer(1, 8))

        # ── Combined interpretation ───────────────────────────────────────────
        story.append(Paragraph("Combined Interpretation", s_sec))
        cnn_txt  = f"tissue shows {'tumor' if cnn['pred_class']==1 else 'normal'} morphology"
        rf_txt   = f"genomic profile indicates {'high' if rf['pred']==1 else 'low'} risk"
        top_gene = rf["top_genes"][0][0]
        interp   = (
            f"For patient {patient_id}, the image analysis detects {cnn_txt} "
            f"({cnn['confidence']*100:.1f}% confidence), while the genomic model finds "
            f"{rf_txt} ({rf['confidence']*100:.1f}% confidence). "
            f"The most influential gene is {top_gene}, "
            f"{'driving' if rf['top_genes'][0][1]>0 else 'suppressing'} the high-risk signal. "
            f"Ground truth risk label: {'HIGH RISK' if risk_label==1 else 'LOW RISK'}."
        )
        story.append(Paragraph(interp, s_interp))
        story.append(PageBreak())

    # Build PDF
    doc.build(story)

    # Cleanup temp files
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    print(f"\n✅ PDF saved → {PDF_OUT}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 6 — Inference + PDF Report Generation")
    print("  10 Patients · CNN + Grad-CAM + RF + SHAP")
    print("=" * 60)

    # Load models
    print("\n🧠 Loading models ...")
    cnn_model  = load_cnn()
    gradcam    = GradCAM(cnn_model)
    rf, scaler = load_rf()
    print("   ✅ EfficientNet-B0 loaded")
    print("   ✅ Random Forest loaded")

    # Select 10 patients
    selected_df, full_df = select_patients()

    # Run inference on each patient
    print("\n🔬 Running inference on 10 patients ...")
    patients_data = []

    for _, row in selected_df.iterrows():
        pid = row["patient_id"]
        print(f"   Processing {pid} ...")

        cnn_result = run_cnn_inference(cnn_model, gradcam, row)
        rf_result  = run_genomics_inference(rf, scaler, row, top_n=5)

        patients_data.append({
            "patient_id" : pid,
            "risk_label" : int(row["risk_label"]),
            "cnn"        : cnn_result,
            "rf"         : rf_result,
        })

        print(f"      CNN : {cnn_result['pred_label']} "
              f"({cnn_result['confidence']*100:.1f}%)  "
              f"RF : {rf_result['pred_label']} "
              f"({rf_result['confidence']*100:.1f}%)")

    # Generate PDF
    print("\n📄 Generating PDF report ...")
    build_pdf(patients_data)

    print("\n" + "=" * 60)
    print("  Step 6 complete. Run step7_app.py next.")
    print("=" * 60)