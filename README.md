Explainable Multi-Modal Biomarker Discovery
Using Histopathology Images and Genomic Data
1. Project Overview
This project builds an explainable AI system that combines histopathology image analysis with genomic risk prediction for cancer biomarker discovery. It predicts tumor presence from tissue patches and identifies key cancer genes using two complementary machine learning pipelines.
2. Problem Statement
Cancer analysis traditionally relies on tissue morphology (histopathology images) and molecular data (gene expression). However, most AI models are black-box systems that lack interpretability, limiting clinical trust. This project addresses this gap by providing:

Tumor detection from histopathology images
Genomic risk prediction from 100 cancer-relevant genes
Explainability via Grad-CAM (image regions) and SHAP (gene importance)
Combined multi-modal patient report with PDF export

3. Dataset — Histopathology Images
Source: PatchCamelyon (PCam) — Kaggle | 96x96 RGB | 262,144 train | 32,768 val | 32,768 test | 50 patient-linked patches
3.2 Genomics Data
50 patients (P1-P50) | 100 real cancer genes | risk_label | 21 high-risk | 29 low-risk | Top gene: BRCA1
4. Pipeline
Step 1–7: dataset prep → data verification → CNN+GradCAM → Random Forest → SHAP → PDF report → Streamlit app
5. Models
EfficientNet-B0: 90.0% val accuracy | Random Forest: 68.0% CV accuracy | Top gene: BRCA1 (0.0583)
6–7. Results & Top SHAP Genes
BRCA1, STAT5, PIK3CA, ALK, FGFR1, TP53, MLH1, NTRK2, GRB2, CDK1
8. Explainability
Grad-CAM (image heatmaps) + SHAP TreeExplainer (gene importance per patient)
9. Deployment
Hugging Face Spaces | Docker | Streamlit | Python 3.11
URL: https://ranjith445-histopath-multimodal-biomarker.hf.space/
13. Author
M. Ranjith Kumar | GitHub: ranjithtkm445-blip | HF: Ranjith445
