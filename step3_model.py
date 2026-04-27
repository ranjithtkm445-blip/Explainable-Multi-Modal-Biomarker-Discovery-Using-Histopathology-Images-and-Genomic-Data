# =============================================================================
# step3_model.py
# Purpose: Train EfficientNet-B0 on 50 patient-linked patches (pcam_label),
#          generate Grad-CAM heatmaps for all 50 patients,
#          save predictions + heatmap paths back to genomics_clean.csv.
# =============================================================================

import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("D:/HISTOPATHOLOGY")
INFO_PATH = BASE_DIR / "dataset_info.json"

with open(INFO_PATH) as f:
    INFO = json.load(f)

PATCH_DIR   = Path(INFO["patch_dir"])
GENOME_CSV  = Path(INFO["genomics_csv"])
MODEL_DIR   = Path(INFO["model_dir"])
GRADCAM_DIR = BASE_DIR / "gradcam"
GENE_NAMES  = INFO["gene_names"]

for d in [MODEL_DIR, GRADCAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE   = 96
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR         = 1e-4
VAL_RATIO  = 0.2      # 80% train / 20% val from 50 patches

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# =============================================================================
# PART A — Dataset
# =============================================================================

class PatientDataset(Dataset):
    """Loads 50 patient PNG patches with pcam_label (0=normal, 1=tumor)."""

    def __init__(self, df: pd.DataFrame, patch_dir: Path, transform=None):
        self.df        = df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.patch_dir / row["image_filename"]).convert("RGB")
        label = int(row["pcam_label"])
        if self.transform:
            img = self.transform(img)
        return img, label, row["patient_id"]


# ── Transforms ────────────────────────────────────────────────────────────────
# Heavy augmentation — essential for 50-sample training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# =============================================================================
# PART B — Model
# =============================================================================

def build_model() -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    All feature layers frozen — only classifier head is trained.
    With 50 samples, pretrained ImageNet features do the heavy lifting.
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 2),
    )
    return model


# =============================================================================
# PART C — Training
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


def train_model(df: pd.DataFrame):
    """Split 50 patients 80/20, train EfficientNet-B0, save best model."""

    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, random_state=SEED,
        stratify=df["pcam_label"]
    )

    print(f"\n📂 Train: {len(train_df)} patches  |  Val: {len(val_df)} patches")

    train_ds = PatientDataset(train_df, PATCH_DIR, train_transform)
    val_ds   = PatientDataset(val_df,   PATCH_DIR, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params : {trainable:,}")

    best_val_acc    = 0.0
    best_model_path = MODEL_DIR / "efficientnet_b0_best.pth"
    history         = []

    print(f"\n🚀 Training for {NUM_EPOCHS} epochs ...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "train_acc"  : round(train_acc,  2),
            "val_loss"   : round(val_loss,   4),
            "val_acc"    : round(val_acc,    2),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
                  f"time={time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_acc"     : val_acc,
                "mean"        : MEAN,
                "std"         : STD,
            }, best_model_path)

    print(f"\n💾 Best model saved  : {best_model_path}")
    print(f"   Best val accuracy : {best_val_acc:.1f}%")

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Reload best weights
    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    return model


# =============================================================================
# PART D — Grad-CAM
# =============================================================================

class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last convolutional block (features[-1]) to capture
    gradients and activations, then produces a heatmap.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook into last conv block of EfficientNet-B0
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Returns:
            pred_class  : int (0=normal, 1=tumor)
            confidence  : float
            heatmap     : np.ndarray (96, 96) float in [0, 1]
        """
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_()

        # Forward
        logits = self.model(img_tensor)
        probs  = torch.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(probs.argmax().item())

        confidence = float(probs[class_idx].item())

        # Backward for target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-CAM computation
        grads  = self.gradients[0]           # (C, H, W)
        acts   = self.activations[0]         # (C, H, W)
        weights = grads.mean(dim=(1, 2))     # global average pooling

        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.clamp(cam, min=0)
        cam = cam.cpu().numpy()

        # Resize to input image size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return class_idx, confidence, cam


def save_gradcam_overlay(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    patient_id: str,
    pred_label: str,
    confidence: float,
    out_path: Path,
):
    """Save side-by-side: original patch | Grad-CAM overlay."""

    # Colormap heatmap
    colormap   = cm.get_cmap("jet")
    heatmap_c  = colormap(heatmap)[:, :, :3]          # (H, W, 3) float
    heatmap_c  = (heatmap_c * 255).astype(np.uint8)

    # Overlay
    orig_float = original_img.astype(np.float32) / 255.0
    heat_float = heatmap_c.astype(np.float32)   / 255.0
    overlay    = (0.55 * orig_float + 0.45 * heat_float)
    overlay    = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original_img)
    axes[0].set_title(f"{patient_id}\nOriginal", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    color = "#D85A30" if pred_label == "tumor" else "#1D9E75"
    axes[1].set_title(
        f"Grad-CAM\n{pred_label} ({confidence*100:.1f}%)",
        fontsize=9, color=color, fontweight="bold"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def run_gradcam(model: nn.Module, df: pd.DataFrame) -> pd.DataFrame:
    """Generate Grad-CAM heatmaps for all 50 patient patches."""

    print("\n🔥 Generating Grad-CAM heatmaps for all 50 patients ...")

    gradcam = GradCAM(model)

    predictions  = []
    confidences  = []
    pred_labels  = []
    heatmap_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grad-CAM"):
        img_path   = PATCH_DIR / row["image_filename"]
        orig_img   = np.array(Image.open(img_path).convert("RGB"))

        # Preprocess for model
        pil_img    = Image.fromarray(orig_img)
        tensor     = val_transform(pil_img)

        pred_class, conf, heatmap = gradcam.generate(tensor)
        pred_label = "tumor" if pred_class == 1 else "normal"

        # Save overlay image
        out_name = f"gradcam_{row['patient_id']}.png"
        out_path = GRADCAM_DIR / out_name
        save_gradcam_overlay(orig_img, heatmap, row["patient_id"],
                             pred_label, conf, out_path)

        predictions.append(pred_class)
        confidences.append(round(conf, 4))
        pred_labels.append(pred_label)
        heatmap_paths.append(str(out_path))

    df["cnn_prediction"]  = predictions
    df["cnn_label"]       = pred_labels
    df["cnn_confidence"]  = confidences
    df["gradcam_path"]    = heatmap_paths

    print(f"\n📊 Inference summary:")
    print(f"   Tumor predicted  : {pred_labels.count('tumor')}")
    print(f"   Normal predicted : {pred_labels.count('normal')}")
    print(f"\n   Sample predictions:")
    print(df[["patient_id", "cnn_label", "cnn_confidence",
              "risk_label"]].head(10).to_string(index=False))
    print(f"\n✅ Grad-CAM images saved → {GRADCAM_DIR}")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 3 — CNN Training + Grad-CAM")
    print("  50 Patient Patches · EfficientNet-B0")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(GENOME_CSV)
    print(f"\n✅ Loaded {len(df)} patients from {GENOME_CSV.name}")

    # Train model on 50 patches
    model = train_model(df)

    # Grad-CAM on all 50 patients
    df = run_gradcam(model, df)

    # Save updated CSV
    df.to_csv(GENOME_CSV, index=False)
    print(f"\n✅ Updated CSV saved → {GENOME_CSV}")

    print("\n" + "=" * 60)
    print("  Step 3 complete. Run step4_genomics.py next.")
    print("=" * 60)