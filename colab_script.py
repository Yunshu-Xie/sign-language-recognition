# -*- coding: utf-8 -*-
"""
WLASL_Transformer_KFold_Final.py

Integrated script for running on Google Colab.
Best performing configuration:
- Transformer Model (Post-Norm, d_model=128, layers=2)
- Features: Raw Coordinates (x,y,z) only (No velocity)
- Data Augmentation: Move, Scale, TimeSample
- 5-Fold Cross Validation (200 Epochs)
- Cosine Annealing Scheduler

Usage:
1. Upload this file to Google Colab.
2. Ensure your dataset is in Google Drive.
3. Run: !python WLASL_Transformer_KFold_Final.py
"""

import os
import sys

# --- 0. Dependency Check (Auto-Install) ---
try:
    import cv2
    print("OpenCV is already installed.")
except ImportError:
    print("OpenCV not found. Installing...")
    os.system("pip install opencv-python")
    import cv2
    print("OpenCV installed successfully.")

import shutil
import json
import math
import random
import numpy as np
from datetime import datetime
import copy
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import ConcatDataset, DataLoader, Subset, RandomSampler
from torchvision.transforms import Compose

# --- 1. Environment Setup (Colab Specific) ---
try:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount("/content/drive")
    
    # === CONFIGURATION ===
    drive_root = "/content/drive/MyDrive"
    
    # 1. Setup slt_tutorials path
    # User specified relative path: drive/Mydrive/slt_tutorials -> /content/drive/MyDrive/slt_tutorials
    slt_tutorials_path = os.path.join(drive_root, "slt_tutorials")
    
    if os.path.exists(slt_tutorials_path):
        sys.path.append(slt_tutorials_path)
        print(f"Added {slt_tutorials_path} to sys.path")
    else:
        print(f"Warning: {slt_tutorials_path} not found. Checking fallback...")
        # Fallback: Maybe it's in the old project path?
        old_path = os.path.join(drive_root, "东大", "課題", "手话", "slt_tutorials")
        if os.path.exists(old_path):
            sys.path.append(old_path)
            print(f"Added {old_path} to sys.path")
        elif os.path.exists("./slt_tutorials"):
            sys.path.append(".")
            print("Added local ./slt_tutorials to sys.path")

    # 2. Setup Dataset
    # Try to find wlasl100.zip in probable locations
    possible_zips = [
        os.path.join(drive_root, "dataset", "wlasl100.zip"),
        os.path.join(drive_root, "wlasl100.zip"),
        os.path.join(slt_tutorials_path, "wlasl100.zip"),
        os.path.join(drive_root, "东大", "課題", "手话", "dataset", "wlasl100.zip")
    ]
    
    zip_source = None
    for p in possible_zips:
        if os.path.exists(p):
            zip_source = p
            break
    
    zip_dest = "./wlasl100.zip"
    
    if zip_source:
        print(f"Found dataset at: {zip_source}")
        if not os.path.exists(zip_dest):
            print(f"Copying dataset...")
            shutil.copyfile(zip_source, zip_dest)
            print("Unzipping dataset...")
            os.system("unzip -q -o ./wlasl100.zip")
            print("Unzip complete.")
        else:
            print("Dataset already unzipped locally.")
    else:
        print("Error: wlasl100.zip not found in Drive. Please check filepath.")

    data_dir = "./wlasl100"

    # 3. Output Dir
    # Save to Drive root result_temp
    result_root = os.path.join(drive_root, "result_temp")
    if not os.path.exists(result_root):
        os.makedirs(result_root, exist_ok=True)

except ImportError:
    print("Not running on Colab? Skipping Drive setup.")
    project_root = "."
    data_dir = os.path.join(project_root, "dataset", "wlasl100")
    result_root = os.path.join(project_root, "result_temp")
    sys.path.append(os.path.join(project_root, "slt_tutorials"))


# --- 2. Data Augmentation ---

class RandomMove(object):
    def __init__(self, shift_range=0.1): self.shift_range = shift_range
    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"]
            # Fix broadcasting: shift [3, 1, 1] for feat [C, T, J]
            shift = np.random.uniform(-self.shift_range, self.shift_range, size=(3, 1, 1))
            data["feature"] = feat + shift
        return data

class RandomScale(object):
    def __init__(self, scale_range=0.1): self.scale_range = scale_range
    def __call__(self, data):
        if "feature" in data:
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
            data["feature"] = data["feature"] * scale
        return data

class RandomTimeSample(object):
    def __init__(self, max_frames=150, speed_range=0.2):
        self.max_frames = max_frames; self.speed_range = speed_range
    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"]
            T = feat.shape[1] # [C, T, J]
            if T < 4: return data
            new_T = int(T * np.random.uniform(1 - self.speed_range, 1 + self.speed_range))
            if new_T > self.max_frames: new_T = self.max_frames
            if new_T < 4: new_T = 4
            indices = np.linspace(0, T-1, new_T)
            indices = np.round(indices).astype(int)
            data["feature"] = feat[:, indices, :]
        return data


# --- 3. Custom Collate ---

def collate_fn_pad(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    max_len = max([b["feature"].shape[1] for b in batch])
    
    features, tokens, masks = [], [], []
    for b in batch:
        feat = b["feature"]
        C, T, J = feat.shape
        pad_len = max_len - T
        if not isinstance(feat, torch.Tensor): feat = torch.from_numpy(feat)
        padded_feat = F.pad(feat, (0, 0, 0, pad_len))
        features.append(padded_feat)
        tokens.append(b["tokens"])
        mask = torch.ones(max_len, dtype=torch.float32)
        if pad_len > 0: mask[T:] = 0
        masks.append(mask)

    return {
        "feature": torch.stack(features),
        "tokens": torch.tensor(tokens),
        "feature_pad_mask": torch.stack(masks)
    }


# --- 4. Model Components ---

class GPoolRecognitionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels # Fix AttributeError
        self.head = nn.Linear(in_channels, out_channels)
        self._init_weight()
    def _init_weight(self):
        nn.init.normal_(self.head.weight, 0, math.sqrt(1. / self.out_channels))
    def forward(self, feature, mask=None):
        if len(feature.shape) == 3: feature = feature.unsqueeze(-1)
        if mask is None:
            feature = F.avg_pool2d(feature, feature.shape[2:])
        else:
            _mask = mask[:, None, :, None]
            num = torch.clamp(_mask.sum([2, 3]), min=1)
            _mask = _mask / num[:, :, None, None]
            feature = (_mask.expand(feature.shape) * feature).sum([2, 3], keepdim=True)
        return self.head(feature.view(feature.shape[0], -1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerIsoSLR(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_channels, d_model)
        self.activation = nn.ReLU()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = GPoolRecognitionHead(d_model, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        # [N, C, T, J] -> [N, T, J, C] -> [N, T, J*C]
        feature = feature.permute([0, 2, 3, 1])
        feature = feature.reshape([feature.shape[0], feature.shape[1], -1])
        feature = self.pos_encoder(self.activation(self.embedding(feature)))
        
        src_key_padding_mask = (feature_pad_mask == 0) if feature_pad_mask is not None else None
        feature = self.transformer_encoder(feature, src_key_padding_mask=src_key_padding_mask)
        return self.head(feature.permute(0, 2, 1), feature_pad_mask)


# --- 5. Training Utils ---

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, num_batches = 0, len(dataloader)
    for batch in dataloader:
        feat, tokens, mask = batch["feature"].to(device), batch["tokens"].to(device), batch["feature_pad_mask"].to(device)
        pred = model(feat, mask)
        loss = loss_fn(pred, tokens.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / num_batches

def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch in dataloader:
            feat, tokens, mask = batch["feature"].to(device), batch["tokens"].to(device), batch["feature_pad_mask"].to(device)
            pred = model(feat, mask)
            total_loss += loss_fn(pred, tokens.view(-1)).item()
            
            # Fix Accuracy Calc
            pred_ids = pred.argmax(dim=1)
            tgt_ids = tokens.view(-1)
            correct += (pred_ids == tgt_ids).sum().item()
            total_samples += tgt_ids.size(0)
            
    return total_loss / num_batches, (correct / total_samples * 100 if total_samples > 0 else 0)

def test_loop(dataloader, model, device):
    model.eval()
    correct, total_samples = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            feat, tokens, mask = batch["feature"].to(device), batch["tokens"].to(device), batch["feature_pad_mask"].to(device)
            pred = model(feat, mask)
            
            pred_ids = pred.argmax(dim=1)
            tgt_ids = tokens.view(-1)
            correct += (pred_ids == tgt_ids).sum().item()
            total_samples += tgt_ids.size(0)
    return correct / total_samples * 100 if total_samples > 0 else 0

def save_checkpoint(ckpt_path, model, optimizer=None):
    checkpoint = { "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None }
    torch.save(checkpoint, ckpt_path)


# --- 6. Main ---

def main():
    try:
        from modules.defines import (
            USE_POSE, USE_LHAND, USE_RHAND, USE_FACE, get_fullbody_landmarks)
        from modules.transforms import (
            SelectLandmarksAndFeature, ReplaceNan, FilterZero, PartsBasedNormalization, ToTensor)
        from modules.dataset import make_wlasl_dataloader
        print("Successfully imported project modules.")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return

    dictionary_path = os.path.join(data_dir, "wlasl_100.json")
    if not os.path.exists(dictionary_path):
        print(f"Dictionary not found at {dictionary_path}")
        return

    with open(dictionary_path, "r") as f: word_dict = json.load(f)
    out_channels = len(word_dict)
    _, use_landmarks = get_fullbody_landmarks()
    
    # 1. Transforms
    common_norm = PartsBasedNormalization(
        pose_head=0, pose_num=len(USE_POSE), pose_origin=[11, 12], pose_unit1=[11], pose_unit2=[12],
        lhand_head=len(USE_POSE), lhand_num=len(USE_LHAND), lhand_origin=[0, 2, 5, 9, 13, 17], lhand_unit1=[0], lhand_unit2=[2, 5, 9, 13, 17],
        rhand_head=len(USE_POSE) + len(USE_LHAND), rhand_num=len(USE_RHAND), rhand_origin=[0, 2, 5, 9, 13, 17], rhand_unit1=[0], rhand_unit2=[2, 5, 9, 13, 17],
        face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND), face_num=len(USE_FACE), face_origin=[0, 2], face_unit1=[7], face_unit2=[42])

    train_transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, ["x", "y", "z"]),
        ReplaceNan(), FilterZero(),
        RandomScale(0.1), RandomMove(0.05), RandomTimeSample(0.2),
        common_norm, ToTensor()])
    
    test_transforms = Compose([
        SelectLandmarksAndFeature(use_landmarks, ["x", "y", "z"]),
        ReplaceNan(), FilterZero(), common_norm, ToTensor()])

    # 2. Data Loading (Standard Train/Val/Test Split)
    print("Loading data...")
    # Explicitly set shuffle
    train_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "train"), os.path.join(data_dir, "labels", "train"), dictionary_path, batch_size=1, shuffle=True, transforms=train_transforms).dataset
    val_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "val"), os.path.join(data_dir, "labels", "val"), dictionary_path, batch_size=1, shuffle=False, transforms=train_transforms).dataset
    test_loader = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "test"), os.path.join(data_dir, "labels", "test"), dictionary_path, batch_size=1, shuffle=False, transforms=test_transforms)
    
    # Create DataLoaders with custom collate
    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)

    # Config
    EPOCHS = 200
    d_model = 128
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}. Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")
    
    now = datetime.now()
    save_dir = os.path.join(result_root, f"Final_SingleRun_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Model
    model = TransformerIsoSLR(141*3, out_channels, d_model=d_model, num_layers=2, dropout=0.1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1) 
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_acc = 0.0
    
    print(f"Starting Training ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        t_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        v_loss, v_acc = val_loop(val_loader, model, loss_fn, device)
        scheduler.step()
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1}/{EPOCHS}: Train {t_loss:.4f}, Val {v_loss:.4f}, Acc {v_acc:.2f}%")

    # Final Test
    print("Training finished. Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    test_acc = test_loop(test_loader, model, device)
    print(f"Best Val Loss: {best_val_loss:.4f} (Acc: {best_acc:.2f}%)")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Summary & Report
    with open(os.path.join(save_dir, "README_Experiment.md"), "w") as f:
        f.write(f"# Final Single Run Results\n\n")
        f.write(f"- **Test Accuracy**: **{test_acc:.2f}%**\n")
        f.write(f"- **Best Val Loss**: {best_val_loss:.4f}\n")
        f.write(f"- **Best Val Acc**: {best_acc:.2f}%\n")
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.title("Val Accuracy"); plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()