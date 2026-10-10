# -*- coding: utf-8 -*-
"""
main.py

Refactored for Transformer-based Isolated Sign Language Recognition.
Configuration: Single Split (Train/Val/Test), Golden Config.
"""

import os
import sys
import shutil
import json
import math
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# --- 1. Path Configuration (Local) ---
root_dir = "."
data_dir = os.path.join(root_dir, "dataset", "wlasl100")
sys.path.append(os.path.join(root_dir, "slt_tutorials"))

# Try importing modules
try:
    from modules.defines import (
        USE_POSE, USE_LHAND, USE_RHAND, USE_FACE, get_fullbody_landmarks)
    from modules.transforms import (
        SelectLandmarksAndFeature, ReplaceNan, FilterZero, PartsBasedNormalization, ToTensor)
    from modules.dataset import (
        make_wlasl_dataloader)
except ImportError:
    print("Error: Could not import modules from slt_tutorials. Please check your path.")
    sys.exit(1)


# --- 2. Data Augmentation ---

class RandomMove(object):
    """Randomly shift the coordinates."""
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"]
            # feat: [C, T, J] (numpy array)
            # Shift shape: [3, 1, 1] to broadcast over [C=3, T, J]
            shift = np.random.uniform(-self.shift_range, self.shift_range, size=(3, 1, 1))
            data["feature"] = feat + shift
        return data

class RandomScale(object):
    """Randomly scale the skeleton."""
    def __init__(self, scale_range=0.1):
        self.scale_range = scale_range

    def __call__(self, data):
        if "feature" in data:
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
            data["feature"] = data["feature"] * scale
        return data

class RandomTimeSample(object):
    """Randomly resample frames."""
    def __init__(self, max_frames=150, speed_range=0.2):
        self.max_frames = max_frames
        self.speed_range = speed_range

    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"] # [C, T, J]
            T = feat.shape[1]
            if T < 4: return data

            new_T = int(T * np.random.uniform(1 - self.speed_range, 1 + self.speed_range))
            if new_T > self.max_frames: new_T = self.max_frames
            if new_T < 4: new_T = 4
            
            indices = np.linspace(0, T-1, new_T)
            indices = np.round(indices).astype(int)
            data["feature"] = feat[:, indices, :]
        return data

class RotateToHorizontal(object):
    """
    Rotates the skeleton so that the line connecting Left Shoulder (11) 
    and Right Shoulder (12) becomes horizontal (parallel to x-axis).
    """
    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"] 
            
            # Safety Check: Ensure we have enough joints
            if feat.shape[1] < 13:
                return data

            l_shoulder_idx = 11
            r_shoulder_idx = 12
            
            # We need to perform rotation frame by frame
            # feat shape: [T, J, C] usually in this stage (before ToTensor)
            T, J, C = feat.shape
            
            rotated_feat = np.empty_like(feat)
            
            for t in range(T):
                ls = feat[t, l_shoulder_idx, :2] # (x, y)
                rs = feat[t, r_shoulder_idx, :2] # (x, y)
                
                if np.sum(ls) == 0 or np.sum(rs) == 0:
                    rotated_feat[t] = feat[t]
                    continue
                
                delta_y = rs[1] - ls[1]
                delta_x = rs[0] - ls[0]
                angle = np.arctan2(delta_y, delta_x)
                
                ca = np.cos(angle)
                sa = np.sin(angle)
                
                x = feat[t, :, 0]
                y = feat[t, :, 1]
                
                rotated_feat[t, :, 0] = x * ca + y * sa
                rotated_feat[t, :, 1] = -x * sa + y * ca
                
                if C > 2:
                    rotated_feat[t, :, 2:] = feat[t, :, 2:]
            
            data["feature"] = rotated_feat
        return data


class AddCentralVelocity(object):
    """
    Computes velocity using central difference: v[t] = (x[t+1] - x[t-1]) / 2.
    For boundaries:
    v[0] = x[1] - x[0] (Forward diff)
    v[T-1] = x[T-1] - x[T-2] (Backward diff)
    Concatenates velocity to features: [T, J, C] -> [T, J, 2*C]
    """
    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"] # [T, J, C]
            T, J, C = feat.shape
            
            if T < 3:
                # Too short for central diff, fallback to simple diff or zeros
                velocity = np.zeros_like(feat)
            else:
                velocity = np.zeros_like(feat)
                # Central Difference
                velocity[1:-1] = 0.5 * (feat[2:] - feat[:-2])
                # Boundaries
                velocity[0] = feat[1] - feat[0]
                velocity[-1] = feat[-1] - feat[-2]
            
            # Concatenate
            data["feature"] = np.concatenate([feat, velocity], axis=-1)
        return data


# --- 3. Custom Collate Function ---

def collate_fn_pad(batch):
    """
Pads batch of variable length sequences to the max length in the batch.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Get max length
    # batch[i]["feature"] shape: [C, T, J]
    max_len = max([b["feature"].shape[1] for b in batch])
    
    features = []
    tokens = []
    masks = []
    
    for b in batch:
        feat = b["feature"] # [C, T, J]
        C, T, J = feat.shape
        
        pad_len = max_len - T
        
        if not isinstance(feat, torch.Tensor):
            feat = torch.from_numpy(feat)
            
        padded_feat = F.pad(feat, (0, 0, 0, pad_len))
        features.append(padded_feat)
        
        tokens.append(b["tokens"])
        
        mask = torch.ones(max_len, dtype=torch.float32)
        if pad_len > 0:
            mask[T:] = 0
        masks.append(mask)

    features = torch.stack(features)
    tokens = torch.tensor(tokens)
    masks = torch.stack(masks)
    
    return {
        "feature": features,
        "tokens": tokens,
        "feature_pad_mask": masks
    }


# --- 4. Model Components ---

class GPoolRecognitionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.head = nn.Linear(in_channels, out_channels)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.head.weight, 0, math.sqrt(1. / self.out_channels))

    def forward(self, feature, mask=None):
        if len(feature.shape) == 3:
            feature = feature.unsqueeze(-1)
        
        if mask is None:
            feature = F.avg_pool2d(feature, feature.shape[2:])
        else:
            _mask = mask[:, None, :, None]
            num = _mask.sum([2, 3])
            num = torch.clamp(num, min=1) 
            _mask = _mask / num[:, :, None, None]
            _mask = _mask.expand(feature.shape)
            feature = (_mask * feature).sum([2, 3], keepdim=True)
            
        feature = feature.view(feature.shape[0], -1)
        feature = self.head(feature)
        return feature

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerIsoSLR(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_channels, d_model)
        self.activation = nn.ReLU()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = GPoolRecognitionHead(d_model, out_channels)

    def forward(self, feature, feature_pad_mask=None):
        feature = feature.permute([0, 2, 3, 1])
        N, T, J, C = feature.shape
        feature = feature.reshape([N, T, -1]) 

        feature = self.embedding(feature)
        feature = self.activation(feature)
        feature = self.pos_encoder(feature)

        if feature_pad_mask is not None:
            src_key_padding_mask = (feature_pad_mask == 0)
        else:
            src_key_padding_mask = None

        feature = self.transformer_encoder(feature, src_key_padding_mask=src_key_padding_mask)
        feature = feature.permute(0, 2, 1) 
        logits = self.head(feature, feature_pad_mask)
        return logits


# --- 5. Training/Validation Functions ---

def train_loop_iso(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"].to(device)
        tokens = batch_sample["tokens"].to(device)
        feature_pad_mask = batch_sample["feature_pad_mask"].to(device)

        pred = model(feature, feature_pad_mask=feature_pad_mask)
        loss = loss_fn(pred, tokens.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
            
    train_loss /= num_batches
    return train_loss

def val_loop_iso(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    val_loss = 0
    correct = 0
    total_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"].to(device)
            tokens = batch_sample["tokens"].to(device)
            feature_pad_mask = batch_sample["feature_pad_mask"].to(device)

            pred = model(feature, feature_pad_mask=feature_pad_mask)
            val_loss += loss_fn(pred, tokens.view(-1)).item()
            
            pred_ids = pred.argmax(dim=1)
            tgt_ids = tokens.view(-1)
            correct += (pred_ids == tgt_ids).sum().item()
            total_samples += tgt_ids.size(0)
            
    val_loss /= num_batches
    accuracy = correct / total_samples * 100 if total_samples > 0 else 0.0
    return val_loss, accuracy

def test_loop_iso(dataloader, model, device):
    size = len(dataloader.dataset)
    correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"].to(device)
            tokens = batch_sample["tokens"].to(device)
            feature_pad_mask = batch_sample["feature_pad_mask"].to(device)

            pred = model(feature, feature_pad_mask=feature_pad_mask)
            
            pred_ids = pred.argmax(dim=1)
            tgt_ids = tokens.view(-1)
            correct += (pred_ids == tgt_ids).sum().item()
            total_samples += tgt_ids.size(0)
            
    return correct / total_samples * 100 if total_samples > 0 else 0


# --- 6. Main Execution ---

def main():
    
    # 1. Setup Dictionary
    dictionary_path = os.path.join(data_dir, "wlasl_100.json")
    if not os.path.exists(dictionary_path):
        print(f"Error: Dictionary not found at {dictionary_path}")
        sys.exit(1)

    with open(dictionary_path, "r") as fread:
        word_dict = json.load(fread)
    out_channels = len(word_dict)

    # 2. Setup Transforms
    _, use_landmarks = get_fullbody_landmarks()
    use_features = ["x", "y", "z"]

    # Training Transforms
    train_transforms = Compose([
        RotateToHorizontal(),
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero(),
        RandomScale(0.1),
        RandomMove(0.05),
        RandomTimeSample(speed_range=0.2),
        PartsBasedNormalization(
            pose_head=0, pose_num=len(USE_POSE), pose_origin=[11, 12], pose_unit1=[11], pose_unit2=[12],
            lhand_head=len(USE_POSE), lhand_num=len(USE_LHAND), lhand_origin=[0, 2, 5, 9, 13, 17], lhand_unit1=[0], lhand_unit2=[2, 5, 9, 13, 17],
            rhand_head=len(USE_POSE) + len(USE_LHAND), rhand_num=len(USE_RHAND), rhand_origin=[0, 2, 5, 9, 13, 17], rhand_unit1=[0], rhand_unit2=[2, 5, 9, 13, 17],
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND), face_num=len(USE_FACE), face_origin=[0, 2], face_unit1=[7], face_unit2=[42]),
        AddCentralVelocity(), # Added after Norm
        ToTensor()])

    # Val/Test Transforms
    test_transforms = Compose([
        RotateToHorizontal(),
        SelectLandmarksAndFeature(use_landmarks, use_features),
        ReplaceNan(),
        FilterZero(),
        PartsBasedNormalization(
            pose_head=0, pose_num=len(USE_POSE), pose_origin=[11, 12], pose_unit1=[11], pose_unit2=[12],
            lhand_head=len(USE_POSE), lhand_num=len(USE_LHAND), lhand_origin=[0, 2, 5, 9, 13, 17], lhand_unit1=[0], lhand_unit2=[2, 5, 9, 13, 17],
            rhand_head=len(USE_POSE) + len(USE_LHAND), rhand_num=len(USE_RHAND), rhand_origin=[0, 2, 5, 9, 13, 17], rhand_unit1=[0], rhand_unit2=[2, 5, 9, 13, 17],
            face_head=len(USE_POSE) + len(USE_LHAND) + len(USE_RHAND), face_num=len(USE_FACE), face_origin=[0, 2], face_unit1=[7], face_unit2=[42]),
        AddCentralVelocity(), # Added after Norm
        ToTensor()])

    # Input dimension: (x, y, z) + (vx, vy, vz) = 6
    in_channels = 141 * 6 

    # 3. Load Data (Separate Splits)
    print("Loading data...")
    train_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "train"), os.path.join(data_dir, "labels", "train"),
                                     dictionary_path, batch_size=1, shuffle=True, transforms=train_transforms).dataset
    val_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "val"), os.path.join(data_dir, "labels", "val"),
                                   dictionary_path, batch_size=1, shuffle=False, transforms=test_transforms).dataset
    test_loader = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "test"), os.path.join(data_dir, "labels", "test"),
                                        dictionary_path, batch_size=1, shuffle=False, transforms=test_transforms)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)

    # 4. Config & Model
    EPOCHS = 200
    d_model = 128
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}. Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")
    
    now = datetime.now()
    save_dir = os.path.join(root_dir, "result_temp", f"SingleSplit_Final_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Model
    model = TransformerIsoSLR(in_channels, out_channels, d_model=d_model, num_layers=2, dropout=0.2).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction="mean", label_smoothing=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_acc = 0.0
    
    # 5. Training Loop
    print(f"Starting Training ({EPOCHS} Epochs)...")
    for epoch in range(EPOCHS):
        t_loss = train_loop_iso(train_loader, model, loss_fn, optimizer, device)
        v_loss, v_acc = val_loop_iso(val_loader, model, loss_fn, device)
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

    # 6. Evaluation
    print("Training finished. Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), weights_only=True))
    test_acc = test_loop_iso(test_loader, model, device)
    
    print(f"Best Val Loss: {best_val_loss:.4f} (Acc: {best_acc:.2f}%)")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # 7. Reporting & Plotting
    import csv
    csv_path = os.path.join(save_dir, "training_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Acc"])
        for i in range(len(history['train_loss'])):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['val_acc'][i]])
    print(f"Training history saved to {csv_path}")

    with open(os.path.join(save_dir, "README_Experiment.md"), "w") as f:
        f.write(f"# Final Single Run Results\n\n")
        f.write(f"## Config\n- Model: Transformer (dim=128, layers=2, drop=0.2)\n")
        f.write(f"- Epochs: {EPOCHS}\n\n")
        f.write(f"## Results\n")
        f.write(f"- **Test Accuracy**: **{test_acc:.2f}%**\n")
        f.write(f"- **Best Val Loss**: {best_val_loss:.4f}\n")
        f.write(f"- **Best Val Acc**: {best_acc:.2f}%\n")
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val')
    plt.title("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='green')
    plt.title("Accuracy"); plt.legend()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()