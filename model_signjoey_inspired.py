# -*- coding: utf-8 -*-
"""
S_3_4_signjoey_inspired_transformer.py

Optimized Transformer-based Isolated Sign Language Recognition.
Inspired by "Sign Language Transformers" (Camgoz et al., CVPR'20).

Key Improvements:
1. Pre-Layer Normalization (norm_first=True) for better convergence.
2. Noam Learning Rate Scheduler (Inverse Square Root Decay).
3. Xavier/Glorot Initialization.
4. K-Fold Cross Validation (5 Folds, 200 Epochs).
"""

import os
import sys
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
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import Compose

# --- 1. Path Configuration ---
root_dir = "."
data_dir = os.path.join(root_dir, "dataset", "wlasl100")
sys.path.append(os.path.join(root_dir, "slt_tutorials"))

from modules.defines import (
    USE_POSE, USE_LHAND, USE_RHAND, USE_FACE, get_fullbody_landmarks)
from modules.transforms import (
    SelectLandmarksAndFeature, ReplaceNan, FilterZero, PartsBasedNormalization, ToTensor)
from modules.dataset import (
    make_wlasl_dataloader)


# --- 2. Data Augmentation ---

class RandomMove(object):
    def __init__(self, shift_range=0.1): self.shift_range = shift_range
    def __call__(self, data):
        if "feature" in data:
            feat = data["feature"]
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
    # batch[i]["feature"]: [C, T, J]
    max_len = max([b["feature"].shape[1] for b in batch])
    
    features, tokens, masks = [], [], []
    for b in batch:
        feat = b["feature"]
        C, T, J = feat.shape
        pad_len = max_len - T
        if not isinstance(feat, torch.Tensor): feat = torch.from_numpy(feat)
        padded_feat = F.pad(feat, (0, 0, 0, pad_len)) # Pad T dim (2nd last)
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


# --- 4. Model Components (Updated) ---

class GPoolRecognitionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.head = nn.Linear(in_channels, out_channels)
        self._init_weight()
    def _init_weight(self):
        # Xavier Initialization
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

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
        
        # IMPROVEMENT 1: Pre-Layer Normalization (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = GPoolRecognitionHead(d_model, out_channels)
        self._init_weights()

    def _init_weights(self):
        # IMPROVEMENT 3: Xavier Initialization for Linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature, feature_pad_mask=None):
        feature = feature.permute([0, 2, 3, 1])
        N, T, J, C = feature.shape
        feature = feature.reshape([N, T, -1]) 

        feature = self.pos_encoder(self.activation(self.embedding(feature)))
        
        src_key_padding_mask = (feature_pad_mask == 0) if feature_pad_mask is not None else None
        feature = self.transformer_encoder(feature, src_key_padding_mask=src_key_padding_mask)
        
        return self.head(feature.permute(0, 2, 1), feature_pad_mask)


# --- 5. Training Utils (Updated Scheduler) ---

# IMPROVEMENT 2: Noam Scheduler
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

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
            loss = loss_fn(pred, tokens.view(-1))
            total_loss += loss.item()
            
            # Correct Accuracy Calculation
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

if __name__ == "__main__":
    dictionary_path = os.path.join(data_dir, "wlasl_100.json")
    with open(dictionary_path, "r") as f:
        word_dict = json.load(f)
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

    # 2. Data Loading (K-Fold)
    print("Loading data...")
    train_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "train"), os.path.join(data_dir, "labels", "train"), dictionary_path, batch_size=1, shuffle=True, transforms=train_transforms).dataset
    val_ds = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "val"), os.path.join(data_dir, "labels", "val"), dictionary_path, batch_size=1, shuffle=False, transforms=train_transforms).dataset
    test_loader = make_wlasl_dataloader(os.path.join(data_dir, "tracks", "test"), os.path.join(data_dir, "labels", "test"), dictionary_path, batch_size=1, shuffle=False, transforms=test_transforms)
    
    full_dataset = ConcatDataset([train_ds, val_ds])
    
    # K-Fold Config
    K_FOLDS = 5
    EPOCHS = 200
    BATCH_SIZE = 32
    d_model = 128 # Golden config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}. Starting {K_FOLDS}-Fold CV ({EPOCHS} Epochs).")
    
    now = datetime.now()
    save_dir = os.path.join(root_dir, "result_temp", f"SignJoey_KFold_{now.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    indices = np.random.permutation(len(full_dataset))
    folds = np.array_split(indices, K_FOLDS)
    fold_results = {}

    # --- CHANGED: Run only Fold 1 ---
    for k in range(1):
        print(f"\n--- Single Split Training (Fold {k+1}) ---")
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(K_FOLDS) if i != k])
        
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)
        
        model = TransformerIsoSLR(141*3, out_channels, d_model=d_model, num_layers=2, dropout=0.1).to(device) # Dropout back to 0.1 for Pre-Norm
        
        # Noam Scheduler Setup
        # warm_steps should be ~10% of total steps. Steps per epoch = len(train_idx) / BATCH_SIZE approx 45.
        # Total steps = 200 * 45 = 9000. Warmup = 900.
        warmup_steps = 1000 
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # LR set by scheduler
        scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, factor=1.0)
        loss_fn = CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1) 
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            t_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
            v_loss, v_acc = val_loop(val_loader, model, loss_fn, device)
            # Scheduler steps are usually per-batch in Noam, but per-epoch is fine for small data
            scheduler.step() 
            
            history['train_loss'].append(t_loss); history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{k+1}_best.pth"))
            
            if (epoch+1) % 20 == 0: # Print less frequently
                print(f"Ep {epoch+1}: Train {t_loss:.4f}, Val {v_loss:.4f}, Acc {v_acc:.2f}%")

        # Test
        model.load_state_dict(torch.load(os.path.join(save_dir, f"fold{k+1}_best.pth"), weights_only=True))
        test_acc = test_loop(test_loader, model, device)
        print(f"Fold {k+1} Best Val: {best_val_loss:.4f}, Test Acc: {test_acc:.2f}%")
        fold_results[f"fold{k+1}"] = {'best_val': best_val_loss, 'test_acc': test_acc, 'history': history}

    # Summary
    avg_acc = np.mean([r['test_acc'] for r in fold_results.values()])
    print(f"\n=== Average Test Accuracy: {avg_acc:.2f}% ===")
    
    # Save Report
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(f"# SignJoey Inspired Results\n\nAvg Test Acc: **{avg_acc:.2f}%**\n")
        for k, v in fold_results.items(): f.write(f"- {k}: {v['test_acc']:.2f}%\n")
    
    # Plot
    plt.figure(figsize=(10, 5))
    for k, v in fold_results.items(): plt.plot(v['history']['val_acc'], label=k, alpha=0.5)
    plt.legend(); plt.title("K-Fold Accuracy"); plt.savefig(os.path.join(save_dir, "kfold_acc.png"))
