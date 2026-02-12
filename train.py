"""
SekoKuva-Mobile-Net — Training Pipeline
========================================
Copyright (c) 2026 BC Bertenex Oy

This script trains SekoKuva-Mobile-Net on OpenImages V7 (or any image dataset).
Designed to run on CSC LUMI supercomputer or any machine with a GPU.

Training enhancements (all enabled by default):
  - Mixed Precision (AMP): ~2× speedup on GPUs with Tensor Cores
  - CutMix + MixUp: +2-5% accuracy through advanced augmentation
  - Class-Balanced Sampling: handles uneven class distributions
  - Progressive Resolution: 112→160→224, faster convergence
  - SWA (Stochastic Weight Averaging): +1-2% accuracy at the end
  - Gradient Accumulation: simulate larger batch sizes

Usage:
    # Single GPU (all enhancements on by default)
    python train.py --data_dir /path/to/openimages --epochs 100

    # Disable specific enhancements
    python train.py --data_dir /path/to/data --no_amp --no_cutmix

    # LUMI (multiple GPUs)
    srun python train.py --data_dir /path/to/openimages --epochs 100
"""

import os
import argparse
import time
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import datasets, transforms

from sekokuva_mobile_net.model import sekokuva_mobilenet


# =============================================================================
# Data Transforms (Augmentation)
# =============================================================================

def get_training_transforms(input_size: int = 224):
    """
    Data augmentation for training.
    
    These transforms randomly modify each image slightly every time it's shown
    to the network. This prevents the network from memorizing specific images
    and forces it to learn general features.
    
    Think of it as showing a flashcard to a student, but each time:
    - You hold it at a slightly different angle (RandomRotation)
    - You cover a random part (RandomResizedCrop)  
    - You sometimes flip it (RandomHorizontalFlip)
    - You change the lighting slightly (ColorJitter)
    """
    return transforms.Compose([
        # Random crop and resize: forces the network to recognize objects
        # at different scales and positions. A leaf might be big or small,
        # centered or in a corner.
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        
        # Flip horizontally 50% of the time.
        # A healthy leaf is still healthy when mirrored.
        transforms.RandomHorizontalFlip(),
        
        # Slight rotation: objects in photos aren't always perfectly aligned
        transforms.RandomRotation(15),
        
        # Vary brightness, contrast, saturation slightly.
        # Photos taken in Uganda will have very different lighting than
        # the training data — this helps the model cope.
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        
        # Convert to tensor (numbers the network can process)
        transforms.ToTensor(),
        
        # Normalize to standard range.
        # These specific numbers (mean and std) center the data around 0,
        # which helps training converge faster.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_validation_transforms(input_size: int = 224):
    """
    Transforms for validation — NO randomness.
    We want consistent evaluation, so we just resize and normalize.
    """
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),  # Slightly larger
        transforms.CenterCrop(input_size),           # Then center crop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# CutMix + MixUp — Advanced Data Augmentation
# =============================================================================
#
# These techniques create "mixed" training examples:
#
# CutMix: Cut a rectangle from one image and paste it onto another.
#   The label becomes proportional to area: 70% cat + 30% dog.
#   Forces the network to look at ALL parts of the image, not just the center.
#
# MixUp: Blend two images together like a double exposure.
#   The label becomes proportional to blend: 60% cat + 40% dog.
#   Creates smoother decision boundaries between classes.
#
# Both reduce overfitting and typically add +2-5% validation accuracy.

def rand_bbox(size, lam):
    """Generate a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def apply_cutmix_or_mixup(images, labels, num_classes,
                           cutmix_prob=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
    """
    Randomly apply either CutMix or MixUp to a batch.
    
    Returns the (possibly modified) images and soft target labels.
    About half the time it does CutMix, half the time MixUp.
    """
    batch_size = images.size(0)
    targets_one_hot = F.one_hot(labels, num_classes).float()

    if random.random() < cutmix_prob:
        # ---- CutMix ----
        # Draw a random mixing ratio from Beta distribution
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        rand_index = torch.randperm(batch_size).to(images.device)
        # Cut a random rectangle and paste from shuffled images
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to actual pasted area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    else:
        # ---- MixUp ----
        # Blend two images: new = λ·imageA + (1-λ)·imageB
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        rand_index = torch.randperm(batch_size).to(images.device)
        images = lam * images + (1 - lam) * images[rand_index]

    # Mix labels the same way: new_label = λ·labelA + (1-λ)·labelB
    mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[rand_index]
    return images, mixed_targets


# =============================================================================
# Class-Balanced Sampling
# =============================================================================
#
# Real datasets are rarely balanced — you might have 1000 images of "cat"
# but only 50 of "pangolin". Without balancing, the network mostly learns
# about common classes and ignores rare ones.
#
# WeightedRandomSampler gives each image a sampling probability inversely
# proportional to its class frequency. Rare classes get sampled MORE often,
# common classes less. Every class gets roughly equal representation per epoch.

def make_class_balanced_sampler(dataset):
    """Create a WeightedRandomSampler that balances class frequencies."""
    targets = [s[1] for s in dataset.samples]
    class_counts = Counter(targets)
    total = len(targets)
    
    # Weight per class = total / (num_classes * count_for_this_class)
    class_weights = {cls: total / (len(class_counts) * count)
                     for cls, count in class_counts.items()}
    
    # Weight per sample = weight of its class
    sample_weights = [class_weights[t] for t in targets]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    scaler=None, num_classes=None, use_cutmix=True,
                    grad_accum_steps=1):
    """
    Train for one full pass through the dataset.
    
    Supports:
    - AMP (Automatic Mixed Precision) via scaler — ~2× speedup on Tensor Core GPUs
    - CutMix/MixUp augmentation via use_cutmix — better regularization
    - Gradient accumulation via grad_accum_steps — simulate larger batch sizes
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply CutMix or MixUp augmentation
        mixed_targets = None
        if use_cutmix and num_classes is not None:
            images, mixed_targets = apply_cutmix_or_mixup(
                images, labels, num_classes
            )
        
        # Forward pass with Automatic Mixed Precision (AMP)
        # AMP runs computations in float16 where safe, giving ~2× speedup
        # on GPUs with Tensor Cores (RTX 30/40 series, LUMI AMD MI250X).
        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            outputs = model(images)
            if mixed_targets is not None:
                # Soft cross-entropy for mixed CutMix/MixUp labels
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -torch.sum(mixed_targets * log_probs, dim=1).mean()
            else:
                loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps  # Scale for accumulation
        
        # Backward pass with gradient scaling (prevents underflow in fp16)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation: step optimizer every N mini-batches.
        # This simulates a larger batch size without using more VRAM.
        # E.g., batch_size=64 × grad_accum_steps=4 = effective batch 256.
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Track progress (use unscaled loss for logging)
        running_loss += loss.item() * grad_accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")
    
    # Flush remaining accumulated gradients
    if len(dataloader) % grad_accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SekoKuva-Mobile-Net")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset (ImageFolder format: data_dir/class_name/image.jpg)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Where to save trained models")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (adjust based on GPU memory)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Initial learning rate")
    parser.add_argument("--input_size", type=int, default=224,
                        help="Input image size (square)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Training enhancement flags (all ON by default — disable with --no_xxx)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--no_cutmix", action="store_true",
                        help="Disable CutMix/MixUp augmentation")
    parser.add_argument("--no_balanced_sampling", action="store_true",
                        help="Disable class-balanced sampling")
    parser.add_argument("--no_progressive", action="store_true",
                        help="Disable progressive resolution (112→160→224)")
    parser.add_argument("--no_swa", action="store_true",
                        help="Disable Stochastic Weight Averaging")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × this)")
    parser.add_argument("--swa_lr", type=float, default=0.01,
                        help="SWA learning rate")
    args = parser.parse_args()
    
    # Resolve feature flags
    use_amp = not args.no_amp
    use_cutmix = not args.no_cutmix
    use_balanced = not args.no_balanced_sampling
    use_progressive = not args.no_progressive
    use_swa = not args.no_swa
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # AMP only makes sense on CUDA (float16 Tensor Cores)
    if use_amp and device.type != "cuda":
        use_amp = False
        print("AMP disabled (no CUDA device)")
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Print active enhancements
    print(f"\nTraining enhancements:")
    print(f"  Mixed precision (AMP):  {'ON' if use_amp else 'OFF'}")
    print(f"  CutMix/MixUp:           {'ON' if use_cutmix else 'OFF'}")
    print(f"  Balanced sampling:      {'ON' if use_balanced else 'OFF'}")
    print(f"  Progressive resolution: {'ON' if use_progressive else 'OFF'}")
    print(f"  SWA:                    {'ON' if use_swa else 'OFF'}")
    print(f"  Gradient accumulation:  {args.grad_accum_steps}x")
    
    # =========================================================================
    # Progressive Resolution Schedule
    # =========================================================================
    # Start training at low resolution for speed, gradually increase.
    # Early epochs learn rough features (edges, textures) that don't need
    # full detail. Fine details come later at full resolution.
    if use_progressive:
        e1 = int(args.epochs * 0.30)  # First 30% at low res
        e2 = int(args.epochs * 0.60)  # 30-60% at medium res
        resolution_schedule = [
            (0,  e1, 112),
            (e1, e2, 160),
            (e2, args.epochs, args.input_size),
        ]
        print(f"  Resolution: 112px (ep 0-{e1}) → 160px (ep {e1}-{e2}) "
              f"→ {args.input_size}px (ep {e2}-{args.epochs})")
    else:
        resolution_schedule = [(0, args.epochs, args.input_size)]
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    initial_res = resolution_schedule[0][2]
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    print(f"\nLoading training data from: {train_dir}")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_training_transforms(initial_res)
    )
    
    print(f"Loading validation data from: {val_dir}")
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_validation_transforms(args.input_size)  # Always validate at full res
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Found {len(train_dataset)} training images in {num_classes} classes")
    print(f"Found {len(val_dataset)} validation images")
    
    # Save class names for later use in the app
    class_names_path = os.path.join(args.output_dir, "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(train_dataset.classes, f, indent=2)
    print(f"Saved class names to {class_names_path}")
    
    # =========================================================================
    # Class-Balanced Sampling
    # =========================================================================
    sampler = make_class_balanced_sampler(train_dataset) if use_balanced else None
    if use_balanced:
        targets = [s[1] for s in train_dataset.samples]
        counts = Counter(targets)
        min_cls = min(counts.values())
        max_cls = max(counts.values())
        print(f"  Class balance: min {min_cls} → max {max_cls} images "
              f"(ratio {max_cls/min_cls:.1f}x, now balanced)")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),   # Don't shuffle when using sampler
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,             # faster GPU transfer
        drop_last=True               # drop incomplete last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # =========================================================================
    # Create Model
    # =========================================================================
    model = sekokuva_mobilenet(num_classes=num_classes, input_size=args.input_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SekoKuva-Mobile-Net: {total_params:,} parameters")
    
    # Loss function: Cross-Entropy with label smoothing
    # Label smoothing (0.1) prevents the network from being overconfident.
    # Instead of learning "this is 100% a banana", it learns
    # "this is 90% a banana, 10% could be something else."
    # This makes the features more robust for transfer learning.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer: SGD with momentum
    # SGD (Stochastic Gradient Descent) is the workhorse optimizer.
    # Momentum (0.9) adds "inertia" — like a ball rolling downhill,
    # it doesn't change direction on every bump.
    # Weight decay (1e-4) gently penalizes large weights, preventing overfitting.
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Learning rate schedule: Cosine Annealing
    # Starts at max learning rate, gradually decreases to near-zero
    # following a cosine curve. Like starting with big brush strokes
    # and finishing with fine detail.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # =========================================================================
    # SWA Setup — Stochastic Weight Averaging
    # =========================================================================
    # In the final 25% of training, average the model weights across
    # multiple checkpoints. This finds a flatter minimum in the loss
    # landscape, which generalizes better. Like taking the average answer
    # from several good students instead of just one.
    swa_model = None
    swa_scheduler = None
    swa_start = int(args.epochs * 0.75)
    
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        print(f"  SWA starts at epoch {swa_start}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
    
    # =====================================================================
    # TRAINING LOOP
    # =====================================================================
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    current_res = initial_res
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # ---- Progressive Resolution: check if we need to change ----
        for res_start, res_end, res_size in resolution_schedule:
            if res_start <= epoch < res_end and res_size != current_res:
                current_res = res_size
                print(f"\n>>> Resolution change → {current_res}x{current_res}")
                train_dataset.transform = get_training_transforms(current_res)
                # Recreate DataLoader so workers pick up the new transform
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
                break
        
        # ---- Train ----
        in_swa = use_swa and epoch >= swa_start
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler,
            num_classes=num_classes,
            use_cutmix=use_cutmix,
            grad_accum_steps=args.grad_accum_steps,
        )
        
        # ---- Validate ----
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # ---- Learning Rate / SWA step ----
        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            current_lr = args.swa_lr
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        
        elapsed = time.time() - start_time
        
        # Print summary
        swa_tag = " [SWA]" if in_swa else ""
        res_tag = f" [{current_res}px]" if use_progressive else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.0f}s{res_tag}{swa_tag}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "num_classes": num_classes,
            "input_size": args.input_size,
        }
        
        # Always save latest
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
            print(f"  ★ New best validation accuracy: {best_val_acc:.1f}%")
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))
    
    # =====================================================================
    # SWA: Final batch norm update and evaluation
    # =====================================================================
    if use_swa and swa_model is not None:
        print("\nUpdating SWA batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_val_loss, swa_val_acc = validate(swa_model, val_loader, criterion, device)
        print(f"SWA model: Val Loss {swa_val_loss:.4f} | Val Acc {swa_val_acc:.1f}%")
        
        # Save SWA model (uses same state_dict format as regular model)
        swa_checkpoint = {
            "epoch": args.epochs,
            "model_state_dict": swa_model.module.state_dict(),
            "num_classes": num_classes,
            "input_size": args.input_size,
            "val_acc": swa_val_acc,
            "val_loss": swa_val_loss,
            "is_swa": True,
        }
        torch.save(swa_checkpoint, os.path.join(args.output_dir, "swa.pt"))
        
        if swa_val_acc > best_val_acc:
            best_val_acc = swa_val_acc
            torch.save(swa_checkpoint, os.path.join(args.output_dir, "best.pt"))
            print(f"  ★ SWA model is the new best: {best_val_acc:.1f}%")
    
    print("=" * 60)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Next step: run export_tflite.py to convert for mobile deployment")


if __name__ == "__main__":
    main()
