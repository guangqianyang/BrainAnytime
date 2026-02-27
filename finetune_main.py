#!/usr/bin/env python
"""
MultiMAE3D Finetuning for Downstream Tasks

Full finetuning: train the entire model end-to-end.

Tasks: CN vs AD, CN vs MCI, MMSE, AGE
Each task runs with multiple seeds and reports mean +/- std metrics.

Usage:
    # Finetune on all 4 tasks (3 seeds each)
    python finetune_main.py --pretrained ./pretrain_checkpoints/multimae/best_model.pth

    # Specific task only
    python finetune_main.py --pretrained ./pretrain_checkpoints/multimae/best_model.pth --tasks "CN vs AD"
"""

import os
import sys
import gc
import random
import warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from models.multimae3d import create_multimae3d, MultiMAE3D
from downstream_dataloader import create_downstream_dataloader


# =========================================================================
# Utilities
# =========================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise ValueError(f'Boolean value expected, got: {v}')


def setup_logger(log_dir, name, filename):
    """Simple logger setup."""
    import logging
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, filename))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# =========================================================================
# Model: MultiMAE3D encoder + downstream task head
# =========================================================================

class MultiMAE3DForDownstream(nn.Module):
    """
    MultiMAE3D encoder + task head for downstream classification/regression.

    Uses encoder.encode() to get all-patch features, pools to a single vector
    (CLS token or mean pooling), then applies a linear head.
    """

    def __init__(
        self,
        encoder: MultiMAE3D,
        embed_dim: int = 768,
        num_outputs: int = 1,
        pool: str = 'cls',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.num_patches_per_modality = encoder.num_patches
        self.num_global_tokens = encoder.num_global_tokens

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_outputs),
        )

    def forward(self, images: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images:   [B, 4, D, H, W]
            observed: [B, 4] float mask (1.0=present, 0.0=missing)
        Returns:
            logits: [B, num_outputs]
        """
        # encode() returns [B, 1 + 4*num_patches, embed_dim]
        encoder_out = self.encoder.encode(images, observed)

        if self.pool == 'cls':
            features = encoder_out[:, 0]  # CLS token -> [B, D]
        elif self.pool == 'mean':
            # Mean pool over modality tokens with masking for missing modalities
            tokens = encoder_out[:, self.num_global_tokens:]  # [B, 4*N_p, D]
            B, _, D = tokens.shape
            N = self.num_patches_per_modality
            # Build per-token mask: repeat each modality's observed flag N times
            mask = observed.unsqueeze(-1).expand(-1, -1, N)  # [B, 4, N]
            mask = mask.reshape(B, 4 * N).unsqueeze(-1)  # [B, 4*N, 1]
            features = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")

        features = self.norm(features)
        logits = self.head(features)
        return logits


# =========================================================================
# Pretrained weight loading
# =========================================================================

def load_pretrained_weights(model: MultiMAE3D, checkpoint_path: str, device='cpu'):
    """
    Load pretrained encoder weights into a MultiMAE3D model.

    Supports checkpoint formats:
      - 'encoder_state_dict': encoder-only (from periodic/best saves)
      - 'full_model_state_dict': full model (from best_model.pth)
      - 'model_state_dict': full model (from latest.pth)
      - raw state_dict (no wrapper key)
    """
    print(f"Loading pretrained weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Pick the best available state dict
    if 'encoder_state_dict' in ckpt:
        state_dict = ckpt['encoder_state_dict']
        source = 'encoder_state_dict'
    elif 'full_model_state_dict' in ckpt:
        state_dict = ckpt['full_model_state_dict']
        source = 'full_model_state_dict'
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        source = 'model_state_dict'
    else:
        state_dict = ckpt
        source = 'raw'

    # If loading from full model, filter to encoder keys only
    encoder_prefixes = ('encoder.', 'input_adapters.', 'pos_embed', 'global_tokens')
    if source in ('full_model_state_dict', 'model_state_dict', 'raw'):
        state_dict = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in encoder_prefixes)
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Output adapters (decoders) are expected to be missing — we don't use them
    truly_missing = [k for k in missing if not k.startswith('output_adapters.')]

    epoch_info = ckpt.get('epoch', '?')
    loss_info = ckpt.get('loss', '?')
    if isinstance(loss_info, float):
        loss_info = f"{loss_info:.4f}"
    print(f"  Source: {source}, Epoch: {epoch_info}, Pretrain Loss: {loss_info}")
    print(f"  Loaded {len(state_dict)} parameter tensors")
    if truly_missing:
        print(f"  WARNING: {len(truly_missing)} encoder keys missing: {truly_missing[:5]}...")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (ignored)")

    del ckpt
    return model


# =========================================================================
# Metrics
# =========================================================================

def calc_regression_metrics(preds, labels):
    """Calculate MAE, RMSE, Pearson correlation."""
    preds, labels = np.array(preds), np.array(labels)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    if len(preds) > 1 and np.std(preds) > 0 and np.std(labels) > 0:
        r, _ = pearsonr(preds, labels)
    else:
        r = 0.0
    return {'mae': mae, 'rmse': rmse, 'pearson': r}


def calc_classification_metrics(preds, labels, probs):
    """Calculate ACC, AUC, Sensitivity, Specificity, F1."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    preds, labels = np.array(preds), np.array(labels)
    probs = np.array(probs)

    acc = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0

    f1 = f1_score(labels, preds, average='binary')

    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity, specificity = 0.0, 0.0

    return {
        'acc': acc, 'auc': auc,
        'sensitivity': sensitivity, 'specificity': specificity,
        'f1': f1,
    }


def calc_metrics_by_combo(preds, labels, probs, combos, task_type):
    """Calculate metrics grouped by modality combination string."""
    is_cls = task_type in ('CN vs AD', 'CN vs MCI')
    grouped = defaultdict(lambda: {'preds': [], 'labels': [], 'probs': []})
    for i, combo in enumerate(combos):
        grouped[combo]['preds'].append(preds[i])
        grouped[combo]['labels'].append(labels[i])
        grouped[combo]['probs'].append(probs[i])

    results = {}
    for combo, data in grouped.items():
        p = np.array(data['preds'])
        l = np.array(data['labels'])
        pr = np.array(data['probs'])
        if is_cls:
            try:
                m = calc_classification_metrics(p, l, pr)
            except Exception:
                m = {'acc': 0, 'auc': 0, 'sensitivity': 0, 'specificity': 0, 'f1': 0}
        else:
            m = calc_regression_metrics(p, l)
        m['n_samples'] = len(p)
        results[combo] = m
    return results


# =========================================================================
# Training & Evaluation
# =========================================================================

MODALITY_NAMES = ['T1', 'T2', 'Flair', 'PET']


def run_epoch(loader, model, criterion, device, task_type,
              is_training=False, optimizer=None):
    """Run one epoch of training or evaluation."""
    all_preds, all_labels, all_probs = [], [], []
    modality_combos = []
    total_loss, n_batches = 0.0, 0
    is_cls = task_type in ('CN vs AD', 'CN vs MCI')

    model.train() if is_training else model.eval()

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, leave=False,
                          desc='Train' if is_training else 'Eval'):
            images = batch['images'].to(device, non_blocking=True)
            observed = batch['observed'].to(device, non_blocking=True)
            labels = batch['labels'][:, 0].to(device, non_blocking=True)

            logits = model(images, observed)       # [B, 1]
            logits_flat = logits.squeeze(-1)       # [B]
            loss = criterion(logits_flat, labels.float())

            total_loss += loss.item()
            n_batches += 1

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

            # Collect predictions
            if is_cls:
                prob_pos = torch.sigmoid(logits_flat).detach().cpu().numpy()
                pred = (prob_pos > 0.5).astype(int)
                probs_2d = np.stack([1 - prob_pos, prob_pos], axis=1)
                all_preds.extend(pred)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs_2d)
            else:
                pred_vals = logits_flat.detach().cpu().numpy()
                all_preds.extend(pred_vals)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(pred_vals.reshape(-1, 1))

            # Track modality combos (eval only)
            if not is_training:
                B = images.shape[0]
                for i in range(B):
                    present = [
                        MODALITY_NAMES[j]
                        for j in range(4)
                        if observed[i, j] > 0.5
                    ]
                    combo_str = ('+'.join(sorted(present))
                                 if present else 'None')
                    modality_combos.append(combo_str)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, all_preds, all_labels, all_probs, modality_combos


# =========================================================================
# Single task + seed pipeline
# =========================================================================

def train_and_evaluate(args, task_type, seed, device):
    """
    Full train/val/test pipeline for one (task, seed) combination.
    Returns a dict of test metrics.
    """
    seed_everything(seed)
    torch.cuda.empty_cache()

    is_cls = task_type in ('CN vs AD', 'CN vs MCI')

    # ---- Data loaders ----
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        cache_data=False,
        image_size=tuple(args.image_size),
        base_dir=args.base_dir,
        modalities=args.modalities,
        intersection=args.intersection,
    )

    print(f"\nLoading data for task={task_type}, seed={seed}, mode=finetune")
    train_loader = create_downstream_dataloader(
        excel_path=args.train_excel, labels=[task_type],
        augmentation=True, shuffle=True,
        phase='train', modality_dropout=True, expand_val_combinations=False,
        **loader_kwargs,
    )
    val_loader = create_downstream_dataloader(
        excel_path=args.val_excel, labels=[task_type],
        augmentation=False, shuffle=False,
        phase='val', modality_dropout=False, expand_val_combinations=True,
        **loader_kwargs,
    )
    test_loader = create_downstream_dataloader(
        excel_path=args.test_excel, labels=[task_type],
        augmentation=False, shuffle=False,
        phase='test', modality_dropout=False, expand_val_combinations=False,
        exclusive_modalities=False,
        **loader_kwargs,
    )
    print(f"  Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # ---- Model ----
    encoder = create_multimae3d(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
    )

    # Load pretrained weights
    if args.pretrained and os.path.isfile(args.pretrained):
        load_pretrained_weights(encoder, args.pretrained, device='cpu')
    else:
        print("  No pretrained weights loaded (training from scratch)")

    model = MultiMAE3DForDownstream(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_outputs=1,
        pool=args.pool,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # ---- Freeze encoder if requested ----
    freeze_epochs = getattr(args, 'freeze_epochs', 0)
    if freeze_epochs > 0:
        # Freeze all pretrained encoder parameters
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"  Model: {total_params:,} total, {trainable_params:,} trainable "
              f"(encoder frozen: {encoder_params:,} params for first {freeze_epochs} epochs)")
    else:
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"  Model: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Helper: build optimizer + scheduler ----
    warmup_ep = args.warmup_epochs
    total_ep = args.epochs

    def build_optimizer_and_scheduler(model, lr, remaining_epochs, warmup):
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr,
                                      weight_decay=args.weight_decay)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(remaining_epochs - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.lr, total_ep, warmup_ep)

    # Criterion
    criterion = (nn.BCEWithLogitsLoss() if is_cls
                 else nn.MSELoss()).to(device)

    # ---- Training loop ----
    best_metric = 0.0 if is_cls else float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(total_ep):
        # Unfreeze encoder after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"\n  >>> Epoch {epoch}: Unfreezing pretrained encoder <<<")
            for param in model.encoder.parameters():
                param.requires_grad = True
            unfrozen_trainable = sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)
            print(f"  Trainable params: {unfrozen_trainable:,} (all parameters)")
            # Rebuild optimizer & scheduler for joint training phase
            remaining = total_ep - epoch
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, args.lr, remaining, warmup_ep)
            print(f"  New optimizer created: lr={args.lr}, "
                  f"remaining_epochs={remaining}, warmup={warmup_ep}")

        # Train
        train_loss, tr_preds, tr_labels, tr_probs, _ = run_epoch(
            train_loader, model, criterion, device, task_type,
            is_training=True, optimizer=optimizer,
        )
        # Validate
        val_loss, val_preds, val_labels, val_probs, _ = run_epoch(
            val_loader, model, criterion, device, task_type,
            is_training=False,
        )

        scheduler.step()
        torch.cuda.empty_cache()

        # Compute metrics
        if is_cls:
            tr_m = calc_classification_metrics(tr_preds, tr_labels, tr_probs)
            val_m = calc_classification_metrics(val_preds, val_labels, val_probs)
            current = val_m['acc']
            improved = current > best_metric
        else:
            tr_m = calc_regression_metrics(tr_preds, tr_labels)
            val_m = calc_regression_metrics(val_preds, val_labels)
            current = val_m['mae']
            improved = current < best_metric

        if improved:
            best_metric = current
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0

            # Save best checkpoint
            mode_suffix = 'freeze_then_finetune' if freeze_epochs > 0 else 'finetune'
            save_dir = os.path.join(_BASE_DIR, 'saves', f'multimae_{mode_suffix}')
            os.makedirs(save_dir, exist_ok=True)
            task_str = task_type.replace(' ', '_')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'task': task_type,
                'seed': seed,
                'best_metric': best_metric,
                'freeze_epochs': freeze_epochs,
            }, os.path.join(save_dir, f'{task_str}_seed_{seed}_best.pth'))
        else:
            patience_counter += 1

        # Print progress periodically or on improvement
        if (epoch + 1) % 5 == 0 or improved:
            if is_cls:
                print(
                    f"  Epoch {epoch+1:3d}/{total_ep} | "
                    f"TrLoss: {train_loss:.4f}, TrAcc: {tr_m['acc']*100:.1f}% | "
                    f"ValAcc: {val_m['acc']*100:.1f}%, "
                    f"ValAUC: {val_m['auc']*100:.1f}%"
                    f"{'  ***' if improved else ''}"
                )
            else:
                print(
                    f"  Epoch {epoch+1:3d}/{total_ep} | "
                    f"TrLoss: {train_loss:.4f}, TrMAE: {tr_m['mae']:.4f} | "
                    f"ValMAE: {val_m['mae']:.4f}, "
                    f"ValPearson: {val_m['pearson']:.4f}"
                    f"{'  ***' if improved else ''}"
                )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # ---- Test evaluation ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n  Evaluating on test set...")
    test_loss, test_preds, test_labels, test_probs, test_combos = run_epoch(
        test_loader, model, criterion, device, task_type,
        is_training=False,
    )

    # Overall test metrics
    if is_cls:
        test_m = calc_classification_metrics(
            test_preds, test_labels, test_probs)
        print(
            f"  Test: Acc={test_m['acc']*100:.2f}%, "
            f"AUC={test_m['auc']*100:.2f}%, "
            f"Sen={test_m['sensitivity']*100:.2f}%, "
            f"Spe={test_m['specificity']*100:.2f}%, "
            f"F1={test_m['f1']*100:.2f}%"
        )
    else:
        test_m = calc_regression_metrics(test_preds, test_labels)
        print(
            f"  Test: MAE={test_m['mae']:.4f}, "
            f"RMSE={test_m['rmse']:.4f}, "
            f"Pearson={test_m['pearson']:.4f}"
        )

    # Per-modality-combination breakdown
    combo_results = calc_metrics_by_combo(
        test_preds, test_labels, test_probs, test_combos, task_type)
    if combo_results:
        print(f"\n  Per-modality-combination results:")
        for combo in sorted(combo_results.keys()):
            r = combo_results[combo]
            n = r['n_samples']
            if is_cls:
                print(f"    {combo:25s} (n={n:3d}) | "
                      f"Acc={r['acc']*100:.1f}%, AUC={r['auc']*100:.1f}%")
            else:
                print(f"    {combo:25s} (n={n:3d}) | "
                      f"MAE={r['mae']:.4f}, Pearson={r['pearson']:.4f}")

        # Save per-combo results to Excel
        freeze_epochs = getattr(args, 'freeze_epochs', 0)
        mode_tag = (f"freeze{freeze_epochs}_finetune"
                    if freeze_epochs > 0 else "finetune")
        _save_combo_results(combo_results, task_type, seed, mode_tag, is_cls)

    # Cleanup
    del model, encoder, optimizer, train_loader, val_loader, test_loader
    del best_model_state
    torch.cuda.empty_cache()
    gc.collect()

    return test_m


def _save_combo_results(combo_results, task_type, seed, mode, is_cls):
    """Save per-modality-combination results to Excel."""
    results_dir = os.path.join(_BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for combo in sorted(combo_results.keys()):
        r = combo_results[combo]
        row = {'Modality': combo, 'N': r['n_samples']}
        if is_cls:
            row.update({
                'Acc': r['acc'] * 100,
                'AUC': r['auc'] * 100,
                'Sensitivity': r['sensitivity'] * 100,
                'Specificity': r['specificity'] * 100,
                'F1': r['f1'] * 100,
            })
        else:
            row.update({
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'Pearson': r['pearson'],
            })
        rows.append(row)

    task_str = task_type.replace(' ', '_')
    path = os.path.join(
        results_dir,
        f'multimae_{mode}_{task_str}_seed_{seed}_by_combo.xlsx',
    )
    pd.DataFrame(rows).to_excel(path, index=False)
    print(f"  Saved: {path}")


# =========================================================================
# Argument parsing
# =========================================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='MultiMAE3D Finetuning for Downstream Tasks')

    # Mode
    p.add_argument('--mode', type=str, default='finetune',
                   choices=['finetune', 'freeze_then_finetune'],
                   help='finetune: train all parameters end-to-end; '
                        'freeze_then_finetune: freeze encoder for N epochs then unfreeze')
    p.add_argument(
        '--pretrained', type=str,
        default=os.path.join(
            _BASE_DIR, 'pretrain_checkpoints', 'multimae', 'best_model.pth'),
        help='Path to pretrained MultiMAE checkpoint')

    # Tasks & seeds
    p.add_argument('--tasks', type=str, nargs='+',
                   default=['CN vs AD', 'CN vs MCI', 'MMSE', 'AGE'],
                   help='Tasks to evaluate')
    p.add_argument('--n_seeds', type=int, default=3,
                   help='Number of random seeds per task')

    # Data
    p.add_argument('--train_excel', type=str,
                   default='./data/Downstream/'
                           'ADNI_Division/modality_data_train.xlsx')
    p.add_argument('--val_excel', type=str,
                   default='./data/Downstream/'
                           'ADNI_Division/modality_data_val.xlsx')
    p.add_argument('--test_excel', type=str,
                   default='./data/Downstream/'
                           'ADNI_Division/modality_data_test.xlsx')
    p.add_argument('--base_dir', type=str,
                   default='./data/Downstream/ADNI/')
    p.add_argument('--modalities', type=str, nargs='+',
                   default=['T1', 'T2', 'Flair', 'PET'])
    p.add_argument('--intersection', type=str2bool, default=False)
    p.add_argument('--image_size', type=int, nargs=3,
                   default=[128, 128, 128])
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=8)

    # MultiMAE encoder architecture (must match pretrained checkpoint)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--embed_dim', type=int, default=768)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--decoder_embed_dim', type=int, default=384)
    p.add_argument('--decoder_depth', type=int, default=2)
    p.add_argument('--decoder_num_heads', type=int, default=12)

    # Downstream head
    p.add_argument('--pool', type=str, default='cls',
                   choices=['cls', 'mean'],
                   help='Feature pooling: cls token or mean pool')
    p.add_argument('--dropout', type=float, default=0.1)

    # Training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-5,
                   help='Learning rate')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=15,
                   help='Early stopping patience')
    p.add_argument('--freeze_epochs', type=int, default=0,
                   help='Number of epochs to freeze pretrained encoder '
                        '(0 = no freeze, full finetune from start)')

    # Device
    p.add_argument('--device', type=int, default=0)

    return p.parse_args()


# =========================================================================
# Main: loop over tasks x seeds
# =========================================================================

def main():
    args = parse_args()
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print(f"MultiMAE3D Downstream Evaluation")
    print(f"  Mode         : {args.mode}")
    print(f"  Tasks        : {args.tasks}")
    print(f"  Seeds        : {args.n_seeds}")
    print(f"  Pretrained   : {args.pretrained}")
    print(f"  Pool         : {args.pool}")
    print(f"  Device       : {device}")
    print(f"  LR           : {args.lr}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    if args.freeze_epochs > 0:
        print(f"  Freeze epochs: {args.freeze_epochs} (encoder frozen, then joint training)")
    print("=" * 80)

    # Logger
    log_dir = os.path.join(_BASE_DIR, 'logs')
    logger = setup_logger(log_dir, 'multimae_ft',
                          f'multimae_{args.mode}.txt')

    all_results = {}

    for task_type in args.tasks:
        print(f"\n{'='*80}")
        print(f"TASK: {task_type}")
        print(f"{'='*80}")

        is_cls = task_type in ('CN vs AD', 'CN vs MCI')
        seed_results = []

        for seed in range(args.n_seeds):
            print(f"\n--- Seed {seed} ---")
            metrics = train_and_evaluate(args, task_type, seed, device)
            seed_results.append(metrics)

        all_results[task_type] = seed_results

        # Per-task summary
        print(f"\n{task_type} Summary ({args.n_seeds} seeds):")
        summary_str = f"[{args.mode}] {task_type}: "
        if is_cls:
            for key in ['acc', 'auc', 'sensitivity', 'specificity', 'f1']:
                vals = [r[key] * 100 for r in seed_results]
                msg = f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}%"
                print(f"  {key:>12s}: {msg}")
                summary_str += f"{key}={msg}, "
        else:
            for key in ['mae', 'rmse', 'pearson']:
                vals = [r[key] for r in seed_results]
                msg = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
                print(f"  {key:>12s}: {msg}")
                summary_str += f"{key}={msg}, "
        logger.info(summary_str)

    # ---- Final summary table ----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary_rows = []

    for task_type in args.tasks:
        results = all_results[task_type]
        is_cls = task_type in ('CN vs AD', 'CN vs MCI')
        row = {'Task': task_type, 'Mode': args.mode}

        if is_cls:
            for key in ['acc', 'auc', 'sensitivity', 'specificity', 'f1']:
                vals = [r[key] * 100 for r in results]
                row[f'{key}_mean'] = np.mean(vals)
                row[f'{key}_std'] = np.std(vals)
                row[key] = f"{np.mean(vals):.2f}+/-{np.std(vals):.2f}"
            # Per-seed values
            for i, r in enumerate(results):
                row[f'seed_{i}_acc'] = r['acc'] * 100
                row[f'seed_{i}_auc'] = r['auc'] * 100

            vals_acc = [r['acc'] * 100 for r in results]
            vals_auc = [r['auc'] * 100 for r in results]
            print(f"  {task_type:12s} | "
                  f"Acc: {np.mean(vals_acc):.2f}+/-{np.std(vals_acc):.2f}% | "
                  f"AUC: {np.mean(vals_auc):.2f}+/-{np.std(vals_auc):.2f}%")
        else:
            for key in ['mae', 'rmse', 'pearson']:
                vals = [r[key] for r in results]
                row[f'{key}_mean'] = np.mean(vals)
                row[f'{key}_std'] = np.std(vals)
                row[key] = f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"
            for i, r in enumerate(results):
                row[f'seed_{i}_mae'] = r['mae']
                row[f'seed_{i}_pearson'] = r['pearson']

            vals_mae = [r['mae'] for r in results]
            vals_r = [r['pearson'] for r in results]
            print(f"  {task_type:12s} | "
                  f"MAE: {np.mean(vals_mae):.4f}+/-{np.std(vals_mae):.4f} | "
                  f"Pearson: {np.mean(vals_r):.4f}+/-{np.std(vals_r):.4f}")

        summary_rows.append(row)

    # Save summary Excel
    results_dir = os.path.join(_BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    freeze_epochs = getattr(args, 'freeze_epochs', 0)
    summary_tag = (f"freeze{freeze_epochs}_finetune"
                   if freeze_epochs > 0 else "finetune")
    summary_path = os.path.join(
        results_dir, f'multimae_{summary_tag}_summary.xlsx')
    pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
