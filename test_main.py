#!/usr/bin/env python
"""
MultiMAE3D Test-Only Evaluation

Load saved finetuned checkpoints and evaluate on the test set.
Reuses model/data/metric utilities from finetune_main.py.

Usage:
    # Test all tasks for finetune mode
    python test_main.py --mode finetune

    # Test a specific task
    python test_main.py --mode finetune --tasks "CN vs AD"

    # Test with custom checkpoint directory
    python test_main.py --mode finetune --checkpoint_dir ./saves/multimae_finetune/
"""

import os
import sys
import gc
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from models.multimae3d import create_multimae3d, MultiMAE3D
from downstream_dataloader import create_downstream_dataloader
from finetune_main import (
    seed_everything,
    str2bool,
    MultiMAE3DForDownstream,
    run_epoch,
    calc_classification_metrics,
    calc_regression_metrics,
    calc_metrics_by_combo,
    _save_combo_results,
)


# =========================================================================
# Test-only evaluation for a single (task, seed)
# =========================================================================

def test_evaluate(args, task_type, seed, device, checkpoint_path):
    """Load a saved checkpoint and evaluate on test set."""
    seed_everything(seed)
    torch.cuda.empty_cache()

    is_cls = task_type in ('CN vs AD', 'CN vs MCI')

    # ---- Test data loader ----
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

    print(f"\nLoading test data for task={task_type}, seed={seed}")
    test_loader = create_downstream_dataloader(
        excel_path=args.test_excel, labels=[task_type],
        augmentation=False, shuffle=False,
        phase='test', modality_dropout=False, expand_val_combinations=False,
        exclusive_modalities=False,
        **loader_kwargs,
    )
    print(f"  Test: {len(test_loader.dataset)} samples")

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

    model = MultiMAE3DForDownstream(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_outputs=1,
        pool=args.pool,
        dropout=args.dropout,
    ).to(device)

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded (epoch={ckpt.get('epoch', '?')}, "
          f"best_metric={ckpt.get('best_metric', '?')})")

    # Criterion
    criterion = (nn.BCEWithLogitsLoss() if is_cls
                 else nn.MSELoss()).to(device)

    # ---- Test evaluation ----
    print("  Evaluating on test set...")
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

        # Save per-combo results
        mode_tag = args.mode
        _save_combo_results(combo_results, task_type, seed,
                            f"test_{mode_tag}", is_cls)

    # Cleanup
    del model, encoder, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return test_m


# =========================================================================
# Argument parsing
# =========================================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description='MultiMAE3D Test-Only Evaluation')

    # Mode & checkpoints
    p.add_argument('--mode', type=str, default='finetune',
                   choices=['finetune', 'freeze_then_finetune'],
                   help='Which training mode checkpoints to load')
    p.add_argument('--checkpoint_dir', type=str, default=None,
                   help='Directory containing saved checkpoints. '
                        'Defaults to saves/multimae_{mode}/')

    # Tasks & seeds
    p.add_argument('--tasks', type=str, nargs='+',
                   default=['CN vs AD', 'CN vs MCI', 'MMSE', 'AGE'],
                   help='Tasks to evaluate')
    p.add_argument('--n_seeds', type=int, default=3,
                   help='Number of random seeds per task')

    # Data
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

    # MultiMAE encoder architecture (must match checkpoint)
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
                   choices=['cls', 'mean'])
    p.add_argument('--dropout', type=float, default=0.1)

    # Device
    p.add_argument('--device', type=int, default=0)

    return p.parse_args()


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Resolve checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            _BASE_DIR, 'saves', f'multimae_{args.mode}')

    print("=" * 80)
    print(f"MultiMAE3D Test-Only Evaluation")
    print(f"  Mode           : {args.mode}")
    print(f"  Tasks          : {args.tasks}")
    print(f"  Seeds          : {args.n_seeds}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print(f"  Test data      : {args.test_excel}")
    print(f"  Pool           : {args.pool}")
    print(f"  Device         : {device}")
    print("=" * 80)

    all_results = {}

    for task_type in args.tasks:
        print(f"\n{'='*80}")
        print(f"TASK: {task_type}")
        print(f"{'='*80}")

        is_cls = task_type in ('CN vs AD', 'CN vs MCI')
        seed_results = []
        task_str = task_type.replace(' ', '_')

        for seed in range(args.n_seeds):
            ckpt_name = f'{task_str}_seed_{seed}_best.pth'
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)

            if not os.path.isfile(ckpt_path):
                print(f"\n--- Seed {seed} --- SKIPPED (checkpoint not found: {ckpt_name})")
                continue

            print(f"\n--- Seed {seed} ---")
            metrics = test_evaluate(args, task_type, seed, device, ckpt_path)
            seed_results.append(metrics)

        if not seed_results:
            print(f"  No checkpoints found for {task_type}, skipping.")
            continue

        all_results[task_type] = seed_results

        # Per-task summary
        n = len(seed_results)
        print(f"\n{task_type} Summary ({n} seeds):")
        if is_cls:
            for key in ['acc', 'auc', 'sensitivity', 'specificity', 'f1']:
                vals = [r[key] * 100 for r in seed_results]
                print(f"  {key:>12s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}%")
        else:
            for key in ['mae', 'rmse', 'pearson']:
                vals = [r[key] for r in seed_results]
                print(f"  {key:>12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # ---- Final summary table ----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    summary_rows = []

    for task_type in args.tasks:
        if task_type not in all_results:
            continue
        results = all_results[task_type]
        is_cls = task_type in ('CN vs AD', 'CN vs MCI')
        row = {'Task': task_type, 'Mode': args.mode, 'N_seeds': len(results)}

        if is_cls:
            for key in ['acc', 'auc', 'sensitivity', 'specificity', 'f1']:
                vals = [r[key] * 100 for r in results]
                row[f'{key}_mean'] = np.mean(vals)
                row[f'{key}_std'] = np.std(vals)
                row[key] = f"{np.mean(vals):.2f}+/-{np.std(vals):.2f}"
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
    if summary_rows:
        results_dir = os.path.join(_BASE_DIR, 'results')
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(
            results_dir, f'multimae_test_{args.mode}_summary.xlsx')
        pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    print("=" * 80)


if __name__ == '__main__':
    main()
