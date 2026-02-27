"""
MultiMAE3D Pretraining Script with Cross-Modal Prediction + Anatomy-Aware Masking

Usage:
    # Single GPU
    python train_multimae.py --batch_size 4

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 train_multimae.py --batch_size 4

    # With cross-modal prediction + anatomy-aware masking
    torchrun --nproc_per_node=8 train_multimae.py --batch_size 4 \
        --enable_cross_modal --use_anatomy_masking --atlas_path altas/AAL116_standard.nii.gz
"""

import os
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.multimae3d import create_multimae3d
from pretrain_dataloader_v2 import MultiModalPretrainDataset
from anatomy_masking import (
    AnatomyAwareMasking,
    create_ema_teacher,
    update_ema_teacher,
    extract_teacher_attention,
)


# =============================================================================
# Distributed setup
# =============================================================================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        try:
            torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] Warning: cleanup failed: {e}")
            try:
                dist.destroy_process_group()
            except Exception:
                pass


# =============================================================================
# Training loop
# =============================================================================

def cosine_ema_momentum(epoch, total_epochs, start=0.996, end=1.0):
    """Cosine schedule for EMA momentum: start → end over training."""
    progress = epoch / max(total_epochs, 1)
    return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2


def cross_modal_lambda_schedule(epoch, warmup_epochs, target_lambda):
    """λ=0 for first warmup_epochs, then linear ramp to target over next warmup_epochs."""
    if epoch <= warmup_epochs:
        return 0.0
    ramp_progress = min(1.0, (epoch - warmup_epochs) / max(warmup_epochs, 1))
    return target_lambda * ramp_progress


def train_one_epoch(
    model, dataloader, optimizer, epoch, writer,
    rank=0, device="cuda", global_step=0, grad_clip=0.5,
    enable_cross_modal=False, cross_modal_lambda=0.1,
    cross_modal_warmup_epochs=10, total_epochs=1200,
    ema_momentum_start=0.996, ema_momentum_end=1.0,
    anatomy_masking=None, anatomy_ema_teacher=None,
):
    model.train()
    model_inner = model.module if hasattr(model, "module") else model

    # Compute schedules for this epoch
    ema_momentum = cosine_ema_momentum(
        epoch, total_epochs, ema_momentum_start, ema_momentum_end,
    )
    effective_lambda = cross_modal_lambda_schedule(
        epoch, cross_modal_warmup_epochs, cross_modal_lambda,
    ) if enable_cross_modal else 0.0

    # Anatomy masking: compute mask probabilities for this epoch
    mask_probs = None
    if anatomy_masking is not None:
        mask_probs = anatomy_masking.get_mask_probs(epoch, total_epochs)
        if mask_probs is not None:
            mask_probs = mask_probs.to(device)

    use_dynamic_anatomy = (
        anatomy_masking is not None
        and anatomy_ema_teacher is not None
        and anatomy_masking.importance_mode in ('dynamic', 'combined')
    )

    total_loss = 0.0
    total_cross_loss = 0.0
    per_mod_losses = {name: 0.0 for name in ["T1", "T2", "Flair", "PET"]}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))

    for batch_idx, batch in enumerate(pbar):
        images = batch["images"].to(device)      # [B, 4, 128, 128, 128]
        observed = batch["observed"].to(device)   # [B, 4]

        # Forward with anatomy-aware masking
        output = model(images, observed, return_loss=True, patch_mask_probs=mask_probs)
        mae_loss = output["loss"]
        cross_loss = output.get("cross_modal_loss", torch.tensor(0.0, device=device))

        # Combined loss
        loss = mae_loss + effective_lambda * cross_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )

        optimizer.step()

        # EMA update of cross-modal teacher (after optimizer step)
        if enable_cross_modal:
            model_inner.update_teacher(ema_momentum)

        # EMA update of anatomy masking teacher (every step)
        if anatomy_ema_teacher is not None:
            update_ema_teacher(anatomy_ema_teacher, model, momentum=anatomy_masking.ema_momentum)

        # Periodically extract teacher attention and update dynamic importance
        iteration = global_step + batch_idx
        if use_dynamic_anatomy and iteration > 0 and iteration % anatomy_masking.attention_update_freq == 0:
            tb = min(anatomy_masking.teacher_batch_size, images.shape[0])
            with torch.no_grad():
                attn = extract_teacher_attention(
                    anatomy_ema_teacher, images[:tb], observed[:tb],
                )
            anatomy_masking.update_dynamic_importance(attn)
            # Recompute mask probs with updated dynamic importance
            new_probs = anatomy_masking.get_mask_probs(epoch, total_epochs)
            if new_probs is not None:
                mask_probs = new_probs.to(device)

        # Logging
        mae_val = mae_loss.item()
        cross_val = cross_loss.item() if torch.is_tensor(cross_loss) else cross_loss
        combined_val = loss.item()
        total_loss += mae_val
        total_cross_loss += cross_val
        num_batches += 1

        for name, mod_loss in output["per_modality_loss"].items():
            per_mod_losses[name] += mod_loss.item()

        if rank == 0 and writer is not None:
            step = global_step + batch_idx
            writer.add_scalar("Train/Batch/MAE_Loss", mae_val, step)
            writer.add_scalar("Train/Batch/Total_Loss", combined_val, step)
            if enable_cross_modal:
                writer.add_scalar("Train/Batch/Cross_Modal_Loss", cross_val, step)
                writer.add_scalar("Train/Batch/Cross_Lambda", effective_lambda, step)
                writer.add_scalar("Train/Batch/EMA_Momentum", ema_momentum, step)
            writer.add_scalar("Train/Batch/Grad_Norm", grad_norm.item(), step)
            writer.add_scalar("Train/Batch/LR", optimizer.param_groups[0]["lr"], step)
            for name, mr in output["mask_ratios"].items():
                writer.add_scalar(f"Train/Batch/MaskRatio_{name}", mr, step)

        if rank == 0:
            postfix = {"mae": f"{mae_val:.4f}"}
            if enable_cross_modal and effective_lambda > 0:
                postfix["cross"] = f"{cross_val:.4f}"
            pbar.set_postfix(postfix)

    avg_loss = total_loss / max(num_batches, 1)
    avg_cross_loss = total_cross_loss / max(num_batches, 1)
    avg_mod_losses = {k: v / max(num_batches, 1) for k, v in per_mod_losses.items()}

    if rank == 0 and writer is not None:
        writer.add_scalar("Train/Epoch/MAE_Loss", avg_loss, epoch)
        writer.add_scalar("Train/Epoch/Total_Loss",
                          avg_loss + effective_lambda * avg_cross_loss, epoch)
        if enable_cross_modal:
            writer.add_scalar("Train/Epoch/Cross_Modal_Loss", avg_cross_loss, epoch)
        for name, ml in avg_mod_losses.items():
            writer.add_scalar(f"Train/Epoch/Loss_{name}", ml, epoch)
        writer.add_scalar("Train/Epoch/LR", optimizer.param_groups[0]["lr"], epoch)

        # Log anatomy masking curriculum info
        if anatomy_masking is not None:
            info = anatomy_masking.get_curriculum_info(epoch, total_epochs)
            writer.add_scalar("Anatomy/Phase", info['phase'], epoch)
            writer.add_scalar("Anatomy/Temperature", info['temperature'], epoch)
            if 'importance_max' in info:
                writer.add_scalar("Anatomy/Importance_Max", info['importance_max'], epoch)
                writer.add_scalar("Anatomy/Importance_Min", info['importance_min'], epoch)
                writer.add_scalar("Anatomy/Importance_Mean", info['importance_mean'], epoch)
            if 'prob_ratio' in info:
                writer.add_scalar("Anatomy/Prob_MaxMinRatio", info['prob_ratio'], epoch)

    return avg_loss, avg_mod_losses, num_batches


# =============================================================================
# Checkpoint management
# =============================================================================

def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, best_loss,
    global_step, save_dir, rank=0, save_freq=100,
    anatomy_masking=None, anatomy_ema_teacher=None,
):
    if rank != 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    model_inner = model.module if hasattr(model, "module") else model

    # Always save latest (for resume)
    resume_ckpt = {
        "epoch": epoch,
        "model_state_dict": model_inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "best_loss": best_loss,
        "global_step": global_step,
    }
    if anatomy_masking is not None:
        resume_ckpt["anatomy_masking_state"] = anatomy_masking.state_dict()
    if anatomy_ema_teacher is not None:
        ema_inner = anatomy_ema_teacher.module if hasattr(anatomy_ema_teacher, "module") else anatomy_ema_teacher
        resume_ckpt["anatomy_ema_teacher_state_dict"] = ema_inner.state_dict()
    torch.save(resume_ckpt, os.path.join(save_dir, "latest.pth"))

    # Periodic save (encoder only, for downstream)
    if epoch % save_freq == 0:
        encoder_ckpt = {
            "epoch": epoch,
            "encoder_state_dict": {
                k: v for k, v in model_inner.state_dict().items()
                if k.startswith("encoder.") or k.startswith("input_adapters.") or k.startswith("pos_embed") or k.startswith("global_tokens")
            },
            "loss": loss,
        }
        torch.save(encoder_ckpt, os.path.join(save_dir, f"encoder_epoch_{epoch}.pth"))

    # Save best
    if loss <= best_loss:
        encoder_ckpt = {
            "epoch": epoch,
            "encoder_state_dict": {
                k: v for k, v in model_inner.state_dict().items()
                if k.startswith("encoder.") or k.startswith("input_adapters.") or k.startswith("pos_embed") or k.startswith("global_tokens")
            },
            "full_model_state_dict": model_inner.state_dict(),
            "loss": loss,
        }
        torch.save(encoder_ckpt, os.path.join(save_dir, "best_model.pth"))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MultiMAE3D Pretraining")

    # Data
    parser.add_argument("--excel_dir", type=str,
                        default="./data/Match_data_path/pretraining_processed")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--augmentation", action="store_true", default=True)
    parser.add_argument("--no_augmentation", action="store_false", dest="augmentation")

    # Model
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--decoder_embed_dim", type=int, default=384)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--decoder_num_heads", type=int, default=12)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--use_dirichlet", action="store_true", default=True)
    parser.add_argument("--no_dirichlet", action="store_false", dest="use_dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)

    # Cross-modal mutual prediction
    parser.add_argument("--enable_cross_modal", action="store_true", default=False,
                        help="Enable cross-level mutual prediction (MRI↔PET)")
    parser.add_argument("--cross_modal_lambda", type=float, default=0.1,
                        help="Weight for cross-modal loss (search: 0.01-1.0)")
    parser.add_argument("--cross_modal_warmup_epochs", type=int, default=10,
                        help="Epochs with λ=0 before linear ramp")
    parser.add_argument("--ema_momentum_start", type=float, default=0.996,
                        help="EMA momentum at start of training (for cross-modal teacher)")
    parser.add_argument("--ema_momentum_end", type=float, default=1.0,
                        help="EMA momentum at end of training (for cross-modal teacher)")

    # Anatomy-aware masking
    parser.add_argument("--use_anatomy_masking", action="store_true", default=False,
                        help="Enable anatomy-aware adaptive masking")
    parser.add_argument("--atlas_path", type=str, default="altas/AAL116_standard.nii.gz",
                        help="Path to brain atlas NIfTI file (128^3, in data space)")
    parser.add_argument("--importance_mode", type=str, default="combined",
                        choices=["static", "dynamic", "combined"],
                        help="Importance scoring mode: static (AD prior), dynamic (EMA attention), combined")
    parser.add_argument("--anatomy_w_high", type=float, default=3.0,
                        help="Importance weight for AD-critical regions")
    parser.add_argument("--anatomy_w_mid", type=float, default=1.5,
                        help="Importance weight for other gray matter regions")
    parser.add_argument("--anatomy_w_low", type=float, default=0.3,
                        help="Importance weight for non-brain patches")
    parser.add_argument("--anatomy_temp_target", type=float, default=1.0,
                        help="Target temperature for masking softmax (lower = more focused)")
    parser.add_argument("--anatomy_temp_start", type=float, default=5.0,
                        help="Starting temperature at Phase 2 onset")
    parser.add_argument("--anatomy_phase1_end", type=float, default=0.2,
                        help="End of Phase 1 (uniform masking) as fraction of total epochs")
    parser.add_argument("--anatomy_phase2_end", type=float, default=0.7,
                        help="End of Phase 2 (transition) as fraction of total epochs")
    parser.add_argument("--anatomy_ema_momentum", type=float, default=0.998,
                        help="EMA momentum for anatomy masking teacher model")
    parser.add_argument("--attention_update_freq", type=int, default=200,
                        help="Extract teacher attention every N iterations")
    parser.add_argument("--teacher_batch_size", type=int, default=2,
                        help="Batch size for teacher attention extraction")
    parser.add_argument("--dynamic_weight", type=float, default=0.5,
                        help="Weight of dynamic importance in combined mode [0, 1]")

    # Training
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    # Save
    parser.add_argument("--save_dir", type=str, default="./pretrain_checkpoints/multimae")
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="./logs/multimae")
    parser.add_argument("--resume", type=str, default="", help="Path to latest.pth for resume (restores epoch/optimizer)")
    parser.add_argument("--pretrain_weights", type=str, default="",
                        help="Path to pretrained checkpoint — loads model weights only, starts from epoch 1")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=" * 70)
        print("MultiMAE3D Pretraining")
        if args.enable_cross_modal:
            print("  + Cross-Modal Mutual Prediction ENABLED")
        if args.use_anatomy_masking:
            print("  + Anatomy-Aware Adaptive Masking ENABLED")
            print(f"    Atlas: {args.atlas_path}")
            print(f"    Mode: {args.importance_mode}")
            print(f"    Weights: high={args.anatomy_w_high}, mid={args.anatomy_w_mid}, low={args.anatomy_w_low}")
            print(f"    Temperature: {args.anatomy_temp_start} -> {args.anatomy_temp_target}")
            print(f"    Curriculum: Phase1 end={args.anatomy_phase1_end}, Phase2 end={args.anatomy_phase2_end}")
        print("=" * 70)
        print(f"World size: {world_size}, Device: {device}")
        print(f"Config: {vars(args)}")
        print("=" * 70)

    # Dataset
    dataset = MultiModalPretrainDataset(
        excel_dir=args.excel_dir,
        image_size=(args.img_size, args.img_size, args.img_size),
        augmentation=args.augmentation,
        modality_dropout_prob=0.0,  # No dropout — natural missing is enough
        min_modalities=1,
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    if rank == 0:
        print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Model
    model = create_multimae3d(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mask_ratio=args.mask_ratio,
        use_dirichlet=args.use_dirichlet,
        dirichlet_alpha=args.dirichlet_alpha,
        drop_path_rate=args.drop_path_rate,
        enable_cross_modal=args.enable_cross_modal,
    )
    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/1e6:.1f}M)")

    # Anatomy-aware masking setup (before DDP wrapping)
    anatomy_masking = None
    anatomy_ema_teacher = None

    if args.use_anatomy_masking:
        anatomy_masking = AnatomyAwareMasking(
            img_size=args.img_size,
            patch_size=args.patch_size,
            atlas_path=args.atlas_path,
            w_high=args.anatomy_w_high,
            w_mid=args.anatomy_w_mid,
            w_low=args.anatomy_w_low,
            temperature_target=args.anatomy_temp_target,
            temperature_start=args.anatomy_temp_start,
            phase1_end=args.anatomy_phase1_end,
            phase2_end=args.anatomy_phase2_end,
            ema_momentum=args.anatomy_ema_momentum,
            attention_update_freq=args.attention_update_freq,
            teacher_batch_size=args.teacher_batch_size,
            importance_mode=args.importance_mode,
            dynamic_weight=args.dynamic_weight,
        )

        if rank == 0:
            info = anatomy_masking.get_curriculum_info(1, args.epochs)
            print(f"Anatomy masking initialized: {anatomy_masking.num_patches} patches, "
                  f"importance range [{info.get('importance_min', 'N/A'):.3f}, "
                  f"{info.get('importance_max', 'N/A'):.3f}]")

        # Create EMA teacher for dynamic importance (before DDP)
        if args.importance_mode in ('dynamic', 'combined'):
            anatomy_ema_teacher = create_ema_teacher(model)
            if rank == 0:
                print(f"Anatomy EMA teacher created (momentum={args.anatomy_ema_momentum})")

    # DDP wrapping (after EMA teacher creation)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer (only trainable params — excludes teacher EMA parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # LR Scheduler: linear warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard
    writer = None
    if rank == 0:
        log_dir = os.path.join(args.log_dir, f"seed_{args.seed}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")

    # Resume
    start_epoch = 1
    best_loss = float("inf")
    global_step = 0

    if args.pretrain_weights and os.path.isfile(args.pretrain_weights):
        # Load model weights only — epoch/optimizer/scheduler stay fresh (start from epoch 1)
        if rank == 0:
            print(f"Loading pretrained weights: {args.pretrain_weights}")
        ckpt = torch.load(args.pretrain_weights, map_location=f"cuda:{local_rank}")
        model_inner = model.module if hasattr(model, "module") else model
        state_dict = ckpt.get("model_state_dict", ckpt.get("full_model_state_dict", ckpt))
        missing, unexpected = model_inner.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"  Loaded weights from epoch {ckpt.get('epoch', '?')}")
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        # Initialize cross-modal teacher from the loaded student weights
        if args.enable_cross_modal:
            model_inner.init_teacher_from_student()
            if rank == 0:
                print("  Cross-modal teacher initialized from loaded student weights")
        # Initialize anatomy EMA teacher from loaded weights
        if anatomy_ema_teacher is not None:
            anatomy_ema_teacher.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print("  Anatomy EMA teacher initialized from pretrained weights")
        if rank == 0:
            print(f"  Training from epoch 1 with fresh optimizer/scheduler")
        del ckpt

    elif args.resume and os.path.isfile(args.resume):
        # Full resume: restore model + optimizer + scheduler + epoch counter
        if rank == 0:
            print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}")
        model_inner = model.module if hasattr(model, "module") else model
        missing, unexpected = model_inner.load_state_dict(ckpt["model_state_dict"], strict=False)
        if rank == 0 and (missing or unexpected):
            print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        if args.enable_cross_modal:
            model_inner.init_teacher_from_student()
            if rank == 0:
                print("  Cross-modal teacher re-initialized from loaded student weights")
        # Restore anatomy masking state
        if anatomy_masking is not None and "anatomy_masking_state" in ckpt:
            anatomy_masking.load_state_dict(ckpt["anatomy_masking_state"])
            if rank == 0:
                print("  Restored anatomy masking state")
        # Restore anatomy EMA teacher
        if anatomy_ema_teacher is not None and "anatomy_ema_teacher_state_dict" in ckpt:
            anatomy_ema_teacher.load_state_dict(ckpt["anatomy_ema_teacher_state_dict"])
            if rank == 0:
                print("  Restored anatomy EMA teacher state")
        if rank == 0:
            print(f"Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.4f}")
        del ckpt

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss, avg_mod_losses, num_batches = train_one_epoch(
            model, dataloader, optimizer, epoch, writer,
            rank=rank, device=device, global_step=global_step,
            grad_clip=args.grad_clip,
            enable_cross_modal=args.enable_cross_modal,
            cross_modal_lambda=args.cross_modal_lambda,
            cross_modal_warmup_epochs=args.cross_modal_warmup_epochs,
            total_epochs=args.epochs,
            ema_momentum_start=args.ema_momentum_start,
            ema_momentum_end=args.ema_momentum_end,
            anatomy_masking=anatomy_masking,
            anatomy_ema_teacher=anatomy_ema_teacher,
        )
        global_step += num_batches
        elapsed = time.time() - t0

        # Step LR scheduler
        scheduler.step()

        if rank == 0:
            mod_str = ", ".join(f"{k}: {v:.4f}" for k, v in avg_mod_losses.items() if v > 0)
            phase_str = ""
            if anatomy_masking is not None:
                info = anatomy_masking.get_curriculum_info(epoch, args.epochs)
                phase_str = f" | Phase {info['phase']}, tau={info['temperature']:.2f}"
            print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | {mod_str}{phase_str} | Time: {elapsed:.1f}s")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, avg_loss, best_loss,
            global_step, args.save_dir, rank, args.save_freq,
            anatomy_masking=anatomy_masking, anatomy_ema_teacher=anatomy_ema_teacher,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            if rank == 0:
                print(f"  -> New best loss: {best_loss:.4f}")

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()

    cleanup_distributed()
    if writer is not None:
        writer.add_scalar("Final/Best_Loss", best_loss, 0)
        writer.close()

    if rank == 0:
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
