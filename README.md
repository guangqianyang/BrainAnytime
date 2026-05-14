# BrainAnytime

Official implementation of **BrainAnytime: Anatomy-Aware Cross-Modal Pretraining for Brain Image Analysis with Arbitrary Modality Availability**.

## Congrats: This paper has been early accepted (top 9%) by MICCAI 2026.

## Pretrained Weights

**The pretrained model weights is available at [google drive](https://drive.google.com/file/d/1L49zJ_Apj2jJe88_iy6jLcmd6KUlnc5h/view?usp=sharing).**

## Overview

BrainAnytime is a self-supervised pretraining framework for multi-modal 3D brain imaging (T1, T2, Flair, PET) that handles **arbitrary missing modality combinations** at both training and inference time.

### Key Features

- **Multi-modal Masked Autoencoder (MultiMAE3D)**: Shared ViT encoder with per-modality input/output adapters, supporting 4 modalities (T1, T2, Flair, PET)
- **Cross-Modal Mutual Prediction**: EMA teacher-student framework for MRI-PET cross-level feature alignment
- **Anatomy-Aware Adaptive Masking**: Three-phase curriculum masking guided by AAL116 brain atlas and AD-relevant region priors
- **Missing Modality Robustness**: Handles arbitrary missing modality combinations via attention masking and observed indicators

## Project Structure

```
BrainAnytime/
├── models/
│   ├── multimae3d.py              # MultiMAE3D model architecture
│   └── multimae3d_utils.py        # Patchify, masking, positional embeddings
├── anatomy_masking.py             # Anatomy-aware adaptive masking module
├── pretrain_dataloader_v2.py      # Multi-modal pretraining data loader
├── train_multimae.py              # Pretraining script (single/multi-GPU DDP)
├── finetune_main.py               # Downstream finetuning (CN vs AD, CN vs MCI, MMSE, AGE)
├── test_main.py                   # Test-only evaluation
└── altas/
    └── AAL116_standard.nii.gz     # AAL116 brain atlas (128x128x128)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torchio
- nibabel
- timm
- einops
- tensorboardX
- scikit-learn
- pandas
- scipy
- tqdm

## Data Preparation

Organize your data as follows:

```
./data/
├── Match_data_path/
│   └── pretraining_processed/     # Pretraining Excel files
│       ├── modality_data_A4.xlsx
│       ├── modality_data_ADNIDOD.xlsx
│       ├── modality_data_AIBL.xlsx
│       ├── modality_data_BraTS.xlsx
│       └── modality_data_NACC.xlsx
├── Pretrain/                      # Preprocessed NIfTI files for pretraining
└── Downstream/
    └── ADNI/                      # Downstream task data
        └── ADNI_Division/
            ├── modality_data_train.xlsx
            ├── modality_data_val.xlsx
            └── modality_data_test.xlsx
```

Each Excel file should contain columns for subject IDs and file paths to the corresponding NIfTI images for each modality.

## Usage

### Pretraining

```bash
# Single GPU
python train_multimae.py --batch_size 4

# Multi-GPU DDP (8 GPUs)
torchrun --nproc_per_node=8 train_multimae.py \
    --batch_size 16 \
    --enable_cross_modal \
    --use_anatomy_masking \
    --atlas_path altas/AAL116_standard.nii.gz
```

### Downstream Finetuning

```bash
# Finetune on all tasks (3 seeds each)
python finetune_main.py \
    --pretrained ./pretrain_checkpoints/multimae/best_model.pth

# Specific task only
python finetune_main.py \
    --pretrained ./pretrain_checkpoints/multimae/best_model.pth \
    --tasks "CN vs AD"
```

### Testing

```bash
# Test all tasks for finetune mode
python test_main.py --mode finetune

# Test a specific task
python test_main.py --mode finetune --tasks "CN vs AD"
```

## Downstream Tasks

| Task | Type | Metric |
|------|------|--------|
| CN vs AD | Classification | ACC, AUC, Sensitivity, Specificity, F1 |
| CN vs MCI | Classification | ACC, AUC, Sensitivity, Specificity, F1 |
| MMSE | Regression | MAE, RMSE, Pearson |
| AGE | Regression | MAE, RMSE, Pearson |

## License

This project is released for academic research purposes only.

## Citation

Citation information will be provided upon paper acceptance.
