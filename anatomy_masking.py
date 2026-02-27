"""
Anatomy-Aware Adaptive Masking for MultiMAE3D Pretraining.

Four components:
1. Patch-Region Mapping: Maps 512 patches (8x8x8 grid) to AAL116 brain atlas regions
2. Region Importance Scoring: Static (AD prior) + Dynamic (EMA teacher attention)
3. Mask Probability Generation: Softmax with temperature control
4. Curriculum Scheduler: Three-phase training schedule

Usage:
    masking = AnatomyAwareMasking(
        img_size=128, patch_size=16,
        atlas_path='altas/AAL116_standard.nii.gz',
    )

    # In training loop:
    mask_probs = masking.get_mask_probs(epoch, total_epochs)
    output = model(images, observed, patch_mask_probs=mask_probs)

    # EMA teacher attention update (every N iterations):
    attn = extract_teacher_attention(ema_teacher, images, observed)
    masking.update_dynamic_importance(attn)
"""

import os
import math
import copy
import numpy as np
import torch

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# =============================================================================
# AAL116 Atlas: Label-to-Name Mapping + AD Importance
# =============================================================================

AAL116_LABEL_NAMES = {
    1: 'Precentral_L', 2: 'Precentral_R',
    3: 'Frontal_Sup_L', 4: 'Frontal_Sup_R',
    5: 'Frontal_Sup_Orb_L', 6: 'Frontal_Sup_Orb_R',
    7: 'Frontal_Mid_L', 8: 'Frontal_Mid_R',
    9: 'Frontal_Mid_Orb_L', 10: 'Frontal_Mid_Orb_R',
    11: 'Frontal_Inf_Oper_L', 12: 'Frontal_Inf_Oper_R',
    13: 'Frontal_Inf_Tri_L', 14: 'Frontal_Inf_Tri_R',
    15: 'Frontal_Inf_Orb_L', 16: 'Frontal_Inf_Orb_R',
    17: 'Rolandic_Oper_L', 18: 'Rolandic_Oper_R',
    19: 'Supp_Motor_Area_L', 20: 'Supp_Motor_Area_R',
    21: 'Olfactory_L', 22: 'Olfactory_R',
    23: 'Frontal_Sup_Medial_L', 24: 'Frontal_Sup_Medial_R',
    25: 'Frontal_Med_Orb_L', 26: 'Frontal_Med_Orb_R',
    27: 'Rectus_L', 28: 'Rectus_R',
    29: 'Insula_L', 30: 'Insula_R',
    31: 'Cingulum_Ant_L', 32: 'Cingulum_Ant_R',
    33: 'Cingulum_Mid_L', 34: 'Cingulum_Mid_R',
    35: 'Cingulum_Post_L', 36: 'Cingulum_Post_R',
    37: 'Hippocampus_L', 38: 'Hippocampus_R',
    39: 'ParaHippocampal_L', 40: 'ParaHippocampal_R',
    41: 'Amygdala_L', 42: 'Amygdala_R',
    43: 'Calcarine_L', 44: 'Calcarine_R',
    45: 'Cuneus_L', 46: 'Cuneus_R',
    47: 'Lingual_L', 48: 'Lingual_R',
    49: 'Occipital_Sup_L', 50: 'Occipital_Sup_R',
    51: 'Occipital_Mid_L', 52: 'Occipital_Mid_R',
    53: 'Occipital_Inf_L', 54: 'Occipital_Inf_R',
    55: 'Fusiform_L', 56: 'Fusiform_R',
    57: 'Postcentral_L', 58: 'Postcentral_R',
    59: 'Parietal_Sup_L', 60: 'Parietal_Sup_R',
    61: 'Parietal_Inf_L', 62: 'Parietal_Inf_R',
    63: 'SupraMarginal_L', 64: 'SupraMarginal_R',
    65: 'Angular_L', 66: 'Angular_R',
    67: 'Precuneus_L', 68: 'Precuneus_R',
    69: 'Paracentral_Lobule_L', 70: 'Paracentral_Lobule_R',
    71: 'Caudate_L', 72: 'Caudate_R',
    73: 'Putamen_L', 74: 'Putamen_R',
    75: 'Pallidum_L', 76: 'Pallidum_R',
    77: 'Thalamus_L', 78: 'Thalamus_R',
    79: 'Heschl_L', 80: 'Heschl_R',
    81: 'Temporal_Sup_L', 82: 'Temporal_Sup_R',
    83: 'Temporal_Pole_Sup_L', 84: 'Temporal_Pole_Sup_R',
    85: 'Temporal_Mid_L', 86: 'Temporal_Mid_R',
    87: 'Temporal_Pole_Mid_L', 88: 'Temporal_Pole_Mid_R',
    89: 'Temporal_Inf_L', 90: 'Temporal_Inf_R',
    91: 'Cerebelum_Crus1_L', 92: 'Cerebelum_Crus1_R',
    93: 'Cerebelum_Crus2_L', 94: 'Cerebelum_Crus2_R',
    95: 'Cerebelum_3_L', 96: 'Cerebelum_3_R',
    97: 'Cerebelum_4_5_L', 98: 'Cerebelum_4_5_R',
    99: 'Cerebelum_6_L', 100: 'Cerebelum_6_R',
    101: 'Cerebelum_7b_L', 102: 'Cerebelum_7b_R',
    103: 'Cerebelum_8_L', 104: 'Cerebelum_8_R',
    105: 'Cerebelum_9_L', 106: 'Cerebelum_9_R',
    107: 'Cerebelum_10_L', 108: 'Cerebelum_10_R',
    109: 'Vermis_1_2', 110: 'Vermis_3',
    111: 'Vermis_4_5', 112: 'Vermis_6',
    113: 'Vermis_7', 114: 'Vermis_8',
    115: 'Vermis_9', 116: 'Vermis_10',
}

# AD-relevant regions: base name (without _L/_R) -> importance level
# Based on Braak staging and AD pathology literature
AD_REGION_IMPORTANCE = {
    # Hippocampus (Braak III-IV)
    'Hippocampus': 'high',
    # Parahippocampal / Entorhinal cortex (Braak I-II, earliest involvement)
    'ParaHippocampal': 'high',
    # Amygdala (Braak III-IV)
    'Amygdala': 'high',
    # Posterior cingulate cortex (early metabolic changes in AD)
    'Cingulum_Post': 'high',
    # Precuneus (default mode network hub, early amyloid deposition)
    'Precuneus': 'high',
    # Inferior temporal (early cortical atrophy)
    'Temporal_Inf': 'high',
    # Middle temporal
    'Temporal_Mid': 'high',
    # Fusiform gyrus
    'Fusiform': 'high',
    # Angular gyrus (default mode network)
    'Angular': 'high',
    # Medial orbitofrontal (default mode network)
    'Frontal_Med_Orb': 'high',
    # Temporal poles
    'Temporal_Pole_Sup': 'high',
    'Temporal_Pole_Mid': 'high',
    # Insula
    'Insula': 'high',
    # Thalamus (subcortical relay)
    'Thalamus': 'high',
    # Caudate (striatal amyloid)
    'Caudate': 'high',
}


def _get_region_importance(region_name):
    """Match an AAL116 region name to its AD importance level."""
    base = region_name
    if base.endswith('_L') or base.endswith('_R'):
        base = base[:-2]
    return AD_REGION_IMPORTANCE.get(base, 'mid')


# =============================================================================
# Patch-Region Mapping
# =============================================================================

def build_patch_region_mapping(atlas_data, img_size, patch_size):
    """Build mapping from 3D patches to atlas regions.

    For each patch, computes the fraction of voxels belonging to each region.
    Patch ordering matches einops rearrange:
      "b c (nd pd) (nh ph) (nw pw) -> b (nd nh nw) c pd ph pw"
      patch_index = d_idx * (grid_h * grid_w) + h_idx * grid_w + w_idx

    Args:
        atlas_data: [D, H, W] integer numpy array (0 = background)
        img_size: (D, H, W) tuple
        patch_size: (pd, ph, pw) tuple

    Returns:
        membership: [N_patches, K] float32 tensor (region membership fractions)
        region_labels: sorted list of unique non-zero integer labels
    """
    grid = tuple(img_size[i] // patch_size[i] for i in range(3))
    N = grid[0] * grid[1] * grid[2]

    labels = sorted([int(l) for l in np.unique(atlas_data) if l > 0])
    K = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    membership = np.zeros((N, K), dtype=np.float32)
    voxels_per_patch = patch_size[0] * patch_size[1] * patch_size[2]

    patch_idx = 0
    for d in range(grid[0]):
        for h in range(grid[1]):
            for w in range(grid[2]):
                block = atlas_data[
                    d * patch_size[0]:(d + 1) * patch_size[0],
                    h * patch_size[1]:(h + 1) * patch_size[1],
                    w * patch_size[2]:(w + 1) * patch_size[2],
                ].flatten()

                for label in labels:
                    count = np.sum(block == label)
                    if count > 0:
                        membership[patch_idx, label_to_idx[label]] = count / voxels_per_patch

                patch_idx += 1

    return torch.from_numpy(membership), labels


# =============================================================================
# Main Class
# =============================================================================

class AnatomyAwareMasking:
    """Anatomy-aware adaptive masking with curriculum learning.

    Args:
        img_size: Input volume size (default 128)
        patch_size: Patch size (default 16)
        atlas_path: Path to AAL116 atlas NIfTI (128x128x128, labels 1-116)
        w_high / w_mid / w_low: Importance weights for AD-critical / gray matter / non-brain regions
        temperature_target: Final temperature for softmax (lower = more focused masking)
        temperature_start: Starting temperature at Phase 2 onset
        phase1_end: End of Phase 1 (uniform masking) as fraction of total epochs
        phase2_end: End of Phase 2 (transition) as fraction of total epochs
        ema_momentum: EMA momentum for teacher model updates
        attention_update_freq: Extract teacher attention every N training iterations
        teacher_batch_size: Number of samples for teacher attention extraction
        importance_mode: 'static', 'dynamic', or 'combined'
        dynamic_weight: Weight of dynamic importance in combined mode [0, 1]
    """

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        atlas_path=None,
        w_high=3.0,
        w_mid=1.5,
        w_low=0.3,
        temperature_target=1.0,
        temperature_start=5.0,
        phase1_end=0.2,
        phase2_end=0.7,
        ema_momentum=0.998,
        attention_update_freq=200,
        teacher_batch_size=2,
        importance_mode='combined',
        dynamic_weight=0.5,
    ):
        self.img_size = (img_size,) * 3 if isinstance(img_size, int) else tuple(img_size)
        self.patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else tuple(patch_size)
        self.grid = tuple(self.img_size[i] // self.patch_size[i] for i in range(3))
        self.num_patches = self.grid[0] * self.grid[1] * self.grid[2]

        self.w_high = w_high
        self.w_mid = w_mid
        self.w_low = w_low

        self.temperature_target = temperature_target
        self.temperature_start = temperature_start
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end

        self.ema_momentum = ema_momentum
        self.attention_update_freq = attention_update_freq
        self.teacher_batch_size = teacher_batch_size

        self.importance_mode = importance_mode
        self.dynamic_weight = dynamic_weight

        # Internal state
        self.patch_region_membership = None   # [N, K]
        self.region_labels = None             # list[int]
        self.static_importance = None         # [N]
        self.dynamic_region_importance = None  # [K] running average
        self.dynamic_patch_importance = None   # [N] fallback without atlas

        if atlas_path is not None:
            self._load_atlas(atlas_path)
            self._compute_static_importance()

    # -----------------------------------------------------------------
    # Atlas loading and static importance
    # -----------------------------------------------------------------

    def _load_atlas(self, atlas_path):
        if not HAS_NIBABEL:
            raise ImportError("nibabel required for atlas loading: pip install nibabel")
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Atlas not found: {atlas_path}")

        atlas_img = nib.load(atlas_path)
        atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int32)

        if atlas_data.shape != self.img_size:
            raise ValueError(
                f"Atlas shape {atlas_data.shape} != expected {self.img_size}. "
                f"Resample the atlas to match your data dimensions."
            )

        self.patch_region_membership, self.region_labels = build_patch_region_mapping(
            atlas_data, self.img_size, self.patch_size
        )

    def _compute_static_importance(self):
        """Compute per-patch static importance: s_i = sum_k(r_{i,k} * w_k)."""
        if self.patch_region_membership is None:
            return

        K = len(self.region_labels)
        region_weights = torch.zeros(K)
        for i, label in enumerate(self.region_labels):
            name = AAL116_LABEL_NAMES.get(label, f"Region_{label}")
            level = _get_region_importance(name)
            if level == 'high':
                region_weights[i] = self.w_high
            elif level == 'mid':
                region_weights[i] = self.w_mid
            else:
                region_weights[i] = self.w_low

        self.static_importance = self.patch_region_membership @ region_weights  # [N]

        # Penalize non-brain patches (< 10% brain coverage)
        brain_coverage = self.patch_region_membership.sum(dim=1)
        non_brain = brain_coverage < 0.1
        self.static_importance[non_brain] = self.w_low * 0.5

    # -----------------------------------------------------------------
    # Dynamic importance from EMA teacher
    # -----------------------------------------------------------------

    def _aggregate_to_regions(self, patch_attention):
        """Aggregate per-patch attention to region level.

        w_k = (1/|P_k|) * sum_{i in P_k} a_i
        """
        if self.patch_region_membership is None:
            return None
        M = self.patch_region_membership  # [N, K]
        numerator = M.t() @ patch_attention  # [K]
        denominator = M.sum(dim=0).clamp(min=1e-8)  # [K]
        return numerator / denominator

    def update_dynamic_importance(self, patch_attention):
        """Update dynamic importance from EMA teacher CLS attention.

        Aggregates to region level for smoothing if atlas available,
        otherwise uses raw per-patch attention.
        """
        patch_attention = patch_attention.detach().cpu()
        momentum = 0.9

        if self.patch_region_membership is not None:
            region_imp = self._aggregate_to_regions(patch_attention)
            if self.dynamic_region_importance is None:
                self.dynamic_region_importance = region_imp
            else:
                self.dynamic_region_importance = (
                    momentum * self.dynamic_region_importance
                    + (1 - momentum) * region_imp
                )
        else:
            if self.dynamic_patch_importance is None:
                self.dynamic_patch_importance = patch_attention
            else:
                self.dynamic_patch_importance = (
                    momentum * self.dynamic_patch_importance
                    + (1 - momentum) * patch_attention
                )

    def _get_dynamic_scores(self):
        """Convert dynamic importance to per-patch scores."""
        if self.dynamic_region_importance is not None and self.patch_region_membership is not None:
            return self.patch_region_membership @ self.dynamic_region_importance
        return self.dynamic_patch_importance

    # -----------------------------------------------------------------
    # Temperature and curriculum
    # -----------------------------------------------------------------

    def get_temperature(self, epoch, total_epochs):
        """Three-phase curriculum temperature.

        Phase 1 (0 to phase1_end):      tau = inf  (uniform masking)
        Phase 2 (phase1_end to phase2_end): cosine anneal start -> target
        Phase 3 (phase2_end to 1.0):     tau = target (stable)
        """
        progress = epoch / max(total_epochs, 1)
        if progress < self.phase1_end:
            return float('inf')
        elif progress < self.phase2_end:
            phase_progress = (progress - self.phase1_end) / (self.phase2_end - self.phase1_end)
            return self.temperature_target + 0.5 * (
                self.temperature_start - self.temperature_target
            ) * (1.0 + math.cos(math.pi * phase_progress))
        else:
            return self.temperature_target

    # -----------------------------------------------------------------
    # Combined importance scores
    # -----------------------------------------------------------------

    def get_importance_scores(self):
        """Get combined per-patch importance scores based on importance_mode."""
        static = self.static_importance
        dynamic = self._get_dynamic_scores()

        if self.importance_mode == 'static':
            return static
        elif self.importance_mode == 'dynamic':
            return dynamic if dynamic is not None else static
        else:  # combined
            if static is None and dynamic is None:
                return None
            if dynamic is None:
                return static
            if static is None:
                return dynamic
            # Normalize both to [0, 1] before combining
            s_norm = (static - static.min()) / (static.max() - static.min() + 1e-8)
            d_norm = (dynamic - dynamic.min()) / (dynamic.max() - dynamic.min() + 1e-8)
            alpha = self.dynamic_weight
            return (1 - alpha) * s_norm + alpha * d_norm

    # -----------------------------------------------------------------
    # Main API
    # -----------------------------------------------------------------

    def get_mask_probs(self, epoch, total_epochs):
        """Get per-patch masking probabilities.

        Returns:
            [N_patches] tensor (sums to 1), or None for uniform masking.
            Higher value = more likely to be masked.
        """
        tau = self.get_temperature(epoch, total_epochs)
        if tau == float('inf'):
            return None

        scores = self.get_importance_scores()
        if scores is None:
            return None

        return torch.softmax(scores / tau, dim=0)

    def get_curriculum_info(self, epoch, total_epochs):
        """Get curriculum state for logging."""
        tau = self.get_temperature(epoch, total_epochs)
        progress = epoch / max(total_epochs, 1)

        if progress < self.phase1_end:
            phase = 1
        elif progress < self.phase2_end:
            phase = 2
        else:
            phase = 3

        info = {'phase': phase, 'temperature': tau if tau != float('inf') else -1.0}

        scores = self.get_importance_scores()
        if scores is not None:
            info['importance_min'] = scores.min().item()
            info['importance_max'] = scores.max().item()
            info['importance_mean'] = scores.mean().item()

        probs = self.get_mask_probs(epoch, total_epochs)
        if probs is not None:
            info['prob_max'] = probs.max().item()
            info['prob_min'] = probs.min().item()
            info['prob_ratio'] = (probs.max() / probs.min().clamp(min=1e-10)).item()

        return info

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def state_dict(self):
        return {
            'static_importance': self.static_importance,
            'dynamic_region_importance': self.dynamic_region_importance,
            'dynamic_patch_importance': self.dynamic_patch_importance,
        }

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        self.static_importance = state_dict.get('static_importance')
        self.dynamic_region_importance = state_dict.get('dynamic_region_importance')
        self.dynamic_patch_importance = state_dict.get('dynamic_patch_importance')


# =============================================================================
# EMA Teacher Utilities
# =============================================================================

@torch.no_grad()
def create_ema_teacher(model):
    """Create an EMA copy of the model (no gradients)."""
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, momentum=0.998):
    """Update EMA teacher: theta_t = m * theta_t + (1 - m) * theta_s."""
    student_model = student.module if hasattr(student, 'module') else student
    teacher_model = teacher.module if hasattr(teacher, 'module') else teacher
    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)


@torch.no_grad()
def extract_teacher_attention(teacher, images, observed, num_global_tokens=1):
    """Extract CLS-to-patch attention from the EMA teacher's last encoder layer.

    Runs a full (unmasked) forward pass through the teacher encoder and
    extracts attention weights from the final transformer block.

    Args:
        teacher: EMA teacher model (MultiMAE3D, not DDP-wrapped)
        images: [B, 4, D, H, W]
        observed: [B, 4]
        num_global_tokens: number of CLS tokens (default 1)

    Returns:
        patch_attention: [num_patches] averaged CLS attention scores
    """
    from models.multimae3d_utils import patchify

    teacher_model = teacher.module if hasattr(teacher, 'module') else teacher
    teacher_model.eval()

    B = images.shape[0]
    device = images.device
    batch = teacher_model._split_modalities(images)

    # Tokenize all patches (no masking)
    tokens_list = []
    for i, name in enumerate(teacher_model.MODALITY_NAMES):
        patches = patchify(batch[name], teacher_model.patch_size)
        tok = teacher_model.input_adapters[name](patches)
        pos_emb = teacher_model.pos_embed.expand(B, -1, -1)
        tok = tok + pos_emb
        mod_mask = observed[:, i:i + 1].unsqueeze(-1)
        tok = tok * mod_mask
        tokens_list.append(tok)

    input_tokens = torch.cat(tokens_list, dim=1)

    if teacher_model.num_global_tokens > 0:
        cls = teacher_model.global_tokens.unsqueeze(0).expand(B, -1, -1)
        input_tokens = torch.cat([cls, input_tokens], dim=1)

    # Attention mask for missing modalities
    total_tokens = input_tokens.shape[1]
    num_patches = teacher_model.num_patches
    attn_mask = torch.zeros(B, 1, 1, total_tokens, device=device)
    mod_offset = num_global_tokens
    for i in range(len(teacher_model.MODALITY_NAMES)):
        start, end = mod_offset, mod_offset + num_patches
        missing = (observed[:, i] < 0.5)
        if missing.any():
            attn_mask[missing, :, :, start:end] = float("-inf")
        mod_offset = end
    if (attn_mask == 0).all():
        attn_mask = None

    # Forward through encoder layers 0..L-2
    x = input_tokens
    for block in teacher_model.encoder[:-1]:
        x = block(x, attn_mask=attn_mask)

    # Last layer: manually extract attention weights
    last_block = teacher_model.encoder[-1]
    x_norm = last_block.norm1(x)
    B_, N_, C_ = x_norm.shape
    num_heads = last_block.attn.num_heads
    head_dim = C_ // num_heads

    qkv = last_block.attn.qkv(x_norm)
    qkv = qkv.reshape(B_, N_, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)

    attn_weights = (q @ k.transpose(-2, -1)) * last_block.attn.scale
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = attn_weights.softmax(dim=-1)  # [B, heads, N, N]

    # CLS (token 0) attention to patch tokens (skip global tokens)
    cls_attn = attn_weights[:, :, 0, num_global_tokens:]  # [B, heads, 4*num_patches]
    cls_attn = cls_attn.mean(dim=1)  # avg over heads: [B, 4*num_patches]

    # Reshape to per-modality and average
    num_modalities = len(teacher_model.MODALITY_NAMES)
    per_mod_attn = cls_attn.reshape(B, num_modalities, num_patches)

    observed_expanded = observed.unsqueeze(-1)  # [B, 4, 1]
    weighted_attn = (per_mod_attn * observed_expanded).sum(dim=1)  # [B, num_patches]
    count = observed.sum(dim=1, keepdim=True).clamp(min=1)
    avg_attn = weighted_attn / count
    patch_attention = avg_attn.mean(dim=0)  # [num_patches]

    return patch_attention
