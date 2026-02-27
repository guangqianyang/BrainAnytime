"""
MultiMAE 3D Utility Functions
- Patchify / Unpatchify
- Patch shuffling for masking
- 3D sinusoidal positional embeddings
- Dirichlet masking with missing modality support
"""

from typing import Union, Tuple, Dict, List

import torch
import torch.nn as nn
from torch.distributions import Dirichlet
from einops import rearrange


def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)


def calc_patchified_dim(
    img_size: Union[int, Tuple[int, int, int]],
    patch_size: Union[int, Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    img_size = to_3tuple(img_size)
    patch_size = to_3tuple(patch_size)
    return tuple(img_size[i] // patch_size[i] for i in range(3))


def patchify(
    image: torch.Tensor,
    patch_size: Union[int, Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Convert image to patches.
    image: [B, C, D, H, W]
    returns: [B, num_patches, C, pd, ph, pw]
    """
    patch_size = to_3tuple(patch_size)
    img_size = image.shape[-3:]
    patchified_dim = calc_patchified_dim(img_size, patch_size)
    patches = rearrange(
        image,
        "b c (nd pd) (nh ph) (nw pw) -> b (nd nh nw) c pd ph pw",
        pd=patch_size[0],
        ph=patch_size[1],
        pw=patch_size[2],
        nd=patchified_dim[0],
        nh=patchified_dim[1],
        nw=patchified_dim[2],
    )
    return patches


def unpatchify(
    patches: torch.Tensor,
    img_size: Union[int, Tuple[int, int, int]],
    patch_size: Union[int, Tuple[int, int, int]],
) -> torch.Tensor:
    """
    Convert patches back to image.
    patches: [B, num_patches, C, pd, ph, pw]
    returns: [B, C, D, H, W]
    """
    patch_size = to_3tuple(patch_size)
    img_size = to_3tuple(img_size)
    patchified_dim = calc_patchified_dim(img_size, patch_size)
    image = rearrange(
        patches,
        "b (nd nh nw) c pd ph pw -> b c (nd pd) (nh ph) (nw pw)",
        pd=patch_size[0],
        ph=patch_size[1],
        pw=patch_size[2],
        nd=patchified_dim[0],
        nh=patchified_dim[1],
        nw=patchified_dim[2],
    )
    return image


def shuffle_patches(
    patches: torch.Tensor,
    permutations: torch.Tensor = None,
    mask_probs: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffle patches along the patch dimension.

    When mask_probs is None: uniform random shuffle.
    When mask_probs is provided: Gumbel-top-k weighted shuffle.
      Patches with higher mask_probs end up at higher indices (masked),
      patches with lower mask_probs end up at lower indices (visible).

    Args:
        patches: [B, N, ...]
        permutations: optional pre-computed permutation indices [B, N]
        mask_probs: optional [N] per-patch masking probability (sums to 1)

    Returns:
        (shuffled_patches, perm_indices)
    """
    batch_size, num_patches = patches.shape[0], patches.shape[1]
    if permutations is not None:
        perm_idx = permutations
    else:
        rand = torch.rand(batch_size, num_patches, device=patches.device)

        if mask_probs is not None:
            # Gumbel-top-k trick for weighted sampling without replacement.
            # key_i = log(p_i) + Gumbel(0,1)_i
            # Top-k of keys = sample from Multinomial(p, k)
            # After ascending argsort: low keys → visible, high keys → masked.
            mask_probs = mask_probs.to(patches.device)
            gumbel = -torch.log(-torch.log(rand.clamp(1e-20, 1.0 - 1e-20)))
            log_probs = torch.log(mask_probs.clamp(min=1e-20))  # [N]
            keys = gumbel + log_probs.unsqueeze(0)  # [B, N]
            perm_idx = torch.argsort(keys, dim=1)
        else:
            perm_idx = torch.argsort(rand, dim=1)

    shuffled = patches[torch.arange(batch_size, device=patches.device)[:, None], perm_idx]
    return shuffled, perm_idx


def unshuffle_patches(
    patches: torch.Tensor,
    perm_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse of shuffle_patches.
    """
    batch_size = patches.shape[0]
    inv_idx = torch.argsort(perm_idx, dim=1)
    return patches[torch.arange(batch_size, device=patches.device)[:, None], inv_idx]


def build_3d_sincos_position_embedding(
    grid_size: Tuple[int, int, int],
    embed_dim: int,
    temperature: float = 10000.0,
) -> nn.Parameter:
    """
    Build 3D sinusoidal positional embedding.
    returns: [1, num_patches, embed_dim] (frozen parameter)
    """
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size

    assert embed_dim % 6 == 0, \
        f"embed_dim ({embed_dim}) must be divisible by 6 for 3D sincos pos embed"

    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)

    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_d = torch.einsum("m,d->md", grid_d.flatten(), omega)

    # Expand to full grid: [H*W*D, pos_dim] for each axis
    # Use meshgrid ordering to get correct spatial layout
    grid_h_idx, grid_w_idx, grid_d_idx = torch.meshgrid(
        torch.arange(h), torch.arange(w), torch.arange(d), indexing="ij"
    )
    grid_h_flat = grid_h_idx.flatten()  # [H*W*D]
    grid_w_flat = grid_w_idx.flatten()
    grid_d_flat = grid_d_idx.flatten()

    pos_emb = torch.cat([
        torch.sin(out_h[grid_h_flat]),
        torch.cos(out_h[grid_h_flat]),
        torch.sin(out_w[grid_w_flat]),
        torch.cos(out_w[grid_w_flat]),
        torch.sin(out_d[grid_d_flat]),
        torch.cos(out_d[grid_d_flat]),
    ], dim=1)[None, :, :]  # [1, num_patches, embed_dim]

    pos_emb = nn.Parameter(pos_emb)
    pos_emb.requires_grad = False
    return pos_emb


def generate_dirichlet_mask_ratios(
    num_modalities: int,
    alpha: float,
    overall_mask_ratio: float,
) -> torch.Tensor:
    """
    Sample per-modality mask ratios from a Dirichlet distribution.
    The total visible budget is distributed among modalities.

    Returns: [num_modalities] tensor of per-modality mask ratios
    """
    dirichlet = Dirichlet(torch.tensor([float(alpha)] * num_modalities))
    visible_ratio = 1.0 - overall_mask_ratio
    total_visible = visible_ratio * num_modalities
    visible_per_mod = total_visible * dirichlet.sample()
    # Clamp to [0, 1]
    mask_ratios = (1.0 - visible_per_mod).clamp(0.0, 1.0)
    return mask_ratios


def compute_mask_ratios(
    modality_names: List[str],
    observed: torch.Tensor,
    mask_ratio: float = 0.75,
    use_dirichlet: bool = True,
    dirichlet_alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Compute per-modality mask ratios, respecting observed mask.
    Missing modalities (observed=0) get mask_ratio=1.0.
    Present modalities get Dirichlet or uniform masking.

    Args:
        modality_names: list of modality names, e.g. ['T1', 'T2', 'Flair', 'PET']
        observed: [M] bool/float tensor, 1.0=present, 0.0=missing
            NOTE: This is per-sample, called once per sample in the batch.
            For simplicity, we use the same mask ratio for the whole batch
            (based on which modalities are present in the majority of the batch).
        mask_ratio: overall target mask ratio for present modalities
        use_dirichlet: whether to use Dirichlet distribution
        dirichlet_alpha: Dirichlet concentration parameter

    Returns:
        dict mapping modality_name -> mask_ratio (float)
    """
    ratios = {}
    present_mods = [name for i, name in enumerate(modality_names) if observed[i] > 0.5]
    missing_mods = [name for i, name in enumerate(modality_names) if observed[i] <= 0.5]

    # Missing modalities: fully masked
    for name in missing_mods:
        ratios[name] = 1.0

    # Present modalities: Dirichlet or uniform
    if len(present_mods) > 0:
        if use_dirichlet and len(present_mods) > 1:
            # Dirichlet masking among present modalities
            dir_ratios = generate_dirichlet_mask_ratios(
                num_modalities=len(present_mods),
                alpha=dirichlet_alpha,
                overall_mask_ratio=mask_ratio,
            )
            for i, name in enumerate(present_mods):
                ratios[name] = dir_ratios[i].item()
        else:
            # Uniform masking
            for name in present_mods:
                ratios[name] = mask_ratio

    return ratios


def mask_data(
    batch: Dict[str, torch.Tensor],
    modality_names: List[str],
    observed: torch.Tensor,
    mask_ratio: float = 0.75,
    patch_size: Union[int, Tuple[int, int, int]] = 16,
    use_dirichlet: bool = True,
    dirichlet_alpha: float = 1.0,
    patch_mask_probs: torch.Tensor = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, float],
]:
    """
    Core masking function for MultiMAE pretraining.

    For each modality:
    - Patchify the image
    - Shuffle patches (optionally weighted by anatomy importance)
    - Split into selected (visible) and masked based on mask_ratio
    - Missing modalities (observed=0) get 100% masking

    Args:
        batch: dict mapping modality name -> [B, 1, D, H, W] tensor
        modality_names: ordered list of modality names
        observed: [B, M] tensor indicating which modalities are present
        mask_ratio: target mask ratio for present modalities
        patch_size: patch size for patchification
        use_dirichlet: whether to use Dirichlet distribution
        dirichlet_alpha: Dirichlet concentration parameter
        patch_mask_probs: optional [N_patches] per-patch masking probability
            from anatomy-aware masking. When provided, uses Gumbel-top-k
            weighted sampling instead of uniform random shuffling.
            Higher probability = more likely to be masked.

    Returns:
        selected_patches: {modality: [B, num_selected, C, pd, ph, pw]}
        masked_patches: {modality: [B, num_masked, C, pd, ph, pw]}
        perm_indices: {modality: [B, num_patches]}
        mask_ratios: {modality: float}
    """
    patch_size = to_3tuple(patch_size)
    batch_size = observed.shape[0]

    # Union strategy: if ANY sample in the batch has a modality, it gets
    # partial masking. Samples where this modality is missing contribute
    # zero-valued patches (harmless in encoder, excluded from loss).
    # This ensures no information is wasted when modalities are present
    # in a minority of samples.
    batch_observed = (observed.max(dim=0).values > 0.5).float()  # [M]
    mask_ratios = compute_mask_ratios(
        modality_names=modality_names,
        observed=batch_observed,
        mask_ratio=mask_ratio,
        use_dirichlet=use_dirichlet,
        dirichlet_alpha=dirichlet_alpha,
    )

    selected_patches = {}
    masked_patches = {}
    perm_indices = {}

    for mod_name in modality_names:
        # Patchify: [B, 1, D, H, W] -> [B, num_patches, 1, pd, ph, pw]
        patches = patchify(batch[mod_name], patch_size)
        num_patches = patches.shape[1]

        # Shuffle patches (weighted by anatomy importance if provided)
        shuffled, perm_idx = shuffle_patches(patches, mask_probs=patch_mask_probs)
        perm_indices[mod_name] = perm_idx

        # Split into selected and masked
        mod_mask_ratio = mask_ratios[mod_name]
        num_selected = int((1.0 - mod_mask_ratio) * num_patches)
        # Ensure at least 0 selected (for fully masked modalities)
        num_selected = max(0, num_selected)

        selected_patches[mod_name] = shuffled[:, :num_selected]
        masked_patches[mod_name] = shuffled[:, num_selected:]

    return selected_patches, masked_patches, perm_indices, mask_ratios
