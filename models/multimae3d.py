"""
MultiMAE3D: Multi-modal Masked Autoencoder for 3D Medical Images

Architecture:
- Per-modality input adapters (Conv3D patch embedding)
- Shared ViT encoder
- Per-modality output adapters (cross-attn decoder)
- Handles arbitrary missing modalities via observed mask

Based on MultiMAE_reference, simplified for our use case:
- Fixed input size 128^3, 4 modalities (T1, T2, Flair, PET)
- Pure reconstruction pretraining (MSE loss)
- No Hydra/Lightning dependencies
"""

import copy
import math
from typing import Union, Tuple, Dict, List, Optional
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from einops import rearrange

from models.multimae3d_utils import (
    to_3tuple,
    calc_patchified_dim,
    patchify,
    unpatchify,
    shuffle_patches,
    unshuffle_patches,
    build_3d_sincos_position_embedding,
    mask_data,
)


# =============================================================================
# Input Adapter: Conv3D patch embedding (per modality)
# =============================================================================

class PatchedInputAdapter(nn.Module):
    """
    Converts a single-channel 3D volume into patch tokens.
    Input:  [B, N_selected, 1, pd, ph, pw] (selected shuffled patches)
    Output: [B, N_selected, embed_dim]
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = to_3tuple(patch_size)
        self.embed_dim = embed_dim

        # Conv3D projection: each patch -> embed_dim
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C, pd, ph, pw] selected patches (already patchified & shuffled)
        returns: [B, N, embed_dim]
        """
        B, N = x.shape[0], x.shape[1]
        # Merge batch and patch dims for Conv3D
        x = rearrange(x, "b n c d h w -> (b n) c d h w")
        x = self.proj(x)  # [(B*N), embed_dim, 1, 1, 1]
        x = x.flatten(2)  # [(B*N), embed_dim, 1]
        x = x.squeeze(-1)  # [(B*N), embed_dim]
        x = rearrange(x, "(b n) d -> b n d", b=B)
        return x


# =============================================================================
# Cross Attention (for decoder)
# =============================================================================

class CrossAttention(nn.Module):
    """Cross attention: query attends to context (encoder output)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Transformer blocks with attention mask support
# =============================================================================

class Mlp(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MaskedAttention(nn.Module):
    """Multi-head self-attention with optional additive attention mask."""

    def __init__(self, dim, num_heads=12, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, C]
        attn_mask: [B, 1, 1, N] additive mask, -inf for tokens to ignore (column masking)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(nn.Module):
    """Pre-LN Transformer block with optional attention mask support.
    Used for both encoder (with mask) and decoder (without mask).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MaskedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Cross-Modal Predictor (for cross-level mutual prediction)
# =============================================================================

class CrossModalPredictor(nn.Module):
    """3-layer MLP predictor for cross-modal feature prediction.

    Maps features from one modality space to another.
    Structure: Linear(D, 2D) → GELU → Linear(2D, 2D) → GELU → Linear(2D, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Output Adapter: Decoder (per modality)
# =============================================================================

class SpatialOutputAdapter(nn.Module):
    """
    Per-modality decoder.
    Takes encoder tokens, adds mask tokens, applies cross-attention + self-attention,
    then projects back to patch pixel space.

    Architecture:
    1. Project encoder tokens from encoder_dim -> decoder_dim
    2. Create mask tokens for masked positions
    3. Add positional embedding to query (mask + selected tokens)
    4. Cross-attention: query attends to encoder context
    5. Self-attention transformer blocks
    6. Linear projection to patch pixel dimension
    """

    def __init__(
        self,
        out_channels: int = 1,
        img_size: Union[int, Tuple[int, int, int]] = 128,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        encoder_embed_dim: int = 768,
        embed_dim: int = 384,
        num_heads: int = 12,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        self.patchified_dim = calc_patchified_dim(self.img_size, self.patch_size)
        self.num_patches = self.patchified_dim[0] * self.patchified_dim[1] * self.patchified_dim[2]

        # Project encoder tokens to decoder dimension
        self.proj_context = nn.Linear(encoder_embed_dim, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (sincos, frozen)
        self.pos_embed = build_3d_sincos_position_embedding(
            grid_size=self.patchified_dim,
            embed_dim=embed_dim,
        )

        # Cross-attention + MLP (MultiMAE style)
        self.xattn = CrossAttention(
            dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.context_norm = norm_layer(embed_dim)
        self.query_norm = norm_layer(embed_dim)
        self.out_norm = norm_layer(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

        # Self-attention transformer blocks (decoder: no attention mask needed)
        self.blocks = nn.Sequential(*[
            MaskedBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ]) if depth > 0 else nn.Identity()

        # Output projection: decoder_dim -> patch_pixels
        dim_patch = self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * out_channels
        self.out_proj = nn.Linear(embed_dim, dim_patch)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        task_range: Tuple[int, int],
        perm_idx: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Args:
            encoder_tokens: [B, total_visible_tokens, encoder_dim] (last layer output)
            task_range: (start, end) indices of this modality's tokens in the concat
            perm_idx: [B, num_patches] permutation indices for this modality
            num_patches: total number of patches for this modality

        Returns:
            output: [B, num_patches, out_channels, pd, ph, pw] (all patches, unshuffled order)
        """
        B = encoder_tokens.shape[0]

        # 1. Project encoder tokens to decoder dim
        context = self.proj_context(encoder_tokens)

        # 2. Extract this modality's selected tokens from the context
        num_selected = task_range[1] - task_range[0]
        selected_tokens = context[:, task_range[0]:task_range[1]]

        # 3. Create mask tokens for masked positions
        num_masked = num_patches - num_selected
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # 4. Concatenate: [selected, masked] in shuffled order
        query = torch.cat([selected_tokens, mask_tokens], dim=1)  # [B, num_patches, dim]

        # 5. Add positional embedding (following the permutation order)
        pos_emb = self.pos_embed.expand(B, -1, -1)  # [B, num_patches, dim]
        pos_emb_shuffled = pos_emb[torch.arange(B, device=pos_emb.device)[:, None], perm_idx]
        query = query + pos_emb_shuffled

        # 6. Cross-attention + MLP
        x = self.xattn(self.query_norm(query), self.context_norm(context))
        x = x + self.mlp(self.out_norm(x))

        # 7. Self-attention blocks
        if self.depth > 0:
            x = self.blocks(x)

        # 8. Project to patch pixel space
        x = self.out_proj(x)  # [B, num_patches, patch_pixels]

        # 9. Reshape to patch format
        x = rearrange(
            x,
            "b n (c pd ph pw) -> b n c pd ph pw",
            c=self.out_channels,
            pd=self.patch_size[0],
            ph=self.patch_size[1],
            pw=self.patch_size[2],
        )

        # 10. Unshuffle back to spatial order
        x = unshuffle_patches(x, perm_idx)

        return x


# =============================================================================
# MultiMAE3D: Main Model
# =============================================================================

class MultiMAE3D(nn.Module):
    """
    Multi-modal Masked Autoencoder for 3D Medical Images.

    Handles 4 modalities (T1, T2, Flair, PET) with arbitrary missing modalities.

    Forward pass:
    1. Split stacked input into per-modality volumes
    2. Patchify and mask each modality (missing → 100% masked)
    3. Tokenize visible patches via per-modality input adapters
    4. Add positional embeddings + CLS token
    5. Concatenate all visible tokens → shared ViT encoder
    6. Per-modality decoder → reconstruct masked patches
    7. Compute MSE loss only on present modalities' masked patches
    """

    MODALITY_NAMES = ["T1", "T2", "Flair", "PET"]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 128,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 2,
        decoder_num_heads: int = 12,
        mask_ratio: float = 0.75,
        use_dirichlet: bool = True,
        dirichlet_alpha: float = 1.0,
        num_global_tokens: int = 1,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        enable_cross_modal: bool = False,
    ):
        super().__init__()

        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.mask_ratio = mask_ratio
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.num_global_tokens = num_global_tokens
        self.enable_cross_modal = enable_cross_modal

        self.patchified_dim = calc_patchified_dim(self.img_size, self.patch_size)
        self.num_patches = self.patchified_dim[0] * self.patchified_dim[1] * self.patchified_dim[2]

        # ----- Input adapters (per modality) -----
        self.input_adapters = nn.ModuleDict({
            name: PatchedInputAdapter(
                in_channels=1,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )
            for name in self.MODALITY_NAMES
        })

        # ----- Encoder positional embedding (sincos, frozen) -----
        self.pos_embed = build_3d_sincos_position_embedding(
            grid_size=self.patchified_dim,
            embed_dim=embed_dim,
        )

        # ----- CLS token -----
        if num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.zeros(num_global_tokens, embed_dim))
            nn.init.normal_(self.global_tokens, std=0.02)

        # ----- Shared Transformer encoder (ModuleList for attn_mask support) -----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.encoder = nn.ModuleList([
            MaskedBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                act_layer=nn.GELU,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # ----- Output adapters / decoders (per modality) -----
        self.output_adapters = nn.ModuleDict({
            name: SpatialOutputAdapter(
                out_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                encoder_embed_dim=embed_dim,
                embed_dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                depth=decoder_depth,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
            for name in self.MODALITY_NAMES
        })

        # Initialize weights
        self._initialize_weights()

        # ----- Cross-modal mutual prediction components -----
        if self.enable_cross_modal:
            # Teacher encoder (EMA copy of student) — no gradients
            self.teacher_input_adapters = copy.deepcopy(self.input_adapters)
            for p in self.teacher_input_adapters.parameters():
                p.requires_grad = False

            self.teacher_encoder = copy.deepcopy(self.encoder)
            for p in self.teacher_encoder.parameters():
                p.requires_grad = False

            # Teacher global tokens stored as buffer (auto-moves with .to(device))
            if self.num_global_tokens > 0:
                self.register_buffer(
                    "teacher_global_tokens",
                    self.global_tokens.data.clone(),
                )

            # Cross-modal predictors (student-only, learnable)
            self.predictor_mri_to_pet = CrossModalPredictor(embed_dim)
            self.predictor_pet_to_mri = CrossModalPredictor(embed_dim)
            # Initialize predictor weights
            self.predictor_mri_to_pet.apply(self._init_weights)
            self.predictor_pet_to_mri.apply(self._init_weights)

    def _initialize_weights(self):
        self.apply(self._init_weights)
        # Special init for Conv3D projection (following MAE)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
            if isinstance(m, nn.Conv3d):
                if ".proj" in name:
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _split_modalities(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split stacked [B, 4, D, H, W] into per-modality dict {name: [B, 1, D, H, W]}."""
        return {
            name: images[:, i:i+1]
            for i, name in enumerate(self.MODALITY_NAMES)
        }

    # -----------------------------------------------------------------
    # Cross-modal mutual prediction helpers
    # -----------------------------------------------------------------

    def _encode_with(
        self,
        selected_patches: Dict[str, torch.Tensor],
        perm_indices: Dict[str, torch.Tensor],
        observed: torch.Tensor,
        input_adapters: nn.ModuleDict,
        global_tokens,
        encoder_blocks: nn.ModuleList,
    ):
        """
        Shared encoding logic used by both student and teacher.

        Returns:
            encoder_output: [B, total_tokens, D] or None
            task_ranges: OrderedDict {modality_name: (start, end)}
        """
        B = observed.shape[0]
        device = observed.device

        tokens = {}
        for name in self.MODALITY_NAMES:
            sel = selected_patches[name]
            if sel.shape[1] == 0:
                continue
            tok = input_adapters[name](sel)
            perm = perm_indices[name]
            pos_emb = self.pos_embed.expand(B, -1, -1)
            pos_emb_selected = pos_emb[
                torch.arange(B, device=device)[:, None], perm[:, :sel.shape[1]]
            ]
            tok = tok + pos_emb_selected
            tokens[name] = tok

        token_list = []
        task_ranges = OrderedDict()
        offset = self.num_global_tokens

        for name in self.MODALITY_NAMES:
            if name in tokens:
                n_tok = tokens[name].shape[1]
                task_ranges[name] = (offset, offset + n_tok)
                token_list.append(tokens[name])
                offset += n_tok
            else:
                task_ranges[name] = (offset, offset)

        if len(token_list) == 0:
            return None, task_ranges

        input_tokens = torch.cat(token_list, dim=1)

        if self.num_global_tokens > 0 and global_tokens is not None:
            if global_tokens.dim() == 2:
                cls = global_tokens.unsqueeze(0).expand(B, -1, -1)
            else:
                cls = global_tokens.expand(B, -1, -1)
            input_tokens = torch.cat([cls, input_tokens], dim=1)

        # Column masking for missing modalities
        total_tokens = input_tokens.shape[1]
        attn_mask = torch.zeros(B, 1, 1, total_tokens, device=device)
        for i, name in enumerate(self.MODALITY_NAMES):
            start, end = task_ranges[name]
            if start == end:
                continue
            missing = (observed[:, i] < 0.5)
            if missing.any():
                attn_mask[missing, :, :, start:end] = float("-inf")
        if (attn_mask == 0).all():
            attn_mask = None

        encoder_output = input_tokens
        for block in encoder_blocks:
            encoder_output = block(encoder_output, attn_mask=attn_mask)

        return encoder_output, task_ranges

    def _compute_cross_modal_loss(
        self,
        selected_patches: Dict[str, torch.Tensor],
        perm_indices: Dict[str, torch.Tensor],
        observed: torch.Tensor,
        student_encoder_output: torch.Tensor,
        task_ranges: OrderedDict,
    ) -> torch.Tensor:
        """
        Cross-level mutual prediction loss (simplified global-average-pooling version).

        Two groups:
          - MRI group: all T1 + T2 + Flair tokens  →  z_MRI  (D-dim vector)
          - PET group: all PET tokens               →  z_PET  (D-dim vector)

        Predictions (student → teacher):
          - predictor_mri_to_pet(z_MRI_student) → predict z_PET_teacher
          - predictor_pet_to_mri(z_PET_student) → predict z_MRI_teacher

        Loss: negative cosine similarity, averaged over paired samples only.
        """
        B = observed.shape[0]
        device = observed.device

        # Paired = has at least one MRI modality AND PET
        has_mri = (observed[:, :3].sum(dim=1) > 0.5)  # [B]
        has_pet = (observed[:, 3] > 0.5)               # [B]
        is_paired = has_mri & has_pet                   # [B]

        if not is_paired.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # --- Teacher forward (no gradients) ---
        with torch.no_grad():
            teacher_gt = (
                self.teacher_global_tokens
                if self.num_global_tokens > 0 else None
            )
            teacher_output, _ = self._encode_with(
                selected_patches, perm_indices, observed,
                self.teacher_input_adapters, teacher_gt,
                self.teacher_encoder,
            )
        if teacher_output is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # --- Build group masks [B, L] ---
        total_tokens = student_encoder_output.shape[1]
        mri_mask = torch.zeros(B, total_tokens, device=device)
        pet_mask = torch.zeros(B, total_tokens, device=device)

        # MRI group: T1 (idx 0), T2 (idx 1), Flair (idx 2)
        for idx, name in enumerate(["T1", "T2", "Flair"]):
            start, end = task_ranges[name]
            if start < end:
                mri_mask[:, start:end] = observed[:, idx:idx+1].expand(-1, end - start)

        # PET group: idx 3
        start, end = task_ranges["PET"]
        if start < end:
            pet_mask[:, start:end] = observed[:, 3:4].expand(-1, end - start)

        # --- Global average pooling per group ---
        mri_count = mri_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pet_count = pet_mask.sum(dim=1, keepdim=True).clamp(min=1)

        z_mri_s = (student_encoder_output * mri_mask.unsqueeze(-1)).sum(dim=1) / mri_count  # [B, D]
        z_pet_s = (student_encoder_output * pet_mask.unsqueeze(-1)).sum(dim=1) / pet_count  # [B, D]

        z_mri_t = (teacher_output * mri_mask.unsqueeze(-1)).sum(dim=1) / mri_count  # [B, D]
        z_pet_t = (teacher_output * pet_mask.unsqueeze(-1)).sum(dim=1) / pet_count  # [B, D]

        # --- L2 normalize onto unit hypersphere ---
        z_mri_s = F.normalize(z_mri_s, dim=-1)
        z_pet_s = F.normalize(z_pet_s, dim=-1)
        z_mri_t = F.normalize(z_mri_t, dim=-1)
        z_pet_t = F.normalize(z_pet_t, dim=-1)

        # --- Cross-modal predictions + normalize ---
        pred_pet = F.normalize(self.predictor_mri_to_pet(z_mri_s), dim=-1)  # [B, D]
        pred_mri = F.normalize(self.predictor_pet_to_mri(z_pet_s), dim=-1)  # [B, D]

        # --- Negative cosine similarity: L = 2 - 2·cos(pred, target) ---
        loss_m2p = 2 - 2 * (pred_pet * z_pet_t.detach()).sum(dim=-1)  # [B]
        loss_p2m = 2 - 2 * (pred_mri * z_mri_t.detach()).sum(dim=-1)  # [B]

        # Average only over paired samples
        paired_f = is_paired.float()
        n_paired = paired_f.sum().clamp(min=1)

        loss_m2p = (loss_m2p * paired_f).sum() / n_paired
        loss_p2m = (loss_p2m * paired_f).sum() / n_paired

        return 0.5 * (loss_m2p + loss_p2m)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA update: θ_teacher ← m·θ_teacher + (1-m)·θ_student."""
        if not self.enable_cross_modal:
            return

        for p_s, p_t in zip(
            self.input_adapters.parameters(),
            self.teacher_input_adapters.parameters(),
        ):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)

        if self.num_global_tokens > 0:
            self.teacher_global_tokens.mul_(momentum).add_(
                self.global_tokens.data, alpha=1 - momentum
            )

        for p_s, p_t in zip(
            self.encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)

    @torch.no_grad()
    def init_teacher_from_student(self):
        """Copy current student weights to teacher (call after loading checkpoint)."""
        if not self.enable_cross_modal:
            return

        for p_s, p_t in zip(
            self.input_adapters.parameters(),
            self.teacher_input_adapters.parameters(),
        ):
            p_t.data.copy_(p_s.data)

        if self.num_global_tokens > 0:
            self.teacher_global_tokens.copy_(self.global_tokens.data)

        for p_s, p_t in zip(
            self.encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            p_t.data.copy_(p_s.data)

    def forward(
        self,
        images: torch.Tensor,
        observed: torch.Tensor,
        return_loss: bool = True,
        patch_mask_probs: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 4, D, H, W] stacked multi-modal 3D volumes
            observed: [B, 4] float tensor, 1.0=present, 0.0=missing
            return_loss: if True, compute and return reconstruction loss
            patch_mask_probs: optional [N_patches] per-patch masking probability
                from anatomy-aware masking (higher = more likely to be masked)

        Returns:
            dict with:
                'loss': scalar MSE loss (if return_loss=True)
                'per_modality_loss': {name: loss} for each present modality
                'mask_ratios': {name: float} actual mask ratios used
        """
        B = images.shape[0]
        device = images.device

        # 1. Split into per-modality dict
        batch = self._split_modalities(images)

        # 2. Mask data (patchify + shuffle + split)
        #    When patch_mask_probs is provided, uses anatomy-aware weighted sampling
        selected_patches, masked_patches, perm_indices, mask_ratios = mask_data(
            batch=batch,
            modality_names=self.MODALITY_NAMES,
            observed=observed,
            mask_ratio=self.mask_ratio,
            patch_size=self.patch_size,
            use_dirichlet=self.use_dirichlet if self.training else False,
            dirichlet_alpha=self.dirichlet_alpha,
            patch_mask_probs=patch_mask_probs if self.training else None,
        )

        # 3-6. Student encoding (tokenize → concat → attn mask → encoder)
        encoder_output, task_ranges = self._encode_with(
            selected_patches, perm_indices, observed,
            self.input_adapters, self.global_tokens, self.encoder,
        )

        if encoder_output is None:
            return {
                "loss": torch.tensor(0.0, device=device),
                "cross_modal_loss": torch.tensor(0.0, device=device),
                "per_modality_loss": {},
                "mask_ratios": mask_ratios,
            }

        # 7. Per-modality decoder
        reconstructed = {}
        for name in self.MODALITY_NAMES:
            reconstructed[name] = self.output_adapters[name](
                encoder_tokens=encoder_output,
                task_range=task_ranges[name],
                perm_idx=perm_indices[name],
                num_patches=self.num_patches,
            )
            # reconstructed[name]: [B, num_patches, 1, pd, ph, pw] in spatial order

        # 8. Compute reconstruction loss (MSE, only on present modalities' masked patches)
        if return_loss:
            total_loss = torch.tensor(0.0, device=device)
            per_mod_loss = {}
            num_present = 0

            for i, name in enumerate(self.MODALITY_NAMES):
                # Only compute loss on present modalities
                mod_observed = observed[:, i]  # [B]
                if mod_observed.sum() < 0.5:
                    continue

                # Ground truth: all patches in spatial order
                gt_patches = patchify(batch[name], self.patch_size)  # [B, num_patches, 1, pd, ph, pw]
                pred_patches = reconstructed[name]  # [B, num_patches, 1, pd, ph, pw]

                # Create per-patch mask: 1 = masked (should reconstruct), 0 = visible
                perm = perm_indices[name]
                num_selected = selected_patches[name].shape[1]
                # In shuffled order: first num_selected are visible, rest masked
                # Convert to spatial order mask (vectorized, no Python loop)
                mask = torch.ones(B, self.num_patches, device=device)
                if num_selected > 0:
                    selected_perm = perm[:, :num_selected]  # [B, num_selected]
                    mask.scatter_(1, selected_perm, 0.0)

                # Per-sample observed mask: zero out loss for missing samples
                sample_mask = mod_observed.float()  # [B]

                # Patch normalization (per-patch zero-mean unit-variance, like original MAE)
                gt_mean = gt_patches.mean(dim=(2, 3, 4, 5), keepdim=True)
                gt_var = gt_patches.var(dim=(2, 3, 4, 5), keepdim=True)
                gt_patches_norm = (gt_patches - gt_mean) / (gt_var + 1e-6).sqrt()

                # Compute MSE on masked patches only (against normalized targets)
                per_patch_mse = ((pred_patches - gt_patches_norm) ** 2).mean(dim=(2, 3, 4, 5))  # [B, num_patches]
                masked_mse = (per_patch_mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B]
                mod_loss = (masked_mse * sample_mask).sum() / sample_mask.sum().clamp(min=1)

                per_mod_loss[name] = mod_loss
                total_loss = total_loss + mod_loss
                num_present += 1

            if num_present > 0:
                total_loss = total_loss / num_present

            # 9. Cross-modal mutual prediction loss
            cross_modal_loss = torch.tensor(0.0, device=device)
            if self.enable_cross_modal:
                cross_modal_loss = self._compute_cross_modal_loss(
                    selected_patches, perm_indices, observed,
                    encoder_output, task_ranges,
                )

            return {
                "loss": total_loss,
                "cross_modal_loss": cross_modal_loss,
                "per_modality_loss": per_mod_loss,
                "mask_ratios": mask_ratios,
            }

        return {
            "reconstructed": reconstructed,
            "cross_modal_loss": torch.tensor(0.0, device=device),
            "mask_ratios": mask_ratios,
        }

    def encode(
        self,
        images: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode without masking (for downstream use).
        Returns encoder output tokens [B, num_global + 4*num_patches, embed_dim].
        """
        B = images.shape[0]
        device = images.device
        batch = self._split_modalities(images)

        tokens_list = []
        offset = self.num_global_tokens

        for i, name in enumerate(self.MODALITY_NAMES):
            img = batch[name]  # [B, 1, D, H, W]
            patches = patchify(img, self.patch_size)  # [B, num_patches, 1, pd, ph, pw]

            # Tokenize all patches (no masking)
            tok = self.input_adapters[name](patches)  # [B, num_patches, embed_dim]

            # Add positional embedding
            pos_emb = self.pos_embed.expand(B, -1, -1)
            tok = tok + pos_emb

            # Zero out tokens for missing modalities
            mod_mask = observed[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
            tok = tok * mod_mask

            tokens_list.append(tok)
            offset += self.num_patches

        input_tokens = torch.cat(tokens_list, dim=1)

        # Add CLS token
        if self.num_global_tokens > 0:
            cls = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
            input_tokens = torch.cat([cls, input_tokens], dim=1)

        # Build attention mask: prevent attending to tokens from missing modalities
        total_tokens = input_tokens.shape[1]
        attn_mask = torch.zeros(B, 1, 1, total_tokens, device=device)
        mod_offset = self.num_global_tokens
        for i, name in enumerate(self.MODALITY_NAMES):
            start = mod_offset
            end = mod_offset + self.num_patches
            missing = (observed[:, i] < 0.5)  # [B]
            if missing.any():
                attn_mask[missing, :, :, start:end] = float("-inf")
            mod_offset = end
        if (attn_mask == 0).all():
            attn_mask = None

        # Encode with attention mask
        encoder_output = input_tokens
        for block in self.encoder:
            encoder_output = block(encoder_output, attn_mask=attn_mask)

        return encoder_output


def create_multimae3d(
    img_size: int = 128,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    decoder_embed_dim: int = 384,
    decoder_depth: int = 2,
    decoder_num_heads: int = 12,
    mask_ratio: float = 0.75,
    use_dirichlet: bool = True,
    enable_cross_modal: bool = False,
    **kwargs,
) -> MultiMAE3D:
    """Factory function to create MultiMAE3D with default ViT-B config."""
    return MultiMAE3D(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mask_ratio=mask_ratio,
        use_dirichlet=use_dirichlet,
        enable_cross_modal=enable_cross_modal,
        **kwargs,
    )
