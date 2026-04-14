"""FACET v3: Local-biased transformer for backbone torsion prediction.

Architecture:
  - Input: secondary shifts (6) + masks (6) + AA embedding (32) + flags (6) = 50 per residue
  - Encoder: 3 dilated conv1d layers (receptive field ±4) → 2 RoPE self-attention layers
  - Heads: coarse-to-fine Ramachandran (36×36 grid + circular residual),
           SS (H/E/C) with soft conditioning, order/disorder, angle auxiliary
  - ~1.5M parameters

Designed for extension:
  - Phase 2: retrieval tokens via cross-attention (add retrieval_tokens input to encoder)
  - Phase 3: pre-training with masked shift reconstruction (freeze encoder, swap heads)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Coarse Ramachandran grid utilities
# ─────────────────────────────────────────────────────────────

RAMA_GRID = 36  # 36×36 = 1296 bins, each 10° × 10°
RAMA_N_BINS = RAMA_GRID * RAMA_GRID  # 1296
RAMA_BIN_WIDTH = 2 * math.pi / RAMA_GRID  # ~0.1745 rad = 10°


def angle_to_bin(angle_rad: torch.Tensor) -> torch.Tensor:
    """Convert angle in radians [-π, π) to bin index [0, 35]."""
    a = (angle_rad + math.pi) % (2 * math.pi)
    return (a / RAMA_BIN_WIDTH).long().clamp(0, RAMA_GRID - 1)


def angles_to_bin_idx(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Convert (phi, psi) to flat bin index [0, 1295]."""
    return angle_to_bin(phi) * RAMA_GRID + angle_to_bin(psi)


def bin_idx_to_centers(bin_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flat bin index to (phi_center, psi_center) in radians."""
    phi_bin = bin_idx // RAMA_GRID
    psi_bin = bin_idx % RAMA_GRID
    phi_c = -math.pi + (phi_bin.float() + 0.5) * RAMA_BIN_WIDTH
    psi_c = -math.pi + (psi_bin.float() + 0.5) * RAMA_BIN_WIDTH
    return phi_c, psi_c


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class FACETv3Config:
    """Configuration for FACET v3."""

    # Input
    n_atoms: int = 6  # H, HA, N, CA, CB, C
    n_aa: int = 21  # 20 AAs + padding
    aa_embed_dim: int = 32
    n_flags: int = 6  # gly, pro, pre_pro, aromatic_nbr, aromatic, missing_nbr

    # Encoder
    hidden_dim: int = 128
    n_conv_layers: int = 3
    conv_kernel: int = 5
    conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    n_attn_layers: int = 2
    n_attn_heads: int = 4
    dropout: float = 0.1

    # Coarse-to-fine torsion head
    rama_grid: int = 36  # 36×36 = 1296 bins
    fine_kappa_max: float = 50.0  # within-bin vM, never needs to be large

    # SS head
    n_ss_classes: int = 3  # H, E, C
    ss_condition_dim: int = 32  # projection before conditioning torsion

    # Order head
    order_classes: int = 1  # binary logit

    # Retrieval augmentation (Phase 2)
    retrieval_dim: int = 0  # 0 = no retrieval, 9 = retrieval summary features

    # Training
    context_dropout: float = 0.15
    atom_dropout: float = 0.10

    @property
    def input_dim(self) -> int:
        return self.n_atoms * 2 + self.aa_embed_dim + self.n_flags  # 6+6+32+6 = 50


# ─────────────────────────────────────────────────────────────
# Rotary positional embedding
# ─────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """RoPE for 1D sequences."""

    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, D/2)
        return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor of shape (B, H, L, D)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    # Broadcast cos/sin from (L, D/2) to (1, 1, L, D/2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ─────────────────────────────────────────────────────────────
# Encoder components
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Dilated causal conv1d with pre-norm residual."""

    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=padding, dilation=dilation)
        self.gate_conv = nn.Conv1d(dim, dim, kernel, padding=padding, dilation=dilation)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x) * torch.sigmoid(self.gate_conv(x))  # GLU-style gating
        x = self.proj(x)
        x = x.transpose(1, 2)  # (B, L, D)
        return residual + self.dropout(x)


class RoPEAttentionLayer(nn.Module):
    """Multi-head self-attention with RoPE and pre-norm."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, L, D_h)

        # Apply RoPE to q, k
        cos, sin = self.rope(L, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return residual + self.dropout(self.out(out))


class FeedForward(nn.Module):
    """Pre-norm feed-forward with SwiGLU."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, inner)
        self.w2 = nn.Linear(dim, inner)
        self.w3 = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return residual + self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ─────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────

class CoarseFineRamaHead(nn.Module):
    """Coarse-to-fine Ramachandran torsion head.

    Coarse: categorical over 36×36 = 1296 bins (10° resolution).
    Fine: circular residual + small von Mises concentration within the bin.

    No I₀ overflow possible: kappa is capped at ~50, used only for
    within-bin (±5°) precision. Coarse CE handles basin selection.
    """

    def __init__(self, dim: int, grid: int = 36, fine_kappa_max: float = 50.0):
        super().__init__()
        n_bins = grid * grid
        self.grid = grid
        self.n_bins = n_bins
        self.fine_kappa_max = fine_kappa_max

        # Coarse: hidden → bins
        self.coarse_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_bins),
        )
        # Fine: hidden → (sin_dphi, cos_dphi, sin_dpsi, cos_dpsi, kappa_phi, kappa_psi)
        self.fine_head = nn.Linear(dim, 6)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            h: (B, D) hidden state

        Returns:
            coarse_logits: (B, 1296)
            fine_delta_phi, fine_delta_psi: (B,) residual angles in radians
            fine_kappa_phi, fine_kappa_psi: (B,) within-bin concentrations
        """
        coarse_logits = self.coarse_head(h)  # (B, 1296)

        fine_raw = self.fine_head(h)  # (B, 6)
        # Residual as atan2(sin, cos) — unconstrained, wraps naturally
        fine_delta_phi = torch.atan2(fine_raw[:, 0], fine_raw[:, 1])
        fine_delta_psi = torch.atan2(fine_raw[:, 2], fine_raw[:, 3])
        # Clamp residual to ±half bin width (no point predicting outside the bin)
        half_bw = RAMA_BIN_WIDTH / 2
        fine_delta_phi = fine_delta_phi.clamp(-half_bw, half_bw)
        fine_delta_psi = fine_delta_psi.clamp(-half_bw, half_bw)
        # Small kappa for within-bin precision
        fine_kappa_phi = F.softplus(fine_raw[:, 4]).clamp(max=self.fine_kappa_max) + 0.1
        fine_kappa_psi = F.softplus(fine_raw[:, 5]).clamp(max=self.fine_kappa_max) + 0.1

        return {
            "coarse_logits": coarse_logits,
            "fine_delta_phi": fine_delta_phi,
            "fine_delta_psi": fine_delta_psi,
            "fine_kappa_phi": fine_kappa_phi,
            "fine_kappa_psi": fine_kappa_psi,
        }


class SSHead(nn.Module):
    """Secondary structure prediction: H/E/C."""

    def __init__(self, dim: int, n_classes: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 3), NOT log-softmax."""
        return self.head(h)


class OrderHead(nn.Module):
    """Order/disorder binary prediction."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logit (B, 1)."""
        return self.head(h)


class Chi1Head(nn.Module):
    """3-class chi1 rotamer prediction (g+, g-, trans)."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 3)."""
        return self.head(h)


class AngleAuxHead(nn.Module):
    """Auxiliary sin/cos angle regression for gradient sharpening."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 4)  # sin_phi, cos_phi, sin_psi, cos_psi

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


# ─────────────────────────────────────────────────────────────
# FACET v3 Model
# ─────────────────────────────────────────────────────────────

class FACETv3(nn.Module):
    """FACET v3: local-biased transformer for backbone torsion prediction.

    Encoder:
      dilated conv stem (3 layers, GLU gating) → 2 RoPE self-attention + FFN layers

    Heads:
      - Coarse-to-fine Ramachandran (36×36 grid + circular residual)
      - SS (H/E/C) with soft conditioning into torsion head
      - Order/disorder
      - Angle auxiliary (sin/cos regression)

    Architected for Phase 2/3 extension:
      - Retrieval: add cross-attention layer after self-attention
      - Pre-training: swap heads for masked shift reconstruction
    """

    def __init__(self, config: FACETv3Config | None = None):
        super().__init__()
        if config is None:
            config = FACETv3Config()
        self.config = config
        D = config.hidden_dim

        # Input projection
        self.aa_embed = nn.Embedding(config.n_aa, config.aa_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(config.input_dim, D)

        # Local conv stem
        self.conv_layers = nn.ModuleList([
            ConvBlock(D, config.conv_kernel, d, config.dropout)
            for d in config.conv_dilations
        ])

        # Global self-attention layers
        self.attn_layers = nn.ModuleList()
        for _ in range(config.n_attn_layers):
            self.attn_layers.append(nn.ModuleList([
                RoPEAttentionLayer(D, config.n_attn_heads, config.dropout),
                FeedForward(D, mult=4, dropout=config.dropout),
            ]))

        self.encoder_norm = nn.LayerNorm(D)

        # SS head (predicts first, conditions torsion)
        self.ss_head = SSHead(D, config.n_ss_classes)
        # SS soft conditioning: project SS logits → hidden dim for additive modulation
        self.ss_proj = nn.Sequential(
            nn.Linear(config.n_ss_classes, config.ss_condition_dim),
            nn.GELU(),
            nn.Linear(config.ss_condition_dim, D),
            nn.Sigmoid(),  # gate in [0, 1]
        )

        # Torsion head (operates on SS-conditioned hidden state)
        self.rama_head = CoarseFineRamaHead(D, config.rama_grid, config.fine_kappa_max)

        # Retrieval projection (Phase 2): project retrieval features and fuse
        if config.retrieval_dim > 0:
            self.retrieval_proj = nn.Sequential(
                nn.Linear(config.retrieval_dim, D),
                nn.GELU(),
                nn.Linear(D, D),
            )

        # Auxiliary heads (operate on unconditioned hidden state)
        self.order_head = OrderHead(D)
        self.angle_aux = AngleAuxHead(D)
        self.chi1_head = Chi1Head(D)

    def encode(
        self,
        shifts: torch.Tensor,
        masks: torch.Tensor,
        aa_idx: torch.Tensor,
        flags: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a protein sequence.

        Args:
            shifts: (B, L, 6) secondary shifts
            masks: (B, L, 6) observation masks
            aa_idx: (B, L) AA indices (1-20, 0=pad)
            flags: (B, L, 6) special flags

        Returns:
            (B, L, D) encoded representations
        """
        # Build per-residue features
        aa_feat = self.aa_embed(aa_idx)  # (B, L, 32)
        x = torch.cat([shifts, masks, aa_feat, flags], dim=-1)  # (B, L, 50)
        x = self.input_proj(x)  # (B, L, D)

        # Local conv stem
        for conv in self.conv_layers:
            x = conv(x)

        # Global self-attention
        for attn, ffn in self.attn_layers:
            x = attn(x)
            x = ffn(x)

        return self.encoder_norm(x)

    def forward(
        self,
        shifts: torch.Tensor,
        masks: torch.Tensor,
        aa_idx: torch.Tensor,
        flags: torch.Tensor,
        center_idx: int | None = None,
        retrieval_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            shifts: (B, L, 6) secondary shifts
            masks: (B, L, 6) observation masks
            aa_idx: (B, L) AA indices
            flags: (B, L, 6) special flags
            center_idx: position of center residue to predict.
                If None, uses L//2 (pentapeptide mode).
            retrieval_features: (B, retrieval_dim) optional retrieval summary.

        Returns:
            Dict with coarse_logits, fine_delta_phi/psi, fine_kappa_phi/psi,
            ss_logits, order_logits, angle_aux, confidence
        """
        h_seq = self.encode(shifts, masks, aa_idx, flags)  # (B, L, D)

        # Extract center residue
        if center_idx is None:
            center_idx = shifts.shape[1] // 2
        h = h_seq[:, center_idx, :]  # (B, D)

        # Fuse retrieval evidence (additive, before any head)
        if retrieval_features is not None and hasattr(self, "retrieval_proj"):
            h = h + self.retrieval_proj(retrieval_features)

        # SS prediction (from retrieval-augmented encoder output)
        ss_logits = self.ss_head(h)  # (B, 3)

        # Soft SS conditioning: modulate hidden state before torsion head
        ss_gate = self.ss_proj(ss_logits)  # (B, D) values in [0, 1]
        h_cond = h * (0.5 + ss_gate)  # multiplicative gating centered at 1.0

        # Torsion prediction (from SS-conditioned state)
        rama = self.rama_head(h_cond)

        # Auxiliary predictions (from unconditioned state)
        order_logits = self.order_head(h)  # (B, 1)
        angle_sincos = self.angle_aux(h)  # (B, 4)
        chi1_logits = self.chi1_head(h)  # (B, 3)

        # Confidence: negative entropy of coarse distribution
        coarse_probs = F.softmax(rama["coarse_logits"], dim=-1)
        log_probs = F.log_softmax(rama["coarse_logits"], dim=-1)
        entropy = -(coarse_probs * log_probs).sum(dim=-1)  # (B,)
        # Scale: max entropy = log(1296) ≈ 7.17; confident ≈ 0-2
        confidence = -entropy  # higher = more confident

        return {
            **rama,
            "ss_logits": ss_logits,
            "order_logits": order_logits,
            "angle_sincos": angle_sincos,
            "chi1_logits": chi1_logits,
            "confidence": confidence,
            "coarse_probs": coarse_probs,
        }

    def predict(
        self,
        shifts: torch.Tensor,
        masks: torch.Tensor,
        aa_idx: torch.Tensor,
        flags: torch.Tensor,
        center_idx: int | None = None,
        retrieval_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference: decode (phi, psi) from coarse + fine.

        Returns dict with phi, psi (radians), ss_pred, confidence, coarse_probs.
        """
        with torch.no_grad():
            out = self.forward(shifts, masks, aa_idx, flags, center_idx, retrieval_features)

        # Decode: top coarse bin → center + fine residual
        top_bin = out["coarse_logits"].argmax(dim=-1)  # (B,)
        phi_c, psi_c = bin_idx_to_centers(top_bin)
        phi_c, psi_c = phi_c.to(shifts.device), psi_c.to(shifts.device)

        phi = phi_c + out["fine_delta_phi"]
        psi = psi_c + out["fine_delta_psi"]

        # Wrap to [-π, π)
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        psi = (psi + math.pi) % (2 * math.pi) - math.pi

        return {
            "phi": phi,
            "psi": psi,
            "ss_pred": out["ss_logits"].argmax(dim=-1),
            "chi1_pred": out["chi1_logits"].argmax(dim=-1),
            "chi1_probs": F.softmax(out["chi1_logits"], dim=-1),
            "confidence": out["confidence"],
            "coarse_probs": out["coarse_probs"],
        }


