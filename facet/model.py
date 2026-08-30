"""FACET v3: Local-biased transformer for backbone torsion prediction.

Architecture:
  - Input: secondary shifts (6) + masks (6) + AA embedding (32) + flags (6) = 50 per residue
  - Encoder: 3 dilated conv1d layers (kernel 5, dilations 1/2/4: receptive field ±14,
    i.e. the whole window) → 2 RoPE self-attention layers
  - Heads: coarse-to-fine Ramachandran (36×36 grid + circular residual),
           SS (H/E/C) with soft conditioning, order/disorder, angle auxiliary
  - 1,293,378 parameters (production configuration; see docs/METHOD.md)

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

    # Auxiliary expected-error head (scale-up plan §3.1 calibration fix).
    # When enabled, a dedicated scalar head predicts log(1 + angular_err_deg)
    # trained against the decoded (coarse+fine) prediction error. At inference,
    # ``confidence = -expected_error_log``, replacing the weak entropy-based
    # signal that collapsed PRECISE tier to 0.1% coverage.
    use_error_head: bool = False

    # v4 heads (2026-04-17). See 260416_facet.md §10 / coil brainstorm.
    # use_order_head: legacy binary "is in regular SS" classifier; labels are
    #   derived from phi/psi region, so it's a leaky copy of SSHead. Disable
    #   for v4 in favor of VarianceHead.
    # use_variance_head: regresses log(1 + combined_std_deg) from ensemble
    #   phi_var + psi_var. True rigidity proxy (matches aux-error head units).
    # use_basin_head: 4-class Ramachandran basin classifier (alpha / beta /
    #   PPII / other). Sharper than SSHead's H/E/C because coil is a dustbin.
    use_order_head: bool = True
    use_variance_head: bool = False
    use_basin_head: bool = False
    n_basin_classes: int = 4

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


class ExpectedErrorHead(nn.Module):
    """Predicts log(1 + angular_err_deg) for the decoded (phi, psi).

    Output is an unbounded scalar; train with smooth-L1 against the actual
    decoded error. At inference, ``confidence = -expected_error_log`` —
    monotone in expected accuracy, concentrates on confident residues far
    more tightly than coarse-softmax entropy.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)  # (B,)


class VarianceHead(nn.Module):
    """Predicts log(1 + combined_angular_std_deg) — a true rigidity proxy
    from the ensemble (unlike the legacy OrderHead whose target was derived
    from phi/psi region and was effectively a 2-way SSHead copy).

    Same output unit as ExpectedErrorHead so both can be compared directly:
    one says 'how wrong will my prediction be?' (model self-uncertainty),
    the other says 'how flexible is this residue in its NMR ensemble?'
    (physical disorder). Agreement between them is the ideal.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)  # (B,)


class BasinHead(nn.Module):
    """4-class Ramachandran basin classifier: alpha_R / beta / PPII / other.

    Coarse-grained partner to the 1296-way CoarseFineRamaHead: on flex
    residues where exact (phi, psi) is ill-defined, the basin assignment
    is often still well-defined (e.g., 'mostly PPII'), which is what
    spectroscopists actually want for coil regions.
    """

    def __init__(self, dim: int, n_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


def assign_basin(phi_rad: torch.Tensor, psi_rad: torch.Tensor) -> torch.Tensor:
    """Assign each (phi, psi) in radians to a 4-class Ramachandran basin.

    Returns (N,) int64 tensor:
      0 = alpha_R (right-handed helix): phi∈[-180,-30], psi∈[-100, 30]
      1 = beta (strand): phi∈[-180,-90],  psi∈[90, 180] ∪ [-180,-150]
      2 = PPII: phi∈[-90,-30], psi∈[90, 180]
      3 = other (left-handed alpha, turns, Gly)
    """
    phi_deg = phi_rad * (180.0 / math.pi)
    psi_deg = psi_rad * (180.0 / math.pi)

    alpha = (phi_deg >= -180) & (phi_deg < -30) & (psi_deg >= -100) & (psi_deg < 30)
    # PPII: sub-region of β
    ppii = (phi_deg >= -90) & (phi_deg < -30) & (psi_deg >= 90) & (psi_deg <= 180)
    # β: wide strand region minus PPII subregion
    beta_upper = (phi_deg >= -180) & (phi_deg < -90) & (psi_deg >= 90) & (psi_deg <= 180)
    beta_lower = (phi_deg >= -180) & (phi_deg < -90) & (psi_deg >= -180) & (psi_deg < -150)
    beta = (beta_upper | beta_lower) & ~ppii

    label = torch.full_like(phi_rad, 3, dtype=torch.long)  # default 'other'
    label = torch.where(alpha, torch.zeros_like(label), label)
    label = torch.where(beta, torch.ones_like(label), label)
    label = torch.where(ppii, torch.full_like(label, 2), label)
    return label


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
        if config.use_order_head:
            self.order_head = OrderHead(D)
        self.angle_aux = AngleAuxHead(D)
        self.chi1_head = Chi1Head(D)
        if config.use_error_head:
            self.error_head = ExpectedErrorHead(D)
        if config.use_variance_head:
            self.variance_head = VarianceHead(D)
        if config.use_basin_head:
            self.basin_head = BasinHead(D, config.n_basin_classes)

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
        angle_sincos = self.angle_aux(h)  # (B, 4)
        chi1_logits = self.chi1_head(h)  # (B, 3)

        # Entropy of coarse distribution (kept as a fallback / diagnostic).
        coarse_probs = F.softmax(rama["coarse_logits"], dim=-1)
        log_probs = F.log_softmax(rama["coarse_logits"], dim=-1)
        entropy = -(coarse_probs * log_probs).sum(dim=-1)  # (B,)

        out = {
            **rama,
            "ss_logits": ss_logits,
            "angle_sincos": angle_sincos,
            "chi1_logits": chi1_logits,
            "coarse_probs": coarse_probs,
            "entropy": entropy,
        }
        if hasattr(self, "order_head"):
            out["order_logits"] = self.order_head(h)  # (B, 1)
        if hasattr(self, "variance_head"):
            out["variance_log"] = self.variance_head(h)  # (B,)
        if hasattr(self, "basin_head"):
            out["basin_logits"] = self.basin_head(h)  # (B, 4)

        # Confidence: prefer the learned expected-error head when present.
        # Fall back to negative entropy otherwise (old behavior, backwards
        # compatible with pre-error-head checkpoints).
        if hasattr(self, "error_head"):
            expected_error_log = self.error_head(h)  # (B,)
            out["expected_error_log"] = expected_error_log
            out["confidence"] = -expected_error_log
        else:
            out["confidence"] = -entropy

        return out

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
            "confidence": out["confidence"],
            "coarse_probs": out["coarse_probs"],
        }


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────

def _soft_coarse_loss(
    logits: torch.Tensor,
    hist: torch.Tensor,
    mode: str,
    label_smoothing: float,
) -> torch.Tensor:
    """Soft-target coarse loss. ``mode`` ∈ {kl, argmax, both}.

    - ``kl``: KL(hist || softmax(logits)) — matches full distribution, but
      argmax(logits) can drift far from hist.argmax. Fine for retrieval-style
      downstream consumers, bad for deployment that uses argmax as a point.
    - ``argmax``: CE(logits, hist.argmax) — aligns with deployment's argmax
      decoder. Default for coil Phase 1.1 (point-metric recovery).
    - ``both``: 0.5·KL + 0.5·argmax-CE.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if mode == "kl":
        return F.kl_div(log_probs, hist, reduction="none").sum(dim=-1)
    argmax_bin = hist.argmax(dim=-1)
    argmax_ce = F.cross_entropy(logits, argmax_bin,
                                label_smoothing=label_smoothing, reduction="none")
    if mode == "argmax":
        return argmax_ce
    if mode == "both":
        kl = F.kl_div(log_probs, hist, reduction="none").sum(dim=-1)
        return 0.5 * kl + 0.5 * argmax_ce
    raise ValueError(f"Unknown soft_target_mode: {mode!r}")


def coarse_fine_rama_loss(
    pred: dict[str, torch.Tensor],
    true_phi: torch.Tensor,
    true_psi: torch.Tensor,
    valid: torch.Tensor,
    label_smoothing: float = 0.01,
    sample_weight: torch.Tensor | None = None,
    rama_hist: torch.Tensor | None = None,
    has_soft_target: torch.Tensor | None = None,
    soft_target_mode: str = "kl",
    soft_target_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute coarse CE + fine von Mises NLL.

    ``sample_weight`` is optional (B,) per-sample weights applied to both
    the coarse and fine terms before averaging. Used for synthetic/exp
    mixing in FACET Exp 4 — synthetic samples are down-weighted during
    training so their (noisier) labels contribute less.

    ``soft_target_mode`` controls how soft-target (flex) residues are
    supervised — ``kl``/``argmax``/``both``. See ``_soft_coarse_loss``.

    Returns dict with 'coarse', 'fine', 'total' losses.
    """
    device = true_phi.device

    if valid.sum() == 0:
        zero = torch.tensor(0.0, device=device)
        return {"coarse": zero, "fine": zero, "total": zero}

    phi_v = true_phi[valid]
    psi_v = true_psi[valid]

    # Coarse target: split into soft-target (ensemble histogram) and hard-target (one-hot)
    logits_v = pred["coarse_logits"][valid]
    target_bin = angles_to_bin_idx(phi_v, psi_v)  # (N,) — always needed for fine loss

    # Determine which samples use soft (KL) vs hard (CE) coarse loss
    use_soft = (
        rama_hist is not None
        and has_soft_target is not None
        and valid.sum() > 0
    )
    if use_soft:
        soft_mask = has_soft_target[valid]
        n_soft = int(soft_mask.sum())
    else:
        soft_mask = None
        n_soft = 0

    if n_soft > 0 and n_soft < int(valid.sum()):
        # Mixed batch: some soft, some hard
        hard_mask = ~soft_mask
        soft_logits = logits_v[soft_mask]
        soft_targets = rama_hist[valid][soft_mask].to(device).float()
        soft_loss = _soft_coarse_loss(
            soft_logits, soft_targets, soft_target_mode, label_smoothing,
        ) * soft_target_weight
        hard_logits = logits_v[hard_mask]
        hard_targets = target_bin[hard_mask]
        hard_loss = F.cross_entropy(hard_logits, hard_targets,
                                    label_smoothing=label_smoothing, reduction="none")
        all_per_sample = torch.zeros(int(valid.sum()), device=device)
        all_per_sample[soft_mask] = soft_loss
        all_per_sample[hard_mask] = hard_loss
        if sample_weight is not None:
            w = sample_weight[valid].to(device).float()
            coarse_loss = (all_per_sample * w).sum() / w.sum().clamp(min=1e-6)
        else:
            coarse_loss = all_per_sample.mean()
    elif n_soft > 0:
        # All soft
        soft_targets = rama_hist[valid].to(device).float()
        per_sample = _soft_coarse_loss(
            logits_v, soft_targets, soft_target_mode, label_smoothing,
        ) * soft_target_weight
        if sample_weight is not None:
            w = sample_weight[valid].to(device).float()
            coarse_loss = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        else:
            coarse_loss = per_sample.mean()
    else:
        # All hard (original path)
        if sample_weight is None:
            coarse_loss = F.cross_entropy(
                logits_v, target_bin,
                label_smoothing=label_smoothing,
            )
        else:
            w = sample_weight[valid].to(device).float()
            per_sample_ce = F.cross_entropy(
                logits_v, target_bin,
                label_smoothing=label_smoothing, reduction="none",
            )
            coarse_loss = (per_sample_ce * w).sum() / w.sum().clamp(min=1e-6)

    # Fine target: residual from bin center
    phi_c, psi_c = bin_idx_to_centers(target_bin)
    phi_c, psi_c = phi_c.to(device), psi_c.to(device)
    r_phi = phi_v - phi_c  # true residual
    r_psi = psi_v - psi_c

    # Fine loss: von Mises NLL on residual
    # log p = kappa * cos(r - delta) - log(2π I₀(κ))
    # With kappa ≤ 50, I₀ is safe to compute directly
    d_phi = pred["fine_delta_phi"][valid]
    d_psi = pred["fine_delta_psi"][valid]
    k_phi = pred["fine_kappa_phi"][valid]
    k_psi = pred["fine_kappa_psi"][valid]

    log_p_phi = k_phi * torch.cos(r_phi - d_phi) - torch.log(2 * math.pi * torch.i0(k_phi))
    log_p_psi = k_psi * torch.cos(r_psi - d_psi) - torch.log(2 * math.pi * torch.i0(k_psi))
    per_sample_fine = -(log_p_phi + log_p_psi)

    # Soft-target residues (coil, wide ensemble spread) should NOT be pinned
    # to the ensemble mean via fine NLL — that would undo the whole point of
    # the soft target. Mask them out of the fine-loss average.
    if use_soft and soft_mask is not None:
        fine_mask = ~soft_mask  # True for hard-target residues
    else:
        fine_mask = torch.ones_like(per_sample_fine, dtype=torch.bool)

    if fine_mask.sum() == 0:
        fine_loss = torch.tensor(0.0, device=device)
    elif sample_weight is None:
        fine_loss = per_sample_fine[fine_mask].mean()
    else:
        w = sample_weight[valid].to(device).float()[fine_mask]
        fine_loss = (per_sample_fine[fine_mask] * w).sum() / w.sum().clamp(min=1e-6)

    return {
        "coarse": coarse_loss,
        "fine": fine_loss,
        "total": coarse_loss + 0.5 * fine_loss,
    }


def _weighted_mean(per_sample: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return per_sample.mean()
    w = weight.to(per_sample.device).float()
    return (per_sample * w).sum() / w.sum().clamp(min=1e-6)


def ss_loss(
    ss_logits: torch.Tensor,
    true_ss: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy SS loss (optionally per-sample weighted)."""
    if sample_weight is None:
        return F.cross_entropy(ss_logits, true_ss)
    per_sample = F.cross_entropy(ss_logits, true_ss, reduction="none")
    return _weighted_mean(per_sample, sample_weight)


def order_loss(
    order_logits: torch.Tensor,
    true_order: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy order loss (optionally per-sample weighted)."""
    if sample_weight is None:
        return F.binary_cross_entropy_with_logits(order_logits.squeeze(-1), true_order)
    per_sample = F.binary_cross_entropy_with_logits(
        order_logits.squeeze(-1), true_order, reduction="none",
    )
    return _weighted_mean(per_sample, sample_weight)


def angle_aux_loss(
    angle_sincos: torch.Tensor,
    true_phi: torch.Tensor,
    true_psi: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on sin/cos of angles for gradient sharpening."""
    if valid.sum() == 0:
        return torch.tensor(0.0, device=true_phi.device)

    pred = angle_sincos[valid]  # (N, 4)
    targets = torch.stack([
        torch.sin(true_phi[valid]),
        torch.cos(true_phi[valid]),
        torch.sin(true_psi[valid]),
        torch.cos(true_psi[valid]),
    ], dim=-1)
    if sample_weight is None:
        return F.mse_loss(pred, targets)
    per_sample = F.mse_loss(pred, targets, reduction="none").mean(dim=-1)
    return _weighted_mean(per_sample, sample_weight[valid])


def variance_loss(
    pred: dict[str, torch.Tensor],
    true_phi_var: torch.Tensor,
    true_psi_var: torch.Tensor,
    valid_var: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 on predicted log(1 + combined_angular_std_deg) vs true
    ensemble spread. ``valid_var`` masks residues where phi_var/psi_var
    are defined (NaN-guarded; disordered residues with no ensemble get
    dropped from this loss).
    """
    if valid_var.sum() == 0 or "variance_log" not in pred:
        return torch.tensor(0.0, device=true_phi_var.device)

    with torch.no_grad():
        pv = true_phi_var[valid_var].clamp(min=0.0)
        sv = true_psi_var[valid_var].clamp(min=0.0)
        combined_std_rad = torch.sqrt(pv + sv)
        combined_std_deg = combined_std_rad * (180.0 / math.pi)
        target = torch.log1p(combined_std_deg)

    pred_log_var = pred["variance_log"][valid_var]
    per_sample = F.smooth_l1_loss(pred_log_var, target, reduction="none")

    if sample_weight is None:
        return per_sample.mean()
    return _weighted_mean(per_sample, sample_weight[valid_var])


def basin_loss(
    pred: dict[str, torch.Tensor],
    true_phi: torch.Tensor,
    true_psi: torch.Tensor,
    valid: torch.Tensor,
    label_smoothing: float = 0.01,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy on 4-class Ramachandran basin derived from (phi, psi)."""
    if valid.sum() == 0 or "basin_logits" not in pred:
        return torch.tensor(0.0, device=true_phi.device)

    with torch.no_grad():
        basin_target = assign_basin(true_phi[valid], true_psi[valid])

    logits = pred["basin_logits"][valid]
    if sample_weight is None:
        return F.cross_entropy(logits, basin_target, label_smoothing=label_smoothing)
    per_sample = F.cross_entropy(logits, basin_target,
                                 label_smoothing=label_smoothing, reduction="none")
    return _weighted_mean(per_sample, sample_weight[valid])


def expected_error_loss(
    pred: dict[str, torch.Tensor],
    true_phi: torch.Tensor,
    true_psi: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 on predicted log(1 + err_deg) vs decoded angular error.

    The target is computed from the fully decoded (coarse argmax + fine
    residual) prediction path, detached so the error head does not leak a
    shortcut gradient back into the torsion head.
    """
    if valid.sum() == 0 or "expected_error_log" not in pred:
        return torch.tensor(0.0, device=true_phi.device)

    with torch.no_grad():
        top_bin = pred["coarse_logits"][valid].argmax(dim=-1)  # (N,)
        phi_c, psi_c = bin_idx_to_centers(top_bin)
        phi_c = phi_c.to(true_phi.device)
        psi_c = psi_c.to(true_phi.device)
        phi_pred = phi_c + pred["fine_delta_phi"][valid].detach()
        psi_pred = psi_c + pred["fine_delta_psi"][valid].detach()

        dphi = torch.abs(phi_pred - true_phi[valid])
        dphi = torch.minimum(dphi, 2 * math.pi - dphi)
        dpsi = torch.abs(psi_pred - true_psi[valid])
        dpsi = torch.minimum(dpsi, 2 * math.pi - dpsi)
        err_rad = torch.sqrt((dphi ** 2 + dpsi ** 2) / 2)
        err_deg = err_rad * (180.0 / math.pi)
        target = torch.log1p(err_deg)  # log(1 + err_deg)

    pred_log_err = pred["expected_error_log"][valid]
    per_sample = F.smooth_l1_loss(pred_log_err, target, reduction="none")

    if sample_weight is None:
        return per_sample.mean()
    return _weighted_mean(per_sample, sample_weight[valid])


def chi1_loss(
    chi1_logits: torch.Tensor,
    true_chi1: torch.Tensor,
    chi1_valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy chi1 rotamer loss, masked to residues with defined chi1.

    ``class_weight`` (optional, shape ``[3]``) applies per-class weights to
    the CE loss. Audit §3.3: plain CE collapses to majority class (g-,
    ~47% of data). Inverse-frequency weights recover minority-class
    accuracy at a small cost to majority-class accuracy.
    """
    if chi1_valid.sum() == 0:
        return torch.tensor(0.0, device=chi1_logits.device)
    logits = chi1_logits[chi1_valid]
    targets = true_chi1[chi1_valid]
    cw = class_weight.to(logits.device) if class_weight is not None else None
    if sample_weight is None:
        return F.cross_entropy(logits, targets, weight=cw)
    w = sample_weight[chi1_valid].to(logits.device).float()
    per_sample = F.cross_entropy(logits, targets, reduction="none", weight=cw)
    return (per_sample * w).sum() / w.sum().clamp(min=1e-6)


def facet_v3_loss(
    pred: dict[str, torch.Tensor],
    true_phi: torch.Tensor,
    true_psi: torch.Tensor,
    true_ss: torch.Tensor,
    true_order: torch.Tensor,
    valid: torch.Tensor,
    lambda_ss: float = 0.3,
    lambda_order: float = 0.1,
    lambda_aux: float = 0.2,
    lambda_chi1: float = 0.15,
    lambda_error: float = 0.1,
    lambda_variance: float = 0.1,
    lambda_basin: float = 0.2,
    label_smoothing: float = 0.01,
    sample_weight: torch.Tensor | None = None,
    true_chi1: torch.Tensor | None = None,
    chi1_valid: torch.Tensor | None = None,
    chi1_class_weight: torch.Tensor | None = None,
    rama_hist: torch.Tensor | None = None,
    has_soft_target: torch.Tensor | None = None,
    soft_target_mode: str = "kl",
    soft_target_weight: float = 1.0,
    true_phi_var: torch.Tensor | None = None,
    true_psi_var: torch.Tensor | None = None,
    valid_var: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined multi-task loss for FACET v3.

    ``sample_weight`` (optional) is a (B,) tensor of per-sample multipliers.
    When provided, all sub-losses use a weighted mean. Intended for
    synthetic/experimental mixing (Exp 4).

    Returns dict with all individual losses + 'total'.
    """
    rama = coarse_fine_rama_loss(
        pred, true_phi, true_psi, valid, label_smoothing, sample_weight=sample_weight,
        rama_hist=rama_hist, has_soft_target=has_soft_target,
        soft_target_mode=soft_target_mode,
        soft_target_weight=soft_target_weight,
    )
    ss = ss_loss(pred["ss_logits"], true_ss, sample_weight=sample_weight)
    aux = angle_aux_loss(
        pred["angle_sincos"], true_phi, true_psi, valid, sample_weight=sample_weight,
    )

    total = rama["total"] + lambda_ss * ss + lambda_aux * aux

    result = {
        "total": total,
        "rama_coarse": rama["coarse"],
        "rama_fine": rama["fine"],
        "ss": ss,
        "angle_aux": aux,
    }

    # OrderHead (legacy — only fires when the head is present)
    if "order_logits" in pred:
        ordr = order_loss(pred["order_logits"], true_order, sample_weight=sample_weight)
        result["order"] = ordr
        result["total"] = result["total"] + lambda_order * ordr

    # Chi1 rotamer loss (optional — only fires when chi1 targets are provided)
    if true_chi1 is not None and chi1_valid is not None and "chi1_logits" in pred:
        c1 = chi1_loss(pred["chi1_logits"], true_chi1, chi1_valid,
                       sample_weight, class_weight=chi1_class_weight)
        result["chi1"] = c1
        result["total"] = result["total"] + lambda_chi1 * c1

    # Expected-error aux head loss (only fires when the head is present).
    if "expected_error_log" in pred:
        err = expected_error_loss(
            pred, true_phi, true_psi, valid, sample_weight=sample_weight,
        )
        result["expected_error"] = err
        result["total"] = result["total"] + lambda_error * err

    # VarianceHead loss: regress log(1 + ensemble_std_deg) against actual
    # ensemble spread. Only fires when the head is present and variance
    # targets are provided.
    if ("variance_log" in pred and true_phi_var is not None
            and true_psi_var is not None and valid_var is not None):
        v = variance_loss(pred, true_phi_var, true_psi_var, valid_var,
                          sample_weight=sample_weight)
        result["variance"] = v
        result["total"] = result["total"] + lambda_variance * v

    # BasinHead loss: 4-class CE on derived Ramachandran basin.
    if "basin_logits" in pred:
        b = basin_loss(pred, true_phi, true_psi, valid,
                       label_smoothing=label_smoothing,
                       sample_weight=sample_weight)
        result["basin"] = b
        result["total"] = result["total"] + lambda_basin * b

    return result
