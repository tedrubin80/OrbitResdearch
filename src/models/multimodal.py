"""Multi-modal model combining orbit positions with solar wind data.

Dual-encoder architecture with attention-based fusion for predicting
orbit perturbations during geomagnetic storms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Attention layer that lets one modality attend to another."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_len, d_model) — primary modality
            context: (batch, seq_len, d_model) — modality to attend to

        Returns:
            (batch, seq_len, d_model) — attended query
        """
        attended, _ = self.attn(query, context, context)
        return self.norm(query + attended)


class SolarWindOrbitModel(nn.Module):
    """Residual gated multi-modal model: output = base_prediction + gate * perturbation.

    Architecture:
        Orbit LSTM encoder → base prediction (LSTM-equivalent)
        Solar wind LSTM encoder → solar features
        Cross-modal attention (orbit attends to solar wind)
        Attention-weighted summary → perturbation head
        Sigmoid gate controls perturbation strength
        Output: base + gate * perturbation

    The model can never be worse than standalone LSTM because the gate
    can learn to output ~0, reducing to pure orbit-only prediction.

    Two-phase training:
        Phase 1: Freeze solar/perturbation/gate, train orbit encoder + base_head.
        Phase 2: Unfreeze everything, fine-tune with lower LR.
    """

    def __init__(
        self,
        orbit_input_dim: int = 6,
        solar_input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        horizon: int = 360,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            orbit_input_dim: Orbit features (x,y,z,vx,vy,vz)
            solar_input_dim: Solar wind features (Bx,By,Bz,speed,density,Kp,Dst)
            hidden_dim: Encoder hidden dimension
            num_layers: Number of LSTM layers
            nhead: Attention heads for cross-modal fusion
            horizon: Prediction steps
            output_dim: Output features (x,y,z)
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.output_dim = output_dim

        # --- Orbit encoder (same as standalone LSTM) ---
        self.orbit_proj = nn.Linear(orbit_input_dim, hidden_dim)
        self.orbit_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.orbit_norm = nn.LayerNorm(hidden_dim * 2)

        # Base prediction head: final hidden states -> trajectory (LSTM-equivalent)
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

        # --- Solar wind encoder ---
        self.solar_proj = nn.Linear(solar_input_dim, hidden_dim)
        self.solar_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.solar_norm = nn.LayerNorm(hidden_dim * 2)

        # --- Cross-modal attention: orbit attends to solar wind ---
        self.cross_attention = CrossModalAttention(
            d_model=hidden_dim * 2,
            nhead=nhead,
            dropout=dropout,
        )

        # --- Attention-weighted summary (learned, not mean pool) ---
        self.attn_weight = nn.Linear(hidden_dim * 2, 1)

        # --- Perturbation head: deeper MLP producing correction signal ---
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

        # --- Gate: sigmoid controlling perturbation strength ---
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        orbit_input: torch.Tensor,
        solar_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            orbit_input: (batch, seq_len, orbit_input_dim)
            solar_input: (batch, seq_len, solar_input_dim)

        Returns:
            (batch, horizon, output_dim) — predicted positions
        """
        # Encode orbit sequence
        orbit_emb = self.orbit_proj(orbit_input)
        orbit_encoded, (h, _) = self.orbit_encoder(orbit_emb)
        orbit_encoded = self.orbit_norm(orbit_encoded)

        # Base prediction from final hidden states (like standalone LSTM)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (batch, hidden*2)
        base = self.base_head(h_cat).view(-1, self.horizon, self.output_dim)

        # Encode solar wind
        solar_emb = self.solar_proj(solar_input)
        solar_encoded, _ = self.solar_encoder(solar_emb)
        solar_encoded = self.solar_norm(solar_encoded)

        # Cross-attention: orbit features attend to solar wind
        attended = self.cross_attention(orbit_encoded, solar_encoded)

        # Attention-weighted summary (not mean pool)
        attn_scores = torch.softmax(self.attn_weight(attended), dim=1)
        summary = (attended * attn_scores).sum(dim=1)  # (batch, hidden*2)

        # Perturbation: learned correction from solar wind context
        perturbation = self.perturbation_head(summary).view(-1, self.horizon, self.output_dim)

        # Gate: per-element sigmoid controlling correction strength
        gate = self.gate_net(h_cat).view(-1, self.horizon, self.output_dim)

        # Residual output: base + gated perturbation
        return base + gate * perturbation

    def freeze_solar_branch(self):
        """Phase 1: freeze solar wind encoder, cross-attention, perturbation, and gate."""
        for module in [self.solar_proj, self.solar_encoder, self.solar_norm,
                       self.cross_attention, self.attn_weight, self.perturbation_head, self.gate_net]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        """Phase 2: unfreeze everything for fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True


class SolarWindClassifier(nn.Module):
    """Binary classifier for detecting geomagnetic storm impacts.

    Predicts whether a spacecraft's orbit will be significantly
    perturbed (above threshold) given current solar wind conditions.
    Useful as a complementary model to the regression approach.
    """

    def __init__(
        self,
        solar_input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=solar_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, solar_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            solar_input: (batch, seq_len, solar_input_dim)

        Returns:
            (batch, 1) — probability of significant perturbation
        """
        _, (h, _) = self.encoder(solar_input)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return torch.sigmoid(self.classifier(h))
