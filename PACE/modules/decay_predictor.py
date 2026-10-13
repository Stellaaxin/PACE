import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityDecayPredictor(nn.Module):
    """
    Predict per-node positive decay rates lambda_i for one modality.
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(ModalityDecayPredictor, self).__init__()
        mid_dim = max(hidden_dim // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x, umask=None):
        # x: [B, N, D] -> lambda: [B, N]
        lam = self.mlp(x).squeeze(-1)
        lam = F.softplus(lam)
        if umask is not None:
            lam = lam * umask.float()
        return lam


class DecayRatePredictor(nn.Module):
    """
    Module 3: modality-specific decay predictors (no parameter sharing).
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(DecayRatePredictor, self).__init__()
        self.t_predictor = ModalityDecayPredictor(hidden_dim, dropout)
        self.a_predictor = ModalityDecayPredictor(hidden_dim, dropout)
        self.v_predictor = ModalityDecayPredictor(hidden_dim, dropout)

    def forward(self, h_t, h_a, h_v, umask=None):
        lambda_t = self.t_predictor(h_t, umask) if h_t is not None else None
        lambda_a = self.a_predictor(h_a, umask) if h_a is not None else None
        lambda_v = self.v_predictor(h_v, umask) if h_v is not None else None
        return lambda_t, lambda_a, lambda_v
