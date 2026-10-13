import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalContinuousGraph(nn.Module):
    """
    Module 4 (first-stage): causal continuous graph propagation with
    learned semantic relevance, exponential time decay, and top-k sparsification.
    """
    def __init__(self, hidden_dim, dropout=0.1, topk=8, time_beta=1.0):
        super(CausalContinuousGraph, self).__init__()
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.time_beta = nn.Parameter(torch.tensor(float(time_beta)))

        rel_mid = max(hidden_dim // 2, 64)
        # Memory-efficient relevance scorer:
        # rel_ij ~= f_i(x_i) + f_j(x_j), avoiding explicit [B,N,N,2D] concat.
        self.rel_i = nn.Sequential(
            nn.Linear(hidden_dim, rel_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rel_mid, 1),
        )
        self.rel_j = nn.Sequential(
            nn.Linear(hidden_dim, rel_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rel_mid, 1),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_causal_mask(self, timestamps, umask):
        # timestamps: [B, N]
        t_target = timestamps.unsqueeze(2)  # [B, N, 1]
        t_source = timestamps.unsqueeze(1)  # [B, 1, N]
        delta_t = t_target - t_source       # [B, N, N], source -> target
        causal_mask = delta_t > 0
        if umask is not None:
            valid = umask.bool()
            valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)
            causal_mask = causal_mask & valid_pair
        return causal_mask, torch.clamp(delta_t, min=0.0)

    def _topk_mask(self, edge_score, causal_mask):
        if self.topk is None or self.topk <= 0:
            return causal_mask
        bsz, seq_len, _ = edge_score.size()
        k = min(self.topk, seq_len)
        masked_score = edge_score.masked_fill(~causal_mask, -1e9)
        topk_vals, topk_idx = torch.topk(masked_score, k=k, dim=-1)
        keep = topk_vals > -1e8
        topk_mask = torch.zeros_like(causal_mask)
        topk_mask.scatter_(-1, topk_idx, keep)
        return causal_mask & topk_mask

    def forward(self, x, lambda_vec, timestamps, umask=None):
        # x: [B, N, D], lambda_vec: [B, N], timestamps: [B, N]
        if x is None or lambda_vec is None or timestamps is None:
            return x

        bsz, seq_len, dim = x.size()
        rel_i_logit = self.rel_i(x)                              # [B, N, 1]
        rel_j_logit = self.rel_j(x)                              # [B, N, 1]
        rel_logit = rel_i_logit + rel_j_logit.transpose(1, 2)    # [B, N, N]
        rel = torch.sigmoid(rel_logit)

        causal_mask, delta_t = self._build_causal_mask(timestamps.float(), umask)
        lambda_source = lambda_vec.unsqueeze(1)                  # [B, 1, N]
        decay = torch.exp(-lambda_source * delta_t)
        edge_score = rel * decay
        edge_score = edge_score * causal_mask.float()
        sparse_mask = self._topk_mask(edge_score, causal_mask)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        semantic_score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(dim)
        time_bias = -lambda_source * delta_t
        score = self.leakyrelu(semantic_score + rel_logit + self.time_beta * time_bias)

        # If a target has no valid historical source (e.g., first utterance),
        # fallback to self-loop to avoid NaN in softmax.
        has_edge = sparse_mask.any(dim=-1, keepdim=True)         # [B, N, 1]
        eye = torch.eye(seq_len, device=x.device, dtype=torch.bool).unsqueeze(0).expand(bsz, -1, -1)
        fallback_mask = eye & (~has_edge)
        final_mask = sparse_mask | fallback_mask

        score = score.masked_fill(~final_mask, -1e9)
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        out = self.norm(x + self.dropout(out))

        if umask is not None:
            out = out * umask.unsqueeze(-1).float()
        return out
