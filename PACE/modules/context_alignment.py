import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContextFeatureAligner(nn.Module):
    """
    Module 2: project each modality into a shared semantic space and align
    modalities with lightweight cross-modal attention.
    """
    def __init__(self, hidden_dim, n_head, dropout, alpha_init=0.1, gamma_max=0.1):
        super(ContextFeatureAligner, self).__init__()
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.gamma_max = gamma_max

        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_head, dropout=dropout
        )
        self.text_to_visual = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_head, dropout=dropout
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_dropout = nn.Dropout(dropout * 0.5)
        self.dropout = nn.Dropout(dropout)

        # alpha controls cross-modal residual strength; initialized small.
        alpha_init = min(max(alpha_init, 1e-4), 1 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(alpha_init), dtype=torch.float32))
        # gamma controls time embedding strength; initialized to 0 (weak injection).
        self.gamma_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _inv_sigmoid(p):
        return math.log(p / (1.0 - p))

    def _time_embedding(self, timestamps, umask):
        if timestamps is None:
            return None
        t = timestamps.float()
        if umask is None:
            valid = torch.ones_like(t)
        else:
            valid = umask.float()
        max_t = (t * valid).max(dim=1, keepdim=True).values
        max_t = torch.where(max_t > 0, max_t, torch.ones_like(max_t))
        t_norm = (t / max_t).unsqueeze(-1) * valid.unsqueeze(-1)
        return self.time_mlp(t_norm)

    def forward(self, text_feats, audio_feats, visual_feats, umask=None, timestamps=None):
        h_t = self.text_norm(text_feats)
        h_a = self.audio_norm(audio_feats) if audio_feats is not None else None
        h_v = self.visual_norm(visual_feats) if visual_feats is not None else None

        time_emb = self._time_embedding(timestamps, umask)
        if time_emb is not None:
            gamma = self.gamma_max * torch.tanh(self.gamma_raw)
            time_delta = self.time_dropout(time_emb) * gamma
            h_t = h_t + time_delta
            if h_a is not None:
                h_a = h_a + time_delta
            if h_v is not None:
                h_v = h_v + time_delta

        key_padding_mask = umask.eq(0) if umask is not None else None
        contexts = []
        # nn.MultiheadAttention in older torch versions expects [L, B, D].
        h_t_seq = h_t.transpose(0, 1)
        if h_a is not None:
            h_a_seq = h_a.transpose(0, 1)
            a_ctx, _ = self.text_to_audio(
                query=h_t_seq, key=h_a_seq, value=h_a_seq, key_padding_mask=key_padding_mask
            )
            contexts.append(a_ctx.transpose(0, 1))
        if h_v is not None:
            h_v_seq = h_v.transpose(0, 1)
            v_ctx, _ = self.text_to_visual(
                query=h_t_seq, key=h_v_seq, value=h_v_seq, key_padding_mask=key_padding_mask
            )
            contexts.append(v_ctx.transpose(0, 1))

        if contexts:
            alpha = torch.sigmoid(self.alpha_logit)
            cross_ctx = torch.stack(contexts, dim=0).mean(dim=0)
            h_base = self.base_norm(h_t + self.dropout(alpha * cross_ctx))
        else:
            h_base = self.base_norm(h_t)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).type_as(h_base)
            h_base = h_base * valid
            h_t = h_t * valid
            if h_a is not None:
                h_a = h_a * valid
            if h_v is not None:
                h_v = h_v * valid

        valid_mask = umask.float() if umask is not None else None

        def pairwise_dissonance(x, y):
            if x is None or y is None:
                return None
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            dis = 1.0 - torch.sum(x_norm * y_norm, dim=-1)
            if valid_mask is not None:
                dis = dis * valid_mask
            return dis

        pair_stats = {}
        dis_terms = []

        ta = pairwise_dissonance(h_t, h_a)
        if ta is not None:
            pair_stats["text_audio"] = ta
            dis_terms.append(ta)

        tv = pairwise_dissonance(h_t, h_v)
        if tv is not None:
            pair_stats["text_visual"] = tv
            dis_terms.append(tv)

        av = pairwise_dissonance(h_a, h_v)
        if av is not None:
            pair_stats["audio_visual"] = av
            dis_terms.append(av)

        if dis_terms:
            dis_map = torch.stack(dis_terms, dim=0).mean(dim=0)
            if valid_mask is not None:
                valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
                dis_mean = (dis_map.sum(dim=1) / valid_count).mean()
            else:
                dis_mean = dis_map.mean()
        else:
            dis_map = None
            dis_mean = h_base.new_tensor(0.0)

        align_info = {
            "dissonance_map": dis_map,
            "dissonance_mean": dis_mean,
            "pair_dissonance": pair_stats,
        }
        return h_base, h_t, h_a, h_v, align_info


