# PACE model with prototype-guided causal differential graph reasoning.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import f1_score
from modules.context_alignment import ContextFeatureAligner
from modules.decay_predictor import DecayRatePredictor
from modules.causal_continuous_graph import CausalContinuousGraph

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)

        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context, atten_score = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out), atten_score


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_b, mask, speaker_emb=None):
        if speaker_emb != None:
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
        for i in range(self.layers):
            x_b, atten_score = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        return x_b, atten_score

class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DiffGraphAttentionLayer(nn.Module):
    """
    Differential attention graph layer.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, relation=True, num_relation=-1,
                 relation_dim=10, depth=1):
        super(DiffGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  
        self.out_features = out_features  
        self.alpha = alpha
        self.concat = concat
        self.relation = relation
        self.internal_dim = out_features // 2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Projections for pos and neg
        self.a_left_pos = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_right_pos = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_left_neg = nn.Linear(self.internal_dim, 1, bias=False)
        self.a_right_neg = nn.Linear(self.internal_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.a_left_pos.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_right_pos.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_left_neg.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_right_neg.weight, gain=1.414)

        if self.relation:
            self.relation_embedding = nn.Embedding(6, relation_dim)
            self.a_rel_pos = nn.Linear(relation_dim, 1, bias=False)
            self.a_rel_neg = nn.Linear(relation_dim, 1, bias=False)
            nn.init.xavier_uniform_(self.a_rel_pos.weight, gain=1.414)
            nn.init.xavier_uniform_(self.a_rel_neg.weight, gain=1.414)
            # Same-speaker edges are biased toward positive, cross-speaker edges toward negative.
            self.register_buffer(
                "relation_sign_prior",
                torch.tensor([0.0, 1.0, 0.5, 0.5, -0.5, -0.5], dtype=torch.float32),
            )

        # Lambda parameters
        self.lambda_left1 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_right1 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_left2 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_right2 = nn.Parameter(torch.zeros(self.internal_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.lambda_init = lambda_init_fn(depth)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj, proto_bias=None):
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W)  # (B, N, D_out)
        Wh = Wh.view(-1, Wh.size(1), 2, self.internal_dim)  # (B, N, 2, internal_dim)
        Wh_pos = Wh[:, :, 0, :]
        Wh_neg = Wh[:, :, 1, :]

        # Compute e_pos and e_neg
        left_proj_pos = self.a_left_pos(Wh_pos)  # (B, N, 1)
        right_proj_pos = self.a_right_pos(Wh_pos)  # (B, N, 1)
        e_pos = left_proj_pos + right_proj_pos.transpose(1, 2)  # (B, N, N)

        left_proj_neg = self.a_left_neg(Wh_neg)  # (B, N, 1)
        right_proj_neg = self.a_right_neg(Wh_neg)  # (B, N, 1)
        e_neg = left_proj_neg + right_proj_neg.transpose(1, 2)  # (B, N, N)

        if self.relation:
            long_adj = adj.clone().type(torch.long).to(h.device)
            relation_emb = self.relation_embedding(long_adj)  # (B, N, N, relation_dim)
            rel_proj_pos = self.a_rel_pos(relation_emb).squeeze(-1)  # (B, N, N)
            rel_proj_neg = self.a_rel_neg(relation_emb).squeeze(-1)  # (B, N, N)
            rel_polarity = torch.tanh(rel_proj_pos - rel_proj_neg + self.relation_sign_prior[long_adj])
            e_pos = e_pos + rel_proj_pos
            e_neg = e_neg + rel_proj_neg

        if proto_bias is not None:
            proto_bias = proto_bias.to(device=h.device, dtype=e_pos.dtype)
            e_pos = e_pos + proto_bias
            e_neg = e_neg - proto_bias

        e_pos = self.leakyrelu(e_pos)  # (B, N, N)
        e_neg = self.leakyrelu(e_neg)  # (B, N, N)

        # Softmax with masking if relation
        if self.relation:
            zero_vec = -9e15 * torch.ones_like(e_pos)
            e_pos_masked = torch.where(adj > 0, e_pos, zero_vec)
            e_neg_masked = torch.where(adj > 0, e_neg, zero_vec)
            attention_pos = F.softmax(e_pos_masked, dim=2)
            attention_neg = F.softmax(e_neg_masked, dim=2)
        else:
            attention_pos = F.softmax(e_pos, dim=2)
            attention_neg = F.softmax(e_neg, dim=2)

        # Lambda computation
        lambda_1 = torch.exp(torch.sum(self.lambda_left1 * self.lambda_right1, dim=-1).float()).type_as(Wh)
        lambda_2 = torch.exp(torch.sum(self.lambda_left2 * self.lambda_right2, dim=-1).float()).type_as(Wh)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention
        attention = attention_pos - lambda_full * attention_neg
        if self.relation:
            # Signed gate: amplify disagreement when the relation prior suggests heterophily.
            attention = attention + 0.25 * rel_polarity * (attention_pos - attention_neg)

        attention = F.dropout(attention, self.dropout, training=self.training)

        # Split the message into a consensus channel and a disagreement channel.
        Wh_homo = 0.5 * (Wh_pos + Wh_neg)
        Wh_hetero = Wh_pos - Wh_neg
        Wh_v = torch.cat((Wh_homo + 0.25 * Wh_hetero, Wh_homo - 0.25 * Wh_hetero), dim=-1)  # (B, N, out_features)
        h_prime = torch.matmul(attention, Wh_v)  # (B, N, out_features)
        if self.relation:
            diff_msg = torch.matmul(attention_pos - attention_neg, Wh_hetero)
            h_prime = h_prime + 0.15 * torch.cat((diff_msg, -diff_msg), dim=-1)

        h_prime = self.layer_norm(h_prime)
        h_prime = h_prime * (1 - self.lambda_init)

        if self.concat:
            return F.gelu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DiffRGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=2, num_relation=-1, depth=1):
        """Dense version of GAT."""
        super(DiffRGCN, self).__init__()
        self.dropout = dropout
        self.attentions = [DiffGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, relation=True,
                                               num_relation=num_relation, depth=depth) for _ in range(nheads)]  # 婢舵艾銇斿▔銊﹀壈閸?
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = DiffGraphAttentionLayer(nfeat * nheads, nhid, dropout=dropout, alpha=alpha, concat=True,
                                           relation=True, num_relation=num_relation, depth=depth)  # 閹垹顦查崚鐗堫劀鐢摜娣惔?

        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj, proto_bias=None):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # (B,N,num_head*N_out)
        attened_outputs = []
        attention_weights = []
        for att_module in self.attentions:
            att_out, att_w = att_module(x, adj, proto_bias=proto_bias)
            # Graphplt(att_w)
            attened_outputs.append(att_out)
            attention_weights.append(att_w)
        x = torch.cat(attened_outputs, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        att_out, att_w = self.out_att(x, adj, proto_bias=proto_bias)
        attention_weights.append(att_w)
        x = F.gelu(att_out)  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x, attention_weights

def Graphplt(Attention):
    Attention = Attention[-1]
    # attention = F.softmax(attention, dim=2)
    attention = Attention.cpu().detach().numpy()
    num = len(attention)
    n = math.ceil(math.sqrt(num))
    m = math.ceil(num / n)
    fig = plt.figure(figsize=(20 * n, 20 * m), dpi=75)
    for i in range(num):
        axs = fig.add_subplot(n, m, i+1)
        sns.heatmap(attention[i], cmap='coolwarm', annot=True, fmt='.2f', ax=axs)
    plt.tight_layout()
    plt.show()

class custom_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,theta):
        ctx.save_for_backward(input,theta)
        return input/(1-theta.item())

    @staticmethod
    def backward(ctx,grad_output):
        input,theta=ctx.saved_tensors
        input_grad=1/(1-theta.item())*grad_output.clone()

        return input_grad,None

class Modality_drop(nn.Module):

    def __init__(self, dim_list, p_exe=0.7, device='cuda'):
        super().__init__()
        self.dim_list=dim_list
        self.p_exe=p_exe
        self.device=device

    def execute_drop(self, fead_list, q):
        num_mod = len(fead_list)
        B = fead_list[0].shape[0]
        L = fead_list[0].shape[1]
        D = fead_list[0].shape[2]
        exe_drop = torch.rand(1, device=self.device) >= 1-self.p_exe  # 閻?torch.rand 閺囨寧宕?np.random
        if not exe_drop:
            return fead_list, torch.ones([B],dtype=torch.int32,device=self.device)

        d_sum = sum(self.dim_list)
        q_sum = sum(self.dim_list * q)
        theta = q_sum / d_sum

        mask = torch.distributions.Bernoulli(1-q).sample([B,1]).permute(2,1,0).contiguous().reshape(num_mod,B,-1).to(device=self.device)  # [num_mod, B, 1]
        concat_list = torch.stack(fead_list, dim=0)  # [num_mod, B, L, D]
        concat_list = torch.mul(concat_list, mask.unsqueeze(2))  # broadcast mask to [num_mod, B, 1, 1]
        concat_list = custom_autograd.apply(concat_list, theta)
        mask = torch.transpose(mask, 0, 1).squeeze(-1)  # [B, num_mod]
        update_flag = torch.sum(mask, dim=1) > 0
        broadcast_update = update_flag.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, B, 1, 1]
        cleaned_concat = torch.masked_select(concat_list, broadcast_update).reshape(num_mod, -1, L, D)
        cleaned_fea = torch.chunk(cleaned_concat, num_mod, dim=0)
        cleaned_fea = [_.squeeze(0) for _ in cleaned_fea]  # list of [valid_B, L, D]
        return cleaned_fea, update_flag

def calcu_q(performances, q_base, fix_lambda):
    num_mod = len(performances)
    q = torch.zeros(num_mod, device=performances.device)
    relu = nn.ReLU(inplace=True)
    softmax = nn.Softmax(dim=0)  
    epsilon = 1e-5  
    for i in range(num_mod):
        ratios = []
        for j in range(num_mod):
            if i == j:
                continue
            if performances[j] < epsilon: 
                continue
            ratio = performances[i] / (performances[j] + epsilon) - 1 
            ratios.append(ratio)
        if ratios:
            ratios_tensor = torch.tensor(ratios, device=performances.device)
            norm_ratios = softmax(relu(ratios_tensor)) 
            avg_ratio = torch.sum(norm_ratios * ratios_tensor) / len(ratios)
            if avg_ratio > 0:
                q[i] = q_base * (1 + fix_lambda * avg_ratio)
    q = torch.clamp(q, min=0.0, max=1.0)
    return q


class PACE(nn.Module):
    def __init__(self, dataset, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, use_adam_drop=True,
                 q_base=0.3, lam=0.9, p_exe=0.5, causal_topk=8, causal_time_beta=1.0,
                 fusion_num_experts=4, fusion_perf_momentum=0.9, fusion_res_alpha_init=0.1,
                 use_module5=True, enable_mask_aux=False, mask_ratio=0.15,
                 use_causal_graph=True,
                 dissonance_beta=0.25, use_proto_nodes=False, prototype_path="",
                 proto_top_k=2, proto_audio_scale=0.04, proto_visual_scale=0.06,
                 proto_temperature=8.0, proto_mode="feature", proto_start_epoch=10,
                 proto_safe_conf=0.34, proto_safe_margin=0.65, proto_safe_gate_temp=10.0,
                 protect_visual_drop=False, protect_visual_strength=0.55,
                 proto_anchor="learned", proto_anchor_detach=True,
                 proto_adapter_margin_gate=False, proto_adapter_margin=0.65,
                 proto_adapter_gate_temp=8.0, proto_diff_graph=False,
                 proto_diff_scale=0.06):
        super(PACE, self).__init__()
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dataset = dataset
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        self.textf_input = nn.Linear(D_text, hidden_dim)
        self.acouf_input = nn.Linear(D_audio, hidden_dim)
        self.visuf_input = nn.Linear(D_visual, hidden_dim)
        self.modal_aligner = ContextFeatureAligner(hidden_dim=hidden_dim, n_head=n_head, dropout=dropout)
        self.decay_predictor = DecayRatePredictor(hidden_dim=hidden_dim, dropout=dropout)
        self.last_decay_rates = {}
        self.use_causal_graph = use_causal_graph
        self.causal_topk = causal_topk
        self.causal_time_beta = causal_time_beta
        self.use_proto_nodes = use_proto_nodes
        self.proto_top_k = proto_top_k
        self.proto_audio_scale = proto_audio_scale
        self.proto_visual_scale = proto_visual_scale
        self.proto_temperature = proto_temperature
        self.proto_mode = proto_mode
        self.proto_start_epoch = proto_start_epoch
        self.proto_safe_conf = proto_safe_conf
        self.proto_safe_margin = proto_safe_margin
        self.proto_safe_gate_temp = proto_safe_gate_temp
        self.protect_visual_drop = protect_visual_drop
        self.protect_visual_strength = protect_visual_strength
        self.proto_anchor = proto_anchor
        self.proto_anchor_detach = proto_anchor_detach
        self.proto_adapter_margin_gate = proto_adapter_margin_gate
        self.proto_adapter_margin = proto_adapter_margin
        self.proto_adapter_gate_temp = proto_adapter_gate_temp
        self.proto_diff_graph = proto_diff_graph
        self.proto_diff_scale = proto_diff_scale
        self.current_epoch = 0
        self.last_proto_info = {}
        if self.use_proto_nodes:
            rng_state = torch.random.get_rng_state()
            try:
                proto_init = self._load_or_init_prototypes(
                    n_classes=n_classes,
                    hidden_dim=hidden_dim,
                    prototype_path=prototype_path,
                )
                self.emotion_prototypes = nn.Parameter(proto_init)
                if self.proto_mode in ("feature", "both", "visual_graph", "adapter"):
                    self.proto_a_proj = nn.Linear(hidden_dim, hidden_dim)
                    self.proto_v_proj = nn.Linear(hidden_dim, hidden_dim)
                    self.proto_a_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                    self.proto_v_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                    self.proto_a_norm = nn.LayerNorm(hidden_dim)
                    self.proto_v_norm = nn.LayerNorm(hidden_dim)
                    nn.init.constant_(self.proto_a_gate.bias, -2.0)
                    nn.init.constant_(self.proto_v_gate.bias, -2.0)
                    self.proto_adapter_out = nn.Linear(hidden_dim, n_classes)
                    nn.init.zeros_(self.proto_adapter_out.weight)
                    nn.init.zeros_(self.proto_adapter_out.bias)
                else:
                    self.proto_a_proj = None
                    self.proto_v_proj = None
                    self.proto_a_gate = None
                    self.proto_v_gate = None
                    self.proto_a_norm = None
                    self.proto_v_norm = None
                    self.proto_adapter_out = None
            finally:
                torch.random.set_rng_state(rng_state)
        else:
            self.register_parameter("emotion_prototypes", None)
            self.proto_a_proj = None
            self.proto_v_proj = None
            self.proto_a_gate = None
            self.proto_v_gate = None
            self.proto_a_norm = None
            self.proto_v_norm = None
            self.proto_adapter_out = None

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.agate = EnhancedFilterModule(hidden_dim)
        self.vgate = EnhancedFilterModule(hidden_dim)

        self.gatTer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatT = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)

        self.gatAer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatA = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatVer = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)
        self.gatV = DiffRGCN(hidden_dim, hidden_dim, num_relation=4)

        self.causal_t = CausalContinuousGraph(hidden_dim, dropout=dropout, topk=causal_topk, time_beta=causal_time_beta)
        self.causal_a = CausalContinuousGraph(hidden_dim, dropout=dropout, topk=causal_topk, time_beta=causal_time_beta)
        self.causal_v = CausalContinuousGraph(hidden_dim, dropout=dropout, topk=causal_topk, time_beta=causal_time_beta)
        self.use_module5 = use_module5
        self.enable_mask_aux = enable_mask_aux
        self.mask_ratio = mask_ratio
        self.fusion_perf_momentum = fusion_perf_momentum
        self.last_fusion_info = {}
        if self.use_module5:
            raise RuntimeError("PACE main configuration uses prototype-calibrated additive fusion; module-5 is disabled.")
        self.relation_fusion = None
        self.fusion_res_alpha_logit = None
        self.register_buffer("modal_perf_ema", torch.full((3,), 0.5))

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        if self.use_module5:
            self.fused_output_layer = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes)
                )
        else:
            self.fused_output_layer = None
        if self.enable_mask_aux:
            self.mask_mlp_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.mask_classifier = nn.Linear(hidden_dim, n_classes)
        else:
            self.mask_mlp_fusion = None
            self.mask_classifier = None

        self.use_adam_drop = use_adam_drop
        self.q_base = q_base
        self.lam = lam
        self.p_exe = p_exe
        self.dissonance_beta = dissonance_beta
        self.d = [hidden_dim, hidden_dim, hidden_dim]
        self.modality_drop = Modality_drop(dim_list=torch.tensor(self.d), p_exe=self.p_exe)
        self.last_alignment_info = {}
        self.last_proto_aux_logits = {}

    def _load_or_init_prototypes(self, n_classes, hidden_dim, prototype_path):
        if prototype_path and os.path.exists(prototype_path):
            loaded = torch.load(prototype_path, map_location="cpu")
            if isinstance(loaded, dict):
                for key in ("prototypes", "prototype", "weight", "centers"):
                    if key in loaded:
                        loaded = loaded[key]
                        break
            loaded = torch.as_tensor(loaded, dtype=torch.float32)
            if loaded.dim() == 2 and loaded.size(0) == n_classes and loaded.size(1) == hidden_dim:
                return F.normalize(loaded, p=2, dim=-1)

        if self.proto_anchor == "external":
            raise FileNotFoundError("proto_anchor='external' requires a valid --prototype_path with shape [num_classes, hidden_dim]")

        proto = torch.empty(n_classes, hidden_dim)
        nn.init.xavier_uniform_(proto)
        return F.normalize(proto, p=2, dim=-1)

    def _prototype_bank(self):
        if self.proto_anchor == "text_head" and hasattr(self, "t_output_layer"):
            linear = self.t_output_layer[-1]
            if isinstance(linear, nn.Linear) and linear.weight.size(0) == self.n_classes:
                p = linear.weight
                if self.proto_anchor_detach:
                    p = p.detach()
                return F.normalize(p, p=2, dim=-1)
        return F.normalize(self.emotion_prototypes, p=2, dim=-1)

    def _proto_enhance(self, x, umask, scale, gate_layer, proj_layer, norm_layer, name):
        if (not self.use_proto_nodes) or x is None or scale <= 0:
            return x

        p = self._prototype_bank()
        x_norm = F.normalize(x, p=2, dim=-1)
        score = torch.matmul(x_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)
        msg = torch.matmul(route, p)
        gate = torch.sigmoid(gate_layer(torch.cat([x, msg], dim=-1)))
        delta = gate * proj_layer(msg)
        out = norm_layer(x + float(scale) * delta)
        if umask is not None:
            out = out * umask.unsqueeze(-1).float()

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
            valid_route = route[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                gate_mean = gate[valid].mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                self.last_proto_info[name + "_entropy"] = entropy.detach()
                self.last_proto_info[name + "_gate"] = gate_mean.detach()
                self.last_proto_info[name + "_hist"] = hist.detach()
        return out

    def _proto_visual_graph_rescue(self, visual, umask, scale):
        if (not self.use_proto_nodes) or visual is None or scale <= 0:
            return visual

        p = self._prototype_bank()
        v_norm = F.normalize(visual, p=2, dim=-1)
        score = torch.matmul(v_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)

        proto_msg = torch.matmul(route, p)
        proto_msg = self.proto_v_proj(proto_msg)
        gate_in = torch.cat([visual, proto_msg], dim=-1)
        gate = torch.sigmoid(self.proto_v_gate(gate_in))
        rescued = visual + float(scale) * gate * proto_msg

        if umask is not None:
            rescued = torch.where(umask.unsqueeze(-1).bool(), rescued, visual)

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(visual.size()[:2], dtype=torch.bool, device=visual.device)
            valid_route = route[valid]
            valid_gate = gate[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                gate_mean = valid_gate.mean()
                top_conf = valid_route.max(dim=-1).values.mean()
                self.last_proto_info["visual_gate"] = gate_mean.detach()
                self.last_proto_info["visual_entropy"] = entropy.detach()
                self.last_proto_info["visual_conf"] = top_conf.detach()
                self.last_proto_info["visual_active"] = (valid_gate.mean(dim=-1) > 0.1).float().mean().detach()
                self.last_proto_info["visual_hist"] = hist.detach()

        return rescued

    def _proto_logit_residual(self, x, umask, scale, name):
        if (not self.use_proto_nodes) or x is None or scale <= 0:
            return None

        p = self._prototype_bank()
        x_norm = F.normalize(x, p=2, dim=-1)
        score = torch.matmul(x_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)
        residual = route - (1.0 / float(p.size(0)))

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
            valid_route = route[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                top_conf = valid_route.max(dim=-1).values.mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                self.last_proto_info[name + "_entropy"] = entropy.detach()
                self.last_proto_info[name + "_gate"] = top_conf.detach()
                self.last_proto_info[name + "_hist"] = hist.detach()

        return float(scale) * residual

    def _proto_safe_logit_residual(self, x, umask, base_logits, scale, name):
        if (not self.use_proto_nodes) or x is None or base_logits is None or scale <= 0:
            return None

        p = self._prototype_bank()
        x_norm = F.normalize(x, p=2, dim=-1)
        score = torch.matmul(x_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        dense_route = F.softmax(score, dim=-1)

        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)
        residual = route - (1.0 / float(p.size(0)))

        proto_conf = dense_route.max(dim=-1).values
        base_top2 = torch.topk(base_logits.detach(), k=2, dim=-1).values
        base_margin = base_top2[..., 0] - base_top2[..., 1]
        gate_temp = float(self.proto_safe_gate_temp)
        conf_gate = torch.sigmoid((proto_conf - float(self.proto_safe_conf)) * gate_temp)
        margin_gate = torch.sigmoid((float(self.proto_safe_margin) - base_margin) * gate_temp)
        gate = conf_gate * margin_gate

        if umask is not None:
            gate = gate * umask.float()

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
            valid_route = dense_route[valid]
            valid_gate = gate[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                self.last_proto_info[name + "_entropy"] = entropy.detach()
                self.last_proto_info[name + "_gate"] = valid_gate.mean().detach()
                self.last_proto_info[name + "_active"] = (valid_gate > 0.1).float().mean().detach()
                self.last_proto_info[name + "_hist"] = hist.detach()

        return float(scale) * gate.unsqueeze(-1) * residual

    def _proto_adapter_logits(self, visual, umask, scale, base_logits=None):
        if (not self.use_proto_nodes) or visual is None or self.proto_adapter_out is None or scale <= 0:
            return None

        p = self._prototype_bank()
        v_norm = F.normalize(visual, p=2, dim=-1)
        score = torch.matmul(v_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)
        proto_msg = torch.matmul(route, p)
        proto_hidden = torch.tanh(self.proto_v_proj(proto_msg))
        residual = self.proto_adapter_out(proto_hidden) * float(scale)
        adapter_gate = None
        if self.proto_adapter_margin_gate and base_logits is not None:
            base_top2 = torch.topk(base_logits.detach(), k=2, dim=-1).values
            base_margin = base_top2[..., 0] - base_top2[..., 1]
            adapter_gate = torch.sigmoid(
                (float(self.proto_adapter_margin) - base_margin) * float(self.proto_adapter_gate_temp)
            )
            residual = residual * adapter_gate.unsqueeze(-1)

        if umask is not None:
            residual = residual * umask.unsqueeze(-1).float()

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(visual.size()[:2], dtype=torch.bool, device=visual.device)
            valid_route = route[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                top_conf = valid_route.max(dim=-1).values.mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                res_norm = residual[valid].norm(dim=-1).mean()
                self.last_proto_info["adapter_entropy"] = entropy.detach()
                self.last_proto_info["adapter_conf"] = top_conf.detach()
                self.last_proto_info["adapter_res_norm"] = res_norm.detach()
                if adapter_gate is not None:
                    self.last_proto_info["adapter_gate"] = adapter_gate[valid].mean().detach()
                self.last_proto_info["adapter_hist"] = hist.detach()

        return residual

    def _proto_pair_bias(self, x, umask, scale, name):
        if (not self.use_proto_nodes) or x is None or scale <= 0:
            return None

        p = self._prototype_bank()
        x_norm = F.normalize(x, p=2, dim=-1)
        score = torch.matmul(x_norm, p.transpose(0, 1)) * float(self.proto_temperature)
        k = min(max(int(self.proto_top_k), 1), p.size(0))
        top_val, top_idx = torch.topk(score, k=k, dim=-1)
        sparse_score = torch.full_like(score, -1e4)
        sparse_score.scatter_(-1, top_idx, top_val)
        route = F.softmax(sparse_score, dim=-1)

        proto_sim = torch.matmul(route, route.transpose(1, 2))
        bias = (proto_sim - (1.0 / float(p.size(0)))) * float(scale)
        if umask is not None:
            pair_mask = umask.unsqueeze(1).float() * umask.unsqueeze(2).float()
            bias = bias * pair_mask

        with torch.no_grad():
            valid = umask.bool() if umask is not None else torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
            valid_route = route[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                top_conf = valid_route.max(dim=-1).values.mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                if umask is not None:
                    pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
                    valid_bias = bias[pair_valid]
                else:
                    valid_bias = bias.reshape(-1)
                self.last_proto_info[name + "_entropy"] = entropy.detach()
                self.last_proto_info[name + "_conf"] = top_conf.detach()
                self.last_proto_info[name + "_bias"] = valid_bias.abs().mean().detach() if valid_bias.numel() > 0 else bias.new_tensor(0.0)
                self.last_proto_info[name + "_hist"] = hist.detach()

        return bias

    def _proto_aux_logits(self, x, umask, name):
        if (not self.use_proto_nodes) or x is None:
            return None

        p = self._prototype_bank()
        x_norm = F.normalize(x, p=2, dim=-1)
        logits = torch.matmul(x_norm, p.transpose(0, 1)) * float(self.proto_temperature)

        with torch.no_grad():
            route = F.softmax(logits, dim=-1)
            valid = umask.bool() if umask is not None else torch.ones(x.size()[:2], dtype=torch.bool, device=x.device)
            valid_route = route[valid]
            if valid_route.numel() > 0:
                entropy = -(valid_route * torch.log(valid_route + 1e-8)).sum(-1).mean()
                top_conf = valid_route.max(dim=-1).values.mean()
                hist = torch.bincount(valid_route.argmax(-1), minlength=p.size(0)).float()
                hist = hist / hist.sum().clamp_min(1.0)
                self.last_proto_info[name + "_entropy"] = entropy.detach()
                self.last_proto_info[name + "_gate"] = top_conf.detach()
                self.last_proto_info[name + "_hist"] = hist.detach()

        self.last_proto_aux_logits[name] = logits
        return logits

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj, label=None, warm_up=1):
        self.last_proto_info = {}
        self.last_proto_aux_logits = {}
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(spk_idx.device)
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().to(spk_idx.device)

        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 0, 2))
        if visuf is not None and acouf is not None:
            visuf = self.visuf_input(visuf.permute(1, 0, 2))
            acouf = self.acouf_input(acouf.permute(1, 0, 2))
        else:
            visuf = None
            acouf = None

        h_base, h_t, h_a, h_v, align_info = self.modal_aligner(textf, acouf, visuf, u_mask, timestamps)
        self.last_alignment_info = align_info
        lambda_t, lambda_a, lambda_v = self.decay_predictor(h_t, h_a, h_v, u_mask)
        # Keep lambda predictions for module-4 integration without breaking current API.
        self.last_decay_rates = {
            'lambda_t': lambda_t,
            'lambda_a': lambda_a,
            'lambda_v': lambda_v,
        }
        textf = h_base
        textf, Cattention_weights = self.gatTer(textf, Cross_semantic_adj)
        textf, Sattention_weights = self.gatT(textf, Self_semantic_adj)
        if self.use_causal_graph:
            textf = self.causal_t(textf, lambda_t, timestamps, u_mask)
        B = textf.shape[0]

        sub_log_prog = []
        if visuf is not None and acouf is not None:
            acouf = h_a
            acouf, attention_weights = self.a_a(acouf, u_mask, spk_embeddings)
            proto_graph_active = (
                self.use_proto_nodes
                and getattr(self, "proto_diff_graph", False)
                and warm_up == 0
                and int(self.current_epoch) >= int(self.proto_start_epoch)
            )
            audio_proto_bias = self._proto_pair_bias(
                acouf, u_mask, getattr(self, "proto_diff_scale", 0.0), "audio_graph"
            ) if proto_graph_active else None
            acouf, Cattention_weights_a = self.gatAer(acouf, Cross_semantic_adj, proto_bias=audio_proto_bias)
            acouf, Sattention_weights_a = self.gatA(acouf, Self_semantic_adj, proto_bias=audio_proto_bias)
            acouf = self.agate(acouf)
            visuf = h_v
            visuf, attention_weights = self.v_v(visuf, u_mask, spk_embeddings)
            visual_proto_bias = self._proto_pair_bias(
                visuf, u_mask, getattr(self, "proto_diff_scale", 0.0), "visual_graph"
            ) if proto_graph_active else None
            visuf, Cattention_weights_v = self.gatVer(visuf, Cross_semantic_adj, proto_bias=visual_proto_bias)
            visuf, Sattention_weights_v = self.gatV(visuf, Self_semantic_adj, proto_bias=visual_proto_bias)
            visuf = self.vgate(visuf)
            if self.use_causal_graph:
                acouf = self.causal_a(acouf, lambda_a, timestamps, u_mask)
                visuf = self.causal_v(visuf, lambda_v, timestamps, u_mask)
            proto_active = self.use_proto_nodes and warm_up == 0 and int(self.current_epoch) >= int(self.proto_start_epoch)
            if proto_active and self.proto_mode == "visual_graph":
                visuf = self._proto_visual_graph_rescue(visuf, u_mask, self.proto_visual_scale)
            if proto_active and self.proto_mode in ("feature", "both"):
                acouf = self._proto_enhance(
                    acouf, u_mask, self.proto_audio_scale,
                    self.proto_a_gate, self.proto_a_proj, self.proto_a_norm, "audio"
                )
                visuf = self._proto_enhance(
                    visuf, u_mask, self.proto_visual_scale,
                    self.proto_v_gate, self.proto_v_proj, self.proto_v_norm, "visual"
                )
            if proto_active and self.proto_mode == "aux":
                self._proto_aux_logits(acouf, u_mask, "audio")
                self._proto_aux_logits(visuf, u_mask, "visual")
            fead_list = [textf, visuf, acouf]
        else:
            fead_list = [textf]

        num_mod = len(fead_list)
        device = textf.device
        update_flag = torch.ones([B], dtype=torch.int32, device=device)
        if self.use_adam_drop and label is not None and warm_up == 0:
            # compute perfs
            t = self.t_output_layer(textf)
            perf_t = self._get_performance(t, label, u_mask)
            if visuf is not None and acouf is not None:
                a = self.a_output_layer(acouf)
                v = self.v_output_layer(visuf)
                perf_a = self._get_performance(a, label, u_mask)
                perf_v = self._get_performance(v, label, u_mask)
                performances = torch.tensor([perf_t, perf_v, perf_a], device=device)
            else:
                performances = torch.tensor([perf_t], device=device)
            q = calcu_q(performances, self.q_base, self.lam)
            dissonance_mean = align_info.get("dissonance_mean") if isinstance(align_info, dict) else None
            if dissonance_mean is not None:
                dis_norm = torch.clamp(dissonance_mean / 2.0, min=0.0, max=1.0)
                dis_scale = torch.clamp(1.0 - self.dissonance_beta * dis_norm, min=0.5, max=1.0)
                q = torch.clamp(q * dis_scale.to(device=device, dtype=q.dtype), 0.0, 1.0)
            if self.protect_visual_drop and visuf is not None and acouf is not None and q.numel() >= 3:
                with torch.no_grad():
                    visual_conf = self.last_proto_info.get("visual_conf")
                    if visual_conf is None:
                        v_logits = self.v_output_layer(visuf)
                        visual_conf = F.softmax(v_logits, dim=-1).max(dim=-1).values
                        valid = u_mask.bool() if u_mask is not None else torch.ones(visuf.size()[:2], dtype=torch.bool, device=device)
                        visual_conf = visual_conf[valid].mean() if visual_conf[valid].numel() > 0 else q.new_tensor(0.0)
                    visual_conf = torch.as_tensor(visual_conf, device=device, dtype=q.dtype).clamp(0.0, 1.0)
                    weak_visual = (1.0 - visual_conf) * float(self.protect_visual_strength)
                    q[1] = torch.clamp(q[1] * (1.0 - weak_visual), 0.0, 1.0)
                    q[0] = torch.clamp(q[0] + self.q_base * 0.25 * weak_visual, 0.0, 1.0)
                    self.last_proto_info["visual_drop_protect"] = weak_visual.detach()
            dim_list = torch.tensor([textf.shape[-1]] * num_mod, device=device)
            self.modality_drop.dim_list = dim_list
            cleaned_fea, update_flag = self.modality_drop.execute_drop(fead_list, q)
            textf = cleaned_fea[0]
            if visuf is not None and acouf is not None:
                visuf = cleaned_fea[1]
                acouf = cleaned_fea[2]

        t = self.t_output_layer(textf)
        if visuf is not None and acouf is not None:
            self.last_fused_feature = torch.cat([textf, acouf, visuf], dim=-1).detach()
            a = self.a_output_layer(acouf)
            v = self.v_output_layer(visuf)
            proto_active = self.use_proto_nodes and warm_up == 0 and int(self.current_epoch) >= int(self.proto_start_epoch)
            if proto_active and self.proto_mode in ("logit", "both"):
                a_res = self._proto_logit_residual(acouf, u_mask, self.proto_audio_scale, "audio")
                v_res = self._proto_logit_residual(visuf, u_mask, self.proto_visual_scale, "visual")
                if a_res is not None:
                    a = a + a_res
                if v_res is not None:
                    v = v + v_res
            baseline_logits = t + a + v

            if self.use_module5:
                # Module 5: relation-aware fusion with adaptive modality balancing.
                if label is not None:
                    perf_t = self._get_performance(t, label, u_mask)
                    perf_a = self._get_performance(a, label, u_mask)
                    perf_v = self._get_performance(v, label, u_mask)
                    current_perf = torch.tensor(
                        [perf_t, perf_a, perf_v],
                        device=device,
                        dtype=self.modal_perf_ema.dtype,
                    )
                    if self.training:
                        with torch.no_grad():
                            self.modal_perf_ema.mul_(self.fusion_perf_momentum).add_(
                                current_perf * (1.0 - self.fusion_perf_momentum)
                            )

                perf_for_fusion = self.modal_perf_ema.detach().to(device=device, dtype=textf.dtype)
                h_fused, fusion_info = self.relation_fusion(textf, acouf, visuf, perf_for_fusion)
                fused_logits = self.fused_output_layer(h_fused)
                # Residual-safe fusion with a controlled fusion residual.
                if warm_up == 1:
                    fusion_alpha = textf.new_tensor(0.0)
                else:
                    fusion_alpha = torch.sigmoid(self.fusion_res_alpha_logit).to(dtype=textf.dtype, device=device)
                all_final_out = baseline_logits + fusion_alpha * fused_logits
                fusion_info["fusion_alpha"] = fusion_alpha.detach()
                self.last_fusion_info = fusion_info
            else:
                all_final_out = baseline_logits
                self.last_fusion_info = {}

            if proto_active and self.proto_mode == "safe_logit":
                v_safe_res = self._proto_safe_logit_residual(
                    visuf, u_mask, all_final_out, self.proto_visual_scale, "visual_safe"
                )
                if v_safe_res is not None:
                    all_final_out = all_final_out + v_safe_res
            if proto_active and self.proto_mode == "adapter":
                proto_res = self._proto_adapter_logits(visuf, u_mask, self.proto_visual_scale, all_final_out)
                if proto_res is not None:
                    all_final_out = all_final_out + proto_res

            sub_log_prog.append(F.log_softmax(t, dim=-1))
            sub_log_prog.append(F.log_softmax(a, dim=-1))
            sub_log_prog.append(F.log_softmax(v, dim=-1))
        else:
            self.last_fusion_info = {}
            self.last_fused_feature = textf.detach()
            all_final_out = t
            sub_log_prog.append(F.log_softmax(t, dim=-1))
        aux_mask_loss = None
        if self.enable_mask_aux and self.training and warm_up == 0 and label is not None and visuf is not None and acouf is not None:
            aux_mask_loss = self.masked_aux_loss(
                h_t=h_t,
                h_a=h_a,
                h_v=h_v,
                u_mask=u_mask,
                timestamps=timestamps,
                label=label,
                mask_ratio=self.mask_ratio,
            )

        all_log_prob = F.log_softmax(all_final_out, dim=-1)
        all_prob = F.softmax(all_final_out, dim=-1)
        return sub_log_prog, all_log_prob, all_prob, all_final_out, update_flag, aux_mask_loss

    def masked_aux_loss(self, h_t, h_a, h_v, u_mask, timestamps, label, mask_ratio=None):
        """
        Module 6: masked self-supervised temporal prediction.
        It uses module-3/4 and an auxiliary fusion/classifier head.
        """
        device = h_t.device
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        h_t_det = h_t.detach()
        h_a_det = h_a.detach()
        h_v_det = h_v.detach()

        lambda_t, lambda_a, lambda_v = self.decay_predictor(h_t_det, h_a_det, h_v_det, u_mask)

        valid_mask = u_mask > 0
        rand_mask = torch.rand_like(u_mask.float()) < mask_ratio
        mask = rand_mask & valid_mask

        if mask.sum() == 0:
            valid_idx = valid_mask.view(-1).nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                return h_t_det.sum() * 0.0
            pick = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=device)]
            mask = mask.view(-1)
            mask[pick] = True
            mask = mask.view_as(u_mask)

        h_t_masked = h_t_det.masked_fill(mask.unsqueeze(-1), 0.0)
        h_a_masked = h_a_det.masked_fill(mask.unsqueeze(-1), 0.0)
        h_v_masked = h_v_det.masked_fill(mask.unsqueeze(-1), 0.0)

        h_t_prime = self.causal_t(h_t_masked, lambda_t, timestamps, u_mask)
        h_a_prime = self.causal_a(h_a_masked, lambda_a, timestamps, u_mask)
        h_v_prime = self.causal_v(h_v_masked, lambda_v, timestamps, u_mask)

        concat = torch.cat([h_t_prime, h_a_prime, h_v_prime], dim=-1)
        aux_feat = self.mask_mlp_fusion(concat)
        aux_logits = self.mask_classifier(aux_feat)

        flat_mask = mask.view(-1)
        if flat_mask.sum() == 0:
            return aux_logits.sum() * 0.0

        flat_logits = aux_logits.view(-1, self.n_classes)[flat_mask]
        flat_labels = label.view(-1)[flat_mask].long()
        return F.cross_entropy(flat_logits, flat_labels)

    def _get_performance(self, logit, label, umask):
        if logit is None:
            return 0.0
        flat_logit = logit.view(-1, self.n_classes)
        flat_label = label.view(-1)
        flat_umask = umask.view(-1) == 1
        valid_preds = torch.argmax(flat_logit[flat_umask], dim=-1)
        valid_labels = flat_label[flat_umask]
        if len(valid_labels) == 0:
            return 0.0
        f1 = f1_score(valid_labels.cpu().numpy(), valid_preds.cpu().numpy(), average='weighted')
        return f1  




# Backward-compatible alias for checkpoints saved before the PACE code layout.
Transformer_Based_Model = PACE

