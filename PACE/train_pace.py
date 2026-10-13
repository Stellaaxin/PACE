# PACE training entry.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "OMP_NUM_THREADS" in os.environ and (not str(os.environ["OMP_NUM_THREADS"]).isdigit()):
    os.environ["OMP_NUM_THREADS"] = "8"
if "MKL_NUM_THREADS" in os.environ and (not str(os.environ["MKL_NUM_THREADS"]).isdigit()):
    os.environ["MKL_NUM_THREADS"] = "8"
import numpy as np, argparse, time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import csv
import json
from collections import Counter
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from dataloader import IEMOCAPDataset, MELDDataset, DailyDialogDataset
from model_pace import MaskedNLLLoss, PACE
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import random
# from vision import TwoD_Tsne, sampleCount, confuPLT

# for FLOPs estimation
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop is not installed, FLOPs will use fallback estimation.")


DATASET_CLASS_PRIORS = {
    "IEMOCAP": [0.086747, 0.144406, 0.227883, 0.160585, 0.127711, 0.252668],
    "MELD": [0.481226, 0.107663, 0.191571, 0.079669, 0.154023, 0.026054, 0.132184],
    "DailyDialog": [0.0017, 0.0034, 0.1251, 0.8310, 0.0099, 0.0177, 0.0112],
}


class ReproIEMOCAPDataset(Dataset):
    """IEMOCAP loader compatible with both 12-field and compact 7-field pickles."""

    def __init__(self, path, windows, train=True, sort_keys=False, verbose=True):
        payload = pickle.load(open(path, 'rb'), encoding='latin1')
        if len(payload) == 12:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.roberta2, self.roberta3, self.roberta4, \
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
            self.testVid = payload
        elif len(payload) == 9:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid = payload
        elif len(payload) == 8:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoVisual, self.trainVid, self.testVid = payload
            self.videoSentence = None
        elif len(payload) == 7:
            self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoVisual, self.trainVid, self.testVid = payload
            self.videoIDs = None
            self.videoSentence = None
        else:
            raise ValueError(f'Unsupported IEMOCAP pickle format: expected 7/8/9/12 fields, got {len(payload)}')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        if sort_keys:
            self.keys = sorted(self.keys)
        self.len = len(self.keys)
        self.windows = windows
        if verbose:
            split = 'train' if train else 'test'
            print(f'[IEMOCAP loader] split={split}, pickle_fields={len(payload)}, dialogs={self.len}, sort_keys={sort_keys}')
            self._print_fingerprint(split)

    def _first_shape(self, store):
        if self.len == 0:
            return None
        first = self.keys[0]
        return tuple(np.asarray(store[first]).shape)

    def _label_counts(self):
        counter = Counter()
        for vid in self.keys:
            counter.update([int(x) for x in self.videoLabels[vid]])
        return [counter.get(i, 0) for i in range(6)]

    def _print_fingerprint(self, split):
        head = self.keys[:3]
        print(
            f'[IEMOCAP fingerprint] split={split}, first_keys={head}, '
            f'text_shape={self._first_shape(self.videoText)}, '
            f'audio_shape={self._first_shape(self.videoAudio)}, '
            f'visual_shape={self._first_shape(self.videoVisual)}, '
            f'label_counts={self._label_counts()}'
        )

    @staticmethod
    def _speaker_to_onehot(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=np.float32)
            if arr.size == 2:
                return arr.tolist()
        if isinstance(x, str):
            return [1.0, 0.0] if x in ('M', '0', 'male', 'Male') else [0.0, 1.0]
        return [1.0, 0.0] if int(x) == 0 else [0.0, 1.0]

    def _speaker_ids(self, data):
        speakers = []
        for item in data:
            qmask = item[3]
            speakers.append(torch.argmax(qmask, dim=-1).tolist())
        return speakers

    def get_semantic_adj(self, data):
        batch_speakers = self._speaker_ids(data)
        max_len = max(len(speaker) for speaker in batch_speakers)
        semantic_adj = []
        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speaker):
                        break
                    if speaker[i] == speaker[j]:
                        if i == j:
                            s[i, j] = 1
                        elif i < j:
                            s[i, j] = 2
                        else:
                            s[i, j] = 3
                    else:
                        if i < j:
                            s[i, j] = 4
                        elif i > j:
                            s[i, j] = 5
            semantic_adj.append(s)
        return torch.stack(semantic_adj)

    def getSelf_semantic_adj(self, data):
        batch_speakers = self._speaker_ids(data)
        max_len = max(len(speaker) for speaker in batch_speakers)
        self_adj = []
        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speaker):
                        break
                    if speaker[i] == speaker[j]:
                        if i == j:
                            s[i, j] = 1
                        elif i > j:
                            s[i, j] = 2
                        else:
                            s[i, j] = 3
            self_adj.append(s)
        return torch.stack(self_adj)

    def getCross_semantic_adj(self, data):
        batch_speakers = self._speaker_ids(data)
        max_len = max(len(speaker) for speaker in batch_speakers)
        cross_adj = []
        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0:
                        continue
                    if j == len(speaker):
                        break
                    if speaker[i] == speaker[j]:
                        if i == j:
                            s[i, j] = 1
                    else:
                        if i < j:
                            s[i, j] = 4
                        elif i > j:
                            s[i, j] = 5
            cross_adj.append(s)
        return torch.stack(cross_adj)

    def __getitem__(self, index):
        vid = self.keys[index]
        speakers = [self._speaker_to_onehot(x) for x in self.videoSpeakers[vid]]
        return torch.FloatTensor(np.array(self.videoText[vid])), \
               torch.FloatTensor(np.array(self.videoVisual[vid])), \
               torch.FloatTensor(np.array(self.videoAudio[vid])), \
               torch.FloatTensor(speakers), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        self_adj = self.getSelf_semantic_adj(data)
        cross_adj = self.getCross_semantic_adj(data)
        semantic_adj = self.get_semantic_adj(data)
        batch = [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist()
                 for i in dat]
        batch.append(torch.LongTensor(self_adj))
        batch.append(torch.LongTensor(cross_adj))
        batch.append(torch.LongTensor(semantic_adj))
        return batch


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class ModelSwa:
    def __init__(self, model):
        self.n_averaged = 0
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.n_averaged == 0:
                self.shadow[name] = param.data.clone()
            else:
                self.shadow[name] = (self.shadow[name] * self.n_averaged + param.data) / (self.n_averaged + 1.0)
        self.n_averaged += 1

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def build_class_balanced_weights(class_priors, beta=0.9999, pseudo_total=10000):
    priors = torch.tensor(class_priors, dtype=torch.float32)
    priors = priors / (priors.sum() + 1e-12)
    counts = (priors * pseudo_total).clamp(min=1.0)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / (weights.mean() + 1e-12)
    return weights


def masked_focal_loss_from_logits(
    logits,
    target,
    mask,
    class_weight=None,
    gamma=2.0,
    logit_adjust=None,
):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_target = target.view(-1).long()
    flat_mask = mask.view(-1) > 0

    if flat_mask.sum() == 0:
        return flat_logits.sum() * 0.0

    valid_logits = flat_logits[flat_mask]
    valid_target = flat_target[flat_mask]

    if logit_adjust is not None:
        valid_logits = valid_logits + logit_adjust.view(1, -1)

    log_probs = F.log_softmax(valid_logits, dim=-1)
    probs = torch.exp(log_probs)
    log_pt = log_probs.gather(dim=1, index=valid_target.unsqueeze(1)).squeeze(1)
    pt = probs.gather(dim=1, index=valid_target.unsqueeze(1)).squeeze(1)
    focal_factor = torch.pow((1.0 - pt).clamp(min=1e-6), gamma)
    loss = -focal_factor * log_pt

    if class_weight is not None:
        sample_weight = class_weight[valid_target]
        loss = loss * sample_weight
        return loss.sum() / (sample_weight.sum() + 1e-8)

    return loss.mean()


def masked_label_smoothing_ce_from_logits(
    logits,
    target,
    mask,
    smoothing=0.1,
    class_weight=None,
    logit_adjust=None,
):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_target = target.view(-1).long()
    flat_mask = mask.view(-1) > 0
    if flat_mask.sum() == 0:
        return flat_logits.sum() * 0.0

    valid_logits = flat_logits[flat_mask]
    valid_target = flat_target[flat_mask]
    if logit_adjust is not None:
        valid_logits = valid_logits + logit_adjust.view(1, -1)

    num_classes = valid_logits.size(-1)
    with torch.no_grad():
        true_dist = torch.full_like(valid_logits, fill_value=smoothing / max(num_classes - 1, 1))
        true_dist.scatter_(1, valid_target.unsqueeze(1), 1.0 - smoothing)

    log_probs = F.log_softmax(valid_logits, dim=-1)
    loss = -(true_dist * log_probs).sum(dim=-1)

    if class_weight is not None:
        sample_weight = class_weight[valid_target]
        loss = loss * sample_weight
        return loss.sum() / (sample_weight.sum() + 1e-8)
    return loss.mean()


def masked_symmetric_kl_from_logits(logits_p, logits_q, mask):
    p = logits_p.view(-1, logits_p.size(-1))
    q = logits_q.view(-1, logits_q.size(-1))
    flat_mask = mask.view(-1) > 0
    if flat_mask.sum() == 0:
        return p.sum() * 0.0
    p = p[flat_mask]
    q = q[flat_mask]
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    p_prob = p_log.exp()
    q_prob = q_log.exp()
    kl_pq = F.kl_div(p_log, q_prob, reduction='none').sum(dim=-1)
    kl_qp = F.kl_div(q_log, p_prob, reduction='none').sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def append_txt(log_path, line):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(str(line) + '\n')

def dump_epoch_csv(csv_path, epoch_records):
    if not epoch_records:
        return
    fieldnames = list(epoch_records[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_records)

def format_flops(flops):
    if flops is None or flops <= 0:
        return "N/A"
    flops = float(flops)
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    return f"{flops:.0f}"


def make_position_timestamps(umask):
    """Build monotonic utterance timestamps when the dataset has no explicit times."""
    steps = torch.arange(umask.size(1), device=umask.device, dtype=umask.dtype)
    return steps.unsqueeze(0).expand_as(umask) * umask


def unpack_batch(data, dataset, cuda=False):
    if cuda:
        data = [d.cuda() for d in data]
    if dataset == "DailyDialog":
        if len(data) == 8:
            textf, qmask, umask, label, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = data
        else:
            textf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = data
            timestamps = make_position_timestamps(umask)
        return textf, None, None, qmask, umask, label, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj

    if len(data) == 10:
        return data

    textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = data
    timestamps = make_position_timestamps(umask)
    return textf, visuf, acouf, qmask, umask, label, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, windows=5):
    trainset = MELDDataset('data/meld_multimodal_features.pkl', windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5, sort_keys=False):
    trainset = ReproIEMOCAPDataset("data/iemocap_multimodal_features.pkl", windows=windows, sort_keys=sort_keys)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = ReproIEMOCAPDataset("data/iemocap_multimodal_features.pkl", train=False, windows=windows, sort_keys=sort_keys)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_DailyDialog_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = DailyDialogDataset("data/dd_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogDataset("data/dd_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def train_or_eval_model(
    model,
    loss_function,
    dataloader,
    epoch,
    optimizer=None,
    train=False,
    dataset=None,
    warm_up_epochs=60,
    lambda_mask=0.1,
    loss_type="nll",
    label_smoothing=0.0,
    rdrop_alpha=0.0,
    focal_gamma=2.0,
    focal_class_weight=None,
    logit_adjust=None,
    grad_clip=0.0,
    ema=None,
    proto_aux_weight=0.0,
    proto_aux_start_epoch=35,
    proto_aux_modalities="audio,visual",
):
    losses, preds, labels, masks, all_transformer_outs, umasks = [], [], [], [], [], []
    warm_up = 1 if epoch < warm_up_epochs else 0
    aux_mask_losses = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    if hasattr(model, 'current_epoch'):
        model.current_epoch = epoch + 1

    batch_times = []
    inference_times = []
    total_samples = 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for data in dataloader:
        batch_start_time = time.time()
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = \
            unpack_batch(data, dataset, cuda=cuda)

        umasks.append(umask)
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        batch_sample_count = umask.sum().item()
        total_samples += batch_sample_count
        
        if not train:
            inference_start = time.time()
        
        log_prob, all_log_prob, all_prob, all_final_out, update_flag, aux_mask_loss = \
            model(textf, visuf, acouf, umask, qmask, lengths, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj, label=label if train else None, warm_up=warm_up)
        
        if not train:
            inference_time = time.time() - inference_start
            if batch_sample_count > 0:
                inference_times.append(inference_time / batch_sample_count)
            else:
                batch_size_actual = textf.size(0) if textf is not None else 1
                if batch_size_actual > 0:
                    inference_times.append(inference_time / batch_size_actual)
        lp_all = all_log_prob.view(-1, all_log_prob.size()[-1])
        labels_ = label.view(-1)
        umask_ = umask.view(-1)

        def compute_supervised_loss(log_prob_local, all_log_prob_local, all_final_out_local):
            lp_all_local = all_log_prob_local.view(-1, all_log_prob_local.size()[-1])
            if loss_type == "cb_focal":
                fusion_local = masked_focal_loss_from_logits(
                    logits=all_final_out_local,
                    target=labels_,
                    mask=umask_,
                    class_weight=focal_class_weight,
                    gamma=focal_gamma,
                    logit_adjust=logit_adjust,
                )
            elif loss_type == "ls_ce":
                fusion_local = masked_label_smoothing_ce_from_logits(
                    logits=all_final_out_local,
                    target=labels_,
                    mask=umask_,
                    smoothing=label_smoothing,
                    class_weight=focal_class_weight,
                    logit_adjust=logit_adjust,
                )
            else:
                fusion_local = loss_function(lp_all_local, labels_, umask_)

            if dataset == "DailyDialog":
                total_local = fusion_local
            else:
                t_loss_local = loss_function(log_prob_local[0].view(-1, log_prob_local[0].size()[-1]), labels_, umask_)
                a_loss_local = loss_function(log_prob_local[1].view(-1, log_prob_local[1].size()[-1]), labels_, umask_)
                v_loss_local = loss_function(log_prob_local[2].view(-1, log_prob_local[2].size()[-1]), labels_, umask_)
                total_local = fusion_local + t_loss_local * (t_loss_local / 10) + a_loss_local * (a_loss_local / 10) + v_loss_local * (v_loss_local / 10)
            return total_local, fusion_local

        loss, fusion_loss = compute_supervised_loss(log_prob, all_log_prob, all_final_out)
        total_loss = loss
        pred_prob = all_prob
        proto_aux_loss = None
        if train and proto_aux_weight > 0 and (epoch + 1) >= proto_aux_start_epoch:
            aux_logits = getattr(model, 'last_proto_aux_logits', {})
            allowed_aux = {
                item.strip().lower()
                for item in str(proto_aux_modalities).split(',')
                if item.strip()
            }
            aux_losses = []
            for proto_name, proto_logits in aux_logits.items():
                if allowed_aux and proto_name.lower() not in allowed_aux:
                    continue
                aux_losses.append(masked_label_smoothing_ce_from_logits(
                    logits=proto_logits,
                    target=labels_,
                    mask=umask_,
                    smoothing=0.0,
                    class_weight=focal_class_weight,
                    logit_adjust=logit_adjust,
                ))
            if aux_losses:
                proto_aux_loss = sum(aux_losses) / len(aux_losses)
                total_loss = total_loss + float(proto_aux_weight) * proto_aux_loss

        if train and rdrop_alpha > 0:
            log_prob2, all_log_prob2, all_prob2, all_final_out2, _, aux_mask_loss2 = model(
                textf, visuf, acouf, umask, qmask, lengths, timestamps,
                Self_semantic_adj, Cross_semantic_adj, Semantic_adj,
                label=label, warm_up=warm_up
            )
            loss2, _ = compute_supervised_loss(log_prob2, all_log_prob2, all_final_out2)
            kl = masked_symmetric_kl_from_logits(all_final_out, all_final_out2, umask_)
            total_loss = 0.5 * (loss + loss2) + rdrop_alpha * kl
            pred_prob = 0.5 * (all_prob + all_prob2)
            if proto_aux_loss is not None:
                total_loss = total_loss + float(proto_aux_weight) * proto_aux_loss

            if aux_mask_loss is not None and aux_mask_loss2 is not None:
                aux_mean = 0.5 * (aux_mask_loss + aux_mask_loss2)
                total_loss = total_loss + lambda_mask * aux_mean
                aux_mask_losses.append(aux_mean.item())
            elif aux_mask_loss is not None:
                total_loss = total_loss + lambda_mask * aux_mask_loss
                aux_mask_losses.append(aux_mask_loss.item())
        else:
            if train and aux_mask_loss is not None:
                total_loss = total_loss + lambda_mask * aux_mask_loss
                aux_mask_losses.append(aux_mask_loss.item())

        lp_ = pred_prob.view(-1, pred_prob.size()[-1])
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        all_transformer_outs.append(all_final_out)
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask_.cpu().numpy())

        losses.append(total_loss.item()*masks[-1].sum())
        if train:
            total_loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
    peak_memory_mb = 0
    if cuda:
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [], 0, 0, 0, 0

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    if inference_times:
        avg_inference_time = np.mean(inference_times)
    else:
        avg_inference_time = 0
        if not train and batch_times:
            avg_batch_time_for_inference = np.mean(batch_times)
            if total_samples > 0:
                avg_inference_time = avg_batch_time_for_inference / (total_samples / len(batch_times)) if len(batch_times) > 0 else 0
    
    avg_aux_mask_loss = round(float(np.mean(aux_mask_losses)), 6) if aux_mask_losses else 0.0
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, all_transformer_outs, umasks, avg_batch_time, avg_inference_time, peak_memory_mb, avg_aux_mask_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='none', choices=['none', 'pace'],
                        help='short reproducible preset; pace is the main PACE single-model configuration')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--seed', type=int, default=2094, help='random seed')
    parser.add_argument('--lr', type=float, default=0.000068, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--windows', type=int, default=20, help='number of windows')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='model output path')
    parser.add_argument('--disable_modality_dropout', action='store_true', default=False, help='disable modality dropout')
    parser.add_argument('--q_base', type=float, default=0.3, help='q_base parameter')
    parser.add_argument('--lam', type=float, default=0.9, help='lam parameter')
    parser.add_argument('--p_exe', type=float, default=0.5, help='p_exe parameter')
    parser.add_argument('--causal_topk', type=int, default=8, help='top-k historical neighbors in module-4 causal graph')
    parser.add_argument('--causal_time_beta', type=float, default=1.0, help='time-bias strength in module-4 causal graph')
    parser.add_argument('--disable_causal_graph', action='store_true', default=False,
                        help='disable module-4 causal temporal graph propagation for ablation')
    parser.add_argument('--dissonance_beta', type=float, default=0.25, help='scale factor for modality-dissonance-aware routing')
    parser.add_argument('--disable_module5', action='store_true', default=False, help='disable relation-aware fusion and use summed modality logits')
    parser.add_argument('--fusion_num_experts', type=int, default=4, help='number of experts in module-5 relation-aware fusion')
    parser.add_argument('--fusion_perf_momentum', type=float, default=0.9, help='EMA momentum of modality performance for module-5 balancing')
    parser.add_argument('--fusion_res_alpha_init', type=float, default=0.1, help='initial residual weight of module-5 fusion branch')
    parser.add_argument('--enable_mask_aux', action='store_true', default=False, help='enable module-6 masked self-supervised temporal prediction')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='mask ratio for module-6')
    parser.add_argument('--lambda_mask', type=float, default=0.1, help='loss weight for module-6 masked auxiliary task')
    parser.add_argument('--use_proto_nodes', action='store_true', default=False,
                        help='enable emotion prototype virtual nodes for weak-modality compensation')
    parser.add_argument('--prototype_path', type=str, default='',
                        help='optional [num_classes, hidden_dim] prototype tensor path')
    parser.add_argument('--proto_anchor', type=str, default='learned', choices=['learned', 'text_head', 'external'],
                        help='prototype node source: learned vectors or text classifier class anchors')
    parser.add_argument('--proto_anchor_detach', action='store_true', default=True,
                        help='detach text-head prototype anchors so visual graph does not perturb the text head directly')
    parser.add_argument('--proto_top_k', type=int, default=2, help='top-k prototype routes per utterance')
    parser.add_argument('--proto_audio_scale', type=float, default=0.04, help='audio prototype residual scale')
    parser.add_argument('--proto_visual_scale', type=float, default=0.06, help='visual prototype residual scale')
    parser.add_argument('--proto_temperature', type=float, default=8.0, help='prototype routing temperature')
    parser.add_argument('--proto_mode', type=str, default='feature', choices=['feature', 'logit', 'both', 'aux', 'safe_logit', 'visual_graph', 'adapter'],
                        help='prototype node injection mode')
    parser.add_argument('--proto_start_epoch', type=int, default=10,
                        help='1-based epoch to activate prototype nodes after warm-up')
    parser.add_argument('--proto_aux_weight', type=float, default=0.0,
                        help='training-only auxiliary loss weight for prototype-node routes')
    parser.add_argument('--proto_aux_start_epoch', type=int, default=35,
                        help='1-based epoch to activate prototype auxiliary loss')
    parser.add_argument('--proto_aux_modalities', type=str, default='audio,visual',
                        help='comma-separated modalities for prototype aux CE, e.g. visual or audio,visual')
    parser.add_argument('--proto_safe_conf', type=float, default=0.34,
                        help='safe_logit confidence threshold for visual-prototype route')
    parser.add_argument('--proto_safe_margin', type=float, default=0.65,
                        help='safe_logit only activates when base logit margin is below this value')
    parser.add_argument('--proto_safe_gate_temp', type=float, default=10.0,
                        help='safe_logit sigmoid gate temperature')
    parser.add_argument('--protect_visual_drop', action='store_true', default=False,
                        help='protect weak visual modality during modality dropout when prototype evidence is weak')
    parser.add_argument('--protect_visual_strength', type=float, default=0.55,
                        help='strength of visual-protected modality-dropout adjustment')
    parser.add_argument('--proto_adapter_margin_gate', action='store_true', default=False,
                        help='apply prototype adapter residual mainly on low-margin boundary samples')
    parser.add_argument('--proto_adapter_margin', type=float, default=0.65,
                        help='base-logit margin threshold for prototype adapter gate')
    parser.add_argument('--proto_adapter_gate_temp', type=float, default=8.0,
                        help='sigmoid temperature for prototype adapter margin gate')
    parser.add_argument('--proto_diff_graph', action='store_true', default=False,
                        help='enable prototype-conditioned differential graph attention for audio/visual branches')
    parser.add_argument('--proto_diff_scale', type=float, default=0.06,
                        help='scale of prototype pairwise bias injected into differential graph attention')
    parser.add_argument('--loss_type', type=str, default='nll', choices=['nll', 'cb_focal', 'ls_ce'], help='training loss for fused head')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for ls_ce loss')
    parser.add_argument('--rdrop_alpha', type=float, default=0.0, help='R-Drop consistency weight; set 0 to disable')
    parser.add_argument(
        '--reg_mode',
        type=str,
        default='manual',
        choices=['manual', 'baseline', 'cb_focal', 'ls_ce', 'cb_focal_rdrop', 'ls_ce_rdrop'],
        help='one-click preset for imbalance-aware training and consistency regularization',
    )
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for focal loss')
    parser.add_argument('--focal_beta', type=float, default=0.9999, help='beta for class-balanced focal weights')
    parser.add_argument('--logit_adjust_tau', type=float, default=0.0, help='tau for logit adjustment by class prior')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='max grad norm; set <=0 to disable')
    parser.add_argument('--use_cosine_scheduler', action='store_true', default=False, help='use cosine annealing LR scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum lr for cosine scheduler')
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA weights for evaluation')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    parser.add_argument('--ema_eval_start_epoch', type=int, default=60, help='start epoch (1-based) to evaluate with EMA shadow weights')
    parser.add_argument('--use_swa', action='store_true', default=False, help='use SWA model averaging')
    parser.add_argument('--swa_start_epoch', type=int, default=70, help='start epoch (1-based) to update SWA')
    parser.add_argument('--swa_eval_start_epoch', type=int, default=80, help='start epoch (1-based) to evaluate with SWA weights')
    parser.add_argument('--warm_up_epochs', type=int, default=60, help='warm up epochs')
    parser.add_argument('--log-dir', type=str, default='./logs', help='directory to store txt/csv logs')
    parser.add_argument('--run-name', type=str, default='', help='optional run name for log file')
    parser.add_argument('--save_topk', type=int, default=0, help='save top-k checkpoints by test fscore each run; 0 disables')
    parser.add_argument('--topk_dir', type=str, default='', help='optional directory for top-k checkpoints; default: <save_model_path>/topk_ckpts/<run_name>')
    parser.add_argument('--sort_dialog_keys', action='store_true', default=False,
                        help='sort dialogue ids before sampling for stricter reproducibility across pickle/set variants')
    parser.add_argument('--probe_data_only', action='store_true', default=False,
                        help='build loaders, print data fingerprints, and exit before training')
    provided_args = set()
    argv_tokens = sys.argv[1:]
    for action in parser._actions:
        for opt in action.option_strings:
            if any(token == opt or token.startswith(opt + '=') for token in argv_tokens):
                provided_args.add(action.dest)
    args = parser.parse_args()
    if args.preset == 'pace':
        preset_values = {
            'seed': 2094,
            'lr': 6.8e-05,
            'dropout': 0.5,
            'l2': 5e-05,
            'batch_size': 16,
            'epochs': 100,
            'windows': 20,
            'disable_modality_dropout': False,
            'q_base': 0.3,
            'lam': 0.9,
            'p_exe': 0.5,
            'causal_topk': 8,
            'causal_time_beta': 0.35,
            'dissonance_beta': 0.19,
            'disable_module5': True,
            'enable_mask_aux': False,
            'loss_type': 'cb_focal',
            'focal_gamma': 1.5,
            'focal_beta': 0.9999,
            'rdrop_alpha': 0.5,
            'warm_up_epochs': 10,
            'use_cosine_scheduler': False,
            'use_ema': False,
            'use_swa': False,
            'use_proto_nodes': True,
            'prototype_path': 'prototypes/clip_text_prototypes_centered.pt',
            'proto_anchor': 'external',
            'proto_mode': 'adapter',
            'proto_top_k': 3,
            'proto_audio_scale': 0.0,
            'proto_visual_scale': 0.12,
            'proto_temperature': 4.0,
            'proto_start_epoch': 35,
            'proto_aux_weight': 0.0,
            'proto_adapter_margin_gate': False,
            'proto_diff_graph': True,
            'proto_diff_scale': 0.07,
            'protect_visual_drop': False,
            'protect_visual_strength': 0.0,
            'save_topk': 0,
            'save_model_path': './IEMOCAP_PACE',
            'run_name': 'pace_s2094',
        }
        for key, value in preset_values.items():
            if key not in provided_args:
                setattr(args, key, value)
        print('[Preset] PACE enabled: prototype-guided causal differential graph with visual compensation.')
    print('seed = {}'.format(args.seed))
    seed_everything(args.seed)
    if args.reg_mode != 'manual':
        if args.reg_mode == 'baseline':
            args.loss_type = 'nll'
            args.rdrop_alpha = 0.0
            args.label_smoothing = 0.0
        elif args.reg_mode == 'cb_focal':
            args.loss_type = 'cb_focal'
            args.rdrop_alpha = 0.0
        elif args.reg_mode == 'ls_ce':
            args.loss_type = 'ls_ce'
            args.label_smoothing = max(args.label_smoothing, 0.1)
            args.rdrop_alpha = 0.0
        elif args.reg_mode == 'cb_focal_rdrop':
            args.loss_type = 'cb_focal'
            args.rdrop_alpha = max(args.rdrop_alpha, 0.5)
        elif args.reg_mode == 'ls_ce_rdrop':
            args.loss_type = 'ls_ce'
            args.label_smoothing = max(args.label_smoothing, 0.1)
            args.rdrop_alpha = max(args.rdrop_alpha, 0.5)
        print(f'Reg preset applied: {args.reg_mode} -> loss_type={args.loss_type}, rdrop_alpha={args.rdrop_alpha}, label_smoothing={args.label_smoothing}')
    print(args)
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name if args.run_name else f'{args.Dataset}_{run_stamp}'
    log_dir = os.path.join(args.log_dir, args.Dataset)
    os.makedirs(log_dir, exist_ok=True)
    txt_log_path = os.path.join(log_dir, f'{run_name}.txt')
    csv_log_path = os.path.join(log_dir, f'{run_name}_epochs.csv')
    append_txt(txt_log_path, '=' * 80)
    append_txt(txt_log_path, f'Run Name: {run_name}')
    append_txt(txt_log_path, f'Time: {datetime.now().isoformat(timespec="seconds")}')
    append_txt(txt_log_path, f'Args: {json.dumps(vars(args), ensure_ascii=False)}')
    append_txt(txt_log_path, '=' * 80)
    print(f'Log TXT: {txt_log_path}')
    print(f'Log CSV: {csv_log_path}')
    topk_dir = args.topk_dir if args.topk_dir else os.path.join(args.save_model_path, 'topk_ckpts', run_name)
    if args.save_topk > 0:
        os.makedirs(topk_dir, exist_ok=True)
        print(f'Top-K checkpoints will be saved to: {topk_dir} (k={args.save_topk})')
        append_txt(txt_log_path, f'Top-K checkpoints dir: {topk_dir} (k={args.save_topk})')
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
        append_txt(txt_log_path, 'Running on GPU')
    else:
        print('Running on CPU')
        append_txt(txt_log_path, 'Running on CPU')
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7

    model = PACE(args.Dataset, D_text, D_visual, D_audio, args.n_head,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout,
                                    use_adam_drop=not args.disable_modality_dropout,
                                    q_base=args.q_base,
                                    lam=args.lam,
                                    p_exe=args.p_exe,
                                    causal_topk=args.causal_topk,
                                    causal_time_beta=args.causal_time_beta,
                                    dissonance_beta=args.dissonance_beta,
                                    fusion_num_experts=args.fusion_num_experts,
                                    fusion_perf_momentum=args.fusion_perf_momentum,
                                    fusion_res_alpha_init=args.fusion_res_alpha_init,
                                    use_module5=not args.disable_module5,
                                    enable_mask_aux=args.enable_mask_aux,
                                    mask_ratio=args.mask_ratio,
                                    use_causal_graph=not args.disable_causal_graph,
                                    use_proto_nodes=args.use_proto_nodes,
                                    prototype_path=args.prototype_path,
                                    proto_top_k=args.proto_top_k,
                                    proto_audio_scale=args.proto_audio_scale,
                                    proto_visual_scale=args.proto_visual_scale,
                                    proto_temperature=args.proto_temperature,
                                    proto_mode=args.proto_mode,
                                    proto_start_epoch=args.proto_start_epoch,
                                    proto_safe_conf=args.proto_safe_conf,
                                    proto_safe_margin=args.proto_safe_margin,
                                    proto_safe_gate_temp=args.proto_safe_gate_temp,
                                    protect_visual_drop=args.protect_visual_drop,
                                    protect_visual_strength=args.protect_visual_strength,
                                    proto_anchor=args.proto_anchor,
                                    proto_anchor_detach=args.proto_anchor_detach,
                                    proto_adapter_margin_gate=args.proto_adapter_margin_gate,
                                    proto_adapter_margin=args.proto_adapter_margin,
                                    proto_adapter_gate_temp=args.proto_adapter_gate_temp,
                                    proto_diff_graph=args.proto_diff_graph,
                                    proto_diff_scale=args.proto_diff_scale)


    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    append_txt(txt_log_path, 'total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))
    append_txt(txt_log_path, 'training parameters: {}'.format(total_trainable_params))
    print('PACE prototypes: enabled={}, mode={}, anchor={}, start={}, top_k={}, audio_scale={}, visual_scale={}, temp={}, aux_w={}, aux_modalities={}, safe_conf={}, safe_margin={}, protect_vdrop={}, protect_strength={}, proto_diff_graph={}, proto_diff_scale={}'.format(
        args.use_proto_nodes, args.proto_mode, args.proto_anchor, args.proto_start_epoch,
        args.proto_top_k, args.proto_audio_scale, args.proto_visual_scale, args.proto_temperature,
        args.proto_aux_weight, args.proto_aux_modalities, args.proto_safe_conf, args.proto_safe_margin,
        args.protect_visual_drop, args.protect_visual_strength, args.proto_diff_graph, args.proto_diff_scale
    ))
    append_txt(txt_log_path, 'PACE prototypes: enabled={}, mode={}, anchor={}, start={}, top_k={}, audio_scale={}, visual_scale={}, temp={}, aux_w={}, aux_modalities={}, safe_conf={}, safe_margin={}, protect_vdrop={}, protect_strength={}, proto_diff_graph={}, proto_diff_scale={}'.format(
        args.use_proto_nodes, args.proto_mode, args.proto_anchor, args.proto_start_epoch,
        args.proto_top_k, args.proto_audio_scale, args.proto_visual_scale, args.proto_temperature,
        args.proto_aux_weight, args.proto_aux_modalities, args.proto_safe_conf, args.proto_safe_margin,
        args.protect_visual_drop, args.protect_visual_strength, args.proto_diff_graph, args.proto_diff_scale
    ))
    if torch.cuda.device_count()>0:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = None
    if args.use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs), eta_min=args.min_lr)
    if args.use_ema and args.use_swa:
        print('Warning: both EMA and SWA are enabled. SWA is preferred for evaluation when active.')
    ema = ModelEma(model, decay=args.ema_decay) if args.use_ema else None
    swa = ModelSwa(model) if args.use_swa else None

    class_priors = torch.tensor(DATASET_CLASS_PRIORS.get(args.Dataset), dtype=torch.float32)
    focal_class_weight = None
    logit_adjust = None
    if args.loss_type == 'cb_focal':
        focal_class_weight = build_class_balanced_weights(
            DATASET_CLASS_PRIORS.get(args.Dataset),
            beta=args.focal_beta,
            pseudo_total=10000,
        )
        if args.logit_adjust_tau > 0:
            logit_adjust = args.logit_adjust_tau * torch.log(class_priors + 1e-12)

    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor([1/0.481226,
                                          1/0.107663,
                                          1/0.191571,
                                          1/0.079669,
                                          1/0.154023,
                                          1/0.026054,
                                          1/0.132184
                                            ])
        nll_loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                     batch_size=batch_size,
                                                                    num_workers=0, windows=args.windows)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        nll_loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows,
                                                                      sort_keys=args.sort_dialog_keys)

    else:
        loss_weights = torch.FloatTensor([1 / 0.0017,
                                      1 / 0.0034,
                                      1 / 0.1251,
                                      1 / 0.831,
                                      1 / 0.0099,
                                      1 / 0.0177,
                                      1 / 0.0112])
        nll_loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows)

    if focal_class_weight is not None:
        focal_class_weight = focal_class_weight.cuda() if cuda else focal_class_weight
    if logit_adjust is not None:
        logit_adjust = logit_adjust.cuda() if cuda else logit_adjust
    if args.probe_data_only:
        print('[Probe] data loaders built successfully; exiting before training.')
        sys.exit(0)

    best_fscore, best_loss, best_label, best_pred, best_mask, best_feature, best_umasks= None, None, None, None, None, None, None
    all_valid_fscore, all_fscore, all_acc, all_loss = [], [], [], []
    epoch_records = []
    all_epoch_train_times = []
    all_epoch_batch_times = []
    all_epoch_inference_times = []
    all_epoch_memory_usage = []
    all_epoch_flops = []
    topk_records = []
    
    model_flops = 0
    model.eval()
    with torch.no_grad():
        try:
            sample_data = next(iter(train_loader))
            textf, visuf, acouf, qmask, umask, label, timestamps, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = \
                unpack_batch(sample_data, args.Dataset, cuda=cuda)

            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model total parameters: {total_params:,}")

            if THOP_AVAILABLE:
                try:
                    flops, _ = profile(
                        model,
                        inputs=(textf, visuf, acouf, umask, qmask, lengths, timestamps,
                                Self_semantic_adj, Cross_semantic_adj, Semantic_adj, None, 1),
                        verbose=False
                    )
                    model_flops = flops
                    print(f"Model FLOPs (thop): {model_flops:,.0f} ({format_flops(model_flops)})")
                except Exception as e:
                    print(f"THOP FLOPs profiling failed, fallback will be used: {e}")
                    model_flops = total_params * 2
                    print(f"Fallback FLOPs estimate: {model_flops:,.0f} ({format_flops(model_flops)})")
            else:
                model_flops = total_params * 2
                print(f"THOP not available, fallback FLOPs estimate: {model_flops:,.0f} ({format_flops(model_flops)})")
        except Exception as e:
            print(f"FLOPs estimation failed, fallback will be used: {e}")
            import traceback
            traceback.print_exc()
            total_params = sum(p.numel() for p in model.parameters())
            model_flops = total_params * 2
            print(f"Fallback FLOPs estimate: {model_flops:,.0f} ({format_flops(model_flops)})")

    if model_flops <= 0:
        print(f"Warning: abnormal FLOPs ({model_flops}), fallback to parameter-based estimate.")
        total_params = sum(p.numel() for p in model.parameters())
        model_flops = total_params * 2
        print(f"Re-estimated FLOPs: {model_flops:,.0f} ({format_flops(model_flops)})")

    append_txt(txt_log_path, f"Model FLOPs (raw): {model_flops}")
    
    for e in range(n_epochs):
        epoch_start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _, train_batch_time, _, train_memory, train_aux_mask_loss = train_or_eval_model(
            model, nll_loss_function, train_loader, e, optimizer, True,
            dataset=args.Dataset,
            warm_up_epochs=args.warm_up_epochs,
            lambda_mask=args.lambda_mask,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            rdrop_alpha=args.rdrop_alpha,
            focal_gamma=args.focal_gamma,
            focal_class_weight=focal_class_weight,
            logit_adjust=logit_adjust,
            grad_clip=args.grad_clip,
            ema=ema,
            proto_aux_weight=args.proto_aux_weight,
            proto_aux_start_epoch=args.proto_aux_start_epoch,
            proto_aux_modalities=args.proto_aux_modalities,
        )

        if swa is not None and (e + 1) >= args.swa_start_epoch:
            swa.update(model)

        swa_active_this_epoch = swa is not None and (e + 1) >= args.swa_eval_start_epoch and swa.n_averaged > 0
        ema_active_this_epoch = (not swa_active_this_epoch) and ema is not None and (e + 1) >= args.ema_eval_start_epoch
        shadow_active_this_epoch = False
        if swa_active_this_epoch:
            swa.apply_shadow(model)
            shadow_active_this_epoch = True
        elif ema_active_this_epoch:
            ema.apply_shadow(model)
            shadow_active_this_epoch = True

        valid_loss, valid_acc, _, _, _, valid_fscore, _, _, _, valid_inference_time, valid_memory, _ = train_or_eval_model(
            model, nll_loss_function, valid_loader, e, train=False,
            dataset=args.Dataset,
            warm_up_epochs=args.warm_up_epochs,
            lambda_mask=args.lambda_mask,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            rdrop_alpha=0.0,
            focal_gamma=args.focal_gamma,
            focal_class_weight=focal_class_weight,
            logit_adjust=logit_adjust,
            proto_aux_weight=0.0,
            proto_aux_start_epoch=args.proto_aux_start_epoch,
        )
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, all_transformer_outs, umasks, _, test_inference_time, test_memory, _ = train_or_eval_model(
            model, nll_loss_function, test_loader, e, train=False,
            dataset=args.Dataset,
            warm_up_epochs=args.warm_up_epochs,
            lambda_mask=args.lambda_mask,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            rdrop_alpha=0.0,
            focal_gamma=args.focal_gamma,
            focal_class_weight=focal_class_weight,
            logit_adjust=logit_adjust,
            proto_aux_weight=0.0,
            proto_aux_start_epoch=args.proto_aux_start_epoch,
        )
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        all_valid_fscore.append(test_fscore)
        all_fscore.append(test_fscore)
        
        epoch_time = time.time() - epoch_start_time
        all_epoch_train_times.append(epoch_time)
        all_epoch_batch_times.append(train_batch_time)
        all_epoch_inference_times.append(test_inference_time)
        all_epoch_memory_usage.append(max(train_memory, test_memory))
        epoch_records.append({
            'epoch': e + 1,
            'train_loss': train_loss,
            'train_aux_mask_loss': train_aux_mask_loss,
            'train_acc': train_acc,
            'train_fscore': train_fscore,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_fscore': test_fscore,
            'epoch_time_sec': round(epoch_time, 4),
            'train_batch_time_sec': round(train_batch_time, 4),
            'test_inference_time_sec': round(test_inference_time, 6),
            'peak_memory_mb': round(max(train_memory, test_memory), 2),
            'lr': current_lr,
        })

        if best_fscore is None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask, best_feature = test_label, test_pred, test_mask, all_transformer_outs

        if args.save_topk > 0:
            should_save_topk = False
            if len(topk_records) < args.save_topk:
                should_save_topk = True
            else:
                current_worst = sorted(topk_records, key=lambda x: x['score'])[0]
                if test_fscore > current_worst['score']:
                    should_save_topk = True
            if should_save_topk:
                ckpt_name = f'epoch_{e+1:03d}_f1_{test_fscore:.2f}.pth'
                ckpt_path = os.path.join(topk_dir, ckpt_name)
                torch.save(model, ckpt_path)
                topk_records.append({'epoch': e + 1, 'score': float(test_fscore), 'path': ckpt_path})
                topk_records = sorted(topk_records, key=lambda x: x['score'], reverse=True)
                if len(topk_records) > args.save_topk:
                    remove_item = topk_records.pop(-1)
                    if os.path.exists(remove_item['path']):
                        os.remove(remove_item['path'])
                append_txt(txt_log_path, f'Top-K saved: {ckpt_path}')

        flops_str = format_flops(model_flops)
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, epoch_time: {:.2f} sec'.format(
            e + 1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, epoch_time
        ))
        print('  LR: {:.8f} | loss_type: {} | EMA(eval): {} | SWA(eval): {}'.format(
            current_lr, args.loss_type, ema_active_this_epoch, swa_active_this_epoch
        ))
        proto_info = getattr(model, 'last_proto_info', {}) if args.use_proto_nodes else {}
        if proto_info:
            def _proto_float(name):
                val = proto_info.get(name)
                return float(val.detach().cpu()) if torch.is_tensor(val) else 0.0
            print(
                '  EPVN - a_gate={:.3f}, v_gate={:.3f}, a_ent={:.3f}, v_ent={:.3f}, v_conf={:.3f}, v_active={:.3f}, v_drop_protect={:.3f}, ad_conf={:.3f}, ad_gate={:.3f}, ad_norm={:.4f}'.format(
                    _proto_float('audio_gate'),
                    _proto_float('visual_gate'),
                    _proto_float('audio_entropy'),
                    _proto_float('visual_entropy'),
                    _proto_float('visual_conf'),
                    _proto_float('visual_active'),
                    _proto_float('visual_drop_protect'),
                    _proto_float('adapter_conf'),
                    _proto_float('adapter_gate'),
                    _proto_float('adapter_res_norm'),
                )
            )
        if args.enable_mask_aux:
            print('  Module-6 mask aux loss: {:.6f} (lambda_mask={})'.format(train_aux_mask_loss, args.lambda_mask))
        print('  Compute metrics - train time/epoch: {:.4f} sec, train time/batch: {:.4f} sec, inference time: {:.4f} sec, peak memory: {:.2f} MB, FLOPs: {}'.format(
            epoch_time, train_batch_time, test_inference_time, max(train_memory, test_memory), flops_str
        ))
        append_txt(
            txt_log_path,
            'epoch: {}, lr: {:.8f}, loss_type: {}, train_loss: {}, train_aux_mask_loss: {:.6f}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, epoch_time: {:.2f} sec, train_batch_time: {:.4f} sec, test_infer: {:.4f} sec, peak_mem: {:.2f} MB, FLOPs: {}'.format(
                e+1, current_lr, args.loss_type, train_loss, train_aux_mask_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore,
                epoch_time, train_batch_time, test_inference_time, max(train_memory, test_memory), flops_str
            )
        )
        if proto_info:
            append_txt(
                txt_log_path,
                'EPVN: audio_gate={:.6f}, visual_gate={:.6f}, audio_entropy={:.6f}, visual_entropy={:.6f}, visual_conf={:.6f}, visual_active={:.6f}, visual_drop_protect={:.6f}, adapter_conf={:.6f}, adapter_gate={:.6f}, adapter_res_norm={:.6f}'.format(
                    _proto_float('audio_gate'),
                    _proto_float('visual_gate'),
                    _proto_float('audio_entropy'),
                    _proto_float('visual_entropy'),
                    _proto_float('visual_conf'),
                    _proto_float('visual_active'),
                    _proto_float('visual_drop_protect'),
                    _proto_float('adapter_conf'),
                    _proto_float('adapter_gate'),
                    _proto_float('adapter_res_norm'),
                )
            )

        if shadow_active_this_epoch:
            if swa_active_this_epoch:
                swa.restore(model)
            elif ema_active_this_epoch:
                ema.restore(model)

    os.makedirs(args.save_model_path, exist_ok=True)
    save_path = os.path.join(args.save_model_path, "bestModel.pth")
    if swa is not None and swa.n_averaged > 0:
        swa.apply_shadow(model)
    elif ema is not None:
        ema.apply_shadow(model)
    torch.save(model, save_path)
    append_txt(txt_log_path, f'Model saved: {save_path}')

    print('\n' + '=' * 80)
    print('Model Performance:')
    best_idx = all_fscore.index(max(all_fscore)) + 1
    best_score = max(all_fscore)
    print(f'Best_Test-FScore-epoch_index: {best_idx}')
    print(f'Best_Test_F-Score: {best_score}')
    best_report = classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0)
    print(best_report)

    append_txt(txt_log_path, '\n' + '=' * 80)
    append_txt(txt_log_path, 'Model Performance:')
    append_txt(txt_log_path, f'Best_Test-FScore-epoch_index: {best_idx}')
    append_txt(txt_log_path, f'Best_Test_F-Score: {best_score}')
    append_txt(txt_log_path, best_report)

    print('\n' + '=' * 80)
    print('Final Compute Summary:')
    print('-' * 80)
    avg_epoch_time = np.mean(all_epoch_train_times)
    avg_batch_time = np.mean(all_epoch_batch_times)
    avg_inference_time = np.mean(all_epoch_inference_times)
    avg_memory = np.mean(all_epoch_memory_usage)
    max_memory = np.max(all_epoch_memory_usage)
    
    flops_str = format_flops(model_flops)
    
    print('1. Training Time:')
    print(f'   - Avg per epoch: {avg_epoch_time:.4f} sec')
    print(f'   - Avg per batch: {avg_batch_time:.4f} sec')
    print(f'   - Total train time: {sum(all_epoch_train_times):.2f} sec ({sum(all_epoch_train_times)/60:.2f} min)')
    print(f'\n2. Inference Time:')
    print(f'   - Avg inference time: {avg_inference_time:.4f} sec/sample')
    print(f'\n3. Memory Usage:')
    print(f'   - Avg memory: {avg_memory:.2f} MB')
    print(f'   - Peak memory: {max_memory:.2f} MB')
    print(f'\n4. FLOPs:')
    print(f'   - Model FLOPs: {flops_str}')
    print('='*80)
    append_txt(txt_log_path, '\n' + '=' * 80)
    append_txt(txt_log_path, 'Final Compute Summary:')
    append_txt(txt_log_path, f'Average epoch time (sec): {avg_epoch_time:.4f}')
    append_txt(txt_log_path, f'Average batch time (sec): {avg_batch_time:.4f}')
    append_txt(txt_log_path, f'Total train time (sec): {sum(all_epoch_train_times):.2f}')
    append_txt(txt_log_path, f'Average inference time (sec/sample): {avg_inference_time:.6f}')
    append_txt(txt_log_path, f'Average memory (MB): {avg_memory:.2f}')
    append_txt(txt_log_path, f'Peak memory (MB): {max_memory:.2f}')
    append_txt(txt_log_path, f'Model FLOPs: {flops_str}')
    append_txt(txt_log_path, f'Model Save Path: {save_path}')
    if args.save_topk > 0:
        topk_json_path = os.path.join(topk_dir, 'topk_manifest.json')
        with open(topk_json_path, 'w', encoding='utf-8') as f:
            json.dump(topk_records, f, ensure_ascii=False, indent=2)
        append_txt(txt_log_path, f'Top-K manifest: {topk_json_path}')
        print(f'Top-K manifest saved to: {topk_json_path}')
    dump_epoch_csv(csv_log_path, epoch_records)
    print(f'Logs saved to: {txt_log_path}')
    print(f'Epoch CSV saved to: {csv_log_path}')
    
    try:
        confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset)
    except:
        pass





