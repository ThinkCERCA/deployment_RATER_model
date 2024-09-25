import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms


def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep


def get_pred(col):
    def row_wise(row):
        return " ".join([str(x) for x in range(row.start, row.end)])

    return row_wise


def make_preds(preds, ids, th=0.303):
    r = []

    for text_id, (obj_pred, reg_pred, cls_pred, eff_pred) in zip(ids, preds):

        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()
        eff_pred = eff_pred.softmax(-1)

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        eff_scores = eff_pred
        pr_scores = (obj_scores * cls_scores) ** 0.5

        pos_inds = pr_scores > 0.5

        if pos_inds.sum() == 0:
            continue

        pr_score, pr_label, pr_eff = pr_scores[pos_inds], cls_labels[pos_inds], eff_scores[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score, th)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]
        pr_eff = pr_eff[keep]

        res = dict(
            id=text_id,
            start=start.cpu().numpy(),
            end=end.cpu().numpy(),
            score_discourse_type=pr_score.cpu().numpy(),
            discourse_type=pr_label.cpu().numpy(),
            score_discourse_effectiveness_0=pr_eff[:, 0].cpu().numpy() + pr_eff[:, 2].cpu().numpy(),
            score_discourse_effectiveness_1=pr_eff[:, 1].cpu().numpy(),
        )

        res = pd.DataFrame(res).sort_values('start').reset_index(drop=True)
        res['predictionstring'] = res.apply(get_pred(' '), axis=1)

        r.append(res)

    return pd.concat(r, axis=0).reset_index(drop=True)


class TextModel(pl.LightningModule):
    def __init__(self, cfg, config_path=None):
        super().__init__()
        self.cfg = cfg

        model_cfg = cfg['model']

        self.config = torch.load(config_path)
        self.backbone = AutoModel.from_config(self.config)

        hidden_size = self.config.hidden_size
        self.fc = nn.Linear(hidden_size, 1 + 2 + 7 + 3)

    def forward(self, inputs):
        x = self.backbone(**inputs).last_hidden_state
        x = self.fc(x)
        return x

    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        inputs = {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
        }

        logits = self(inputs)

        logits = aggregate_tokens_to_words(logits, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:-3]
        eff_pred = logits[..., -3:]

        return obj_pred, reg_pred, cls_pred, eff_pred

    def predict_step(self, data, batch_idx):
        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)

        return obj_pred.detach().cpu(), reg_pred.detach().cpu(), cls_pred.detach().cpu(), eff_pred.detach().cpu()