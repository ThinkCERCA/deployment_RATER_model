import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from train_code.code.score import score_feedback_comp
from train_code.code.ema import EMA
from train_code.code.awp import AWP
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms

def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
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
        return " ".join([str(x) for x in range(row.start,row.end)])
    return row_wise

class TextModel(pl.LightningModule):
    def __init__(self, cfg, config_path=None):

        super().__init__()
        self.cfg = cfg

        model_cfg = cfg['model']

        if config_path:
            self.config = torch.load(config_path)
        else:
            self.config = AutoConfig.from_pretrained(model_cfg['model_name'], output_hidden_states=True)

        self.skip_validation = cfg['skip_validation']

        self.dropout_eval = model_cfg['dropout_eval']
        self.dynamic_positive = model_cfg['dynamic_positive']

        self.awp = None
        self.use_awp = model_cfg['use_awp']
        self.skip_awp = model_cfg['skip_awp']

        if model_cfg['pretrained']:
            self.backbone = AutoModel.from_pretrained(model_cfg['model_name'], config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        if model_cfg['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(model_cfg['dropout'])

        hidden_size = self.config.hidden_size
        self.fc = nn.Linear(hidden_size, 1+2+7+3)

        self.loss = 0
        self.cnt = 0
        self.preds = []
        self.gts = []

        self.ema = None
        
    def train(self, mode=True):
        super().train(mode)

        if self.ema is None:
            self.ema = EMA(self, 0.9)
            self.ema.register()

        if mode and self.dropout_eval:
            for name, m in self.named_modules():
                if 'dropout' in name:
                    m.eval()

    def on_after_backward(self):
        if self.use_awp and self.skip_awp == 0:
            self.awp.attack_backward(self.batch)

    def forward(self, inputs):
        x = self.backbone(**inputs).last_hidden_state
        x = self.dropout(x)
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
    
    def predict(self, data, test_score_thr=0.5):
        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()
        eff_pred = eff_pred.softmax(-1)

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        eff_scores = eff_pred
        pr_scores = (obj_scores * cls_scores)**0.5

        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return pd.DataFrame(columns=['id', 'predictionstring',"start",'end',"score_discourse_type","discourse_type","discourse_effectiveness",
                                        "score_discourse_effectiveness",'score_discourse_effectiveness_0',
                                                'score_discourse_effectiveness_1',
                                                'score_discourse_effectiveness_2'])

        pr_score, pr_label, pr_eff = pr_scores[pos_inds], cls_labels[pos_inds], eff_scores[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]
        pr_eff = pr_eff[keep]

        res = dict(
            id=data['text_id'],
            start=start.cpu().numpy(),
            end=end.cpu().numpy(),
            score_discourse_type=pr_score.cpu().numpy(),
            discourse_type=pr_label.cpu().numpy(),
            score_discourse_effectiveness_0 = pr_eff[:,0].cpu().numpy(),
            score_discourse_effectiveness_1 = pr_eff[:,1].cpu().numpy(),
            score_discourse_effectiveness_2 = pr_eff[:,2].cpu().numpy(),
        )

        res = pd.DataFrame(res).sort_values('start').reset_index(drop=True)
        res['predictionstring'] = res.apply(get_pred(' '),axis=1)
        return res

    def get_losses(self, obj_pred, reg_pred, cls_pred, eff_pred, obj_target, reg_target,
                   cls_target, eff_target, pos_loc):
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        reg_pred = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        reg_loss = -iou.log().sum() / num_total_samples

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        eff_loss = F.cross_entropy(
            eff_pred[pos_loc],
            eff_target[pos_loc],
            reduction='sum') / num_total_samples

        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        return obj_loss, reg_loss, cls_loss, eff_loss

    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred, eff_pred, dynamic_positive):
        obj_target = torch.zeros_like(obj_pred, dtype=torch.float)
        reg_target = torch.zeros_like(reg_pred, dtype=torch.float)
        cls_target = torch.zeros_like(cls_pred, dtype=torch.float)
        eff_target = torch.zeros_like(eff_pred, dtype=torch.float)

        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        eff_target[pos_loc, gt_spans[:, 3]] = 1

        # dynamically assign one more positive
        if dynamic_positive:
            cls_prob = (obj_pred.sigmoid().unsqueeze(1) *
                        cls_pred.sigmoid()).sqrt()
            for start, end, label, eff_label in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                
                with torch.autocast('cuda', enabled=False):
                    cls_cost = F.binary_cross_entropy(_cls_prob,
                                                _cls_gt,
                                                reduction='none').sum(-1)
                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
                eff_target[pos_ind, eff_label] = 1

            pos_loc = (obj_target == 1).nonzero().flatten()

        return obj_target, reg_target, cls_target, eff_target, pos_loc

    def training_step(self, data, batch_idx):

        if self.use_awp:
            if self.awp is None:
                self.awp = AWP(self, self.optimizers().optimizer, scaler=self.scaler)
            
            self.batch = data

        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)
        obj_target, reg_target, cls_target, eff_target, pos_loc = self.build_target(
            data['gt_spans'], obj_pred, reg_pred, cls_pred, eff_pred, self.dynamic_positive)

        obj_loss, reg_loss, cls_loss, eff_loss = self.get_losses(obj_pred, reg_pred,
                                                       cls_pred, eff_pred, obj_target,
                                                       reg_target, cls_target, eff_target,
                                                       pos_loc)
        
        loss = obj_loss + reg_loss + cls_loss + eff_loss

        self.log("obj_loss", obj_loss.item(), prog_bar=True)
        self.log("reg_loss", reg_loss.item(), prog_bar=True)
        self.log("cls_loss", cls_loss.item(), prog_bar=True)
        self.log("eff_loss", eff_loss.item(), prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, data, batch_idx):

        if self.skip_validation > 0:
            return

        preds = self.predict(data, test_score_thr=0.5)

        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)
        obj_target, reg_target, cls_target, eff_target, pos_loc = self.build_target(
            data['gt_spans'], obj_pred, reg_pred, cls_pred, eff_pred, dynamic_positive=False)

        obj_loss, reg_loss, cls_loss, eff_loss = self.get_losses(obj_pred, reg_pred,
                                                       cls_pred, eff_pred, obj_target,
                                                       reg_target, cls_target, eff_target,
                                                       pos_loc)
        loss = obj_loss + reg_loss + cls_loss + eff_loss
        
        gt = pd.DataFrame({
            "id":data['text_id'],
            "start":data["gt_spans"][:,0].cpu().numpy(),
            "end":data["gt_spans"][:,1].cpu().numpy(),
            "discourse_type":data["gt_spans"][:,2].cpu().numpy(),
            "discourse_effectiveness":data["gt_spans"][:,3].cpu().numpy(),
                        })
        gt['predictionstring'] = gt.apply(get_pred(' '),axis=1)

        self.loss += loss
        self.cnt += 1
        self.preds.append(preds)
        self.gts.append(gt)

    def predict_step(self, data, batch_idx):
        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)

        return obj_pred.detach().cpu(), reg_pred.detach().cpu(), cls_pred.detach().cpu(), eff_pred.detach().cpu()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def on_validation_epoch_start(self):
        print('\n')
        self.ema.apply_shadow()

    def on_validation_epoch_end(self):

        if self.skip_awp > 0:
            self.skip_awp -= 1

        if self.skip_validation > 0:

            self.log(f'valid_loss', 6.0, on_epoch=True, prog_bar=True)
            self.log(f'fb1', 0.3/self.skip_validation, on_epoch=True, prog_bar=True)
            self.log(f'score', 0.3/self.skip_validation, on_epoch=True, prog_bar=True)

            self.skip_validation -= 1

            return

        loss = self.loss / self.cnt

        pred_df = pd.concat(self.preds,axis=0).reset_index(drop=True)
        gt_df = pd.concat(self.gts,axis=0).reset_index(drop=True)

        macro_f1, new_macro_f1 = score_feedback_comp(pred_df, gt_df,threshold=0.5, weight_tp_segment=0.5,return_class_scores=False)

        self.log(f'valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'fb1', macro_f1, on_epoch=True, prog_bar=True)
        self.log(f'score', new_macro_f1, on_epoch=True, prog_bar=True)

        self.loss = 0
        self.cnt = 0
        self.preds = []
        self.gts = []

        self.ema.restore()

    def configure_optimizers(self):

        weight_decay = self.cfg['optimizer']['weight_decay']

        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
            },
        ]

        optimizer = eval(self.cfg['optimizer']['name'])(
            optimizer_parameters, **self.cfg['optimizer']['params']
        )

        self.optimizer = optimizer

        if 'scheduler' in self.cfg:
            scheduler_name = self.cfg['scheduler']['name']
            params = self.cfg['scheduler']['params']

            if scheduler_name in ['poly', 'cosine']:
                epoch_steps = self.cfg['dataset_size']
                batch_size = self.cfg['train_loader']['batch_size']
                acc_steps = self.cfg['trainer']['accumulate_grad_batches']
                num_gpus = self.cfg['trainer']['devices']
                
                print(f'- Dataset size : {epoch_steps} - Effective batch size :  {batch_size * acc_steps * num_gpus}')
            
                warmup_steps = self.cfg['scheduler']['warmup'] * epoch_steps// (
                    batch_size * acc_steps * num_gpus
                )
                training_steps = self.cfg['epochs'] * epoch_steps // (
                    batch_size * acc_steps * num_gpus
                )

                print(f"Total warmup steps : {warmup_steps} - Total training steps : {training_steps}")

                if scheduler_name == 'poly':
                    power = params['power']
                    lr_end = params['lr_end']
                    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)
                elif scheduler_name == 'cosine':
                    print(params)
                    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, **params)
                else:
                    raise NotImplemented('not implemented!')
            else:
                scheduler = eval(scheduler_name)(
                    optimizer, **params
                )

            lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': self.cfg['scheduler']['interval'],
                'frequency': 1,
            }

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

        return optimizer