import pickle
import pandas as pd
from torchvision.ops import roi_align, nms
import torch
from code.score import score_feedback_comp
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test', default=False, action='store_true'
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

models = [
    'ema_32_awp_5e-6',
    'ema_32_awp_8e-6',
    'ema_32_awp_5e-6_025',
    'xlarge_ema_32_5e-6_2',
    'xlarge',
]

ws = [1/len(models)]*len(models)

if opt.test:
    models = [m + '_test' for m in models]

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

def make_preds(preds, ids, th):

    r = []

    for text_id, (obj_pred, reg_pred, cls_pred, eff_pred) in zip(ids, preds):

        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()
        eff_pred = eff_pred.softmax(-1)

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        eff_scores = eff_pred
        pr_scores = (obj_scores * cls_scores)**0.5

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
            score_discourse_effectiveness_0 = pr_eff[:,0].cpu().numpy(),
            score_discourse_effectiveness_1 = pr_eff[:,1].cpu().numpy(),
            score_discourse_effectiveness_2 = pr_eff[:,2].cpu().numpy(),
        )

        res = pd.DataFrame(res).sort_values('start').reset_index(drop=True)
        res['predictionstring'] = res.apply(get_pred(' '),axis=1)
        
        r.append(res)

    return pd.concat(r,axis=0).reset_index(drop=True)

final_preds = None

for w, model in zip(ws, models):
    preds = pickle.load(open('preds/' + model + '.p', 'rb'))

    if final_preds is None:
        final_preds = [list(pred) for pred in preds]
        for first in final_preds:
            for i in range(len(first)):
                first[i] *= w

    else:
        for first, second in zip(final_preds, preds):
            for i in range(len(first)):
                first[i] += second[i] * w

if opt.test:
    gt_df = pd.read_csv('data/test_gt_df.csv')
else:
    gt_df = pd.read_csv('data/valid_gt_df.csv')

test_ids = gt_df['id'].drop_duplicates().values.tolist()

pred_df = make_preds(final_preds, test_ids, 0.303)
macro_f1, new_macro_f1 = score_feedback_comp(pred_df, gt_df,threshold=0.5, weight_tp_segment=0.5,return_class_scores=False)
print(macro_f1, new_macro_f1)