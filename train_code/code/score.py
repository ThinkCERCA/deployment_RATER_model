import pandas as pd
import numpy as np

def _tp_score_effectiveness(col=""):
    def row_wise(row):
        gt_eff = row.discourse_effectiveness # ('Adequate' = 0, 'Effective' = 1, 'Ineffective' = 2)
        return row[f'score_discourse_effectiveness_{gt_eff}']
    return row_wise

def calc_overlap(set_pred, set_gt,threshold=0.5):  # Note: If we are to change the score/acceptance criterion, the logic should be changed here
    """
    Calculates if the overlap between prediction and
    ground truth is enough fora potential True positive
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)  # Computes the number of words in the ground truth segment
        len_pred = len(set_pred)  # Computes the number of words in the prediction segment
        inter = len(set_gt & set_pred)  # Computes the length of the intersection
        overlap_1 = inter / len_gt  # Check the ratio of the interseections to both ground truth and prediction
        overlap_2 = inter/ len_pred
        return overlap_1 >= threshold and overlap_2 >= threshold,max(overlap_1,overlap_2)  # If both thresholds satisfied, return True. Criteria for best overlap
    except:  # at least one of the input is NaN
        return False, 0  # Otherwise, discard any overlap and return 0.

def score_feedback_comp_micro(pred_df, gt_df, discourse_type,threshold=0.5,weight_tp_segment=0.5):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type, 
                      ['id', 'predictionstring','discourse_effectiveness']].reset_index(drop=True)  # Get all the ground truth segments of a given discourse type along with their effectiveness label (0, 1, 2)
    pred_df = pred_df.loc[pred_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring','score_discourse_effectiveness_0',
                                               'score_discourse_effectiveness_1',
                                               'score_discourse_effectiveness_2']].reset_index(drop=True)  # Get the prediction segments along with the predicted effectiveness distributions (three positive scalars summing to 1)
    pred_df['pred_id'] = pred_df.index  # Set prediction and ground truth IDs as dataframe index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]  # Split the predictionstring of both ground truth and prediction segments up into a list of word indices for later set operations
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]
    
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )  # Outer join all prediction and ground truth segments to perform pair-wise comparison
    
    overlaps = [calc_overlap(*args,threshold=threshold) for args in zip(joined.predictionstring_pred, 
                                                     joined.predictionstring_gt)]  # Calculate the overlap of each segment - prediction pair using the calc_overlap function above (This returns a Boolean for threshold acceptance and the overlap value (currently max of two overlaps))
    
    joined['discourse_effectiveness'] = joined['discourse_effectiveness'].fillna(0).astype(int)
    joined['overlaps'] = np.asarray([x[0] for x in overlaps])*1  # Convert the overlap Boolean into a 0/1 value and assign the result as a  column in the outer join dataframe.
    joined['overlaps_scores'] = np.asarray([x[1] for x in overlaps])*1  # Add the actual overlap scores to the join dataframe
    joined['effectiveness_TP_score'] = joined.apply(_tp_score_effectiveness(' '),axis=1)  # Call a helper function to fetch the predicted probability, for each prediction, on the correct effectiveness class of the ground truth segment, and add this as a column
    # One - Eff not really needed
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    
    joined = joined.sort_values(["overlaps",'overlaps_scores'],ascending=False).reset_index(drop=True).groupby('gt_id').head(1)  # 1) Sort predictions in descending order by their overlap scores, 2) Group them by the ground truth segment they target, 3) only keep the best prediction for each segment. 
    
    TP = joined[joined.overlaps==1]['gt_id'].nunique() # Then, from the best predictions per group which satisfy the overlap threshold, count the number of satisfied ground truth segments. This returns the number of true positives
    
    TP_weighted = weight_tp_segment*TP + (1-weight_tp_segment)*(joined[joined.overlaps==1]['effectiveness_TP_score'].sum())  # In our weighted setup, we don't simply want the number of such segments. We want to assign a partial fraction of the true positive to effectiveness, and thus we compute an effectiveness component.
    # Therefore, in effect TP_weighted <= TP.
    # Consistently with the original metrics, we assign the lost fraction due to the effectiveness coefficient as a false negative. Therefore, TP + FN remains the same in both weighted and unweighted cases.
    # To avoid rewarding the model for effectiveness prediction in non-overlapping segments, we do NOT apply the weighted system for rejected segments. Otherwise, a wholly non-overlapping segment (a false positive) with an otherwise perfect match to any segment goes unpunished.
    # Hence, FP remains the same as in the unweighted metric. and thus TP + FP weighted drops by how much TP drops only.
    
    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)
    
    TP_gap = TP - TP_weighted  # Compute the amount of TP lost due to the weighting of effectiveness.
    TPandFP_weighted = TPandFP - TP_gap  # Append it to the unweighted TP+FP component
    
    #Calculate microf1
    f1_score_fb1 = 2*TP / (TPandFP + TPandFN)  # Compute F1 scores in the unweighted case
    new_f1_score = 2*TP_weighted / (TPandFP_weighted + TPandFN)  # Analogously compute the weighted case. 
    '''
    Notice that, given an unweighted F1 score = a/b in the unweighted case, the weighted F1 score would equal (a - 2g)/() b - g), where 0 <= g <= a/2 is the TP loss due to the efficiency component. 
    Observe that (a - 2g) / (b - g) <= (a / b) for 0 <= g <= a/2, and thus weighted F1 <= normal F1 .
    '''
    return f1_score_fb1,new_f1_score

def score_feedback_comp(pred_df, gt_df,threshold=0.5, weight_tp_segment=0.5,return_class_scores=False):
    class_scores_fb1 = {}
    new_class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        s_fb1,s = score_feedback_comp_micro(pred_df, gt_df, discourse_type,threshold,weight_tp_segment)
        class_scores_fb1[discourse_type] = s_fb1
        new_class_scores[discourse_type] = s
    
    f1_fb1 = np.mean([v for v in class_scores_fb1.values()])
    new_f1 = np.mean([v for v in new_class_scores.values()])
    if return_class_scores:
        return f1_fb1,class_scores_fb1,new_f1, new_class_scores
    return f1_fb1,new_f1