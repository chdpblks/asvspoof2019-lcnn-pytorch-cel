import torch
import numpy as np
from src.metrics.base_metric import BaseMetric


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


class EERMetric(BaseMetric):

    def __init__(self, device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.reset()

    def reset(self):
        self.all_scores = []
        self.all_labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:

        if 'logits' in kwargs:
            logits = kwargs['logits']
        
        if logits.dim() == 2:
            scores = torch.softmax(logits, dim=1)[:, 1]
        else:
            scores = logits

        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self.all_scores.extend(scores_np)
        self.all_labels.extend(labels_np)

        # default value
        return 0.0

    def compute_final_eer(self):
        '''Compute final EER for all accumulated scores.'''
        if len(self.all_scores) == 0:
            return float('nan')

        all_scores_np = np.array(self.all_scores)
        all_labels_np = np.array(self.all_labels)
        
        bona_cm = all_scores_np[all_labels_np == 1]
        spoof_cm = all_scores_np[all_labels_np == 0]

        if len(bona_cm) == 0 or len(spoof_cm) == 0:
            return float('nan')
        
        try:
            eer, threshold = compute_eer(bona_cm, spoof_cm)
            return eer * 100
        except:
            return float('nan')
