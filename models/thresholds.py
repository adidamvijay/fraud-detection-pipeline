"""
Choosing the score above which a transaction is flagged.

Kept separate so the training job and the evaluation use the same rule. If
they diverged, the flags written to FRAUD_SCORES would not correspond to the
operating point the evaluation reports, and the dashboard would be showing
something nobody had measured.

The rule is fixed: the point on the precision/recall curve that maximises F1.
It was chosen before any results were seen so that the threshold is not
selected to flatter the numbers.

Choosing a threshold needs labels. That is not a contradiction with the model
being unsupervised: the model ranks transactions without labels, and labels
are only used to decide where on that ranking to draw the line. In production
that line would be drawn using whatever labelled history exists, or set by
how many alerts a review team can process per day, which is usually the real
constraint.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(labels, scores):
    """
    Return (threshold, precision, recall, f1) at the F1-maximising point.

    Returns None if there are no positive labels to optimise against.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.sum() == 0:
        return None

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one more precision/recall than threshold.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)

    best = int(np.argmax(f1))
    return (float(thresholds[best]), float(precision[best]),
            float(recall[best]), float(f1[best]))


def threshold_for_alert_rate(scores, alert_rate):
    """
    The threshold that flags a fixed share of transactions.

    The alternative way to pick an operating point, and usually the one that
    matters: a review team can process a certain number of alerts per day,
    which fixes the alert rate regardless of what the curve looks like.
    """
    return float(np.quantile(np.asarray(scores), 1.0 - alert_rate))
