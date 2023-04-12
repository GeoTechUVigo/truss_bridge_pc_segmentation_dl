"""
Daniel Lamas Novoa.
Enxeñaría dos materiais, mecánica aplicada e construción.
Escola de enxeñería industrial.
Grupo de xeotecnoloxía aplicada.
Universidade de Vigo.
https://orcid.org/0000-0001-7275-183X
10/06/2022
"""

from sklearn import metrics
import numpy as np
from ismember import ismember


def semantic_metrics(y_real: list, y_pred: list):
    """
    Metrics for semantic segmentation evaluation. This function calculates
    overall Accuraty, mean Accuracy and mean Intersection over Union.
    It also returns an errors array point by point with 0 if it is a TP, and 1 otherwise.

    :param y_real: real values.
    :param y_pred: predicted values.
    :return: oAcc, mAcc, mIoU, errors.
    """

    # confusion matrix
    cm = metrics.confusion_matrix(y_real, y_pred)
    # True positives
    n_classes = len(cm)
    tp = np.zeros(n_classes)
    for i in range(n_classes):
        tp[i] = cm[i,i]

    # overall accuracy (oAcc): TP/FN (TN and FP are not considered because the data are multi-class)
    oAcc = np.sum(tp)/np.sum(cm)

    # mean accuracy (mAcc): Mean of the accuracies of each class TP/FN
    accs = tp/np.sum(cm, axis=1)
    accs[np.isnan(accs)] = 1.0 # nan because x/0
    mAcc = np.mean(accs)

    # mean intersection over union (mIoU): mean of the intersection over union of each class TP/(TP + FN + FP)
    ious = tp/(np.sum(cm, axis=0) + np.sum(cm, axis=1) - tp)
    ious[np.isnan(ious)] = 1.0 # nan because x/0
    mIoU = np.mean(ious)

    # errors
    errors = y_real != y_pred

    return oAcc, mAcc, mIoU, errors


def instance_metrics(y_real: list, y_pred: list, threashold:float = 0.5):
    """
    Metrics for instance segmentation evaluation. This function calculates
    mean precision (IoU > threashold), mean Recall (with IoU > threashold),
    coverage and weighted coverage.
    In the calculation of mPrec and mRec, an object is considered a true positive
    if the IoU between the instance predicted and the real is > threashold.
    The indexes y_real and y_pred of the same object do not have to be the same.
    It also returns an errors array point by point with 0 if it is a TP, and 1 otherwise.

    :param y_real: real values.
    :param y_pred: predicted values.
    :param threashold: mean IoU for mPrec and mRec. [default: 0.5]
    :return: mPrec, mRec, Cov, WCov, errors
    """

    # Instance: mPrec(IoU>0.5), mREc(IoU>0.5), Cov and WCov

    # Index and number of points of each instance, real and predicted
    ids_real, n_real = np.unique(y_real, return_counts=True)
    ids_pred, n_pred = np.unique(y_pred, return_counts=True)

    # max IoU of each group
    maxIoU = np.zeros(len(ids_real))

    # errors
    errors = np.zeros(y_real.shape, dtype='bool')
    errors[:] = True

    # Analyse each real instance
    for i in range(len(ids_real)):

        # y_pred indexes in this instance
        this_ids_pred, this_n_pred = np.unique(y_pred[y_real == ids_real[i]], return_counts=True)

        # IoUs
        ious = this_n_pred / (n_real[i] + n_pred[ismember(ids_pred, this_ids_pred)[0]]  - this_n_pred)

        # Higher IoU value for this ids_real
        maxIoU[i] = ious.max()

        # check errors
        errors[np.logical_and(y_real== ids_real[i], y_pred==this_ids_pred[ious.argmax()])] = False

    # mPrec
    mPrec = np.sum(maxIoU > threashold) / len(ids_pred)

    # mRecall
    mRecall = np.sum(maxIoU > threashold) / len(ids_real)

    # Cov
    cov = np.mean(maxIoU)

    # wCov
    wCov = np.sum(n_real / np.sum(n_real) * maxIoU)

    return mPrec, mRecall, cov, wCov, errors
