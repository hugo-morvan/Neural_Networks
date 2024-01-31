import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    cM = calcConfusionMatrix(LPred, LTrue)
    acc = calcAccuracyCM(cM)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    TP = np.sum(np.logical_and(LPred == 1, LTrue == 1))
    TN = np.sum(np.logical_and(LPred == 0, LTrue == 0))
    FP = np.sum(np.logical_and(LPred == 1, LTrue == 0))
    FN = np.sum(np.logical_and(LPred == 0, LTrue == 1))
    cM = [[TP, FP], [FN, TN]]
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = (cM[0][0] + cM[1][1]) / (cM[0][0] + cM[1][1] + cM[0][1] + cM[1][0])
    # ============================================
    
    return acc
