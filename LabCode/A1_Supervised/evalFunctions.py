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
    #https://stackoverflow.com/questions/61193476/constructing-a-confusion-matrix-from-data-without-sklearn
    nr_classes = int(max(LTrue) - min(LTrue)) + 1 #find number of classes

    cM = [[sum([(LTrue[i] == true_class) and (LPred[i] == pred_class) for i in range(len(LTrue))])
           for pred_class in range(1, nr_classes + 1)] 
           for true_class in range(1, nr_classes + 1)]
    cM = np.array(cM)
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
    acc = sum(np.diag(cM)) / np.sum(cM)
    # ============================================
    
    return acc
