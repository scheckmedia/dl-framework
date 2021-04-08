import numpy as np


def calculate_scores(cm):
    eps = np.finfo(np.float32).eps
    diag = np.diag(cm)

    accuracy = diag / (cm.sum() + eps)
    precision = diag / (cm.sum(axis=0) + eps)
    recall = diag / (cm.sum(axis=1) + eps)
    class_iou = diag / (cm.sum(axis=1) + cm.sum(axis=0) - diag + eps)
    f1 = 2 * ((precision * recall) / (precision + recall + eps))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'class_iou': class_iou,
        'f1': f1
    }
