import torch
import numpy as np
def iou(y_preds, targets):
    iou_list = []
    for y_pred, target in zip(y_preds,targets):
        y_pred = y_pred.squeeze(0).cpu().detach().numpy().argmax(0)
        target = target.numpy()
        intersection = np.bitwise_and(y_pred, target).astype(float).sum()
        union = np.bitwise_or(y_pred, target).astype(float).sum()
        iou_score = (intersection + 1e-6) / (union + 1e-6) # IoU 계산
        iou_list.append(iou_score)
    return sum(iou_list) / len(iou_list)
