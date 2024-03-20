import torch
import torch.nn.functional as F
import torch.nn as nn


def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()


def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return (1 - Tversky) ** gamma


def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)

    return bce + (1 - Tversky) ** gamma


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()

# def cal_hel(pred, target, eps=1e-6):
#     def edge_loss(pred, target):
#         edge = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
#         edge[edge != 0] = 1
#         # input, kernel_size, stride=None, padding=0
#         numerator = (edge * (pred - target).abs_()).sum([2, 3])
#         denominator = edge.sum([2, 3]) + eps
#         return numerator / denominator
#
#     def region_loss(pred, target):
#         # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
#         numerator_fore = (target - target * pred).sum([2, 3])
#         denominator_fore = target.sum([2, 3]) + eps
#
#         numerator_back = ((1 - target) * pred).sum([2, 3])
#         denominator_back = (1 - target).sum([2, 3]) + eps
#         return numerator_fore / denominator_fore + numerator_back / denominator_back
#
#     edge_loss = edge_loss(pred, target)
#     region_loss = region_loss(pred, target)
#     return (edge_loss + region_loss).mean()

def structure_loss_with_ual(pred, mask):
    return structure_loss(pred, mask) + 0.5 * cal_ual(pred, mask)


class Bce_iou_loss(nn.Module):

    def __init__(self):
        super(Bce_iou_loss, self).__init__()

    def forward(self, pred, mask):
        weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

        pred = torch.sigmoid(pred)
        inter = pred * mask
        union = pred + mask
        iou = 1 - (inter + 1) / (union - inter + 1)

        weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
        weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

        return (weighted_bce + weighted_iou).mean()
