import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def simple_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        return model(gt, image, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def modification_train_val_forward(model: nn.Module, gt=None, image=None, depth=None,seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None
        return model(gt,depth, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred1,pred2 = model.sample(image, **kwargs)
        if time_ensemble:
            """ Here is the function 3, Uncertainty based"""

            preds = torch.concat(model.history, dim=1) #([b, t, 384, 384])
            chunks = torch.chunk(preds, 40, dim=1)
            gt_chunks = torch.cat(
                (chunks[0],chunks[2],chunks[4],chunks[6],chunks[8],chunks[10],
                 chunks[12],chunks[14],chunks[16],chunks[18],chunks[20],
                 chunks[22],chunks[24],chunks[26],chunks[28],chunks[30],
                 chunks[32],chunks[34],chunks[36],chunks[38]),dim=1)
            depth_chunks = torch.cat(
                (chunks[1],chunks[3],chunks[5],chunks[7],chunks[9],chunks[11],
                 chunks[13],chunks[15],chunks[17],chunks[19],chunks[21],
                 chunks[23],chunks[25],chunks[27],chunks[29],chunks[31],
                 chunks[33],chunks[35],chunks[37],chunks[39]),dim=1)
            preds1 = gt_chunks.detach().cpu()
            preds2 = depth_chunks.detach().cpu()
            pred1 = torch.mean(preds1, dim=1, keepdim=True)
            pred2 = torch.mean(preds2, dim=1, keepdim=True)

            def process1(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds1[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p
            def process2(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                ps = F.interpolate(preds2[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = ps * p

                return p
            pred1 = [process1(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred1, gt_sizes))]
            pred2 = [process2(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred2, gt_sizes))]

        return {
            "image": image,
            "pred_gt": pred1,
            "pred_depth": pred2,
            "gt": gt if gt is not None else None,
            "depth": depth if depth is not None else None,
        }


def modification_train_val_forward_e(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is extend function 4, with batch extend."""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            for i in range(2):
                model.sample(image, **kwargs)
                preds = torch.cat([preds, torch.concat(model.history, dim=1).detach().cpu()], dim=1)
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }
