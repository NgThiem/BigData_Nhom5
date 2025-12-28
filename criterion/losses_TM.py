# criterion/losses_TM.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        return self.bce(pred, target).mean()

class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, objectness_mask):
        mask = objectness_mask > 0.5
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[mask.expand_as(pred)].view(-1, 4)
        target = target[mask.expand_as(target)].view(-1, 4)

        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)

        # Chuyển ltrb → xyxy
        pred_x1 = -pred[:, 0]
        pred_y1 = -pred[:, 1]
        pred_x2 = pred[:, 2]
        pred_y2 = pred[:, 3]
        target_x1 = -target[:, 0]
        target_y1 = -target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]

        # Tính GIoU loss (đơn giản hóa)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = pred_area + target_area - inter_area
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # GIoU loss
        loss = 1 - iou.mean()
        return loss
