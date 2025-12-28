# criterion/criterions_TM.py
# criterion/criterions_TM.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, objectness_mask):
        # MỞ RỘNG MASK ĐỂ KHỚP SỐ KÊNH
        mask = (objectness_mask > 0.5).expand_as(pred)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return self.l1_loss(pred[mask], target[mask])


class SetCriterion_TM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.objectness_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.giou_loss = GIoULoss()

    def prepare_targets(self, targets, shape):
        B = len(targets)
        H, W = shape
        device = targets[0]['boxes'].device

        target_objectness = torch.zeros(B, 1, H, W, device=device)
        target_regression = torch.zeros(B, 4, H, W, device=device)

        for b in range(B):
            boxes = targets[b]['boxes']
            if boxes.numel() == 0:
                continue

            boxes_pixel = boxes * torch.tensor([W, H, W, H], device=device)
            x1, y1, x2, y2 = boxes_pixel.unbind(1)

            for i in range(boxes.size(0)):
                x1_i = max(0, min(W - 1, int(x1[i].item())))
                y1_i = max(0, min(H - 1, int(y1[i].item())))
                x2_i = max(0, min(W, int(x2[i].item()) + 1))
                y2_i = max(0, min(H, int(y2[i].item()) + 1))

                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                target_objectness[b, 0, y1_i:y2_i, x1_i:x2_i] = 1.0

                cx = (x1[i] + x2[i]) / 2
                cy = (y1[i] + y2[i]) / 2
                l_val = (cx - x1_i).item()
                t_val = (cy - y1_i).item()
                r_val = (x2_i - cx).item()
                b_val = (y2_i - cy).item()

                target_regression[b, 0, y1_i:y2_i, x1_i:x2_i] = l_val
                target_regression[b, 1, y1_i:y2_i, x1_i:x2_i] = t_val
                target_regression[b, 2, y1_i:y2_i, x1_i:x2_i] = r_val
                target_regression[b, 3, y1_i:y2_i, x1_i:x2_i] = b_val

        return target_objectness, target_regression

    def forward(self, outputs, targets):
        losses = {}
        num_levels = len(outputs['objectness'])

        for level in range(num_levels):
            shape = outputs['objectness'][level].shape[-2:]
            target_obj, target_reg = self.prepare_targets(targets, shape)

            loss_obj = self.objectness_loss(outputs['objectness'][level], target_obj)
            losses[f'loss_ce_{level}'] = loss_obj

            if outputs['lbrts'][level] is not None:
                loss_reg = self.giou_loss(outputs['lbrts'][level], target_reg, target_obj)
                losses[f'loss_giou_{level}'] = loss_reg

        losses['loss_ce'] = torch.stack([losses[f'loss_ce_{l}'] for l in range(num_levels)]).mean()
        if 'loss_giou_0' in losses:
            losses['loss_giou'] = torch.stack([losses[f'loss_giou_{l}'] for l in range(num_levels)]).mean()
        else:
            losses['loss_giou'] = torch.tensor(0.0, device=losses['loss_ce'].device)

        return losses
