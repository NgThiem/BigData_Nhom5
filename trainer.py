
import torch
import pytorch_lightning as pl
import numpy as np
from torchvision.ops import nms
from models import build_model
from criterion.criterions_TM import SetCriterion_TM

def match_name_keywords(n, name_keywords):
    for keyword in name_keywords:
        if keyword in n:
            return True
    return False

class Matching_Trainer(pl.LightningModule):
    def __init__(self, args, datamodule):
        super().__init__()
        self.args = args
        self.datamodule = datamodule
        self.model = build_model(args)
        self.criterion = SetCriterion_TM(args)
        self.val_preds = []
        self.val_gts = []

    def training_step(self, batch, batch_idx):
        return self.each_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        loss = self.each_step(batch, 'val')
        
        image = batch['image']
        exemplars = batch['exemplars']
        boxes = batch['boxes']

        if self.args.template_type in ['roi_align', 'prototype']:
            K = self.args.num_exemplars
            boxes_for_tm = []
            for b in boxes:
                if b.size(0) >= K:
                    boxes_for_tm.append(b[:K])
                else:
                    pad = b[-1:].repeat(K - b.size(0), 1)
                    boxes_for_tm.append(torch.cat([b, pad], dim=0))
            model_input = boxes_for_tm
        else:
            model_input = exemplars

        pred_objectness, pred_regressions, _, _ = self.model(image, model_input)
        
        obj_map = pred_objectness[0].sigmoid()  # [B, 1, H, W]
        ltrb_map = pred_regressions[0]          # [B, 4, H, W]
        B, _, H, W = obj_map.shape

        for i in range(B):
            scores = obj_map[i, 0].view(-1)
            topk_scores, topk_idx = scores.topk(100)
            y = topk_idx // W
            x = topk_idx % W

            l = ltrb_map[i, 0, y, x]
            t = ltrb_map[i, 1, y, x]
            r = ltrb_map[i, 2, y, x]
            b = ltrb_map[i, 3, y, x]

            x1 = (x - l).clamp(min=0, max=W - 1)
            y1 = (y - t).clamp(min=0, max=H - 1)
            x2 = (x + r).clamp(min=0, max=W - 1)
            y2 = (y + b).clamp(min=0, max=H - 1)

            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1).float()
            pred_boxes = pred_boxes * (1024.0 / H)

            keep = nms(pred_boxes, topk_scores, self.args.NMS_iou_threshold)
            pred_count = keep.numel()
            gt_count = boxes[i].size(0)

            self.val_preds.append(pred_count)
            self.val_gts.append(gt_count)

        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            print("WARNING: No predictions collected during validation!")
            return

        preds = np.array(self.val_preds)
        gts = np.array(self.val_gts)
        mae = np.mean(np.abs(preds - gts))
        rmse = np.sqrt(np.mean((preds - gts) ** 2))

        print(f"\n VALIDATION EPOCH END — MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        # Log cho Lightning (hiển thị trên progress bar)
        self.log('val_mae', mae, prog_bar=True, sync_dist=True)
        self.log('val_rmse', rmse, prog_bar=True, sync_dist=True)

        # Reset buffer
        self.val_preds.clear()
        self.val_gts.clear()

    def each_step(self, batch, mode):
        image = batch['image']
        exemplars = batch['exemplars']
        boxes = batch['boxes']

        if self.args.template_type in ['roi_align', 'prototype']:
            K = self.args.num_exemplars
            boxes_for_tm = []
            for b in boxes:
                if b.size(0) >= K:
                    boxes_for_tm.append(b[:K])
                else:
                    pad = b[-1:].repeat(K - b.size(0), 1)
                    boxes_for_tm.append(torch.cat([b, pad], dim=0))
            model_input = boxes_for_tm
        else:
            model_input = exemplars

        pred_objectness, pred_regressions, matching_feature, _ = self.model(image, model_input)

        targets = [{'boxes': b} for b in boxes]
        outputs = {
            'objectness': pred_objectness,
            'lbrts': pred_regressions
        }
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss_dict.values())
        self.log(f'{mode}_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        weight_decay = getattr(self.args, 'weight_decay', 1e-4)
        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, ['backbone']) and p.requires_grad
                ],
                "lr": self.args.lr
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if match_name_keywords(n, ['backbone']) and p.requires_grad
                ],
                "lr": getattr(self.args, 'lr_backbone', 0.0)
            }
        ]

        if getattr(self.args, 'lr_drop', False):
            milestones = [int(self.args.max_epochs * 0.6)]
        else:
            milestones = [self.args.max_epochs + 1]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
