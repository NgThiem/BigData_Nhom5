# datamodules/tmr_sku10_dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TMRSKU10Dataset(Dataset):
    def __init__(self, root, transform=None, max_exemplars=5, split="train", **kwargs):
        self.root = root
        self.transform = transform
        self.max_exemplars = max_exemplars

        ann_file = os.path.join(root, "exemplars.json")
        with open(ann_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            self.data_list = []
            for img_name, entry in raw_data.items():
                self.data_list.append({
                    "img_name": img_name,
                    "exemplar_files": entry["exemplar_files"],
                    "boxes": entry["boxes"]
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = os.path.join(self.root, "images", item["img_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Anh khong ton tai: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2] 
    
        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    
        exemplar_files = item["exemplar_files"]
        boxes = item["boxes"]
    
        min_len = min(len(exemplar_files), len(boxes))
        if min_len == 0:
            exemplars = [image_tensor] * self.max_exemplars
            # Dùng box toàn ảnh đã chuẩn hóa
            boxes_normalized = [[0.0, 0.0, 1.0, 1.0]] * self.max_exemplars
        else:
            exemplar_files = exemplar_files[:min_len]
            boxes = boxes[:min_len]
    
            exemplars = []
            for ex_name in exemplar_files[:self.max_exemplars]:
                ex_path = os.path.join(self.root, "exemplars", ex_name)
                if not os.path.exists(ex_path):
                    ex_tensor = image_tensor
                else:
                    ex_img = Image.open(ex_path).convert("RGB")
                    ex_np = np.array(ex_img)
                    if self.transform is not None:
                        transformed_ex = self.transform(image=ex_np)
                        ex_tensor = transformed_ex["image"]
                    else:
                        ex_tensor = torch.from_numpy(ex_np).permute(2, 0, 1).float() / 255.0
                exemplars.append(ex_tensor)
    
            while len(exemplars) < self.max_exemplars:
                exemplars.append(exemplars[-1] if exemplars else image_tensor)
            
            # XỬ LÝ BOXES: ĐẢM BẢO HỢP LỆ + CHUẨN HÓA
            boxes_normalized = []
            boxes_sync = boxes[:self.max_exemplars]
            while len(boxes_sync) < self.max_exemplars:
                boxes_sync.append(boxes_sync[-1] if boxes_sync else [0, 0, w, h])
            
            for box in boxes_sync:
                x1, y1, x2, y2 = box
                # Clamp into image boundaries
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))  # Ensure width >= 1
                y2 = max(y1 + 1, min(h, y2))  # Ensure height >= 1
                # Normalize to [0, 1]
                boxes_normalized.append([x1 / w, y1 / h, x2 / w, y2 / h])
    
        exemplars = torch.stack(exemplars)
        boxes_tensor = torch.tensor(boxes_normalized, dtype=torch.float32)
    
        return {
            "image": image_tensor,
        "exemplars": exemplars,
        "boxes": boxes_tensor,  # ← ĐÃ CHUẨN HÓA [0,1] VÀ HỢP LỆ
        "img_name": item["img_name"]
    }
