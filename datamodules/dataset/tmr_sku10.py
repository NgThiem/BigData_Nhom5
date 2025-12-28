
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
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Albumentations + ToTensorV2 → trả về Tensor trong dict["image"]
        if self.transform:
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"]  # ✅ ĐÃ LÀ TORCH TENSOR
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        exemplars = []
        for ex_name in item["exemplar_files"][:self.max_exemplars]:
            ex_path = os.path.join(self.root, "exemplars", ex_name)
            ex_img = Image.open(ex_path).convert("RGB")
            ex_np = np.array(ex_img)

            if self.transform:
                transformed_ex = self.transform(image=ex_np)
                ex_tensor = transformed_ex["image"]  # ✅ ĐÃ LÀ TORCH TENSOR
            else:
                ex_tensor = torch.from_numpy(ex_np).permute(2, 0, 1).float() / 255.0

            exemplars.append(ex_tensor)

        while len(exemplars) < self.max_exemplars and len(exemplars) > 0:
            exemplars.append(exemplars[-1])
        if len(exemplars) == 0:
            exemplars = [image_tensor] * self.max_exemplars

        exemplars = torch.stack(exemplars[:self.max_exemplars])
        boxes = torch.tensor(item["boxes"], dtype=torch.float32)

        return {
            "image": image_tensor,
            "exemplars": exemplars,
            "boxes": boxes,
            "img_name": item["img_name"]
        }