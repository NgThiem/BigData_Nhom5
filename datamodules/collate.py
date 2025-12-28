# datamodules/collate.py
import torch

def custom_collate(batch):
    # Các key cần xử lý đặc biệt
    special_keys = {"image", "boxes", "exemplars"}
    
    # Tạo dict kết quả
    result = {}
    
    # Xử lý key đặc biệt
    if "image" in batch[0]:
        result["image"] = torch.stack([x["image"] for x in batch])
    
    if "boxes" in batch[0]:
        result["boxes"] = [x["boxes"].detach().clone() for x in batch]
    
    if "exemplars" in batch[0]:
        result["exemplars"] = [x["exemplars"].detach().clone() for x in batch]
    
    # Tự động thêm các key còn lại (không cần xử lý đặc biệt)
    for key in batch[0].keys():
        if key not in special_keys:
            result[key] = [x[key] for x in batch]
    
    return result