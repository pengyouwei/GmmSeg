import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.affine_transform import random_affine_params, apply_affine_transform


def compute_prior(label, gmm_num=4, img_size=128):
    """计算先验概率分布"""
    prior = np.zeros((gmm_num, img_size, img_size), dtype=np.float32)
    cls_map = {0:0, 1:85, 2:170, 3:255}
    for k in range(gmm_num):
        prior[k] = (label.squeeze().cpu().numpy() == cls_map[k]).astype(np.float32)
    prior = np.clip(prior, a_min=1e-8, a_max=None)
    prior = prior / (np.sum(prior, axis=0, keepdims=True))
    return torch.tensor(prior, dtype=torch.float32)


# 定义数据集ACDCDataset
class ACDCDataset(Dataset):
    """
    自定义数据集类, 用于读取 ACDC 数据集中的 frameXX 和其 _gt 标签。
    """
    def __init__(self, root_dir, phase='training', transform_image=None, transform_label=None, img_size=128, gmm_num=4):
        """
        Args:
            root_dir (str): 数据集的根目录路径。
            phase (str): 'training' 或 'testing', 选择训练或测试集。
            transform (callable, optional): 图像和标签的可选变换。
        """
        self.img_size = img_size
        self.gmm_num = gmm_num
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = []
        self.labels = []

        phase_path = os.path.join(root_dir, phase)
        patients = [entry for entry in os.scandir(phase_path) if entry.is_dir()]
        for patient in patients:
            cfg_path = os.path.join(patient.path, "Info.cfg")
            if not os.path.exists(cfg_path):
                print(f"Warning: Info.cfg not found for patient {patient.name}")
                continue
                
            try:
                with open(cfg_path, "r") as cfg:
                    lines = cfg.readlines()
                ED = lines[0].split(":")[1].strip()
                ES = lines[1].split(":")[1].strip()
            except (IndexError, ValueError) as e:
                print(f"Warning: Error parsing Info.cfg for patient {patient.name}: {e}")
                continue 
            for entry in os.scandir(patient.path):
                if not entry.is_dir() or "4d" in entry.name:
                    continue
                frame = entry.name.split("_")[1][-2:]
                frame = "ED" if int(frame) == int(ED) else "ES"
                slice_num = sum(1 for file in os.listdir(entry.path) if file.endswith(".png"))
                for slice, file in enumerate(os.listdir(entry.path)):
                    if not file.endswith(".png"):
                        continue
                    file_path = os.path.join(entry.path, file)
                    if "gt" not in entry.name:
                        self.images.append({
                            "patient": patient.name,
                            "frame": frame,
                            "slice": slice,
                            "slice_num": slice_num,
                            "path": file_path,
                        })
                    elif "gt" in entry.name:
                        self.labels.append({
                            "patient": patient.name,
                            "frame": frame,
                            "slice": slice,
                            "slice_num": slice_num,
                            "path": file_path,
                        })


        # 确保图像和标签数量一致
        assert len(self.images) == len(self.labels), "Number of images and labels do not match."
        # print(f"ACDCDataset ({phase}) loaded: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = self.transform_image(Image.open(self.images[idx]["path"]).convert('L'))  # [C, H, W]
            label = self.transform_label(Image.open(self.labels[idx]["path"]).convert('L'))  # [C, H, W]
        except Exception as e:
            print(f"Error loading image/label at index {idx}: {e}")
            print(f"Image path: {self.images[idx]['path']}")
            print(f"Label path: {self.labels[idx]['path']}")
            raise
        prior = np.zeros((self.gmm_num, self.img_size, self.img_size), dtype=np.float32) # [4, H, W]
        cls_map = {0:0, 1:85, 2:170, 3:255}
        for k in range(self.gmm_num):
            prior[k] = (label.squeeze().cpu().numpy() == cls_map[k]).astype(np.float32)
        prior = np.clip(prior, a_min=1e-8, a_max=None) # [4, H, W]
        prior = prior / (np.sum(prior, axis=0, keepdims=True)) # 归一化 [4, H, W]
        prior = torch.tensor(prior, dtype=torch.float32) # [4, H, W]

        return {
            "image": {
                "patient": self.images[idx]["patient"],
                "frame": self.images[idx]["frame"],
                "slice": self.images[idx]["slice"],
                "slice_num": self.images[idx]["slice_num"],
                "data": image,  # [C, H, W]
            },
            "label": {
                "patient": self.labels[idx]["patient"],
                "frame": self.labels[idx]["frame"],
                "slice": self.labels[idx]["slice"],
                "slice_num": self.labels[idx]["slice_num"],
                "data": label,  # [C, H, W]
            },
            "prior": {
                "data": prior,  # [4, H, W]
            },
        }
    



# 先验数据集
class PriorDataset(Dataset):
    def __init__(self, root_dir, transform_image=None, transform_label=None, gmm_num=4, slice_num=10, img_size=128):
        self.gmm_num = gmm_num
        self.slice_num = slice_num
        self.img_size = img_size
        self.transform_image = transform_image
        self.transform_label = transform_label
        
        self.images = []
        self.labels = []
        patients = [entry for entry in os.scandir(root_dir) if entry.is_dir()]
        for patient in patients:
            for frame in os.scandir(patient.path):
                img_dir = os.path.join(frame.path, "img")
                seg_dir = os.path.join(frame.path, "seg")
                if not os.path.exists(img_dir) or not os.path.exists(seg_dir):
                    continue
                img_files = sorted([file for file in os.listdir(img_dir) if file.endswith(".png")])
                seg_files = sorted([file for file in os.listdir(seg_dir) if file.endswith(".png")])
                
                # 确保图像和标签文件数量一致
                if len(img_files) != len(seg_files):
                    print(f"Warning: Mismatch in {patient.name}/{frame.name} - {len(img_files)} images vs {len(seg_files)} labels")
                    continue
                    
                for slice, (img_file, seg_file) in enumerate(zip(img_files, seg_files)):
                    img_path = os.path.join(img_dir, img_file)
                    seg_path = os.path.join(seg_dir, seg_file)
                    self.images.append({
                        "patient": patient.name,
                        "frame": frame.name,
                        "slice": slice,
                        "path": img_path,
                    })
                    self.labels.append({
                        "patient": patient.name,
                        "frame": frame.name,
                        "slice": slice,
                        "path": seg_path,
                    })
                    
        # 确保图像和标签数量一致
        assert len(self.images) == len(self.labels), f"Number of images ({len(self.images)}) and labels ({len(self.labels)}) do not match."
        print(f"PriorDataset loaded: {len(self.images)} samples")
                    

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        try:
            image = self.transform_image(Image.open(self.images[idx]["path"]).convert('L'))  # [C, H, W]
            label = self.transform_label(Image.open(self.labels[idx]["path"]).convert('L'))  # [C, H, W]
        except Exception as e:
            print(f"Error loading image/label at index {idx} in PriorDataset: {e}")
            print(f"Image path: {self.images[idx]['path']}")
            print(f"Label path: {self.labels[idx]['path']}")
            raise
            
        prior = compute_prior(label, self.gmm_num, self.img_size)

        return {
            "image": {
                "patient": self.images[idx]["patient"],
                "frame": self.images[idx]["frame"],
                "slice": self.images[idx]["slice"],
                "data": image,  # [C, H, W]
            },
            "label": {
                "patient": self.labels[idx]["patient"],
                "frame": self.labels[idx]["frame"],
                "slice": self.labels[idx]["slice"],
                "data": label,  # [C, H, W]
            },
            "prior": {
                "data": prior,  # [4, H, W]
            },
        }




# 配准数据集
class RRDataset(Dataset):
    def __init__(self, root_dir, phase='training', transform_image=None, transform_label=None, img_size=128, gmm_num=4):
        self.img_size = img_size
        self.gmm_num = gmm_num
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.file_paths = []

        phase_path = os.path.join(root_dir, phase)
        patients = [entry for entry in os.scandir(phase_path) if entry.is_dir()]
        for patient in patients:
            for entry in os.scandir(patient.path):
                if not entry.is_dir() or "4d" in entry.name or "gt" not in entry.name:
                    continue
                for file in os.listdir(entry.path):
                    if not file.endswith(".png"):
                        continue
                    file_path = os.path.join(entry.path, file)
                    self.file_paths.append(file_path)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        obj = Image.open(file_path).convert('L')
        label = self.transform_label(obj) # [C, H, W]
        label_01 = label / 255.0 # [C, H, W]
        
        prior = compute_prior(label, self.gmm_num, self.img_size)
        

        # 生成随机的刚性变换参数
        # fixed = label_01.clone()  # [1, H, W]
        fixed = prior.clone()  # [4, H, W]
        scale, tx, ty = random_affine_params(scale_range=(0.2, 5.0), shift_range=(-20, 20))
        moving = apply_affine_transform(fixed.unsqueeze(0), scale, tx, ty).squeeze(0)  # [C, H, W]
        input_tensor = torch.cat((fixed, moving), dim=0)  # 将固定图像和移动图像拼接在一起
        affine_param = torch.tensor([scale, tx, ty], dtype=torch.float32)  # [3]

        return {
            "fixed_moving": input_tensor,  # [C*2, H, W]
            "affine_param": affine_param,  # [3]
            "label": label_01,  # [C, H, W]
            "prior": prior,  # [4, H, W]
        }


