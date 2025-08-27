import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from config import Config
from data.transform import center_crop_image_label, center_crop_prior_4chs




# 定义数据集ACDCDataset
class ACDCDataset(Dataset):
    def __init__(self, phase='train', transform_image=None, transform_label=None, config: Config=Config()):
        self.root_dir = os.path.join(config.DATASET_DIR, config.DATASET)
        self.config = config
        self.img_size = config.IMG_SIZE
        self.gmm_num = config.GMM_NUM
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = []
        self.labels = []

        group_path = [entry for entry in os.scandir(self.root_dir) if entry.is_dir()]
        for gp in group_path:
            for frame_path in os.scandir(gp):
                prior_path = os.path.join(frame_path.path, f'prior_{self.gmm_num}chs.npy')
                phase_image_path = os.path.join(frame_path.path, f'{phase}_image', 'slices')
                phase_label_path = os.path.join(frame_path.path, f'{phase}_label', 'slices')
                for image_file in os.scandir(phase_image_path):
                    if not image_file.name.endswith('.npy'): continue
                    slice_id = int(image_file.name.split('_')[-1].split('.')[0])
                    self.images.append({
                        'path': image_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                    })
                for label_file in os.scandir(phase_label_path):
                    if not label_file.name.endswith('.npy'): continue
                    slice_id = int(label_file.name.split('_')[-1].split('.')[0])
                    self.labels.append({
                        'path': label_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                    })

        self.images.sort(key=lambda x: x['path'])
        self.labels.sort(key=lambda x: x['path'])
        assert len(self.images) == len(self.labels), "Number of images and labels do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        slice_id = self.images[idx]['slice_id']
        image = np.load(self.images[idx]['path'])
        label = np.load(self.labels[idx]['path'])
        image = Image.fromarray((image * 255).astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))

        # 使用基于左心室中心的裁剪策略
        image, label = center_crop_image_label(image, label, self.img_size)
        
        if self.transform_image:
            image = self.transform_image(image) # [C, H, W]
        if self.transform_label:
            label = self.transform_label(label) # [C, H, W]

        # 直接读取数据集中的prior
        slice_id = min(slice_id, 9) # 限制slice_id在[0, 9]范围内
        prior = np.load(self.images[idx]['prior_path'])[slice_id]
        prior = center_crop_prior_4chs(prior, self.img_size)
        prior = np.clip(prior, a_min=1e-6, a_max=1.0)
        prior = torch.tensor(prior, dtype=torch.float32) # (gmm_num, H, W)

        # 根据label动态生成prior
        # prior = np.zeros((self.gmm_num, self.img_size, self.img_size), dtype=np.float32)
        # label_np = label.squeeze().detach().cpu().numpy()
        # for k in range(self.gmm_num):
        #     prior[k] = (label_np == k).astype(np.float32)
        # prior = np.clip(prior, a_min=1e-6, a_max=1.0)
        # prior = torch.tensor(prior, dtype=torch.float32)
        
        return {
            "image": image,
            "label": label,
            "prior": prior,
        }
    
        


