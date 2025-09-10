import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from config import Config
from data.transform import center_crop_image_label




# 定义数据集ACDCDataset
# The spatial resolution goes from 1.37 to 1.68 mm2/pixel
class ACDCDataset(Dataset):
    def __init__(self, phase='train', transform_image=None, transform_label=None, config: Config=Config()):
        self.root_dir = os.path.join(config.DATASET_DIR, "ACDC_aligned")
        self.config = config
        self.img_size = config.IMG_SIZE
        self.gmm_num = config.GMM_NUM
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = []
        self.labels = []
        self.phase = phase
        self.prior_num_of_patient = config.PRIOR_NUM_OF_PATIENT

        ED_prior_path = os.path.join(self.root_dir, f'ED_prior_{self.prior_num_of_patient}samples_4chs.npy')
        ES_prior_path = os.path.join(self.root_dir, f'ES_prior_{self.prior_num_of_patient}samples_4chs.npy')
        group_path = [entry for entry in os.scandir(self.root_dir) if entry.is_dir()]
        for gp in group_path:
            for frame_path in os.scandir(gp):
                frame_name = frame_path.name
                if frame_name not in ['ED', 'ES']:
                    continue
                prior_path = ED_prior_path if frame_name == "ED" else ES_prior_path
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
        image, label = center_crop_image_label(image, label, self.config.ACDC_CROP_SIZE)
        
        if self.transform_image:
            image = self.transform_image(image) # [C, H, W]
        if self.transform_label:
            label = self.transform_label(label) # [C, H, W]

        
        # 读取数据集中的prior
        max_slice_id = 9
        slice_id = min(slice_id, max_slice_id) # 限制slice_id在[0, max_slice_id]范围内
        prior = np.load(self.images[idx]['prior_path'])[slice_id]
        prior_size = prior.shape[-1]
        if prior_size != self.img_size:
            prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(0)  # (1, gmm_num, H, W)
            prior = F.interpolate(prior, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
            prior = prior.squeeze(0)  # (gmm_num, img_size, img_size)
        else:
            prior = torch.tensor(prior, dtype=torch.float32)
        prior = torch.clamp(prior, min=1e-6, max=1.0)


        # 根据label动态生成prior
        label_prior = np.zeros((self.gmm_num, self.img_size, self.img_size), dtype=np.float32)
        label_np = label.squeeze().detach().cpu().numpy()
        for k in range(self.gmm_num):
            label_prior[k] = (label_np == k).astype(np.float32)
        label_prior = torch.tensor(label_prior, dtype=torch.float32)
        label_prior = torch.clamp(label_prior, min=1e-6, max=1.0)

        return {
            "image": image,
            "label": label,
            "prior": prior,
            "label_prior": label_prior,
        }
        

#  0.955 to 2.64 mm2/pixel
class MMDataset(Dataset):
    def __init__(self, phase='train', transform_image=None, transform_label=None, config: Config=Config()):
        self.root_dir = os.path.join(config.DATASET_DIR, "MM")
        self.config = config
        self.img_size = config.IMG_SIZE
        self.gmm_num = config.GMM_NUM
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = []
        self.labels = []
        self.phase = phase
        self.prior_num_of_patient = config.PRIOR_NUM_OF_PATIENT

        ED_prior_path = os.path.join(self.root_dir, f'ED_prior_{self.prior_num_of_patient}samples_4chs.npy')
        ES_prior_path = os.path.join(self.root_dir, f'ES_prior_{self.prior_num_of_patient}samples_4chs.npy')
        group_path = [entry for entry in os.scandir(self.root_dir) if entry.is_dir()]
        for gp in group_path:
            for frame_path in os.scandir(gp):
                frame_name = frame_path.name
                if frame_name not in ["ED", "ES"]:
                    continue
                prior_path = ED_prior_path if frame_name == "ED" else ES_prior_path
                phase_image_path = os.path.join(frame_path.path, f'{phase}_image', 'slices')
                phase_label_path = os.path.join(frame_path.path, f'{phase}_label', 'slices')
                for image_file in os.scandir(phase_image_path):
                    if not image_file.name.endswith('.npy'): continue
                    slice_id = int(image_file.name.split('_')[-1].split('.')[0])
                    self.images.append({
                        'path': image_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                        'frame_name': frame_name
                    })
                for label_file in os.scandir(phase_label_path):
                    if not label_file.name.endswith('.npy'): continue
                    slice_id = int(label_file.name.split('_')[-1].split('.')[0])
                    self.labels.append({
                        'path': label_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                        'frame_name': frame_name
                    })

        self.images.sort(key=lambda x: x['path'])
        self.labels.sort(key=lambda x: x['path'])
        assert len(self.images) == len(self.labels), "Number of images and labels do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        slice_id = self.images[idx]['slice_id']
        image = np.load(self.images[idx]['path']) # [0.0, 255.0]
        label = np.load(self.labels[idx]['path'])
        image = Image.fromarray(image.astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))

        # 使用基于左心室中心的裁剪策略
        image, label = center_crop_image_label(image, label, self.config.MM_CROP_SIZE)
        
        if self.transform_image:
            image = self.transform_image(image) # [C, H, W]
            image = torch.flip(image, dims=[2])  # 左右翻转
        if self.transform_label:
            label = self.transform_label(label) # [C, H, W]
            label = torch.flip(label, dims=[2])


        # 读取数据集中的prior
        max_slice_id = 9
        slice_id = min(slice_id, max_slice_id)
        prior = np.load(self.images[idx]['prior_path'])[slice_id]
        prior_size = prior.shape[-1]
        if prior_size != self.img_size:
            prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(0)  # (1, gmm_num, H, W)
            prior = F.interpolate(prior, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
            prior = prior.squeeze(0)  # (gmm_num, img_size, img_size)
        else:
            prior = torch.tensor(prior, dtype=torch.float32)
        prior = torch.clamp(prior, min=1e-6, max=1.0)
        # prior = torch.flip(prior, dims=[2])


        # 根据label动态生成prior
        label_prior = np.zeros((self.gmm_num, self.img_size, self.img_size), dtype=np.float32)
        label_np = label.squeeze().detach().cpu().numpy()
        for k in range(self.gmm_num):
            label_prior[k] = (label_np == k).astype(np.float32)
        label_prior = torch.tensor(label_prior, dtype=torch.float32)
        label_prior = torch.clamp(label_prior, min=1e-6, max=1.0)
        
        return {
            "image": image,
            "label": label,
            "prior": prior,
            "label_prior": label_prior
        }
    


class SCDDataset(Dataset):
    def __init__(self, phase='train', transform_image=None, transform_label=None, config: Config=Config()):
        self.root_dir = os.path.join(config.DATASET_DIR, "SCD")
        self.config = config
        self.img_size = config.IMG_SIZE
        self.gmm_num = config.GMM_NUM
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.images = []
        self.labels = []
        self.phase = phase
        self.prior_num_of_patient = config.PRIOR_NUM_OF_PATIENT

        ED_prior_path = os.path.join(self.root_dir, f'ED_prior_{self.prior_num_of_patient}samples_4chs.npy')
        ES_prior_path = os.path.join(self.root_dir, f'ES_prior_{self.prior_num_of_patient}samples_4chs.npy')
        group_path = [entry for entry in os.scandir(self.root_dir) if entry.is_dir()]
        for gp in group_path:
            for frame_path in os.scandir(gp):
                frame_name = frame_path.name
                if frame_name not in ["ED", "ES"]:
                    continue
                prior_path = ED_prior_path if frame_name == "ED" else ES_prior_path
                phase_image_path = os.path.join(frame_path.path, f'{phase}_image', 'slices')
                phase_label_path = os.path.join(frame_path.path, f'{phase}_label', 'slices')
                for image_file in os.scandir(phase_image_path):
                    if not image_file.name.endswith('.npy'): continue
                    slice_id = int(image_file.name.split('_')[-1].split('.')[0])
                    self.images.append({
                        'path': image_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                        'frame_name': frame_name
                    })
                for label_file in os.scandir(phase_label_path):
                    if not label_file.name.endswith('.npy'): continue
                    slice_id = int(label_file.name.split('_')[-1].split('.')[0])
                    self.labels.append({
                        'path': label_file.path,
                        'slice_id': slice_id,
                        'prior_path': prior_path,
                        'frame_name': frame_name
                    })

        self.images.sort(key=lambda x: x['path'])
        self.labels.sort(key=lambda x: x['path'])
        assert len(self.images) == len(self.labels), "Number of images and labels do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        slice_id = self.images[idx]['slice_id']
        image = np.load(self.images[idx]['path']) # [0.0, 255.0]
        label = np.load(self.labels[idx]['path'])
        image = Image.fromarray((image * 255).astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))

        # 使用基于左心室中心的裁剪策略
        image, label = center_crop_image_label(image, label, self.config.SCD_CROP_SIZE)

        if self.transform_image:
            image = self.transform_image(image) # [C, H, W]
            image = torch.rot90(image, k=3, dims=[1, 2]) # 顺时针旋转90度
            # image = torch.flip(image, dims=[2])  # 左右翻转
            
        if self.transform_label:
            label = self.transform_label(label) # [C, H, W]
            label = torch.rot90(label, k=3, dims=[1, 2]) # 顺时针旋转90度
            # label = torch.flip(label, dims=[2])


        # 读取数据集中的prior
        max_slice_id = 9
        slice_id = min(slice_id, max_slice_id)
        prior = np.load(self.images[idx]['prior_path'])[slice_id]
        prior_size = prior.shape[-1]
        if prior_size != self.img_size:
            prior = torch.tensor(prior, dtype=torch.float32).unsqueeze(0)  # (1, gmm_num, H, W)
            prior = F.interpolate(prior, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
            prior = prior.squeeze(0)  # (gmm_num, img_size, img_size)
        else:
            prior = torch.tensor(prior, dtype=torch.float32)
        prior = torch.clamp(prior, min=1e-6, max=1.0)


        # 根据label动态生成prior
        label_prior = np.zeros((self.gmm_num, self.img_size, self.img_size), dtype=np.float32)
        label_np = label.squeeze().detach().cpu().numpy()
        for k in range(self.gmm_num):
            label_prior[k] = (label_np == k).astype(np.float32)
        label_prior = torch.tensor(label_prior, dtype=torch.float32)
        label_prior = torch.clamp(label_prior, min=1e-6, max=1.0)
        
        return {
            "image": image,
            "label": label,
            "prior": prior,
            "label_prior": label_prior
        }