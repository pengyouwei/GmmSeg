import torch
from torch.utils.data import DataLoader
from data.dataset import ACDCDataset
from data.transform import get_image_transform, get_label_transform
from config import Config
import os

                   

def get_loaders(config: Config):
    """
    Initialize and return the training and testing data loaders.
    """
    dataset_name = config.DATASET
    root_dir = os.path.join(config.DATASET_DIR, dataset_name)
    match dataset_name:
        case "ACDC":
            image_transform = get_image_transform(config.IMG_SIZE)
            label_transform = get_label_transform(config.IMG_SIZE)
            train_set = ACDCDataset(root_dir=root_dir, 
                                    phase="training", 
                                    transform_image=image_transform, 
                                    transform_label=label_transform, 
                                    img_size=config.IMG_SIZE)
            test_set = ACDCDataset(root_dir=root_dir, 
                                   phase="testing", 
                                   transform_image=image_transform, 
                                   transform_label=label_transform, 
                                   img_size=config.IMG_SIZE)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True, 
                              drop_last=True)
    test_loader = DataLoader(test_set, 
                             batch_size=config.BATCH_SIZE, 
                             shuffle=True,
                             num_workers=config.NUM_WORKERS, 
                             pin_memory=True, 
                             drop_last=True)

    return train_loader, test_loader



def get_dirichlet_priors(config: Config):
    """加载预计算先验概率 (仍保持为概率形式)，后续配准完成后再映射到浓度参数。
    返回 shape: [N, K, H, W] float32 in [0,1] (不强制精确归一, 下面会归一化)。"""
    priors_path = [entry.path for entry in os.scandir(os.path.join(config.DATASET_DIR, "dirichlet_priors")) if entry.is_file()]
    if not priors_path:
        raise FileNotFoundError("No Dirichlet priors found in the specified directory.")
    priors_path.sort()
    dirichlet_priors = []
    for prior_path in priors_path:
        dirichlet_prior = torch.load(prior_path, map_location=config.DEVICE, weights_only=True)
        dirichlet_priors.append(dirichlet_prior)
    
    dirichlet_priors = torch.stack(dirichlet_priors, dim=0)  # [10, 4, H, W]
    
    print(f"加载先验概率范围: [{dirichlet_priors.min().item():.4f}, {dirichlet_priors.max().item():.4f}]")
    dirichlet_priors = dirichlet_priors.to(config.DEVICE)
    return dirichlet_priors
    