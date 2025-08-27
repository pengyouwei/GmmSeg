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
    match dataset_name:
        case "ACDC_aligned":
            image_transform = get_image_transform(config.IMG_SIZE)
            label_transform = get_label_transform(config.IMG_SIZE)
            train_set = ACDCDataset(phase="train", 
                                    transform_image=image_transform, 
                                    transform_label=label_transform, 
                                    config=config)
            valid_set = ACDCDataset(phase="val", 
                                    transform_image=image_transform, 
                                    transform_label=label_transform, 
                                    config=config)
        case _:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True, 
                              drop_last=True)
    valid_loader = DataLoader(valid_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True, 
                              drop_last=True)

    return train_loader, valid_loader


    