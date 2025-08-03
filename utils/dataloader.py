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