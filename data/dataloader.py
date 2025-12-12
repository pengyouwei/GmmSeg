import torch
from torch.utils.data import DataLoader
from data.dataset import ACDCDataset, MMDataset, SCDDataset, YORKDataset, CombinedDataset
from data.transform import get_image_transform, get_label_transform
from config import Config
import os
import random
import numpy as np

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)                 

def get_loaders(config: Config):
    """
    Initialize and return the training and testing data loaders.
    """
    dataset_name = config.DATASET
    image_transform = get_image_transform(config.IMG_SIZE)
    label_transform = get_label_transform(config.IMG_SIZE)
    def build_single(name: str, phase: str):
        if name == "ACDC":
            return ACDCDataset(phase=phase, transform_image=image_transform, transform_label=label_transform, config=config)
        if name == "MM":
            return MMDataset(phase=phase, transform_image=image_transform, transform_label=label_transform, config=config)
        if name == "SCD":
            return SCDDataset(phase=phase, transform_image=image_transform, transform_label=label_transform, config=config)
        if name == "YORK":
            return YORKDataset(phase=phase, transform_image=image_transform, transform_label=label_transform, config=config)
        raise ValueError(f"Unsupported dataset: {name}")

    # parse combined names: 'ALL' or 'ACDC+MM' / 'ACDC,MM'
    name_upper = (dataset_name or "").upper()
    if name_upper == "ALL":
        selected = ["ACDC", "MM", "SCD", "YORK"]
    elif any(sep in name_upper for sep in ['+', ',']):
        # split by + or , and filter known names
        parts = [p.strip() for sep in ['+', ','] for p in name_upper.split(sep)]
        # The above duplicates splits; better do a two-step
        name_tmp = name_upper.replace('+', ',')
        selected = [p.strip() for p in name_tmp.split(',') if p.strip()]
    else:
        selected = [name_upper]

    if len(selected) == 1 and selected[0] in {"ACDC", "MM", "SCD", "YORK"}:
        sel = selected[0]
        if sel == "ACDC":
            train_set = ACDCDataset(phase="train", transform_image=image_transform, transform_label=label_transform, config=config)
            valid_set = ACDCDataset(phase="val", transform_image=image_transform, transform_label=label_transform, config=config)
            test_set = ACDCDataset(phase="test", transform_image=image_transform, transform_label=label_transform, config=config)
        elif sel == "MM":
            train_set = MMDataset(phase="train", transform_image=image_transform, transform_label=label_transform, config=config)
            valid_set = MMDataset(phase="val", transform_image=image_transform, transform_label=label_transform, config=config)
            test_set = MMDataset(phase="test", transform_image=image_transform, transform_label=label_transform, config=config)
        elif sel == "SCD":
            train_set = SCDDataset(phase="train", transform_image=image_transform, transform_label=label_transform, config=config)
            valid_set = SCDDataset(phase="val", transform_image=image_transform, transform_label=label_transform, config=config)
            test_set = SCDDataset(phase="test", transform_image=image_transform, transform_label=label_transform, config=config)
        elif sel == "YORK":
            train_set = YORKDataset(phase="train", transform_image=image_transform, transform_label=label_transform, config=config)
            valid_set = YORKDataset(phase="val", transform_image=image_transform, transform_label=label_transform, config=config)
            test_set = YORKDataset(phase="test", transform_image=image_transform, transform_label=label_transform, config=config)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    else:
        # Build combined across phases with aligned subsets
        train_pairs = []
        val_pairs = []
        test_pairs = []
        for name in selected:
            if name not in {"ACDC", "MM", "SCD", "YORK"}:
                raise ValueError(f"Unsupported dataset in combination: {name}")
            train_pairs.append((name, build_single(name, "train")))
            val_pairs.append((name, build_single(name, "val")))
            test_pairs.append((name, build_single(name, "test")))

        train_set = CombinedDataset(train_pairs)
        valid_set = CombinedDataset(val_pairs)
        test_set = CombinedDataset(test_pairs)

    g = torch.Generator()
    g.manual_seed(config.SEED)
    train_loader = DataLoader(train_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              pin_memory=True, 
                              drop_last=True)
    valid_loader = DataLoader(valid_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, 
                              pin_memory=True, 
                              drop_last=False)
    test_loader = DataLoader(test_set, 
                             batch_size=config.BATCH_SIZE, 
                             shuffle=False,
                             num_workers=config.NUM_WORKERS, 
                             pin_memory=True, 
                             drop_last=False)

    return train_loader, valid_loader, test_loader

    