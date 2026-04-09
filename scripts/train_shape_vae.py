import os
import csv
import time
import random
import logging
import sys

import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config, format_config
from data.dataset import ACDCPriorDataset
from data.transform import get_image_transform, get_label_transform
from models.shape_vae import ShapeVAE
from utils.shape_pretrain_utils import (
    beta_with_warmup,
    compute_pca_prior_from_dataset,
    load_pca_prior,
    prepare_shape_target_from_label,
    save_pca_prior,
    shape_vae_loss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_shape_loaders(config):
    image_transform = get_image_transform(config.IMG_SIZE)
    label_transform = get_label_transform(config.IMG_SIZE)

    train_set = ACDCPriorDataset(phase='train', transform_image=image_transform, transform_label=label_transform, config=config)
    val_set = ACDCPriorDataset(phase='val', transform_image=image_transform, transform_label=label_transform, config=config)

    g = torch.Generator()
    g.manual_seed(config.SEED)

    use_workers = int(config.NUM_WORKERS)
    train_loader = DataLoader(
        train_set,
        batch_size=int(config.SHAPE_BATCH_SIZE),
        shuffle=True,
        num_workers=use_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        persistent_workers=(use_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config.SHAPE_BATCH_SIZE),
        shuffle=False,
        num_workers=use_workers,
        pin_memory=True,
        persistent_workers=(use_workers > 0),
        drop_last=False,
    )
    return train_loader, val_loader, train_set


def get_logger(log_dir: str):
    logger = logging.getLogger('shape_vae_trainer')
    logger.setLevel(logging.INFO)

    while logger.handlers:
        logger.handlers.pop()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
    logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_path = os.path.join(log_dir, f'{time_str}.log')

    file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(file_handler)

    return logger, console_handler, file_handler, time_str


def maybe_prepare_pca_prior(config, train_dataset, logger):
    pca_path = config.SHAPE_PCA_PATH
    need_recompute = bool(config.SHAPE_RECOMPUTE_PCA) or (not os.path.isfile(pca_path))

    if need_recompute:
        logger.info('Computing PCA prior from ACDCPriorDataset...')
        mean, eigvals = compute_pca_prior_from_dataset(
            dataset=train_dataset,
            num_classes=int(config.SHAPE_NUM_CLASSES),
            latent_dim=int(config.SHAPE_LATENT_DIM),
            bg_index=int(config.SHAPE_BG_INDEX),
            max_samples=int(config.SHAPE_MAX_PCA_SAMPLES),
        )
        save_pca_prior(path=pca_path, mean=mean, eigvals=eigvals)
        logger.info(f'Saved PCA prior to: {pca_path}')

    prior_var, mean, eigvals = load_pca_prior(
        path=pca_path,
        latent_dim=int(config.SHAPE_LATENT_DIM),
        device=config.DEVICE,
    )
    logger.info(f'Loaded PCA prior: mean_dim={mean.shape[0]}, eigvals_dim={eigvals.shape[0]}')
    return prior_var


def run_epoch(model, loader, optimizer, device, config, prior_var, epoch, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    beta = beta_with_warmup(
        epoch=epoch,
        warmup_epochs=int(config.SHAPE_KL_WARMUP_EPOCHS),
        max_beta=float(config.SHAPE_KL_BETA),
        min_beta=0.0,
    )

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=f"Epoch {epoch+1:3}/{config.SHAPE_EPOCHS}[{'train' if train else 'valid'}]", ncols=100):
            label = batch['label'].to(device=device, dtype=torch.float32)
            target = prepare_shape_target_from_label(
                label=label,
                num_classes=int(config.SHAPE_NUM_CLASSES),
                bg_index=int(config.SHAPE_BG_INDEX),
            )

            if train:
                optimizer.zero_grad()

            recon, mu, logvar, _ = model(target)
            loss_out = shape_vae_loss(
                recon=recon,
                target=target,
                mu=mu,
                logvar=logvar,
                beta=beta,
                prior_var=prior_var,
                recon_type=str(config.SHAPE_RECON_TYPE),
            )
            loss = loss_out['total']

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_recon += float(loss_out['recon'].item())
            total_kl += float(loss_out['kl'].item())

    n = max(1, len(loader))
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n,
        'beta': float(beta),
    }


def plot_history(history, out_dir, time_str):
    epochs = history['epoch']

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(epochs, history['train_loss'], label='train_total', linewidth=2)
    ax1.plot(epochs, history['valid_loss'], label='valid_total', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Shape-VAE Total Loss')
    ax1.legend()
    fig1.tight_layout()
    p1 = os.path.join(out_dir, f'shape_vae_total_{time_str}.png')
    fig1.savefig(p1, dpi=200)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(epochs, history['train_recon'], label='train_recon', linewidth=2)
    ax2.plot(epochs, history['valid_recon'], label='valid_recon', linewidth=2)
    ax2.plot(epochs, history['train_kl'], label='train_kl', linewidth=2)
    ax2.plot(epochs, history['valid_kl'], label='valid_kl', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Shape-VAE Components')
    ax2.legend(ncol=2)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, f'shape_vae_components_{time_str}.png')
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)

    return [p1, p2]


def save_history_csv(history, out_dir, time_str):
    fields = ['epoch', 'lr', 'beta', 'train_loss', 'train_recon', 'train_kl', 'valid_loss', 'valid_recon', 'valid_kl']
    path = os.path.join(out_dir, f'history_shape_vae_{time_str}.csv')

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        n = len(history['epoch'])
        for i in range(n):
            w.writerow([history[k][i] for k in fields])

    return path


def main():
    config = get_config()
    set_seed(config.SEED)

    shape_log_dir = os.path.join(config.SHAPE_LOG_DIR, config.DATASET)
    logger, console_handler, file_handler, time_str = get_logger(shape_log_dir)

    logger.info('Starting shape-VAE pretraining...')
    logger.info('Full configuration:\n' + format_config(config))

    train_loader, val_loader, train_dataset = build_shape_loaders(config)
    logger.info(f'Train samples: {len(train_loader.dataset)}')
    logger.info(f'Valid samples: {len(val_loader.dataset)}')

    model = ShapeVAE(
        in_channels=max(1, int(config.SHAPE_NUM_CLASSES) - 1),
        latent_dim=int(config.SHAPE_LATENT_DIM),
        base_channels=int(config.SHAPE_BASE_CHANNELS),
    ).to(config.DEVICE)

    prior_var = maybe_prepare_pca_prior(config=config, train_dataset=train_dataset, logger=logger)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.SHAPE_LR),
        weight_decay=float(config.SHAPE_WEIGHT_DECAY),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=5,
        min_lr=1e-7,
    )

    save_dir = os.path.join(config.SHAPE_SAVE_DIR, config.DATASET)
    os.makedirs(save_dir, exist_ok=True)

    writer = None
    if config.SHAPE_ADD_TENSORBOARD:
        tb_dir = os.path.join(config.LOGS_DIR, 'tensorboard_shape_vae', config.DATASET, time_str)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

    history = {
        'epoch': [],
        'lr': [],
        'beta': [],
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'valid_loss': [],
        'valid_recon': [],
        'valid_kl': [],
    }

    best_val = float('inf')

    for epoch in range(int(config.SHAPE_EPOCHS)):
        train_out = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=config.DEVICE,
            config=config,
            prior_var=prior_var,
            epoch=epoch,
            train=True,
        )
        val_out = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=config.DEVICE,
            config=config,
            prior_var=prior_var,
            epoch=epoch,
            train=False,
        )

        scheduler.step(val_out['loss'])

        lr0 = float(optimizer.param_groups[0]['lr']) if len(optimizer.param_groups) > 0 else np.nan

        history['epoch'].append(epoch + 1)
        history['lr'].append(lr0)
        history['beta'].append(train_out['beta'])
        history['train_loss'].append(train_out['loss'])
        history['train_recon'].append(train_out['recon'])
        history['train_kl'].append(train_out['kl'])
        history['valid_loss'].append(val_out['loss'])
        history['valid_recon'].append(val_out['recon'])
        history['valid_kl'].append(val_out['kl'])

        logger.info(
            f"Epoch [{epoch+1}/{config.SHAPE_EPOCHS}] "
            f"lr={lr0:.6e} beta={train_out['beta']:.6f} | "
            f"Train total={train_out['loss']:.4f} recon={train_out['recon']:.4f} kl={train_out['kl']:.4f} | "
            f"Valid total={val_out['loss']:.4f} recon={val_out['recon']:.4f} kl={val_out['kl']:.4f}"
        )

        if writer is not None:
            writer.add_scalars('shape_vae/total', {'train': train_out['loss'], 'valid': val_out['loss']}, epoch + 1)
            writer.add_scalars('shape_vae/recon', {'train': train_out['recon'], 'valid': val_out['recon']}, epoch + 1)
            writer.add_scalars('shape_vae/kl', {'train': train_out['kl'], 'valid': val_out['kl']}, epoch + 1)
            writer.add_scalar('shape_vae/beta', train_out['beta'], epoch + 1)
            writer.add_scalar('shape_vae/lr', lr0, epoch + 1)

        if np.isfinite(val_out['loss']) and val_out['loss'] < best_val:
            best_val = val_out['loss']
            best_path = os.path.join(save_dir, config.SHAPE_CKPT_BEST)
            torch.save(model.state_dict(), best_path)
            logger.info(f'Saved best checkpoint: {best_path} (val={best_val:.4f})')
        elif not np.isfinite(val_out['loss']):
            logger.warning('Validation loss is non-finite. Skip best checkpoint update for this epoch.')

    last_path = os.path.join(save_dir, config.SHAPE_CKPT_LAST)
    torch.save(model.state_dict(), last_path)
    logger.info(f'Saved last checkpoint: {last_path}')

    csv_path = save_history_csv(history=history, out_dir=shape_log_dir, time_str=time_str)
    fig_paths = plot_history(history=history, out_dir=shape_log_dir, time_str=time_str)

    logger.info(f'Saved training history: {csv_path}')
    logger.info('Saved training curves: ' + ', '.join(fig_paths))
    logger.info('Shape-VAE pretraining completed.')

    if writer is not None:
        writer.close()

    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)


if __name__ == '__main__':
    main()
