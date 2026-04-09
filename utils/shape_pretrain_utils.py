import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def labels_to_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    label: [B,1,H,W] or [B,H,W]
    return: [B,C,H,W]
    """
    if label.dim() == 4:
        label = label[:, 0, :, :]
    label = label.long().clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(label, num_classes=num_classes).permute(0, 3, 1, 2).float()
    return one_hot


def foreground_channels(one_hot: torch.Tensor, bg_index: int = 0) -> torch.Tensor:
    if one_hot.size(1) <= 1:
        return one_hot
    if bg_index == 0:
        return one_hot[:, 1:, :, :]
    keep = [i for i in range(one_hot.size(1)) if i != bg_index]
    return one_hot[:, keep, :, :]


def mask_to_signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    Binary mask -> signed distance map.
    Inside positive, outside negative, boundary near zero.
    """
    m = (mask > 0.5).astype(np.uint8)

    # Empty/full masks make distance transform numerically unstable.
    if m.max() == 0 or m.min() == 1:
        return np.zeros_like(mask, dtype=np.float32)

    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = inside - outside

    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    sdf[boundary > 0] = 0.0
    sdf = np.nan_to_num(sdf, nan=0.0, posinf=0.0, neginf=0.0)
    return sdf.astype(np.float32)


def one_hot_to_sdf(one_hot: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    one_hot: [B,C,H,W], values in {0,1}
    return: [B,C,H,W] signed distance maps
    """
    device = one_hot.device
    dtype = one_hot.dtype

    arr = one_hot.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)

    norm = float(max(h, w)) if normalize else 1.0
    norm = max(norm, 1.0)

    for bi in range(b):
        for ci in range(c):
            sdf = mask_to_signed_distance(arr[bi, ci])
            out[bi, ci] = sdf / norm

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    return torch.from_numpy(out).to(device=device, dtype=dtype)


def prepare_shape_target_from_label(label: torch.Tensor, num_classes: int, bg_index: int = 0) -> torch.Tensor:
    one_hot = labels_to_one_hot(label=label, num_classes=num_classes)
    fg = foreground_channels(one_hot=one_hot, bg_index=bg_index)
    return one_hot_to_sdf(fg, normalize=True)


def save_pca_prior(path: str, mean: np.ndarray, eigvals: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), eigvals=eigvals.astype(np.float32))


def load_pca_prior(path: str, latent_dim: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    pack = np.load(path)
    mean = pack["mean"].astype(np.float32)
    eigvals = pack["eigvals"].astype(np.float32)
    eigvals = np.nan_to_num(eigvals, nan=1.0, posinf=1e6, neginf=1.0)

    if eigvals.size < latent_dim:
        pad = np.ones((latent_dim - eigvals.size,), dtype=np.float32)
        eigvals = np.concatenate([eigvals, pad], axis=0)

    eigvals = np.clip(eigvals[:latent_dim], 1e-6, 1e6)
    prior_var = torch.from_numpy(eigvals).to(device=device, dtype=torch.float32).unsqueeze(0)
    return prior_var, mean, eigvals


def compute_pca_prior_from_dataset(
    dataset: Dataset,
    num_classes: int,
    latent_dim: int,
    bg_index: int = 0,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    total = len(dataset)
    if max_samples > 0:
        total = min(total, max_samples)

    for idx in range(total):
        sample = dataset[idx]
        label = sample["label"]
        if label.dim() == 3:
            label = label.unsqueeze(0)
        elif label.dim() == 2:
            label = label.unsqueeze(0).unsqueeze(0)

        sdf = prepare_shape_target_from_label(label=label, num_classes=num_classes, bg_index=bg_index)
        vec = sdf.squeeze(0).reshape(-1).cpu().numpy().astype(np.float64)
        vectors.append(vec)

    if len(vectors) == 0:
        raise RuntimeError("No samples found when computing PCA prior.")

    x = np.stack(vectors, axis=0)
    mean = x.mean(axis=0)

    if x.shape[0] < 2:
        eigvals = np.ones((latent_dim,), dtype=np.float32)
        return mean.astype(np.float32), eigvals

    x_centered = x - mean[None, :]
    _, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    eigvals = (s ** 2) / max(1, x.shape[0] - 1)
    eigvals = np.nan_to_num(eigvals, nan=1.0, posinf=1e6, neginf=1.0)

    if eigvals.size < latent_dim:
        pad = np.ones((latent_dim - eigvals.size,), dtype=np.float64)
        eigvals = np.concatenate([eigvals, pad], axis=0)

    eigvals = np.clip(eigvals[:latent_dim], 1e-6, 1e6)
    return mean.astype(np.float32), eigvals.astype(np.float32)


def beta_with_warmup(epoch: int, warmup_epochs: int, max_beta: float, min_beta: float = 0.0) -> float:
    if warmup_epochs <= 0:
        return float(max_beta)
    ratio = min(1.0, max(0.0, float(epoch + 1) / float(warmup_epochs)))
    return float(min_beta + (max_beta - min_beta) * ratio)


def shape_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    prior_var: torch.Tensor | None = None,
    recon_type: str = "smooth_l1",
) -> dict:
    if recon_type == "l1":
        recon_loss = F.l1_loss(recon, target)
    elif recon_type == "mse":
        recon_loss = F.mse_loss(recon, target)
    else:
        recon_loss = F.smooth_l1_loss(recon, target)

    var = torch.exp(logvar)
    if prior_var is None:
        prior_var = torch.ones_like(var)
    prior_var = prior_var.clamp_min(1e-6)

    ratio = (var / prior_var).clamp_min(1e-8)
    kl = 0.5 * (mu.pow(2) / prior_var + ratio - torch.log(ratio) - 1.0)
    kl = kl.sum(dim=1).mean()

    total = recon_loss + float(beta) * kl
    return {
        "total": total,
        "recon": recon_loss,
        "kl": kl,
        "beta": float(beta),
    }
