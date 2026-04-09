import os
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from data.dataloader import get_loaders
from utils.train_utils import forward_pass
from models.unet import UNet
from models.regnet import RR_ResNet
from models.align_net import Align_ResNet
from models.scale_net import Scale_ResNet


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_weights(model: torch.nn.Module, path: str, device: torch.device, strict: bool = True) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # torch.load(weights_only=...) is not available in older torch versions
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=strict)


def _to_hw_label(label_tensor: torch.Tensor) -> np.ndarray:
    arr = label_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.argmax(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Unable to convert label to [H,W], got shape {arr.shape}")
    return arr.astype(np.uint8)


def _class_name_for_dataset(ds: str, class_id: int) -> str:
    """Human-friendly class name for each dataset.

    Label IDs in this repo are normalized to:
    - 0: background
    - 1: LV
    - 2: MYO
    - 3: RV (only for ACDC/MM)
    """
    class_id = int(class_id)
    if class_id == 0:
        return "BG"

    ds_upper = str(ds).upper()
    if ds_upper == "SCD":
        mapping = {1: "LV"}
    elif ds_upper == "YORK":
        mapping = {1: "LV", 2: "MYO"}
    else:
        # ACDC/MM or generic
        mapping = {1: "LV", 2: "MYO", 3: "RV"}
    return mapping.get(class_id, f"class{class_id}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Draw t-SNE of feature maps from model")
    p.add_argument("--dataset", type=str, default=Config.DATASET, help="ACDC|MM|SCD|YORK|ALL")
    p.add_argument(
        "--checkpoints_dir",
        type=str,
        default=Config.CHECKPOINTS_DIR,
        help="Base checkpoints directory (default: ./checkpoints)",
    )
    p.add_argument("--start_id", type=int, default=0)
    p.add_argument("--end_id", type=int, default=-1, help="-1 means last")
    p.add_argument("--id", type=int, default=None, help="single sample id (overrides start/end)")
    p.add_argument("--out_dir", type=str, default=os.path.join(Config.RESULTS_DIR, "tsne"))
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_points", type=int, default=2000, help="max number of points to embed (pixels or images)")
    p.add_argument("--mode", type=str, default="pixel", choices=["pixel", "image"], help="'pixel' embed pixel-wise features, 'image' embed pooled image-level features")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--random_seed", type=int, default=0)
    return p


def _build_global_models(config: Config, device: torch.device) -> Tuple[torch.nn.Module, ...]:
    """Build + load models that are global (not per-dataset)."""
    unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(device)
    reg_net = RR_ResNet(input_channels=2).to(device)
    align_net = Align_ResNet(input_channels=1).to(device)
    scale_net = Scale_ResNet(input_channels=2).to(device)

    unet_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", "unet_best.pth")
    reg_ckpt = os.path.join(config.CHECKPOINTS_DIR, "regnet", "regnet_prior_2chs.pth")
    align_ckpt = os.path.join(config.CHECKPOINTS_DIR, "align_net", "align_best.pth")
    scale_ckpt = os.path.join(config.CHECKPOINTS_DIR, "scale_net", "scale_best.pth")

    _load_weights(unet, unet_ckpt, device=device, strict=True)
    _load_weights(reg_net, reg_ckpt, device=device, strict=True)
    _load_weights(align_net, align_ckpt, device=device, strict=True)
    _load_weights(scale_net, scale_ckpt, device=device, strict=True)

    unet.eval(); reg_net.eval(); align_net.eval(); scale_net.eval()
    return unet, reg_net, align_net, scale_net


def _build_gmm_models_for_dataset(config: Config, device: torch.device, ds_name: str) -> Tuple[torch.nn.Module, ...]:
    """Build + load x/z/o nets whose checkpoints are stored per dataset."""
    x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM * config.GMM_NUM * 2).to(device)
    z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
    o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)

    x_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", str(ds_name), "x_best.pth")
    z_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", str(ds_name), "z_best.pth")
    o_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", str(ds_name), "o_best.pth")

    _load_weights(o_net, o_ckpt, device=device, strict=True)
    _load_weights(x_net, x_ckpt, device=device, strict=True)
    _load_weights(z_net, z_ckpt, device=device, strict=True)

    x_net.eval(); z_net.eval(); o_net.eval()
    return x_net, z_net, o_net


def main(args):
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = Config()
    config.DATASET = args.dataset
    config.CHECKPOINTS_DIR = args.checkpoints_dir
    # force deterministic small batch
    config.BATCH_SIZE = 1
    config.NUM_WORKERS = 1

    _, _, test_loader = get_loaders(config)
    dataset_len = len(test_loader.dataset)
    if dataset_len <= 0:
        raise RuntimeError("Empty test dataset")

    if args.id is not None:
        start_id = end_id = int(args.id)
    else:
        start_id = max(0, int(args.start_id))
        end_id = int(args.end_id)
        if end_id < 0:
            end_id = dataset_len - 1
    start_id = max(0, start_id)
    end_id = min(dataset_len - 1, end_id)

    out_base = args.out_dir
    ds_dir = os.path.join(out_base, str(args.dataset))
    _ensure_dir(ds_dir)
    _ensure_dir(os.path.join(ds_dir, "tsne"))

    unet, reg_net, align_net, scale_net = _build_global_models(config, device)
    gmm_cache: Dict[str, Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]] = {}

    all_feats: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    sample_ids: List[int] = []

    # iterate and collect features
    for idx, batch in enumerate(test_loader):
        if idx < start_id:
            continue
        if idx > end_id:
            break

        image = batch["image"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.long)
        prior = batch["prior"].to(device=device, dtype=torch.float32)
        ds = batch["ds"][0]

        ds_name = str(ds)
        if ds_name not in gmm_cache:
            gmm_cache[ds_name] = _build_gmm_models_for_dataset(config, device, ds_name)
        x_net, z_net, o_net = gmm_cache[ds_name]

        with torch.no_grad():
            out = forward_pass(
                image=image,
                label=label,
                prior=prior,
                unet=unet,
                x_net=x_net,
                z_net=z_net,
                o_net=o_net,
                reg_net=reg_net,
                align_net=align_net,
                scale_net=scale_net,
                ds=ds,
                config=config,
                epoch=0,
                epsilon=1e-6,
            )

        # feature_4chs: [B, C, H, W]
        feat = out.get("feature_4chs", None)
        if feat is None:
            raise RuntimeError("forward_pass did not return 'feature_4chs'")
        feat = feat[0].detach().cpu().numpy()  # [C,H,W]
        gt = _to_hw_label(out.get("label", label))

        C, H, W = feat.shape
        if args.mode == "pixel":
            vectors = feat.reshape(C, H * W).T  # [H*W, C]
            labels = gt.reshape(H * W)
        else:  # image mode: global avg pool
            vectors = feat.mean(axis=(1, 2)).reshape(1, C)  # [1, C]
            labels = np.array([np.bincount(gt.flatten(), minlength=4).argmax()])

        all_feats.append(vectors)
        all_labels.append(labels)
        sample_ids.append(idx)

    if len(all_feats) == 0:
        raise RuntimeError("No samples collected for t-SNE")

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    rng = np.random.RandomState(int(args.random_seed))
    n_points = X.shape[0]
    max_pts = max(1, int(args.max_points))
    if n_points > max_pts:
        inds = rng.choice(n_points, size=max_pts, replace=False)
        X_sample = X[inds]
        y_sample = y[inds]
    else:
        X_sample = X
        y_sample = y

    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        raise RuntimeError("scikit-learn is required for t-SNE. Install it with 'pip install scikit-learn'")

    tsne = TSNE(n_components=2, perplexity=max(5.0, min(50.0, float(args.perplexity))), random_state=int(args.random_seed))
    Z = tsne.fit_transform(X_sample)

    # color map: BG gray, LV red, MYO green, RV blue, others -> tab10
    cmap = {
        0: (0.6, 0.6, 0.6),
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
    }

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)
    unique_labels = np.unique(y_sample).astype(np.int64)
    # Deterministic ordering: BG first, then ascending
    unique_labels = np.array(sorted([int(x) for x in unique_labels], key=lambda x: (0 if x == 0 else 1, x)), dtype=np.int64)
    # If dataset=ALL (mixed), use generic ACDC/MM naming for the legend.
    legend_ds = args.dataset if str(args.dataset).upper() != "ALL" else "ACDC"

    for lbl in unique_labels:
        mask = (y_sample == lbl)
        color = cmap.get(int(lbl), None)
        if color is None:
            color = plt.cm.tab10(int(lbl) % 10)
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=[color],
            s=2,
            label=_class_name_for_dataset(legend_ds, int(lbl)),
            alpha=0.8,
        )

    ax.legend(title="class", markerscale=4, fontsize=6)
    ax.set_title(f"t-SNE ({args.dataset}) samples {start_id}-{end_id} mode={args.mode}")
    ax.set_xticks([]); ax.set_yticks([])

    out_png = os.path.join(ds_dir, "tsne", f"tsne_{args.mode}_{start_id:05d}_{end_id:05d}.png")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved t-SNE plot: {out_png}")


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    main(args)
