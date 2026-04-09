import argparse
import csv
import os
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from data.dataloader import get_loaders
from models.unet import UNet
from models.align_net import Align_ResNet
from models.scale_net import Scale_ResNet
from utils.metrics import evaluate_segmentation
from utils.train_utils import forward_pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_weights(model: torch.nn.Module, path: str, device: torch.device, strict: bool = True) -> None:
    if not path:
        raise ValueError("Empty checkpoint path")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=strict)


def _postprocess_pred(ds: str, pred_cls: np.ndarray) -> np.ndarray:
    ds_upper = str(ds).upper()
    if ds_upper == "SCD":
        pred_cls = pred_cls.copy()
        pred_cls[pred_cls == 2] = 0
        pred_cls[pred_cls == 3] = 0
    if ds_upper == "YORK":
        pred_cls = pred_cls.copy()
        pred_cls[pred_cls == 3] = 0
    return pred_cls


@torch.no_grad()
def compute_testset_dice_csv(
    config: Config,
    csv_path: str,
    x_ckpt: str,
    z_ckpt: str,
    o_ckpt: str,
    unet_ckpt: str,
    align_ckpt: str,
    scale_ckpt: str,
) -> str:
    """Run inference on test set and save per-sample Dice to CSV."""

    # Force deterministic ordered iteration for per-sample metrics
    config = replace(config, BATCH_SIZE=1, NUM_WORKERS=1, MODE="test")

    device = config.DEVICE

    _, _, test_loader = get_loaders(config)
    if len(test_loader) <= 0:
        raise RuntimeError("Empty test loader")

    # Models (match training pipeline)
    unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(device)
    x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM * config.GMM_NUM * 2).to(device)
    z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
    o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
    align_net = Align_ResNet(input_channels=1).to(device)
    scale_net = Scale_ResNet(input_channels=2).to(device)

    # Load weights
    _load_weights(unet, unet_ckpt, device=device, strict=True)
    _load_weights(x_net, x_ckpt, device=device, strict=True)
    _load_weights(z_net, z_ckpt, device=device, strict=True)
    _load_weights(o_net, o_ckpt, device=device, strict=True)
    _load_weights(align_net, align_ckpt, device=device, strict=True)
    _load_weights(scale_net, scale_ckpt, device=device, strict=True)

    # Eval mode
    unet.eval()
    x_net.eval()
    z_net.eval()
    o_net.eval()
    align_net.eval()
    scale_net.eval()

    _ensure_dir(os.path.dirname(csv_path) or ".")

    fieldnames = [
        "index",
        "dataset",
        "frame_name",
        "slice_id",
        "dice_mean",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, batch in enumerate(test_loader):
            image = batch["image"].to(device=device, dtype=torch.float32)
            label = batch["label"].to(device=device, dtype=torch.float32)
            prior = batch["prior"].to(device=device, dtype=torch.float32)

            ds = batch["ds"][0]

            out = forward_pass(
                image=image,
                label=label,
                prior=prior,
                unet=unet,
                x_net=x_net,
                z_net=z_net,
                o_net=o_net,
                reg_net=None,
                align_net=align_net,
                scale_net=scale_net,
                ds=ds,
                config=config,
                epoch=None,
                epsilon=1e-6,
            )

            r = out["r"]
            pred_cls = torch.argmax(r, dim=1).detach().cpu().numpy()  # [1,H,W]
            label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()  # [1,H,W]

            pred_cls = _postprocess_pred(ds=ds, pred_cls=pred_cls)

            class_num = int(batch["class_num"][0])

            # Per-sample metrics (B=1)
            pred_hw = pred_cls[0]
            true_hw = label_np[0]
            metrics = evaluate_segmentation(
                y_pred=pred_hw,
                y_true=true_hw,
                num_classes=class_num,
                background=config.METRIC_WITH_BACKGROUND,
                test=False,
            )

            row = {
                "index": idx,
                "dataset": ds,
                "frame_name": batch.get("frame_name", [""])[0],
                "slice_id": int(batch.get("slice_id", torch.tensor([-1]))[0]),
                "dice_mean": float(metrics["Dice"]["mean"]),
            }
            writer.writerow(row)

    return csv_path


def _read_csv_column(csv_path: str, key: str) -> List[float]:
    values: List[float] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get(key, "")
            if v is None or v == "":
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if np.isnan(fv):
                continue
            values.append(fv)
    return values


def plot_violin_from_csv(
    csv_path: str,
    out_path: str,
    dataset: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if dataset is None:
        # Try to infer dataset from first row
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            dataset = (first or {}).get("dataset", "")

    values = _read_csv_column(csv_path, "dice_mean")
    if len(values) == 0:
        raise RuntimeError("No numeric dice_mean values found in CSV")

    _ensure_dir(os.path.dirname(out_path) or ".")

    # Paper-friendly styling: one main color + black outlines
    main_color = "#4C72B0"  # muted blue (commonly used in academic plots)
    line_color = "#000000"

    plt.figure(figsize=(4.2, 4.2), dpi=160)
    parts = plt.violinplot([values], showmeans=True, showextrema=True, showmedians=False)

    # Violin body
    for pc in parts.get("bodies", []):
        pc.set_facecolor(main_color)
        pc.set_edgecolor(line_color)
        pc.set_linewidth(1.0)
        pc.set_alpha(0.35)

    # Violin lines (means/extrema)
    for k in ("cmeans", "cmins", "cmaxes", "cbars"):
        if k in parts and parts[k] is not None:
            try:
                parts[k].set_color(line_color)
                parts[k].set_linewidth(1.2)
            except Exception:
                pass

    # Overlay boxplot (box + whiskers) on top of violin
    plt.boxplot(
        [values],
        positions=[1],
        widths=0.22,
        showfliers=False,
        patch_artist=True,
        medianprops={"linewidth": 1.4, "color": line_color},
        boxprops={"linewidth": 1.2, "edgecolor": line_color, "facecolor": "white"},
        whiskerprops={"linewidth": 1.2, "color": line_color},
        capprops={"linewidth": 1.2, "color": line_color},
    )

    plt.xticks([1], ["Mean"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Dice")

    if title is None:
        ds = dataset or ""
        title = f"Test-set Dice Violin Plot ({ds})" if ds else "Test-set Dice Violin Plot"
    plt.title(title)

    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def build_default_paths(dataset: str, checkpoints_dir: str) -> Dict[str, str]:
    ds = dataset
    return {
        "unet_ckpt": os.path.join(checkpoints_dir, "unet", "unet_best.pth"),
        "x_ckpt": os.path.join(checkpoints_dir, "unet", ds, "x_best.pth"),
        "z_ckpt": os.path.join(checkpoints_dir, "unet", ds, "z_best.pth"),
        "o_ckpt": os.path.join(checkpoints_dir, "unet", ds, "o_best.pth"),
        "align_ckpt": os.path.join(checkpoints_dir, "align_net", "align_best.pth"),
        "scale_ckpt": os.path.join(checkpoints_dir, "scale_net", "scale_best.pth"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-sample test Dice and draw violin plot")
    parser.add_argument("--dataset", type=str, default=Config.DATASET, help="ACDC | MM | SCD | YORK")
    parser.add_argument("--dataset_dir", type=str, default=Config.DATASET_DIR)
    parser.add_argument("--checkpoints_dir", type=str, default=Config.CHECKPOINTS_DIR)
    parser.add_argument("--metric_with_background", action="store_true", help="Include background in Dice")

    parser.add_argument("--csv", type=str, default="", help="Output CSV path (default: results/violin/<DATASET>_test_dice.csv)")
    parser.add_argument("--plot", type=str, default="", help="Output plot path (default: results/violin/<DATASET>_test_dice_violin.png)")
    parser.add_argument("--title", type=str, default="", help="Plot title override")

    parser.add_argument("--only_plot", action="store_true", help="Only plot from existing CSV")

    # Optional: override checkpoints
    parser.add_argument("--unet_ckpt", type=str, default="")
    parser.add_argument("--x_ckpt", type=str, default="")
    parser.add_argument("--z_ckpt", type=str, default="")
    parser.add_argument("--o_ckpt", type=str, default="")
    parser.add_argument("--align_ckpt", type=str, default="")
    parser.add_argument("--scale_ckpt", type=str, default="")

    args = parser.parse_args()

    dataset = args.dataset.upper()

    out_dir = os.path.join("results", "violin")
    _ensure_dir(out_dir)

    csv_path = args.csv or os.path.join(out_dir, f"{dataset}_test_dice.csv")
    plot_path = args.plot or os.path.join(out_dir, f"{dataset}_test_dice_violin.png")

    cfg = Config(
        DATASET=dataset,
        DATASET_DIR=args.dataset_dir,
        CHECKPOINTS_DIR=args.checkpoints_dir,
        METRIC_WITH_BACKGROUND=bool(args.metric_with_background),
    )

    defaults = build_default_paths(dataset=dataset, checkpoints_dir=args.checkpoints_dir)
    paths = {
        "unet_ckpt": args.unet_ckpt or defaults["unet_ckpt"],
        "x_ckpt": args.x_ckpt or defaults["x_ckpt"],
        "z_ckpt": args.z_ckpt or defaults["z_ckpt"],
        "o_ckpt": args.o_ckpt or defaults["o_ckpt"],
        "align_ckpt": args.align_ckpt or defaults["align_ckpt"],
        "scale_ckpt": args.scale_ckpt or defaults["scale_ckpt"],
    }

    if not args.only_plot:
        compute_testset_dice_csv(
            config=cfg,
            csv_path=csv_path,
            x_ckpt=paths["x_ckpt"],
            z_ckpt=paths["z_ckpt"],
            o_ckpt=paths["o_ckpt"],
            unet_ckpt=paths["unet_ckpt"],
            align_ckpt=paths["align_ckpt"],
            scale_ckpt=paths["scale_ckpt"],
        )

    plot_violin_from_csv(
        csv_path=csv_path,
        out_path=plot_path,
        dataset=dataset,
        title=args.title or None,
    )

    print(f"CSV saved: {csv_path}")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
