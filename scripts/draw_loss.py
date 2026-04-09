import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ClipConfig:
    enabled: bool = True
    quantile: float = 0.98
    factor: float = 1.2
    outlier_ratio: float = 10.0


def _to_float(x: str) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def read_history_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        cols: Dict[str, List[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in cols.keys():
                cols[k].append(_to_float(row.get(k, "")))

    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def robust_ylim(values: np.ndarray, clip: ClipConfig) -> Optional[Tuple[float, float]]:
    v = _finite(values)
    if v.size == 0:
        return None

    v_min = float(np.min(v))
    v_max = float(np.max(v))

    if not clip.enabled:
        pad = 0.05 * (v_max - v_min) if v_max > v_min else 1.0
        return v_min - pad, v_max + pad

    q = float(np.quantile(v, clip.quantile))
    q = max(q, 1e-12)  # 防止全 0 的情况

    # 如果最大值远大于分位数，认为存在离群点：限制 y_max 以避免曲线被压平。
    y_max = v_max
    clipped = False
    if v_max > q * clip.outlier_ratio:
        y_max = q * clip.factor
        clipped = True

    # y_min 也做一点 padding，保证曲线贴边不会太难看
    y_min = v_min
    span = max(y_max - y_min, 1e-6)
    y_min = y_min - 0.05 * span
    y_max = y_max + 0.05 * span

    if clipped:
        # 如果裁剪后 y_max 仍然小于真实最大值，图上会“截断”极端点
        # 这是有意为之：更关注后续稳定阶段的变化。
        pass

    return float(y_min), float(y_max)


def save_fig(fig: plt.Figure, out_path: str, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_total_loss(data: Dict[str, np.ndarray], out_path: str, clip: ClipConfig, dpi: int) -> None:
    epochs = data.get("epoch")
    if epochs is None:
        raise ValueError("CSV missing column: epoch")

    y_train = data.get("train_total")
    y_valid = data.get("valid_total")

    # 方形画布（并尽量让坐标轴区域为方形）
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

    if y_train is not None:
        ax.plot(epochs, y_train, label="train_total", linewidth=2)
    if y_valid is not None:
        ax.plot(epochs, y_valid, label="valid_total", linewidth=2)

    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    # y 轴范围：合并 train/valid 一起做鲁棒裁剪
    ys = []
    if y_train is not None:
        ys.append(y_train)
    if y_valid is not None:
        ys.append(y_valid)
    if ys:
        y_all = np.concatenate([np.asarray(y, dtype=np.float64) for y in ys], axis=0)
        lim = robust_ylim(y_all, clip=clip)
        if lim is not None:
            ax.set_ylim(*lim)

    save_fig(fig, out_path, dpi=dpi)


def plot_loss_components(
    data: Dict[str, np.ndarray],
    out_path: str,
    clip: ClipConfig,
    dpi: int,
    include_valid: bool,
) -> None:
    epochs = data.get("epoch")
    if epochs is None:
        raise ValueError("CSV missing column: epoch")

    # 方形画布（并尽量让坐标轴区域为方形）
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

    train_keys = ["train_loss1", "train_loss2", "train_loss3"]
    valid_keys = ["valid_loss1", "valid_loss2", "valid_loss3"]

    colors = ["tab:blue", "tab:orange", "tab:green"]

    ys_for_ylim: List[np.ndarray] = []

    for k, c in zip(train_keys, colors):
        y = data.get(k)
        if y is None:
            continue
        ax.plot(epochs, y, label=k, linewidth=2, color=c)
        ys_for_ylim.append(y)

    if include_valid:
        for k, c in zip(valid_keys, colors):
            y = data.get(k)
            if y is None:
                continue
            ax.plot(epochs, y, label=k, linewidth=2, color=c, linestyle="--", alpha=0.85)
            ys_for_ylim.append(y)

    ax.set_title("Loss Components (loss1 / loss2 / loss3)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2)

    # 固定 y 轴最大值，突出 loss1 的变化
    y_max_fixed = 30.0
    if ys_for_ylim:
        y_all = np.concatenate([np.asarray(y, dtype=np.float64) for y in ys_for_ylim], axis=0)
        y_min = float(np.nanmin(_finite(y_all))) if _finite(y_all).size > 0 else 0.0
        span = max(y_max_fixed - y_min, 1e-6)
        y_min = y_min - 0.05 * span
        ax.set_ylim(y_min, y_max_fixed)
    else:
        ax.set_ylim(0.0, y_max_fixed)

    save_fig(fig, out_path, dpi=dpi)


def plot_dice(data: Dict[str, np.ndarray], out_path: str, dpi: int) -> None:
    epochs = data.get("epoch")
    dice = data.get("valid_dice")
    if epochs is None or dice is None:
        raise ValueError("CSV missing column: epoch or valid_dice")

    # 方形画布（并尽量让坐标轴区域为方形）
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ax.plot(epochs, dice, label="valid_dice", linewidth=2, color="tab:red")

    ax.set_title("Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    v = _finite(dice)
    if v.size > 0:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.02
        ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))

    save_fig(fig, out_path, dpi=dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw loss/dice curves from history.csv")
    parser.add_argument("--csv", default="history.csv", help="Path to history.csv")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: same dir as csv)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--include_valid", action="store_true", help="Also plot valid_loss1/2/3 in components")

    parser.add_argument("--no_clip", action="store_true", help="Do not apply robust y-lim clipping")
    parser.add_argument("--clip_quantile", type=float, default=0.98)
    parser.add_argument("--clip_factor", type=float, default=1.2)
    parser.add_argument("--outlier_ratio", type=float, default=10.0)

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."

    clip = ClipConfig(
        enabled=not args.no_clip,
        quantile=float(args.clip_quantile),
        factor=float(args.clip_factor),
        outlier_ratio=float(args.outlier_ratio),
    )

    data = read_history_csv(csv_path)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    total_png = os.path.join(out_dir, f"{base}_total_loss.png")
    comp_png = os.path.join(out_dir, f"{base}_loss_components.png")
    dice_png = os.path.join(out_dir, f"{base}_dice.png")

    plot_total_loss(data, total_png, clip=clip, dpi=args.dpi)
    plot_loss_components(data, comp_png, clip=clip, dpi=args.dpi, include_valid=args.include_valid)
    plot_dice(data, dice_png, dpi=args.dpi)

    print("Saved:")
    print(" -", total_png)
    print(" -", comp_png)
    print(" -", dice_png)

    if clip.enabled:
        print(
            f"Note: y-axis uses robust clipping (q={clip.quantile}, factor={clip.factor}, outlier_ratio={clip.outlier_ratio}). "
            "Use --no_clip to disable."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
