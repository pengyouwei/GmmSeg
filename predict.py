import os
import argparse
import csv
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import Config
from data.dataloader import get_loaders
from models.unet import UNet
from models.regnet import RR_ResNet
from models.align_net import Align_ResNet
from models.scale_net import Scale_ResNet
from utils.train_utils import forward_pass


def _ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _denorm_image(image: torch.Tensor) -> np.ndarray:
	"""Convert normalized tensor image to float image in [0,1].

	Expects image from dataset transform: Normalize(mean=0.5, std=0.5), so range ~[-1,1].
	"""
	if image.dim() != 4 or image.size(1) != 1:
		raise ValueError(f"Expected image shape [B,1,H,W], got {tuple(image.shape)}")
	img = image[0, 0].detach().cpu().float().numpy()  # [H,W]
	img = img * 0.5 + 0.5
	img = np.clip(img, 0.0, 1.0)
	return img


def _to_hw_label(label_tensor: torch.Tensor) -> np.ndarray:
	"""Convert a torch label tensor to a [H,W] uint8 numpy array.

	Accepts [B,H,W], [B,1,H,W], [H,W], or [1,H,W]/[1,1,H,W] shapes.
	"""
	arr = label_tensor.detach().cpu().numpy()
	# Remove batch dimension if present
	if arr.ndim == 4:
		# [B,1,H,W] or [B,C,H,W]
		arr = arr[0]
	if arr.ndim == 3:
		# [1,H,W] or [C,H,W]
		if arr.shape[0] == 1:
			arr = arr[0]
		else:
			# If multi-channel, take argmax as a fallback
			arr = arr.argmax(axis=0)
	if arr.ndim != 2:
		raise ValueError(f"Unable to convert label to [H,W], got shape {arr.shape}")
	return arr.astype(np.uint8)


def _overlay_segmentation(
	base_gray01: np.ndarray,
	pred_label: np.ndarray,
	alpha: float = 0.5,
) -> np.ndarray:
	"""Overlay predicted labels on grayscale image.

	Colors:
	  1 (LV):  red
	  2 (MYO): green
	  3 (RV):  blue
	Background (0) transparent.
	Returns uint8 RGB image.
	"""
	if base_gray01.ndim != 2:
		raise ValueError(f"base_gray01 must be [H,W], got {base_gray01.shape}")
	if pred_label.shape != base_gray01.shape:
		raise ValueError(
		    f"pred_label shape {pred_label.shape} != image shape {base_gray01.shape}")

	h, w = base_gray01.shape
	base_rgb = np.stack(
	    [base_gray01, base_gray01, base_gray01], axis=-1)  # [H,W,3]

	out = base_rgb.copy()

	# class -> color
	colors = {
		1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # red
		2: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # green
		3: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # blue
	}

	for cls, color in colors.items():
		mask = (pred_label == cls)
		if not np.any(mask):
			continue
		# alpha blend only on foreground
		out[mask] = (1.0 - alpha) * out[mask] + alpha * color

	out = np.clip(out, 0.0, 1.0)
	out_u8 = (out * 255.0).round().astype(np.uint8)
	return out_u8


def _label_to_rgba(pred_label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
	"""Create an RGBA overlay from label map.

	Colors:
	  1 (LV):  red
	  2 (MYO): green
	  3 (RV):  blue
	Background alpha=0.
	"""
	return _label_to_rgba_filtered(pred_label=pred_label, alpha=alpha, allowed_label_ids=(1, 2, 3))


def _label_to_rgba_filtered(
	pred_label: np.ndarray,
	alpha: float = 0.5,
	allowed_label_ids: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
	"""Create an RGBA overlay from label map, optionally filtering label IDs.

	Only labels in `allowed_label_ids` will be rendered; others are treated as background.

	Colors:
	  1 (LV):  red
	  2 (MYO): green
	  3 (RV):  blue
	Background alpha=0.
	"""
	if pred_label.ndim != 2:
		raise ValueError(f"pred_label must be [H,W], got {pred_label.shape}")

	h, w = pred_label.shape
	rgba = np.zeros((h, w, 4), dtype=np.float32)

	allowed = set(int(x) for x in allowed_label_ids)

	# LV
	if 1 in allowed:
		mask = (pred_label == 1)
		rgba[mask, 0] = 1.0
		rgba[mask, 3] = alpha

	# MYO
	if 2 in allowed:
		mask = (pred_label == 2)
		rgba[mask, 1] = 1.0
		rgba[mask, 3] = alpha

	# RV
	if 3 in allowed:
		mask = (pred_label == 3)
		rgba[mask, 2] = 1.0
		rgba[mask, 3] = alpha

	return rgba


def _allowed_labels_for_dataset(ds: str) -> tuple[int, ...]:
	"""Return label IDs to visualize for each dataset.

	- SCD: foreground has LV only
	- YORK: foreground has LV + MYO
	- others (ACDC/MM): LV + MYO + RV
	"""
	ds_upper = str(ds).upper()
	if ds_upper == "SCD":
		return (1,)
	if ds_upper == "YORK":
		return (1, 2)
	return (1, 2, 3)


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
		mapping = {1: "LV", 2: "MYO", 3: "RV"}
	return mapping.get(class_id, f"class{class_id}")


def _entropy_from_r(r: torch.Tensor, eps: float = 1e-12, normalize: bool = True) -> np.ndarray:
	"""Compute per-pixel entropy map from predicted responsibilities r.

	Expected r shape: [B,K,H,W], where r is already a probability simplex.
	Entropy: H = -sum_k p_k log(p_k).
	If normalize=True, returns H/log(K) in [0,1] (approximately).
	"""
	if r.dim() != 4:
		raise ValueError(f"Expected r shape [B,K,H,W], got {tuple(r.shape)}")

	# Safety: ensure normalized probabilities
	p = r.detach().float().clamp_min(eps)
	p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
	ent = -(p * torch.log(p)).sum(dim=1)  # [B,H,W]

	if normalize:
		k = int(p.size(1))
		if k > 1:
			ent = ent / float(np.log(k))

	ent = ent.clamp(min=0.0)
	return ent[0].cpu().numpy()


def _ece_from_r(
	r: torch.Tensor,
	gt_label: np.ndarray,
	n_bins: int = 15,
	allowed_label_ids: tuple[int, ...] = (0, 1, 2, 3),
	include_background: bool = False,
	eps: float = 1e-12,
) -> tuple[float, dict[str, np.ndarray]]:
	"""Compute Expected Calibration Error (ECE) for segmentation from responsibilities r.

	We use the standard confidence-based ECE:
	- confidence per pixel: max_k p_k
	- predicted class: argmax_k p_k
	- correctness: [pred == gt]
	Then bin by confidence and sum: \sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|.

	Returns:
	  (ece, stats) where stats includes per-bin counts/acc/conf/edges.
	"""
	if n_bins <= 0:
		raise ValueError(f"n_bins must be > 0, got {n_bins}")
	if r.dim() != 4:
		raise ValueError(f"Expected r shape [B,K,H,W], got {tuple(r.shape)}")

	# Normalize probabilities defensively.
	p = r.detach().float().clamp_min(eps)
	p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
	probs_full = p[0].cpu().numpy()  # [K,H,W]

	# IMPORTANT for multi-dataset setup:
	# Some datasets (e.g., SCD/YORK) have fewer semantic classes than K.
	# For calibration we must restrict confidence/argmax to the valid label IDs
	# for this dataset; otherwise an "unused" channel can dominate max(prob)
	# and corrupt the reliability curve.
	k = int(probs_full.shape[0])
	allowed_ids = tuple(int(x) for x in allowed_label_ids)
	allowed_ids = tuple(sorted(set(allowed_ids)))
	if any((x < 0 or x >= k) for x in allowed_ids):
		raise ValueError(
			f"allowed_label_ids {allowed_ids} out of range for r with K={k}"
		)

	probs = probs_full[np.array(allowed_ids, dtype=np.int64)]  # [K',H,W]
	den = probs.sum(axis=0, keepdims=True)
	probs = probs / np.clip(den, eps, None)
	conf = probs.max(axis=0)  # [H,W]
	# Map argmax index back to the original label IDs
	pred = np.take(np.array(allowed_ids, dtype=np.int64), probs.argmax(axis=0))

	if gt_label.ndim != 2:
		raise ValueError(f"gt_label must be [H,W], got {gt_label.shape}")
	if gt_label.shape != pred.shape:
		raise ValueError(f"gt_label shape {gt_label.shape} != pred shape {pred.shape}")

	allowed = set(int(x) for x in allowed_label_ids)
	gt = gt_label.astype(np.int64)
	if include_background:
		mask = np.isin(gt, np.array(sorted(allowed), dtype=np.int64))
	else:
		mask = (gt != 0) & np.isin(gt, np.array(sorted(allowed - {0}), dtype=np.int64))

	if not np.any(mask):
		# No valid pixels to score; return NaN-like sentinel and empty stats.
		stats = {
			"bin_edges": np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32),
			"bin_counts": np.zeros((n_bins,), dtype=np.int64),
			"bin_acc": np.zeros((n_bins,), dtype=np.float32),
			"bin_conf": np.zeros((n_bins,), dtype=np.float32),
		}
		return float("nan"), stats

	conf_v = conf[mask].astype(np.float32)
	corr_v = (pred[mask] == gt[mask]).astype(np.float32)

	bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
	# Put conf==1.0 into the last bin
	bin_ids = np.minimum((conf_v * n_bins).astype(np.int64), n_bins - 1)

	bin_counts = np.zeros((n_bins,), dtype=np.int64)
	bin_acc = np.zeros((n_bins,), dtype=np.float32)
	bin_conf = np.zeros((n_bins,), dtype=np.float32)

	for b in range(n_bins):
		m = (bin_ids == b)
		cnt = int(m.sum())
		bin_counts[b] = cnt
		if cnt <= 0:
			continue
		bin_acc[b] = float(corr_v[m].mean())
		bin_conf[b] = float(conf_v[m].mean())

	N = float(bin_counts.sum())
	ece = float(((bin_counts / max(N, 1.0)) * np.abs(bin_acc - bin_conf)).sum())
	stats = {
		"bin_edges": bin_edges,
		"bin_counts": bin_counts,
		"bin_acc": bin_acc,
		"bin_conf": bin_conf,
	}
	return ece, stats


def _accumulate_ece_stats(
	agg: dict[str, np.ndarray],
	stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
	"""Accumulate per-bin sums to compute dataset-level ECE/reliability."""
	if not agg:
		agg = {
			"bin_edges": stats["bin_edges"].copy(),
			"counts": np.zeros_like(stats["bin_counts"], dtype=np.int64),
			"sum_acc": np.zeros_like(stats["bin_acc"], dtype=np.float64),
			"sum_conf": np.zeros_like(stats["bin_conf"], dtype=np.float64),
		}

	counts = stats["bin_counts"].astype(np.int64)
	# stats[bin_acc] and stats[bin_conf] are means; convert to sums by multiplying counts.
	agg["counts"] += counts
	agg["sum_acc"] += stats["bin_acc"].astype(np.float64) * counts
	agg["sum_conf"] += stats["bin_conf"].astype(np.float64) * counts
	return agg


def _ece_from_aggregated_bins(agg: dict[str, np.ndarray]) -> tuple[float, dict[str, np.ndarray]]:
	counts = agg["counts"].astype(np.int64)
	N = float(counts.sum())
	bin_acc = np.zeros_like(counts, dtype=np.float32)
	bin_conf = np.zeros_like(counts, dtype=np.float32)
	mask = counts > 0
	bin_acc[mask] = (agg["sum_acc"][mask] / counts[mask]).astype(np.float32)
	bin_conf[mask] = (agg["sum_conf"][mask] / counts[mask]).astype(np.float32)
	ece = float(((counts / max(N, 1.0)) * np.abs(bin_acc - bin_conf)).sum())
	stats = {
		"bin_edges": agg["bin_edges"],
		"bin_counts": counts,
		"bin_acc": bin_acc,
		"bin_conf": bin_conf,
	}
	return ece, stats


def _ovr_stats_from_r(
	r: torch.Tensor,
	gt_label: np.ndarray,
	class_id: int,
	n_bins: int = 15,
	allowed_label_ids: tuple[int, ...] = (0, 1, 2, 3),
	include_background_pixels: bool = True,
	eps: float = 1e-12,
) -> dict[str, np.ndarray]:
	"""One-vs-rest reliability stats for a single class.

	For each pixel i, define:
	- confidence: p(y=class_id | x_i)
	- correctness: 1[gt_i == class_id]
	Then bin by confidence.

	IMPORTANT for multi-dataset setup:
	If the network outputs K channels but the dataset only uses a subset of label IDs,
	we restrict to `allowed_label_ids` and renormalize before reading p_c.
	"""
	if n_bins <= 0:
		raise ValueError(f"n_bins must be > 0, got {n_bins}")
	if r.dim() != 4:
		raise ValueError(f"Expected r shape [B,K,H,W], got {tuple(r.shape)}")
	if gt_label.ndim != 2:
		raise ValueError(f"gt_label must be [H,W], got {gt_label.shape}")

	# Normalize probabilities defensively.
	p = r.detach().float().clamp_min(eps)
	p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
	probs_full = p[0].cpu().numpy()  # [K,H,W]

	k = int(probs_full.shape[0])
	allowed_ids = tuple(sorted(set(int(x) for x in allowed_label_ids)))
	if any((x < 0 or x >= k) for x in allowed_ids):
		raise ValueError(f"allowed_label_ids {allowed_ids} out of range for r with K={k}")
	if int(class_id) not in allowed_ids:
		raise ValueError(f"class_id={class_id} must be in allowed_label_ids={allowed_ids}")

	# Restrict + renormalize over allowed IDs
	probs = probs_full[np.array(allowed_ids, dtype=np.int64)]  # [K',H,W]
	den = probs.sum(axis=0, keepdims=True)
	probs = probs / np.clip(den, eps, None)

	# Extract p(class_id)
	class_pos = allowed_ids.index(int(class_id))
	p_c = probs[class_pos].astype(np.float32)  # [H,W]

	gt = gt_label.astype(np.int64)
	allowed_set = set(allowed_ids)
	if include_background_pixels:
		mask = np.isin(gt, np.array(sorted(allowed_set), dtype=np.int64))
	else:
		mask = (gt != 0) & np.isin(gt, np.array(sorted(allowed_set - {0}), dtype=np.int64))

	if not np.any(mask):
		return {
			"bin_edges": np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32),
			"bin_counts": np.zeros((n_bins,), dtype=np.int64),
			"bin_acc": np.zeros((n_bins,), dtype=np.float32),
			"bin_conf": np.zeros((n_bins,), dtype=np.float32),
		}

	conf_v = p_c[mask].astype(np.float32)
	corr_v = (gt[mask] == int(class_id)).astype(np.float32)

	bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
	bin_ids = np.minimum((conf_v * n_bins).astype(np.int64), n_bins - 1)

	bin_counts = np.zeros((n_bins,), dtype=np.int64)
	bin_acc = np.zeros((n_bins,), dtype=np.float32)
	bin_conf = np.zeros((n_bins,), dtype=np.float32)

	for b in range(n_bins):
		m = (bin_ids == b)
		cnt = int(m.sum())
		bin_counts[b] = cnt
		if cnt <= 0:
			continue
		bin_acc[b] = float(corr_v[m].mean())
		bin_conf[b] = float(conf_v[m].mean())

	return {
		"bin_edges": bin_edges,
		"bin_counts": bin_counts,
		"bin_acc": bin_acc,
		"bin_conf": bin_conf,
	}


def _save_reliability_diagram(stats: dict[str, np.ndarray], out_path: str, title: str) -> None:
	"""Save a standard reliability diagram.

	Plot (x=mean confidence per bin, y=mean accuracy per bin) and the ideal y=x line.
	Styling is kept consistent with `_save_reliability_diagram_multi`.
	"""
	bin_acc = stats["bin_acc"].astype(np.float32)
	bin_conf = stats["bin_conf"].astype(np.float32)
	bin_counts = stats["bin_counts"].astype(np.int64)

	mask = bin_counts > 0
	idxs = np.where(mask)[0].astype(np.int64)
	x = bin_conf[idxs]
	y = bin_acc[idxs]

	fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
	ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")
	if x.size > 0:
		# Markers
		h = ax.plot(x, y, marker="o", linestyle="None", markersize=3, label="Top-1", zorder=4)[0]
		color = h.get_color()
		# Connect segments: solid for adjacent bins, dashed across empty-bin gaps.
		# Additionally, for empty bins we draw interpolated (hollow) markers on the dashed segment
		# so the "missing" points are visually represented.
		for i in range(int(x.size) - 1):
			left = int(idxs[i])
			right = int(idxs[i + 1])
			is_gap = right != left + 1
			ls = "--" if is_gap else "-"
			ax.plot(
				[x[i], x[i + 1]],
				[y[i], y[i + 1]],
				linestyle=ls,
				color=color,
				linewidth=1.5,
				zorder=3,
			)
			if is_gap:
				# Draw missing-bin points (interpolated along the dashed segment)
				for b in range(left + 1, right):
					t = float(b - left) / float(right - left)
					xm = float(x[i] + t * (x[i + 1] - x[i]))
					ym = float(y[i] + t * (y[i + 1] - y[i]))
					ax.plot(
						[xm],
						[ym],
						marker="o",
						linestyle="None",
						markersize=3,
						markerfacecolor="none",
						markeredgecolor=color,
						markeredgewidth=1.0,
						zorder=4,
					)

	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.set_xlabel("confidence")
	ax.set_ylabel("accuracy")
	ax.set_title(title)
	ax.legend(loc="lower right", fontsize=7)
	fig.tight_layout()
	fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)


def _save_reliability_diagram_multi(
	stats_by_label: dict[str, dict[str, np.ndarray]],
	out_path: str,
	title: str,
) -> None:
	"""Save a merged reliability diagram with multiple curves.

	Each entry in stats_by_label should be a stats dict containing:
	- bin_counts, bin_acc, bin_conf
	"""
	fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
	ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal")

	# Deterministic ordering: BG first (if present), then others alphabetically.
	labels = list(stats_by_label.keys())
	labels_sorted = sorted(labels, key=lambda s: (0 if str(s).upper() == "BG" else 1, str(s)))

	for label in labels_sorted:
		stats = stats_by_label[label]
		bin_acc = stats["bin_acc"].astype(np.float32)
		bin_conf = stats["bin_conf"].astype(np.float32)
		bin_counts = stats["bin_counts"].astype(np.int64)
		mask = bin_counts > 0
		idxs = np.where(mask)[0].astype(np.int64)
		x = bin_conf[idxs]
		y = bin_acc[idxs]
		if x.size == 0:
			continue
		# Markers
		h = ax.plot(x, y, marker="o", linestyle="None", markersize=3, label=str(label), zorder=4)[0]
		color = h.get_color()
		# Connect segments: solid for adjacent bins, dashed across empty-bin gaps
		for i in range(int(x.size) - 1):
			left = int(idxs[i])
			right = int(idxs[i + 1])
			is_gap = right != left + 1
			ls = "--" if is_gap else "-"
			ax.plot(
				[x[i], x[i + 1]],
				[y[i], y[i + 1]],
				linestyle=ls,
				color=color,
				linewidth=1.5,
				zorder=3,
			)
			if is_gap:
				for b in range(left + 1, right):
					t = float(b - left) / float(right - left)
					xm = float(x[i] + t * (x[i + 1] - x[i]))
					ym = float(y[i] + t * (y[i + 1] - y[i]))
					ax.plot(
						[xm],
						[ym],
						marker="o",
						linestyle="None",
						markersize=3,
						markerfacecolor="none",
						markeredgecolor=color,
						markeredgewidth=1.0,
						zorder=4,
					)

	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.set_xlabel("confidence")
	ax.set_ylabel("accuracy")
	ax.set_title(title)
	ax.legend(loc="lower right", fontsize=7)
	fig.tight_layout()
	fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)


def _load_weights(model: torch.nn.Module, path: str, device: torch.device, strict: bool = True) -> None:
	if not path:
		return
	state = torch.load(path, map_location=device, weights_only=True)
	model.load_state_dict(state, strict=strict)


@torch.no_grad()
def visualize_testset_overlays(
	config: Config,
	start_id: int = 0,
	end_id: int = -1,
	out_dir: Optional[str] = None,
	reliability_dir: Optional[str] = None,
	alpha: float = 0.5,
	ece_bins: int = 15,
	ece_include_bg: bool = False,
	calibration_plot: bool = False,
	calibration_mode: str = "top1",
	ovr_plot_bg: bool = False,
	skip_grids: bool = False,
):
	device = config.DEVICE

	unet_ckpt = "checkpoints/unet/unet_best_25.pth"
	x_ckpt = f"checkpoints/unet/{config.DATASET}/x_best.pth"
	z_ckpt = f"checkpoints/unet/{config.DATASET}/z_best.pth"
	o_ckpt = f"checkpoints/unet/{config.DATASET}/o_best.pth"
	reg_ckpt = "checkpoints/regnet/regnet_prior_2chs.pth"
	align_ckpt = "checkpoints/align_net/align_best.pth"
	scale_ckpt = "checkpoints/scale_net/scale_best.pth"


	# Force deterministic, ordered iteration for visualization
	config.BATCH_SIZE = 1
	config.NUM_WORKERS = 1

	_, _, test_loader = get_loaders(config)
	dataset_len = len(test_loader.dataset)
	if dataset_len <= 0:
		raise RuntimeError("Empty test dataset")

	if end_id < 0:
		end_id = dataset_len - 1
	start_id = max(0, int(start_id))
	end_id = min(dataset_len - 1, int(end_id))
	if start_id > end_id:
		raise ValueError(f"start_id ({start_id}) must be <= end_id ({end_id})")

	# Grids (1x4) output directory
	if (not skip_grids) and out_dir is None:
		out_dir = os.path.join(config.RESULTS_DIR, "predict", config.DATASET)
	if (not skip_grids) and out_dir is not None:
		_ensure_dir(out_dir)

	# Reliability (calibration) output directory
	if reliability_dir is None:
		reliability_dir = os.path.join(config.RESULTS_DIR, "reliability")
	_ensure_dir(reliability_dir)

	# Models (match training pipeline)
	unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(device)
	x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM * config.GMM_NUM * 2).to(device)
	z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
	o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
	reg_net = RR_ResNet(input_channels=2).to(device)
	align_net = Align_ResNet(input_channels=1).to(device)
	scale_net = Scale_ResNet(input_channels=2).to(device)

	# Load weights
	_load_weights(unet, unet_ckpt, device=device, strict=True)
	_load_weights(reg_net, reg_ckpt, device=device, strict=True)
	_load_weights(align_net, align_ckpt, device=device, strict=True)
	_load_weights(scale_net, scale_ckpt, device=device, strict=True)
	_load_weights(o_net, o_ckpt, device=device, strict=True)
	if x_ckpt:
		_load_weights(x_net, x_ckpt, device=device, strict=True)
	if z_ckpt:
		_load_weights(z_net, z_ckpt, device=device, strict=True)
	else:
		# Default to best per-dataset checkpoint if exists
		guess = os.path.join(config.CHECKPOINTS_DIR, "unet", config.DATASET, "z_best.pth")
		if os.path.exists(guess):
			_load_weights(z_net, guess, device=device, strict=True)
	if x_ckpt is None:
		guess = os.path.join(config.CHECKPOINTS_DIR, "unet", config.DATASET, "x_best.pth")
		if os.path.exists(guess):
			_load_weights(x_net, guess, device=device, strict=True)

	# Eval mode (no BN stats updates)
	unet.eval()
	x_net.eval()
	z_net.eval()
	o_net.eval()
	reg_net.eval()
	align_net.eval()
	scale_net.eval()

	# Per-dataset accumulators (supports dataset=ALL)
	top1_agg_by_ds: dict[str, dict[str, np.ndarray]] = {}
	ovr_agg_by_ds: dict[str, dict[int, dict[str, np.ndarray]]] = {}
	
	for idx, batch in enumerate(test_loader):
		if idx < start_id:
			continue
		if idx > end_id:
			break

		image = batch["image"].to(device=device, dtype=torch.float32)
		label = batch["label"].to(device=device, dtype=torch.long)
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
			reg_net=reg_net,
			align_net=align_net,
			scale_net=scale_net,
			ds=ds,
			config=config,
			epoch=0,
			epsilon=1e-6,
		)

		image_proc = out["image"]  # possibly rotated (SCD/YORK)
		r = out["r"]              # [B,K,H,W]
		label_proc = out["label"]

		pred = torch.argmax(r, dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # [H,W]
		allowed_labels = _allowed_labels_for_dataset(ds)
		if allowed_labels != (1, 2, 3):
			# For datasets with fewer foreground classes, treat other labels as background
			mask_keep = np.isin(pred, np.array(allowed_labels, dtype=np.uint8))
			pred = np.where(mask_keep, pred, 0).astype(np.uint8)
		gray01 = _denorm_image(image_proc)  # [H,W] in [0,1]
		rgba_pred = _label_to_rgba_filtered(pred, alpha=alpha, allowed_label_ids=allowed_labels)
		unc = _entropy_from_r(r, eps=1e-12, normalize=True)  # [H,W] float
		unc = np.clip(unc, 0.0, 1.0)

		gt = _to_hw_label(label_proc)
		if allowed_labels != (1, 2, 3):
			mask_keep_gt = np.isin(gt, np.array(allowed_labels, dtype=np.uint8))
			gt = np.where(mask_keep_gt, gt, 0).astype(np.uint8)
		rgba_gt = _label_to_rgba_filtered(gt, alpha=alpha, allowed_label_ids=allowed_labels)

		# ---- Calibration (Top-1 / OvR) ----
		ds_name = str(batch.get("ds", ["DS"])[0])
		valid_ids = (0,) + allowed_labels
		
		if calibration_mode in {"top1", "both"}:
			ece_top1, stats_top1 = _ece_from_r(
				r=r,
				gt_label=gt,
				n_bins=ece_bins,
				allowed_label_ids=valid_ids,
				include_background=ece_include_bg,
				eps=1e-12,
			)
			if ds_name not in top1_agg_by_ds:
				top1_agg_by_ds[ds_name] = {}
			top1_agg_by_ds[ds_name] = _accumulate_ece_stats(top1_agg_by_ds[ds_name], stats_top1)
			# Keep per-sample rows only for top1 (optional; lightweight CSV)
			if "top1_rows" not in locals():
				top1_rows = []
			frame_name = batch.get("frame_name", ["unknown"])[0]
			slice_id = int(batch.get("slice_id", torch.tensor([-1]))[0])
			top1_rows.append(
				{
					"idx": idx,
					"ds": ds_name,
					"frame_name": frame_name,
					"slice_id": slice_id,
					"ece_top1": ece_top1,
				}
			)
		
		if calibration_mode in {"ovr", "both"}:
			if ds_name not in ovr_agg_by_ds:
				ovr_agg_by_ds[ds_name] = {}
			class_ids = list(allowed_labels)
			if ovr_plot_bg:
				class_ids = [0] + class_ids
			for class_id in class_ids:
				stats_c = _ovr_stats_from_r(
					r=r,
					gt_label=gt,
					class_id=int(class_id),
					n_bins=ece_bins,
					allowed_label_ids=valid_ids,
					include_background_pixels=True,
					eps=1e-12,
				)
				if int(class_id) not in ovr_agg_by_ds[ds_name]:
					ovr_agg_by_ds[ds_name][int(class_id)] = {}
				ovr_agg_by_ds[ds_name][int(class_id)] = _accumulate_ece_stats(
					ovr_agg_by_ds[ds_name][int(class_id)],
					stats_c,
				)

		if not skip_grids:
			frame_name = batch.get("frame_name", ["unknown"])[0]
			slice_id = int(batch.get("slice_id", torch.tensor([-1]))[0])
			out_name = f"{idx:05d}_{ds_name}_{frame_name}_slice{slice_id}.png"
			out_path = os.path.join(out_dir, out_name)

			# Save a 1x4 grid: image | GT label | prediction | uncertainty(entropy)
			fig, axes = plt.subplots(
				1,
				4,
				figsize=(12, 3),
				dpi=200,
				gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.1]},
			)
			ax_img, ax_gt, ax_pred, ax_unc = axes

			ax_img.imshow(gray01, cmap="gray", vmin=0.0, vmax=1.0)
			ax_img.set_axis_off()

			ax_gt.imshow(gray01, cmap="gray", vmin=0.0, vmax=1.0)
			ax_gt.imshow(rgba_gt, vmin=0.0, vmax=1.0)
			ax_gt.set_axis_off()

			ax_pred.imshow(gray01, cmap="gray", vmin=0.0, vmax=1.0)
			ax_pred.imshow(rgba_pred, vmin=0.0, vmax=1.0)
			ax_pred.set_axis_off()

			im_unc = ax_unc.imshow(unc, cmap="viridis", vmin=0.0, vmax=1.0)
			ax_unc.set_axis_off()
			fig.colorbar(im_unc, ax=ax_unc, fraction=0.046, pad=0.04)

			plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
			fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
			plt.close(fig)

	# Save calibration plots/summaries under results/reliability/<dataset>/
	if calibration_plot:
		# Top-1 summary (per dataset)
		if calibration_mode in {"top1", "both"} and len(top1_agg_by_ds) > 0:
			# per-sample CSV (mixed datasets) -> save once at root
			if "top1_rows" in locals() and len(top1_rows) > 0:
				top1_csv = os.path.join(reliability_dir, "top1_ece_per_sample.csv")
				with open(top1_csv, "w", newline="", encoding="utf-8") as f:
					writer = csv.DictWriter(
						f,
						fieldnames=["idx", "ds", "frame_name", "slice_id", "ece_top1"],
					)
					writer.writeheader()
					writer.writerows(top1_rows)

			for ds_name, agg in top1_agg_by_ds.items():
				ds_dir = os.path.join(reliability_dir, ds_name)
				_ensure_dir(ds_dir)
				agg_ece, agg_stats = _ece_from_aggregated_bins(agg)
				print(f"Top-1 ECE({ds_name}, {'include_bg' if ece_include_bg else 'no_bg'}): {agg_ece:.6f} | bins={ece_bins}")
				_save_reliability_diagram(
					agg_stats,
					out_path=os.path.join(ds_dir, "top1_reliability.png"),
					title=f"{ds_name} Top-1{' +BG' if ece_include_bg else ''} (ECE={agg_ece:.4f})",
				)
				# tiny summary
				summary_csv = os.path.join(ds_dir, "top1_summary.csv")
				with open(summary_csv, "w", newline="", encoding="utf-8") as f:
					writer = csv.DictWriter(
						f,
						fieldnames=["ds", "mode", "bins", "include_bg", "ece", "n_pixels"],
					)
					writer.writeheader()
					writer.writerow(
						{
							"ds": ds_name,
							"mode": "top1",
							"bins": int(ece_bins),
							"include_bg": bool(ece_include_bg),
							"ece": float(agg_ece),
							"n_pixels": int(agg_stats["bin_counts"].sum()),
						}
					)

		# OvR summary (per dataset, per class)
		if calibration_mode in {"ovr", "both"} and len(ovr_agg_by_ds) > 0:
			for ds_name, class_map in ovr_agg_by_ds.items():
				ds_dir = os.path.join(reliability_dir, ds_name)
				_ensure_dir(ds_dir)
				rows = []
				stats_by_label: dict[str, dict[str, np.ndarray]] = {}
				for class_id, agg in class_map.items():
					ece_c, stats_c = _ece_from_aggregated_bins(agg)
					class_name = _class_name_for_dataset(ds_name, int(class_id))
					rows.append(
						{
							"ds": ds_name,
							"class_id": int(class_id),
							"class_name": class_name,
							"mode": "ovr",
							"bins": int(ece_bins),
							"ece": float(ece_c),
							"n_pixels": int(stats_c["bin_counts"].sum()),
						}
					)
					stats_by_label[class_name] = stats_c

				# Save a single merged figure with all OvR curves
				if len(stats_by_label) > 0:
					macro_ece = float(np.mean([float(r["ece"]) for r in rows])) if len(rows) > 0 else float("nan")
					_save_reliability_diagram_multi(
						stats_by_label,
						out_path=os.path.join(ds_dir, "ovr_reliability.png"),
						title=f"{ds_name} OvR (MacroECE={macro_ece:.4f})",
					)
				ovr_csv = os.path.join(ds_dir, "ovr_summary.csv")
				with open(ovr_csv, "w", newline="", encoding="utf-8") as f:
					writer = csv.DictWriter(
						f,
						fieldnames=["ds", "class_id", "class_name", "mode", "bins", "ece", "n_pixels"],
					)
					writer.writeheader()
					writer.writerows(rows)

	if not skip_grids:
		print(f"Saved 1x4 grids (img/gt/pred/unc): {out_dir}  (range: {start_id}-{end_id})")
	if calibration_plot:
		print(f"Saved reliability diagrams: {reliability_dir}  (range: {start_id}-{end_id})")


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Visualize segmentation overlays on test set")
	p.add_argument("--dataset", type=str, default=Config.DATASET, help="ACDC|MM|SCD|YORK|ALL")
	p.add_argument("--dataset_dir", type=str, default=Config.DATASET_DIR)
	p.add_argument("--img_size", type=int, default=Config.IMG_SIZE)
	p.add_argument("--prior_num_of_patient", type=int, default=Config.PRIOR_NUM_OF_PATIENT)
	p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")

	p.add_argument("--start_id", type=int, default=0)
	p.add_argument("--end_id", type=int, default=-1, help="-1 means last")
	p.add_argument("--alpha", type=float, default=0.5, help="foreground overlay alpha")
	p.add_argument("--out_dir", type=str, default=None)
	p.add_argument(
		"--reliability_dir",
		type=str,
		default=os.path.join(Config.RESULTS_DIR, "reliability"),
		help="Base directory to save calibration/reliability plots (per dataset subfolder will be created)",
	)
	p.add_argument("--ece_bins", type=int, default=15, help="number of bins for ECE")
	p.add_argument(
		"--ece_include_bg",
		action="store_true",
		help="include background pixels when computing Top-1 calibration",
	)
	p.add_argument(
		"--calibration_plot",
		action="store_true",
		help="save calibration/reliability plots under reliability_dir/<dataset>/",
	)
	# Backwards compatible alias
	p.add_argument(
		"--ece_plot",
		action="store_true",
		dest="calibration_plot",
		help="alias of --calibration_plot",
	)
	p.add_argument(
		"--calibration_mode",
		type=str,
		default="top1",
		choices=["top1", "ovr", "both"],
		help="Which calibration logic to plot: top1 (max/argmax), ovr (one-vs-rest per class), or both",
	)
	p.add_argument(
		"--ovr_plot_bg",
		action="store_true",
		help="when using OvR, also plot the background (class 0) curve",
	)
	p.add_argument(
		"--skip_grids",
		action="store_true",
		help="skip saving the 1x4 segmentation grids; useful for only drawing calibration plots",
	)
	return p



# # 只画 Top-1 校准图（默认），不保存1x4分割图
# python predict.py --dataset YORK --calibration_plot --skip_grids

# # 画 OvR（每个类别一张），不含背景类；输出在 results/reliability/YORK/
# python predict.py --dataset YORK --calibration_plot --calibration_mode ovr --skip_grids

# # OvR 也把背景(0类)画出来
# python predict.py --dataset YORK --calibration_plot --calibration_mode ovr --ovr_plot_bg --skip_grids

# # 两种都画
# python predict.py --dataset ACDC --calibration_plot --calibration_mode both --skip_grids

# # 自定义校准图输出根目录（仍会按数据集分子文件夹）
# python predict.py --dataset ALL --calibration_plot --calibration_mode both --skip_grids --reliability_dir results/reliability

if __name__ == "__main__":
	args = _build_argparser().parse_args()

	config = Config()
	config.DATASET = args.dataset
	config.DATASET_DIR = args.dataset_dir
	config.IMG_SIZE = args.img_size
	config.PRIOR_NUM_OF_PATIENT = args.prior_num_of_patient
	if args.device is not None:
		config.DEVICE = torch.device(args.device)
	else:
		config.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	visualize_testset_overlays(
		config=config,
		start_id=args.start_id,
		end_id=args.end_id,
		out_dir=args.out_dir,
		reliability_dir=args.reliability_dir,
		alpha=args.alpha,
		ece_bins=args.ece_bins,
		ece_include_bg=args.ece_include_bg,
		calibration_plot=args.calibration_plot,
		calibration_mode=args.calibration_mode,
		ovr_plot_bg=args.ovr_plot_bg,
		skip_grids=args.skip_grids,
	)
