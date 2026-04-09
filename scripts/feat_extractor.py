import os
import argparse
import csv
from typing import Dict, Tuple, List

import numpy as np
import torch

from config import Config
from data.dataloader import get_loaders
from models.unet import UNet
from models.regnet import RR_ResNet
from models.align_net import Align_ResNet
from models.scale_net import Scale_ResNet
from utils.train_utils import forward_pass


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


def _allowed_labels_for_dataset(ds: str) -> Tuple[int, ...]:
	ds_upper = str(ds).upper()
	if ds_upper == "SCD":
		return (1,)
	if ds_upper == "YORK":
		return (1, 2)
	return (1, 2, 3)


def _extract_per_class_features(
	feature_map: np.ndarray,
	label_hw: np.ndarray,
	class_ids: Tuple[int, ...],
) -> Dict[int, np.ndarray]:
	"""Group pixel features by class.

	feature_map: [C,H,W]
	label_hw: [H,W]
	returns: {class_id: [N,C] float32}
	"""
	if feature_map.ndim != 3:
		raise ValueError(f"feature_map must be [C,H,W], got {feature_map.shape}")
	if label_hw.ndim != 2:
		raise ValueError(f"label_hw must be [H,W], got {label_hw.shape}")

	c, h, w = feature_map.shape
	if label_hw.shape != (h, w):
		raise ValueError(f"label shape {label_hw.shape} != feature spatial {(h, w)}")

	feats_by_cls: Dict[int, np.ndarray] = {}
	for cls in class_ids:
		m = (label_hw == int(cls))
		if not np.any(m):
			feats_by_cls[int(cls)] = np.zeros((0, c), dtype=np.float32)
			continue
		# [C, N] -> [N, C]
		vecs = feature_map[:, m].T.astype(np.float32, copy=False)
		feats_by_cls[int(cls)] = vecs
	return feats_by_cls


def _write_pixel_feature_csv(
	csv_path: str,
	feature_map: np.ndarray,
	label_hw: np.ndarray,
	class_order: Tuple[int, ...] = (0, 1, 2, 3),
	feature_dims_out: int = 4,
) -> None:
	"""Write per-pixel features to CSV.

	Header: class,feature0..feature{feature_dims_out-1}
	Rows are sorted by class (class_order) then by pixel_id.

	- pixel_id uses row-major flatten index: y*W + x
	- if C != feature_dims_out: truncate or pad zeros.
	"""
	if feature_map.ndim != 3:
		raise ValueError(f"feature_map must be [C,H,W], got {feature_map.shape}")
	if label_hw.ndim != 2:
		raise ValueError(f"label_hw must be [H,W], got {label_hw.shape}")

	c, h, w = feature_map.shape
	if label_hw.shape != (h, w):
		raise ValueError(f"label shape {label_hw.shape} != feature spatial {(h, w)}")

	# Flatten features to [H*W, C]
	flat_feats = feature_map.reshape(c, h * w).T.astype(np.float32, copy=False)
	flat_label = label_hw.reshape(h * w).astype(np.int64, copy=False)

	feature_dims_out = int(feature_dims_out)
	if feature_dims_out <= 0:
		raise ValueError("feature_dims_out must be > 0")

	_ensure_dir(os.path.dirname(csv_path))
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		wtr = csv.writer(f)
		header = ["class"] + [f"feature{i}" for i in range(feature_dims_out)]
		wtr.writerow(header)

		class_name_map = {
			0: "bg",
			1: "lv",
			2: "myo",
			3: "rv",
		}

		for cls in class_order:
			m = (flat_label == int(cls))
			if not np.any(m):
				continue
			pixel_ids = np.flatnonzero(m)  # already sorted asc
			vecs = flat_feats[pixel_ids]   # [N,C]

			# enforce 4 feature columns as requested
			if vecs.shape[1] > feature_dims_out:
				vecs_out = vecs[:, :feature_dims_out]
			elif vecs.shape[1] < feature_dims_out:
				pad = np.zeros((vecs.shape[0], feature_dims_out - vecs.shape[1]), dtype=np.float32)
				vecs_out = np.concatenate([vecs, pad], axis=1)
			else:
				vecs_out = vecs

			for i in range(int(pixel_ids.shape[0])):
				cls_name = class_name_map.get(int(cls), f"class{int(cls)}")
				row = [cls_name] + [float(x) for x in vecs_out[i]]
				wtr.writerow(row)


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Extract per-class feature vectors for selected samples")
	p.add_argument("--dataset", type=str, default=Config.DATASET, help="ACDC|MM|SCD|YORK|ALL")
	p.add_argument("--start-id", type=int, default=0, dest="start_id", help="start sample index")
	p.add_argument("--end-id", type=int, default=-1, dest="end_id", help="end sample index (-1 means last)")
	p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
	p.add_argument("--out-dir", type=str, default="feats", help="output folder (default: ./feats)")
	return p


def main(args: argparse.Namespace) -> None:
	if args.device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(str(args.device))

	config = Config()
	config.DATASET = str(args.dataset)
	config.DEVICE = device

	# deterministic, ordered iteration
	config.BATCH_SIZE = 1
	config.NUM_WORKERS = 1

	_, _, test_loader = get_loaders(config)
	dataset_len = len(test_loader.dataset)
	if dataset_len <= 0:
		raise RuntimeError("Empty test dataset")

	start_id = max(0, int(args.start_id))
	end_id = int(args.end_id)
	if end_id < 0:
		end_id = dataset_len - 1
	end_id = min(dataset_len - 1, end_id)
	if start_id > end_id:
		raise ValueError(f"start_id ({start_id}) must be <= end_id ({end_id})")

	out_root = os.path.abspath(str(args.out_dir))
	_ensure_dir(out_root)

	# ---- Build global models ----
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

	# Cache per-dataset GMM nets (x/z/o)
	gmm_cache: Dict[str, Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]] = {}
	rows: List[Dict[str, object]] = []

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
			x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM * config.GMM_NUM * 2).to(device)
			z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)
			o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(device)

			x_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", ds_name, "x_best.pth")
			z_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", ds_name, "z_best.pth")
			o_ckpt = os.path.join(config.CHECKPOINTS_DIR, "unet", ds_name, "o_best.pth")

			_load_weights(o_net, o_ckpt, device=device, strict=True)
			_load_weights(x_net, x_ckpt, device=device, strict=True)
			_load_weights(z_net, z_ckpt, device=device, strict=True)

			x_net.eval(); z_net.eval(); o_net.eval()
			gmm_cache[ds_name] = (x_net, z_net, o_net)

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

		feat = out.get("feature_4chs", None)
		if feat is None:
			raise RuntimeError("forward_pass did not return 'feature_4chs'")
		feat_map = feat[0].detach().cpu().numpy()  # [C,H,W]

		gt = _to_hw_label(out.get("label", label))

		# Save the grayscale image (aligned/processed) in the same format
		img_tensor = out.get("image", image)
		img_map = img_tensor[0].detach().cpu().numpy()  # [1,H,W] or [H,W]
		if img_map.ndim == 2:
			img_map = img_map[None, ...]

		# Keep only valid labels for this dataset
		allowed_fg = _allowed_labels_for_dataset(ds_name)
		valid_ids = (0,) + allowed_fg
		if allowed_fg != (1, 2, 3):
			keep = np.isin(gt, np.array(valid_ids, dtype=np.uint8))
			gt = np.where(keep, gt, 0).astype(np.uint8)

		feats_by_cls = _extract_per_class_features(feat_map, gt, class_ids=valid_ids)

		frame_name = batch.get("frame_name", ["unknown"])[0]
		slice_id = int(batch.get("slice_id", torch.tensor([-1]))[0])

		ds_out_dir = os.path.join(out_root, ds_name)
		_ensure_dir(ds_out_dir)

		out_name = f"{idx:05d}_{ds_name}_{frame_name}_slice{slice_id}.npz"
		out_path = os.path.join(ds_out_dir, out_name)
		csv_name = f"{idx:05d}_{ds_name}_{frame_name}_slice{slice_id}.csv"
		csv_path = os.path.join(ds_out_dir, csv_name)
		img_csv_name = f"{idx:05d}_{ds_name}_{frame_name}_slice{slice_id}_img.csv"
		img_csv_path = os.path.join(ds_out_dir, img_csv_name)

		pack: Dict[str, np.ndarray] = {
			"idx": np.array([idx], dtype=np.int32),
			"slice_id": np.array([slice_id], dtype=np.int32),
			"ds": np.array([ds_name]),
			"frame_name": np.array([str(frame_name)]),
			"allowed_label_ids": np.array(list(valid_ids), dtype=np.int32),
			"gt": gt.astype(np.uint8),
			"feature_map": feat_map.astype(np.float32),
		}
		for cls_id, vecs in feats_by_cls.items():
			pack[f"feats_class{int(cls_id)}"] = vecs.astype(np.float32, copy=False)
			if vecs.shape[0] > 0:
				pack[f"mean_class{int(cls_id)}"] = vecs.mean(axis=0).astype(np.float32)
			else:
				pack[f"mean_class{int(cls_id)}"] = np.zeros((feat_map.shape[0],), dtype=np.float32)

		np.savez_compressed(out_path, **pack)

		# Per-pixel CSV (sorted by class: 0-bg,1-lv,2-myo,3-rv)
		_write_pixel_feature_csv(
			csv_path=csv_path,
			feature_map=feat_map.astype(np.float32, copy=False),
			label_hw=gt,
			class_order=(0, 1, 2, 3),
			feature_dims_out=config.FEATURE_NUM,
		)

		# Per-pixel grayscale CSV (same ordering, 1 value per pixel)
		_write_pixel_feature_csv(
			csv_path=img_csv_path,
			feature_map=img_map.astype(np.float32, copy=False),
			label_hw=gt,
			class_order=(0, 1, 2, 3),
			feature_dims_out=1,
		)

		row: Dict[str, object] = {
			"idx": idx,
			"ds": ds_name,
			"frame_name": str(frame_name),
			"slice_id": slice_id,
			"C": int(feat_map.shape[0]),
			"H": int(feat_map.shape[1]),
			"W": int(feat_map.shape[2]),
			"file": os.path.join(ds_name, out_name).replace("\\", "/"),
		}
		for cls_id in valid_ids:
			row[f"n_class{int(cls_id)}"] = int(feats_by_cls[int(cls_id)].shape[0])
		rows.append(row)

		print(f"[{idx}] saved: {out_path} | {csv_path}")

	if len(rows) > 0:
		index_csv = os.path.join(out_root, "index.csv")
		all_fields = set()
		for r in rows:
			all_fields |= set(r.keys())
		class_fields = sorted([f for f in all_fields if f.startswith("n_class")])
		fieldnames = ["idx", "ds", "frame_name", "slice_id", "C", "H", "W", "file", *class_fields]
		with open(index_csv, "w", newline="", encoding="utf-8") as f:
			w = csv.DictWriter(f, fieldnames=fieldnames)
			w.writeheader()
			for r in rows:
				w.writerow(r)
		print(f"Wrote index: {index_csv}")
	else:
		print("No samples processed; nothing written.")


if __name__ == "__main__":
	main(_build_argparser().parse_args())