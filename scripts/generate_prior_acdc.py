import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from config import Config


def get_lv_centroid(label_np):
    # 找到左心室的像素位置
    lv_mask = (label_np == 1)
    
    if not np.any(lv_mask):
        # 如果没有找到左心室，返回图像中心
        h, w = label_np.shape
        return h // 2, w // 2
    
    # 计算质心
    y_coords, x_coords = np.where(lv_mask)
    center_y = round(np.mean(y_coords))
    center_x = round(np.mean(x_coords))
    
    return center_y, center_x

def find_lv_slice_range(label_volume: np.ndarray, min_lv_pixels: int = 20):
    """
    返回包含 LV(==1) 的切片索引范围 [start, end]（闭区间）。
    若没有满足最小像素阈值的切片，返回 None。
    """
    lv_pixels_per_slice = (label_volume == 1).reshape(label_volume.shape[0], -1).sum(axis=1)
    lv_indices = np.where(lv_pixels_per_slice >= min_lv_pixels)[0]
    if lv_indices.size == 0:
        return None
    return int(lv_indices[0]), int(lv_indices[-1])

def choose_slice_index(i: int, prior_slices: int, num_slices: int, lv_range=None) -> int:
    """
    更稳健的切片索引选择：
    - 若提供 lv_range=(start,end)，则在该范围内按 i 均匀采样。
    - 否则对整个体积用 linspace 均匀采样到 [0, num_slices-1]。
    """
    if num_slices <= 0:
        return 0
    if lv_range is not None:
        start, end = lv_range
        start = max(0, min(start, num_slices - 1))
        end = max(start, min(end, num_slices - 1))
        length = end - start + 1
        if length <= 0:
            return max(0, min(int(i * num_slices / prior_slices), num_slices - 1))
        # 在 [start, end] 范围内等距取样
        idx = start + int(np.floor(i * length / prior_slices))
        return max(start, min(idx, end))
    # 覆盖全范围
    lin_idx = np.linspace(0, max(0, num_slices - 1), prior_slices)
    return int(lin_idx[i].round())

def center_crop(image_np, label_np, target_size):
    """
    基于左心室中心进行裁剪；若裁剪区域越界则先填充，再围绕中心裁剪，
    确保左心室中心出现在输出的几何中心。
    Args:
        image_np: np.ndarray, 原始图像 (H, W)
        label_np: np.ndarray, 原始标签 (H, W)
        target_size: int, 裁剪后的正方形尺寸
    Returns:
        cropped_image_np, cropped_label_np
    """
    # 左心室质心（若不存在返回图像中心）
    center_y, center_x = get_lv_centroid(label_np)

    H, W = label_np.shape
    half = target_size // 2

    # 以质心为几何中心的窗口
    y0 = center_y - half
    x0 = center_x - half
    y1 = y0 + target_size
    x1 = x0 + target_size

    # 计算需要的四向填充量（不足即为正的 pad）
    pad_top    = max(0, -y0)
    pad_left   = max(0, -x0)
    pad_bottom = max(0,  y1 - H)
    pad_right  = max(0,  x1 - W)

    if pad_top or pad_left or pad_bottom or pad_right:
        # 图像优先使用反射填充；反射不可行时降级为边缘复制；再兜底为常数0
        eff_mode = 'reflect'
        if (H < 2 or W < 2 or
            pad_top >= H or pad_bottom >= H or
            pad_left >= W or pad_right >= W):
            eff_mode = 'edge'

        if eff_mode == 'reflect':
            image_np = np.pad(image_np,
                              ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode='reflect')
        elif eff_mode == 'edge':
            image_np = np.pad(image_np,
                              ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode='edge')
        else:
            image_np = np.pad(image_np,
                              ((pad_top, pad_bottom), (pad_left, pad_right)),
                              mode='constant', constant_values=0)

        # 标签恒为背景常数填充
        label_np = np.pad(label_np,
                          ((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=0)

        # 更新中心坐标到新坐标系
        center_y += pad_top
        center_x += pad_left
        H += pad_top + pad_bottom
        W += pad_left + pad_right

    # 现在可以安全裁剪，且确保中心在中点
    start_y = center_y - half
    start_x = center_x - half
    end_y = start_y + target_size
    end_x = start_x + target_size

    cropped_image_np = image_np[start_y:end_y, start_x:end_x]
    cropped_label_np = label_np[start_y:end_y, start_x:end_x]

    return cropped_image_np, cropped_label_np




if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    config = Config()

    dataset_dir = "D:/Users/pyw/Desktop/Dataset/ACDC"
    group = ["gp1", "gp2", "gp3", "gp4", "gp5"]
    phase = ["ED", "ES"]

    # Total number of volumes to sample per phase across all groups (proportional by group size)
    # 若为 None 或 <=0 或 >= 可用总量，则使用全部
    total_volume = 25

    # Image size
    imgsz = config.ACDC_CROP_SIZE # 128
    print("Target image size:", imgsz)

    # Number of prior slices, such as [10, 4, imgsz, imgsz]
    prior_slices = 10

    # 是否对每个 prior 槽位内的标签做轻度尺度归一（按 LV 面积对齐到中位数）
    SCALE_TO_MEDIAN_LV = True
    SCALE_CLAMP = (0.8, 1.25)  # 限制缩放因子范围，避免过度缩放

    # 存储所有选中的label切片，用于计算先验
    ED_labels = [[] for _ in range(prior_slices)]
    ES_labels = [[] for _ in range(prior_slices)]
    
    # 记录用到的体积路径
    used_volume_paths = {"ED": [], "ES": []}
    def allocate_counts(available_counts, total_target):
        total_available = sum(available_counts)
        if total_target is None or total_target <= 0 or total_target >= total_available:
            return available_counts[:]
        props = [ac / total_available if total_available > 0 else 0 for ac in available_counts]
        raw = [p * total_target for p in props]
        base = [min(available_counts[i], int(np.floor(raw[i]))) for i in range(len(available_counts))]
        assigned = sum(base)
        remaining = min(total_target, total_available) - assigned
        remainders = [raw[i] - np.floor(raw[i]) for i in range(len(available_counts))]
        order = sorted(range(len(available_counts)), key=lambda i: (remainders[i], available_counts[i]), reverse=True)
        idx_ptr = 0
        while remaining > 0 and any(base[i] < available_counts[i] for i in range(len(base))):
            i = order[idx_ptr % len(order)]
            idx_ptr += 1
            if base[i] < available_counts[i]:
                base[i] += 1
                remaining -= 1
        return base

    ed_selected_vols = 0
    es_selected_vols = 0
    for ph in phase:
        print(f"Processing phase: {ph}")
        gp_img_paths = {}
        gp_lbl_paths = {}
        gp_counts = []
        for gp in group:
            image_dir = os.path.join(dataset_dir, gp, ph, "train_image")
            label_dir = os.path.join(dataset_dir, gp, ph, "train_label")
            img_paths = [entry.path for entry in os.scandir(image_dir) if entry.name.endswith(".npy")]
            lbl_paths = [entry.path for entry in os.scandir(label_dir) if entry.name.endswith(".npy")]
            img_paths.sort(); lbl_paths.sort()
            if len(img_paths) != len(lbl_paths):
                raise ValueError(f"{gp}-{ph} 图像与标签数量不一致: {len(img_paths)} vs {len(lbl_paths)}")
            gp_img_paths[gp] = img_paths
            gp_lbl_paths[gp] = lbl_paths
            gp_counts.append(len(img_paths))

        alloc = allocate_counts(gp_counts, total_volume)
        print(f"Available per group ({ph}): {dict(zip(group, gp_counts))}")
        print(f"Allocated per group ({ph}): {dict(zip(group, alloc))}")

        phase_selected = 0
        for gp, take_n in zip(group, alloc):
            img_list = gp_img_paths[gp]
            lbl_list = gp_lbl_paths[gp]
            if take_n == 0:
                continue
            indexes = random.sample(range(len(img_list)), take_n)
            selected_image_volumes = [img_list[i] for i in indexes]
            selected_label_volumes = [lbl_list[i] for i in indexes]
            print(f"Selected volumes indices of {gp} for phase {ph}: {sorted(indexes)}")
            phase_selected += take_n
            
            # 记录选中的体积路径
            for img_path, lbl_path in zip(selected_image_volumes, selected_label_volumes):
                used_volume_paths[ph].append({
                    "group": gp,
                    "image_path": img_path,
                    "label_path": lbl_path
                })
            
            for img_vol_path, lbl_vol_path in zip(selected_image_volumes, selected_label_volumes):
                image_volume = np.load(img_vol_path)
                label_volume = np.load(lbl_vol_path)
                num_slices = image_volume.shape[0]
                # 仅在 LV 出现的切片范围内均匀采样，减少不同解剖层级的混杂
                lv_range = find_lv_slice_range(label_volume, min_lv_pixels=20)

                for i in range(prior_slices):
                    # 映射到实际的 slice 索引（优先在 LV 范围内等距取样）
                    slice_index = choose_slice_index(i, prior_slices, num_slices, lv_range)
                    image_slice = image_volume[slice_index]  # (H, W) [0, 1]
                    label_slice = label_volume[slice_index]  # (H, W) [0, 1, 2, 3]
                    # 若该切片没有 LV，跳过（保持以 LV 居中为原则）
                    if not np.any(label_slice == 1):
                        continue
                    cropped_image, cropped_label = center_crop(image_slice, label_slice, imgsz) # 裁剪到 (imgsz, imgsz)
                    
                    if ph == "ED":
                        ED_labels[i].append(cropped_label)
                    else:
                        ES_labels[i].append(cropped_label)
        if ph == "ED":
            ed_selected_vols += phase_selected
        else:
            es_selected_vols += phase_selected


    ED_prior = np.zeros((prior_slices, 4, imgsz, imgsz), dtype=np.float32)  # (slice, label, H, W)
    ES_prior = np.zeros((prior_slices, 4, imgsz, imgsz), dtype=np.float32)  # (slice, label, H, W)
    ED_counts = np.zeros((prior_slices,), dtype=np.int32)
    ES_counts = np.zeros((prior_slices,), dtype=np.int32)
    
    def _resize_label_to_factor(label_np: np.ndarray, factor: float, target_size: int) -> np.ndarray:
        factor = float(np.clip(factor, SCALE_CLAMP[0], SCALE_CLAMP[1]))
        new_h = max(1, int(round(target_size * factor)))
        new_w = max(1, int(round(target_size * factor)))
        pil_img = Image.fromarray(label_np.astype(np.uint8), mode='L')
        pil_resized = pil_img.resize((new_w, new_h), resample=Image.NEAREST)
        arr = np.array(pil_resized, dtype=np.uint8)
        if arr.shape[0] >= target_size and arr.shape[1] >= target_size:
            sy = (arr.shape[0] - target_size) // 2
            sx = (arr.shape[1] - target_size) // 2
            return arr[sy:sy+target_size, sx:sx+target_size]
        pad_h = max(0, target_size - arr.shape[0])
        pad_w = max(0, target_size - arr.shape[1])
        arr = np.pad(arr, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant', constant_values=0)
        return arr.astype(np.uint8)

    def _normalize_slot_labels(slot_labels: list) -> np.ndarray:
        if not SCALE_TO_MEDIAN_LV:
            return np.stack(slot_labels, axis=0)
        areas = [int((lbl == 1).sum()) for lbl in slot_labels]
        nonzero = [a for a in areas if a > 0]
        if len(nonzero) == 0:
            return np.stack(slot_labels, axis=0)
        target_area = int(np.median(nonzero))
        out = []
        for lbl, area in zip(slot_labels, areas):
            if area <= 0:
                out.append(lbl)
                continue
            s = np.sqrt(max(1e-6, target_area / float(area)))
            out.append(_resize_label_to_factor(lbl, s, imgsz))
        return np.stack(out, axis=0)

    for i in range(prior_slices):
        if len(ED_labels[i]) > 0:
            ED_stack = _normalize_slot_labels(ED_labels[i])  # (N, H, W)
            ED_counts[i] = ED_stack.shape[0]
            for k in range(4):
                ED_prior[i, k] = np.mean(ED_stack == k, axis=0)  # (H, W)
        else:
            # 无样本则保持为零（均匀先验可按需改为 1/4）
            ED_counts[i] = 0
        if len(ES_labels[i]) > 0:
            ES_stack = _normalize_slot_labels(ES_labels[i])  # (N, H, W)
            ES_counts[i] = ES_stack.shape[0]
            for k in range(4):
                ES_prior[i, k] = np.mean(ES_stack == k, axis=0)  # (H, W)
        else:
            ES_counts[i] = 0

    np.save(f"{config.DATASET_DIR}/ACDC/ED_prior_{int(ed_selected_vols)}samples_4chs.npy", ED_prior)
    np.save(f"{config.DATASET_DIR}/ACDC/ES_prior_{int(es_selected_vols)}samples_4chs.npy", ES_prior)
    
    # 保存用到的体积路径到txt文件，ED和ES合并，分别保存image和label路径
    total_samples = int(ed_selected_vols + es_selected_vols)
    
    # 保存image路径
    with open(f"{config.DATASET_DIR}/ACDC/used_image_paths_{total_volume}samples.txt", 'w', encoding='utf-8') as f:
        for vol_info in used_volume_paths['ED']:
            f.write(f"{vol_info['image_path']}\n")
        for vol_info in used_volume_paths['ES']:
            f.write(f"{vol_info['image_path']}\n")
    
    # 保存label路径
    with open(f"{config.DATASET_DIR}/ACDC/used_label_paths_{total_volume}samples.txt", 'w', encoding='utf-8') as f:
        for vol_info in used_volume_paths['ED']:
            f.write(f"{vol_info['label_path']}\n")
        for vol_info in used_volume_paths['ES']:
            f.write(f"{vol_info['label_path']}\n")
    
    print("ED_prior shape:", ED_prior.shape)  # (slice, label, H, W)
    print("ES_prior shape:", ES_prior.shape)  # (slice, label, H, W)
    print("ED_prior max, min:", ED_prior.max(), ED_prior.min())
    print("ES_prior max, min:", ES_prior.max(), ES_prior.min())
    print("ED per-slice counts:", ED_counts.tolist())
    print("ES per-slice counts:", ES_counts.tolist())
    print("Total volumes used (ED,ES):", int(ed_selected_vols), int(es_selected_vols))
    print("Total slices used (ED,ES):", int(ed_selected_vols * prior_slices), int(es_selected_vols * prior_slices))
    print(f"Image paths saved to: {config.DATASET_DIR}/ACDC/used_image_paths_{total_volume}samples.txt")
    print(f"Label paths saved to: {config.DATASET_DIR}/ACDC/used_label_paths_{total_volume}samples.txt")