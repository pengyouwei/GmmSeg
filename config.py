import torch
import argparse
from dataclasses import dataclass
from dataclasses import asdict
from pprint import pprint
import random


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {'true', '1', 'yes', 'y', 't'}:
        return True
    if s in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {v}')

@dataclass
class Config:
    SEED: int = 42
    DATASET: str = "ACDC"  # Options: ACDC, MM, SCD, YORK
    DATASET_DIR: str = "D:/Users/pyw/Desktop/Dataset"
    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    PRIOR_NUM_OF_PATIENT: int = 25
    LEARNING_RATE: float = 5e-4
    NUM_WORKERS: int = 6
    LOSS1_WEIGHT: float = 1.0
    LOSS2_WEIGHT: float = 10.0
    LOSS3_WEIGHT: float = 1.0
    LOSS3_WEIGHT_END: float = 0.001
    BEST_LOSS: float = float('inf')
    BEST_DICE: float = 0.5
    BEST_IOU: float = 0.5
    BEST_PIXEL_ERROR: float = float('inf')
    BEST_EPOCH: int = -1
    DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINTS_DIR: str = './checkpoints'
    OUTPUT_DIR: str = './output' # Output directory for visualizations
    VISUALIZE: bool = True
    LOGS_DIR: str = './logs'
    ADD_TENSORBOARD: bool = False
    IMG_SIZE: int = 128 # 目标图片大小
    MM_CROP_SIZE: int = 128
    SCD_CROP_SIZE: int = 128
    YORK_CROP_SIZE: int = 128
    ACDC_CROP_SIZE: int = 128
    IN_CHANNELS: int = 1
    FEATURE_NUM: int = 4
    GMM_NUM: int = 4
    IMAGE_SCALE_RANGE: tuple = (0.5, 2.0)
    PRIOR_SCALE_RANGE: tuple = (0.5, 2.0) if DATASET == "MM" else (0.8, 1.25)
    SHIFT_RANGE: tuple = (0, 0)
    ROTATE_RANGE: tuple = (-60, 60)
    METRIC_WITH_BACKGROUND: bool = False  # 评估指标是否包含背景

    # 开始配准的epoch
    START_REG: int = 0

    # 高斯混合模型参数范围
    MU_RANGE: float = 5.0
    LOG_VAR_MIN: float = -4.0
    LOG_VAR_MAX: float = 1.0

    # 浓度参数强度
    PRIOR_INTENSITY: float = 10.0   # 5-100

    # 预测相关
    PREDICT_IMG: str | None = None           # 单张预测图片路径
    PREDICT_DIR: str | None = None           # 批量预测目录
    RESULTS_DIR: str = './results'           # 结果保存目录

    MODE: str = 'train'  # train | test 
    MU_VAR_MODE: str = 'image_global'  # 'pixel' | 'image_global'

    # shape-VAE 预训练配置（独立脚本使用，默认不影响原 train.py）
    SHAPE_NUM_CLASSES: int = 4
    SHAPE_BG_INDEX: int = 0
    SHAPE_LATENT_DIM: int = 16
    SHAPE_BASE_CHANNELS: int = 16
    SHAPE_BATCH_SIZE: int = 16
    SHAPE_EPOCHS: int = 50
    SHAPE_LR: float = 1e-3
    SHAPE_WEIGHT_DECAY: float = 1e-5
    SHAPE_RECON_TYPE: str = 'smooth_l1'  # smooth_l1 | l1 | mse
    SHAPE_KL_BETA: float = 0.01
    SHAPE_KL_WARMUP_EPOCHS: int = 10
    SHAPE_MAX_PCA_SAMPLES: int = 0  # 0 表示使用全部样本
    SHAPE_RECOMPUTE_PCA: bool = False
    SHAPE_PCA_PATH: str = './checkpoints/shape_vae/ACDC/pca_prior.npz'
    SHAPE_SAVE_DIR: str = './checkpoints/shape_vae'
    SHAPE_LOG_DIR: str = './logs/shape_vae'
    SHAPE_ADD_TENSORBOARD: bool = False
    SHAPE_CKPT_BEST: str = 'shape_vae_best.pth'
    SHAPE_CKPT_LAST: str = 'shape_vae_last.pth'


def get_config():
    parser = argparse.ArgumentParser(description="Training Config")

    parser.add_argument('--seed', type=int, default=Config.SEED)
    parser.add_argument('--dataset', type=str, default=Config.DATASET)
    parser.add_argument('--dataset_dir', type=str, default=Config.DATASET_DIR)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--prior_num_of_patient', type=int, default=Config.PRIOR_NUM_OF_PATIENT)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--loss1_weight', type=float, default=Config.LOSS1_WEIGHT)
    parser.add_argument('--loss2_weight', type=float, default=Config.LOSS2_WEIGHT)
    parser.add_argument('--loss3_weight', type=float, default=Config.LOSS3_WEIGHT)
    parser.add_argument('--checkpoints_dir', type=str, default=Config.CHECKPOINTS_DIR)
    parser.add_argument('--visualize', type=bool, default=Config.VISUALIZE)
    parser.add_argument('--add_tensorboard', type=bool, default=Config.ADD_TENSORBOARD)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--logs_dir', type=str, default=Config.LOGS_DIR)
    parser.add_argument('--img_size', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--acdc_crop_size', type=int, default=Config.ACDC_CROP_SIZE)
    parser.add_argument('--mm_crop_size', type=int, default=Config.MM_CROP_SIZE)
    parser.add_argument('--scd_crop_size', type=int, default=Config.SCD_CROP_SIZE)
    parser.add_argument('--york_crop_size', type=int, default=Config.YORK_CROP_SIZE)
    parser.add_argument('--feature_num', type=int, default=Config.FEATURE_NUM)
    parser.add_argument('--in_channels', type=int, default=Config.IN_CHANNELS)
    parser.add_argument('--gmm_num', type=int, default=Config.GMM_NUM)
    parser.add_argument('--image_scale_range', type=float, nargs=2, default=Config.IMAGE_SCALE_RANGE)
    parser.add_argument('--prior_scale_range', type=float, nargs=2, default=Config.PRIOR_SCALE_RANGE)
    parser.add_argument('--shift_range', type=float, nargs=2, default=Config.SHIFT_RANGE)
    parser.add_argument('--rotate_range', type=float, nargs=2, default=Config.ROTATE_RANGE)
    parser.add_argument('--metric_with_background', type=bool, default=Config.METRIC_WITH_BACKGROUND)
    parser.add_argument('--start_reg', type=int, default=Config.START_REG)
    parser.add_argument('--mu_range', type=float, default=Config.MU_RANGE)
    parser.add_argument('--log_var_min', type=float, default=Config.LOG_VAR_MIN)
    parser.add_argument('--log_var_max', type=float, default=Config.LOG_VAR_MAX)
    parser.add_argument('--prior_intensity', type=float, default=Config.PRIOR_INTENSITY)
    parser.add_argument('--predict_img', type=str, default=Config.PREDICT_IMG)
    parser.add_argument('--predict_dir', type=str, default=Config.PREDICT_DIR)
    parser.add_argument('--results_dir', type=str, default=Config.RESULTS_DIR)
    parser.add_argument('--mode', type=str, default=Config.MODE)
    parser.add_argument('--mu_var_mode', type=str, default=Config.MU_VAR_MODE,
                        choices=['pixel', 'image_global', 'dataset_global'])

    parser.add_argument('--shape_num_classes', type=int, default=Config.SHAPE_NUM_CLASSES)
    parser.add_argument('--shape_bg_index', type=int, default=Config.SHAPE_BG_INDEX)
    parser.add_argument('--shape_latent_dim', type=int, default=Config.SHAPE_LATENT_DIM)
    parser.add_argument('--shape_base_channels', type=int, default=Config.SHAPE_BASE_CHANNELS)
    parser.add_argument('--shape_batch_size', type=int, default=Config.SHAPE_BATCH_SIZE)
    parser.add_argument('--shape_epochs', type=int, default=Config.SHAPE_EPOCHS)
    parser.add_argument('--shape_lr', type=float, default=Config.SHAPE_LR)
    parser.add_argument('--shape_weight_decay', type=float, default=Config.SHAPE_WEIGHT_DECAY)
    parser.add_argument('--shape_recon_type', type=str, default=Config.SHAPE_RECON_TYPE,
                        choices=['smooth_l1', 'l1', 'mse'])
    parser.add_argument('--shape_kl_beta', type=float, default=Config.SHAPE_KL_BETA)
    parser.add_argument('--shape_kl_warmup_epochs', type=int, default=Config.SHAPE_KL_WARMUP_EPOCHS)
    parser.add_argument('--shape_max_pca_samples', type=int, default=Config.SHAPE_MAX_PCA_SAMPLES)
    parser.add_argument('--shape_recompute_pca', type=_str2bool, default=Config.SHAPE_RECOMPUTE_PCA)
    parser.add_argument('--shape_pca_path', type=str, default=Config.SHAPE_PCA_PATH)
    parser.add_argument('--shape_save_dir', type=str, default=Config.SHAPE_SAVE_DIR)
    parser.add_argument('--shape_log_dir', type=str, default=Config.SHAPE_LOG_DIR)
    parser.add_argument('--shape_add_tensorboard', type=_str2bool, default=Config.SHAPE_ADD_TENSORBOARD)
    parser.add_argument('--shape_ckpt_best', type=str, default=Config.SHAPE_CKPT_BEST)
    parser.add_argument('--shape_ckpt_last', type=str, default=Config.SHAPE_CKPT_LAST)


    args = parser.parse_args()

    return Config(
        SEED=args.seed,
        DATASET=args.dataset,
        DATASET_DIR=args.dataset_dir,
        BATCH_SIZE=args.batch_size,
        PRIOR_NUM_OF_PATIENT=args.prior_num_of_patient,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        NUM_WORKERS=args.num_workers,
        LOSS1_WEIGHT=args.loss1_weight,
        LOSS2_WEIGHT=args.loss2_weight,
        LOSS3_WEIGHT=args.loss3_weight,
        CHECKPOINTS_DIR=args.checkpoints_dir,
        OUTPUT_DIR=args.output_dir,
        LOGS_DIR=args.logs_dir,
        VISUALIZE=args.visualize,
        ADD_TENSORBOARD=args.add_tensorboard,
        IMG_SIZE=args.img_size,
        ACDC_CROP_SIZE=args.acdc_crop_size,
        MM_CROP_SIZE=args.mm_crop_size,
        SCD_CROP_SIZE=args.scd_crop_size,
        YORK_CROP_SIZE=args.york_crop_size,
        FEATURE_NUM=args.feature_num,
        IN_CHANNELS=args.in_channels,
        GMM_NUM=args.gmm_num,
        IMAGE_SCALE_RANGE=tuple(args.image_scale_range),
        PRIOR_SCALE_RANGE=tuple(args.prior_scale_range),
        SHIFT_RANGE=tuple(args.shift_range),
        ROTATE_RANGE=tuple(args.rotate_range),
        START_REG=args.start_reg,
        METRIC_WITH_BACKGROUND=args.metric_with_background,
        MU_RANGE=args.mu_range,
        LOG_VAR_MIN=args.log_var_min,
        LOG_VAR_MAX=args.log_var_max,
        PRIOR_INTENSITY=args.prior_intensity,
        PREDICT_IMG=args.predict_img,
        PREDICT_DIR=args.predict_dir,
        RESULTS_DIR=args.results_dir,
        MODE=args.mode,
        MU_VAR_MODE=args.mu_var_mode,

        SHAPE_NUM_CLASSES=args.shape_num_classes,
        SHAPE_BG_INDEX=args.shape_bg_index,
        SHAPE_LATENT_DIM=args.shape_latent_dim,
        SHAPE_BASE_CHANNELS=args.shape_base_channels,
        SHAPE_BATCH_SIZE=args.shape_batch_size,
        SHAPE_EPOCHS=args.shape_epochs,
        SHAPE_LR=args.shape_lr,
        SHAPE_WEIGHT_DECAY=args.shape_weight_decay,
        SHAPE_RECON_TYPE=args.shape_recon_type,
        SHAPE_KL_BETA=args.shape_kl_beta,
        SHAPE_KL_WARMUP_EPOCHS=args.shape_kl_warmup_epochs,
        SHAPE_MAX_PCA_SAMPLES=args.shape_max_pca_samples,
        SHAPE_RECOMPUTE_PCA=args.shape_recompute_pca,
        SHAPE_PCA_PATH=args.shape_pca_path,
        SHAPE_SAVE_DIR=args.shape_save_dir,
        SHAPE_LOG_DIR=args.shape_log_dir,
        SHAPE_ADD_TENSORBOARD=args.shape_add_tensorboard,
        SHAPE_CKPT_BEST=args.shape_ckpt_best,
        SHAPE_CKPT_LAST=args.shape_ckpt_last,
        
        # 保留默认值（不可通过命令行修改）
        BEST_LOSS=Config.BEST_LOSS,
        BEST_DICE=Config.BEST_DICE,
        BEST_IOU=Config.BEST_IOU,
        BEST_PIXEL_ERROR=Config.BEST_PIXEL_ERROR,
        BEST_EPOCH=Config.BEST_EPOCH,
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )


def format_config(config) -> str:
    items = asdict(config)
    lines = [f"{k:<20}: {v}" for k, v in items.items()]
    return "\n".join(lines)

if __name__ == "__main__":
    config = get_config()
    print(format_config(config))