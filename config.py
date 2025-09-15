import torch
import argparse
from dataclasses import dataclass
from dataclasses import asdict
from pprint import pprint

@dataclass
class Config:
    DATASET: str = "ACDC"  # Options: ACDC, MM, SCD, YORK
    DATASET_DIR: str = "D:/Users/pyw/Desktop/Dataset"
    BATCH_SIZE: int = 16
    EPOCHS: int = 20
    PRIOR_NUM_OF_PATIENT: int = 25
    LEARNING_RATE: float = 5e-4
    NUM_WORKERS: int = 8
    BEST_LOSS: float = float('inf')
    BEST_DICE: float = 0.8
    BEST_IOU: float = 0.8
    BEST_PIXEL_ERROR: float = float('inf')
    BEST_EPOCH: int = -1
    DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINTS_DIR: str = './checkpoints'
    OUTPUT_DIR: str = './output' # Output directory for visualizations
    LOGS_DIR: str = './logs'
    IMG_SIZE: int = 128 # 目标图片大小
    ACDC_CROP_SIZE: int = 128
    MM_CROP_SIZE: int = 128
    SCD_CROP_SIZE: int = 128
    YORK_CROP_SIZE: int = 128
    IN_CHANNELS: int = 1
    FEATURE_NUM: int = 4
    CLASS_NUM: int = 4
    GMM_NUM: int = 4
    SCALE_RANGE: tuple = (0.5, 2.0)
    SHIFT_RANGE: tuple = (0, 0)
    ROTATE_RANGE: tuple = (-60, 60)
    USE_LABEL_PRIOR: bool = False  # 是否使用标签先验
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




def get_config():
    parser = argparse.ArgumentParser(description="Training Config")

    parser.add_argument('--dataset', type=str, default=Config.DATASET)
    parser.add_argument('--dataset_dir', type=str, default=Config.DATASET_DIR)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--prior_num_of_patient', type=int, default=Config.PRIOR_NUM_OF_PATIENT)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--checkpoints_dir', type=str, default=Config.CHECKPOINTS_DIR)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--logs_dir', type=str, default=Config.LOGS_DIR)
    parser.add_argument('--img_size', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--acdc_crop_size', type=int, default=Config.ACDC_CROP_SIZE)
    parser.add_argument('--mm_crop_size', type=int, default=Config.MM_CROP_SIZE)
    parser.add_argument('--scd_crop_size', type=int, default=Config.SCD_CROP_SIZE)
    parser.add_argument('--york_crop_size', type=int, default=Config.YORK_CROP_SIZE)
    parser.add_argument('--feature_num', type=int, default=Config.FEATURE_NUM)
    parser.add_argument('--in_channels', type=int, default=Config.IN_CHANNELS)
    parser.add_argument('--class_num', type=int, default=Config.CLASS_NUM)
    parser.add_argument('--gmm_num', type=int, default=Config.GMM_NUM)
    parser.add_argument('--scale_range', type=float, nargs=2, default=Config.SCALE_RANGE)
    parser.add_argument('--shift_range', type=float, nargs=2, default=Config.SHIFT_RANGE)
    parser.add_argument('--rotate_range', type=float, nargs=2, default=Config.ROTATE_RANGE)
    parser.add_argument('--metric_with_background', type=bool, default=Config.METRIC_WITH_BACKGROUND)
    parser.add_argument('--start_reg', type=int, default=Config.START_REG)
    parser.add_argument('--use_label_prior', type=bool, default=Config.USE_LABEL_PRIOR)
    parser.add_argument('--mu_range', type=float, default=Config.MU_RANGE)
    parser.add_argument('--log_var_min', type=float, default=Config.LOG_VAR_MIN)
    parser.add_argument('--log_var_max', type=float, default=Config.LOG_VAR_MAX)
    parser.add_argument('--prior_intensity', type=float, default=Config.PRIOR_INTENSITY)
    parser.add_argument('--predict_img', type=str, default=Config.PREDICT_IMG)
    parser.add_argument('--predict_dir', type=str, default=Config.PREDICT_DIR)
    parser.add_argument('--results_dir', type=str, default=Config.RESULTS_DIR)
    
    args = parser.parse_args()

    return Config(
        DATASET=args.dataset,
        DATASET_DIR=args.dataset_dir,
        BATCH_SIZE=args.batch_size,
        PRIOR_NUM_OF_PATIENT=args.prior_num_of_patient,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        NUM_WORKERS=args.num_workers,
        CHECKPOINTS_DIR=args.checkpoints_dir,
        OUTPUT_DIR=args.output_dir,
        LOGS_DIR=args.logs_dir,
        IMG_SIZE=args.img_size,
        ACDC_CROP_SIZE=args.acdc_crop_size,
        MM_CROP_SIZE=args.mm_crop_size,
        SCD_CROP_SIZE=args.scd_crop_size,
        YORK_CROP_SIZE=args.york_crop_size,
        FEATURE_NUM=args.feature_num,
        IN_CHANNELS=args.in_channels,
        CLASS_NUM=args.class_num,
        GMM_NUM=args.gmm_num,
        SCALE_RANGE=tuple(args.scale_range),
        SHIFT_RANGE=tuple(args.shift_range),
        ROTATE_RANGE=tuple(args.rotate_range),
        START_REG=args.start_reg,
        USE_LABEL_PRIOR=args.use_label_prior,
        METRIC_WITH_BACKGROUND=args.metric_with_background,
        MU_RANGE=args.mu_range,
        LOG_VAR_MIN=args.log_var_min,
        LOG_VAR_MAX=args.log_var_max,
        PRIOR_INTENSITY=args.prior_intensity,
        PREDICT_IMG=args.predict_img,
        PREDICT_DIR=args.predict_dir,
        RESULTS_DIR=args.results_dir,
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