import torch
import argparse
from dataclasses import dataclass

@dataclass
class Config:
    DATASET: str = "ACDC_aligned"
    DATASET_DIR: str = "D:/Users/pyw/Desktop/Dataset"
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
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
    IMG_SIZE: int = 128
    IN_CHANNELS: int = 1
    FEATURE_NUM: int = 4
    CLASS_NUM: int = 4
    GMM_NUM: int = 4
    SCALE_RANGE: tuple = (0.5, 2.0)
    SHIFT_RANGE: tuple = (-20, 20)

    # Registration application epoch threshold
    REG_START_EPOCH: int = 5

    # ---- New hyper params for improved normalization & GMM stability ----
    MU_RANGE: float = 3.0                    # clamp mu to [-MU_RANGE, MU_RANGE]
    VAR_MIN: float = 1e-1                    # minimum variance
    VAR_MAX: float = 2.25                    # maximum variance
    PI_TEMPERATURE: float = 1.0              # temperature for softmax of pi
    VAR_TEMP: float = 2.0                    # 温度/缩放: 控制 var 原始输出进入 softplus 的平滑程度
    VAR_MID_BETA: float = 0.2                # 中位(均值)方差拉升正则权重

    # Prior probability -> concentration mapping
    PRIOR_BASE_CONC: float = 2.0             # base concentration after mapping
    PRIOR_MAX_CONC: float = 8.0              # max concentration after mapping

    # ---- Prediction options (no posterior mode now) ----
    PREDICT_IMAGE: str | None = None         # 单张预测图片路径
    PREDICT_DIR: str | None = None           # 批量预测目录
    PREDICT_SAVE_MASK: bool = True           # 保存彩色 mask
    PREDICT_SAVE_OVERLAY: bool = True        # 保存叠加图
    RESULTS_DIR: str = './results'           # 结果保存目录

    # ---- Warmup (learning rate) ----
    WARMUP_EPOCHS: int = 5                  # 前多少个 epoch 做学习率 warmup（0 表示关闭）
    WARMUP_START_FACTOR: float = 0.1        # 初始学习率 = base_lr * start_factor



def get_config():
    parser = argparse.ArgumentParser(description="Training Config")

    parser.add_argument('--dataset', type=str, default=Config.DATASET)
    parser.add_argument('--dataset_dir', type=str, default=Config.DATASET_DIR)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--checkpoints_dir', type=str, default=Config.CHECKPOINTS_DIR)
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR)
    parser.add_argument('--logs_dir', type=str, default=Config.LOGS_DIR)
    parser.add_argument('--img_size', type=int, default=Config.IMG_SIZE)
    parser.add_argument('--feature_num', type=int, default=Config.FEATURE_NUM)
    parser.add_argument('--in_channels', type=int, default=Config.IN_CHANNELS)
    parser.add_argument('--class_num', type=int, default=Config.CLASS_NUM)
    parser.add_argument('--gmm_num', type=int, default=Config.GMM_NUM)
    parser.add_argument('--scale_range', nargs=2, type=float, default=Config.SCALE_RANGE)
    parser.add_argument('--shift_range', nargs=2, type=int, default=Config.SHIFT_RANGE)
    parser.add_argument('--reg_start_epoch', type=int, default=Config.REG_START_EPOCH)
    parser.add_argument('--mu_range', type=float, default=Config.MU_RANGE)
    parser.add_argument('--var_min', type=float, default=Config.VAR_MIN)
    parser.add_argument('--var_max', type=float, default=Config.VAR_MAX)
    parser.add_argument('--pi_temperature', type=float, default=Config.PI_TEMPERATURE)
    parser.add_argument('--var_temp', type=float, default=Config.VAR_TEMP)
    parser.add_argument('--var_mid_beta', type=float, default=Config.VAR_MID_BETA)
    parser.add_argument('--prior_base_conc', type=float, default=Config.PRIOR_BASE_CONC)
    parser.add_argument('--prior_max_conc', type=float, default=Config.PRIOR_MAX_CONC)
    parser.add_argument('--warmup_epochs', type=int, default=Config.WARMUP_EPOCHS)
    parser.add_argument('--warmup_start_factor', type=float, default=Config.WARMUP_START_FACTOR)
    # prediction related
    parser.add_argument('--predict_image', type=str, default=Config.PREDICT_IMAGE)
    parser.add_argument('--predict_dir', type=str, default=Config.PREDICT_DIR)
    parser.add_argument('--predict_save_mask', action='store_true', default=Config.PREDICT_SAVE_MASK)
    parser.add_argument('--predict_save_overlay', action='store_true', default=Config.PREDICT_SAVE_OVERLAY)
    parser.add_argument('--results_dir', type=str, default=Config.RESULTS_DIR)
    parser.add_argument("--x_pth", type=str, default=None)
    parser.add_argument("--z_pth", type=str, default=None)
    parser.add_argument("--o_pth", type=str, default=None)
    
    args = parser.parse_args()

    return Config(
        DATASET=args.dataset,
        DATASET_DIR=args.dataset_dir,
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        LEARNING_RATE=args.lr,
        NUM_WORKERS=args.num_workers,
        CHECKPOINTS_DIR=args.checkpoints_dir,
        OUTPUT_DIR=args.output_dir,
        LOGS_DIR=args.logs_dir,
        IMG_SIZE=args.img_size,
        FEATURE_NUM=args.feature_num,
        IN_CHANNELS=args.in_channels,
        CLASS_NUM=args.class_num,
        GMM_NUM=args.gmm_num,
        SCALE_RANGE=tuple(args.scale_range),
        SHIFT_RANGE=tuple(args.shift_range),
        REG_START_EPOCH=args.reg_start_epoch,
        MU_RANGE=args.mu_range,
        VAR_MIN=args.var_min,
        VAR_MAX=args.var_max,
        PI_TEMPERATURE=args.pi_temperature,
        VAR_TEMP=args.var_temp,
        VAR_MID_BETA=args.var_mid_beta,
        PRIOR_BASE_CONC=args.prior_base_conc,
        PRIOR_MAX_CONC=args.prior_max_conc,
        WARMUP_EPOCHS=args.warmup_epochs,
        WARMUP_START_FACTOR=args.warmup_start_factor,
        PREDICT_IMAGE=args.predict_image,
        PREDICT_DIR=args.predict_dir,
        PREDICT_SAVE_MASK=args.predict_save_mask,
        PREDICT_SAVE_OVERLAY=args.predict_save_overlay,
        RESULTS_DIR=args.results_dir,
        # 保留默认值（不可通过命令行修改）
        BEST_LOSS=Config().BEST_LOSS,
        BEST_DICE=Config().BEST_DICE,
        BEST_IOU=Config().BEST_IOU,
        BEST_PIXEL_ERROR=Config.BEST_PIXEL_ERROR,
        BEST_EPOCH=Config.BEST_EPOCH,
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )



if __name__ == "__main__":
    config = get_config()
    print(config)