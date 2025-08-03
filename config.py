import torch
import argparse
from dataclasses import dataclass

@dataclass
class Config:
    DATASET: str = "ACDC"
    DATASET_DIR: str = "D:/Users/pyw/Desktop/Dataset"
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4
    NUM_WORKERS: int = 8
    BEST_LOSS: float = float('inf')
    BEST_DICE: float = 0.8
    BEST_IOU: float = 0.8
    BEST_PIXEL_ERROR: float = float('inf')
    DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CHECKPOINTS_DIR: str = './checkpoints'
    OUTPUT_DIR: str = './output' # Output directory for visualizations
    LOGS_DIR: str = './logs'
    IMG_SIZE: int = 128
    IN_CHANNELS: int = 1
    FEATURE_NUM: int = 4
    CLASS_NUM: int = 4
    GMM_NUM: int = 4
    SCALE_RANGE: tuple = (0.2, 5.0)
    SHIFT_RANGE: tuple = (-20, 20)



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
    parser.add_argument("--test_dir", type=str, default=None)
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
        # 保留默认值（不可通过命令行修改）
        BEST_LOSS=Config().BEST_LOSS,
        BEST_DICE=Config().BEST_DICE,
        BEST_IOU=Config().BEST_IOU,
        BEST_PIXEL_ERROR=Config.BEST_PIXEL_ERROR,
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )



if __name__ == "__main__":
    config = get_config()
    print(config)