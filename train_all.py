import os
import sys
from config import Config, get_config
from train import Trainer


SUPPORTED_DATASETS = ["SCD", "YORK", "MM"]


def main():
	for dataset in SUPPORTED_DATASETS:
		print(f"\n🚀 Starting training on {dataset} dataset...")
		# 更新配置中的数据集名称和相关路径
		config = get_config()
		config.DATASET = dataset
		# 创建Trainer并开始训练
		trainer = Trainer(config)
		trainer.run(train=True)

	print("\n✅ All datasets finished.")


if __name__ == "__main__":
	# Allow running as module or script
	sys.exit(main())

