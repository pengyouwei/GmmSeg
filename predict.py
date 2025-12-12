import os
import torch
import numpy as np
import matplotlib
from models.regnet import RR_ResNet
from models.unet import UNet
from config import get_config, Config
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from utils.train_utils import standardize_features
# matplotlib.use('Agg')

def get_image_transform(img_size):
    return transforms.Compose([
        transforms.CenterCrop((img_size, img_size)), 
        transforms.ToTensor(),  # 转为张量，范围 [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到 [-1, 1]
    ])

def get_label_transform(img_size):
    return transforms.Compose([
        transforms.CenterCrop((img_size, img_size)),
        transforms.PILToTensor()
    ])


class Predictor:
	def __init__(self, config: Config):
		self.config = config
		self.device = config.DEVICE
		self.transform_image = get_image_transform(config.IMG_SIZE)
		self.unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(self.device)
		self.x_net = UNet(self.config.FEATURE_NUM, self.config.FEATURE_NUM * self.config.GMM_NUM * 2).to(self.config.DEVICE)  # mu, var
		self.z_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # r
		self.o_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # d
		self.load_weights()
		self.unet.eval()
		self.x_net.eval()
		self.z_net.eval()
		self.o_net.eval()

	def load_weights(self):
		unet_weight_path = os.path.join(self.config.CHECKPOINTS_DIR, 'unet', 'unet_best.pth')
		xnet_weight_path = os.path.join(self.config.CHECKPOINTS_DIR, 'unet', self.config.DATASET, 'x_best.pth')
		znet_weight_path = os.path.join(self.config.CHECKPOINTS_DIR, 'unet', self.config.DATASET, 'z_best.pth')
		onet_weight_path = os.path.join(self.config.CHECKPOINTS_DIR, 'unet', self.config.DATASET, 'o_best.pth')
		self.unet.load_state_dict(torch.load(unet_weight_path, map_location=self.device))
		self.x_net.load_state_dict(torch.load(xnet_weight_path, map_location=self.device))
		self.z_net.load_state_dict(torch.load(znet_weight_path, map_location=self.device))
		self.o_net.load_state_dict(torch.load(onet_weight_path, map_location=self.device))
		print("Loaded model weights.")

	def predict_img(self, image_path: str, save: bool = True):
		if image_path.endswith(('.png', '.jpg', '.jpeg')):
			print("Loading image from:", image_path)
			image = Image.open(image_path).convert('L')
			image = self.transform_image(image).unsqueeze(0).to(self.device)  # [1, 1, H, W]
		if image_path.endswith('.npy'):
			print("Loading numpy array from:", image_path)
			image = np.load(image_path)
			image = (image - image.min()) / (image.max() - image.min())  # 归一化到0-1
			image = Image.fromarray((image * 255).astype(np.uint8))
			image = self.transform_image(image).unsqueeze(0).to(self.device)  # [1, 1, H, W]

		with torch.no_grad():
			features = self.unet(image)
			# 保持与训练一致：标准化特征，避免分布偏移
			features = standardize_features(features)
			post = self.z_net(features)
			pred = torch.argmax(post, dim=1).detach().cpu().numpy()[0]
			print(pred.shape, pred.min(), pred.max())
		
		fig, axes = plt.subplots(1, 3, figsize=(10, 5))
		axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
		axes[0].set_title("Input Image")
		axes[0].axis('off')

		axes[1].imshow(pred, cmap='gray')
		axes[1].set_title("Model Output")
		axes[1].axis('off')

		axes[2].imshow(image.squeeze().cpu().numpy(), cmap='gray')
		axes[2].imshow(pred, cmap='jet', alpha=0.4)  # 彩色mask，透明度0.4
		axes[2].set_title("Overlay")
		axes[2].axis('off')

		fig.tight_layout()  # 新增，防止标题被遮挡
		# 先保存，再阻塞显示，避免窗口一闪而过
		if save:
			os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
			# 使用支持的图像扩展名保存（统一为 .png），避免 .npy 等不被后端支持
			base_name = os.path.splitext(os.path.basename(image_path))[0]
			out_path = os.path.join(self.config.RESULTS_DIR, f"{base_name}.png")
			fig.savefig(out_path, dpi=150, bbox_inches='tight')
			print(f"Saved prediction to {out_path}")
		plt.show()


	def predict_dir(self, folder_path: str):
		pass



if __name__ == '__main__':
	config = get_config()
	predictor = Predictor(config)
	if config.PREDICT_IMG:
		predictor.predict_img(config.PREDICT_IMG)
	if config.PREDICT_DIR:
		predictor.predict_dir(config.PREDICT_DIR)
	print("Prediction completed.")