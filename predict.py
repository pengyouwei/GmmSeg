from data.transform import get_image_transform
from utils.loss import GmmLoss, posterior_from_params
from utils.dataloader import get_dirichlet_priors
from utils.train_utils import forward_pass
from models.regnet import RR_ResNet
from models.unet import UNet
from config import get_config, Config
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')


CLASS_COLORS = {
	0: (0, 0, 0),       # 背景 黑
	1: (0, 0, 255),     # 左心室 蓝 (BGR→后续转RGB时处理)
	2: (0, 255, 0),     # 心肌 绿
	3: (255, 0, 0),     # 右心室 红
}


def label_to_color(mask: np.ndarray) -> np.ndarray:
	"""将整数标签mask(H,W)转为彩色RGB(H,W,3)."""
	h, w = mask.shape
	color = np.zeros((h, w, 3), dtype=np.uint8)
	for k, (r, g, b) in CLASS_COLORS.items():
		color[mask == k] = (r, g, b)
	return color


class Predictor:
	def __init__(self, config: Config, device=None):
		self.config = config
		self.device = device or config.DEVICE
		# 模型结构与训练一致
		self.unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(self.device)
		self.x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM *
		                  config.GMM_NUM * 2).to(self.device)
		# Dirichlet 混合模式下无需 z_net
		if not config.USE_DIRICHLET_MIX:
			self.z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(self.device)
		else:
			self.z_net = None
		self.o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(self.device)
		self.reg_net = RR_ResNet(input_channels=config.GMM_NUM).to(self.device)

		# 加载权重（若存在）
		self._load_weights()
		self.unet.eval(); self.x_net.eval(); self.o_net.eval(); self.reg_net.eval()
		if self.z_net is not None:
			self.z_net.eval()

		# 先验库
		try:
			self.dirichlet_priors = get_dirichlet_priors(config)
		except Exception:
			# 若缺失，创建均匀先验
			self.dirichlet_priors = torch.full((10, config.GMM_NUM, config.IMG_SIZE, config.IMG_SIZE),
										   1.0/config.GMM_NUM, device=self.device)

		# 图像 transform
		self.img_transform = get_image_transform(config.IMG_SIZE)

	def _load_weights(self):
		ckpt_dir = self.config.CHECKPOINTS_DIR
		mapping = {
			'unet': ('unet/feature_extraction.pth', self.unet),
			'x_net': ('unet/x_best.pth', self.x_net),
			'z_net': ('unet/z_best.pth', self.z_net),
			'o_net': ('unet/o_best.pth', self.o_net),
			'reg_net': ('regnet/dirichlet_registration.pth', self.reg_net),
		}
		for name, (rel, module) in mapping.items():
			path = os.path.join(ckpt_dir, rel)
			if os.path.isfile(path):
				try:
					state = torch.load(path, map_location=self.device, weights_only=True)
					module.load_state_dict(state, strict=False)
				except Exception:
					pass

	def _infer_tensor(self, img_tensor: torch.Tensor):
		"""前向推理，返回 (pred_mask, prob_map[K,H,W])。
		Dirichlet 模式下 prob_map 为 posterior r；传统模式为 π。"""
		with torch.no_grad():
			# 构造假 slice_info (batch=1)
			slice_info = torch.tensor([0])
			slice_num = torch.tensor([1])
			features, mu, var, pi, d1, d0 = forward_pass(image=img_tensor,
									 unet=self.unet,
									 x_net=self.x_net,
									 z_net=self.z_net,
									 o_net=self.o_net,
									 reg_net=self.reg_net,
									 dirichlet_priors=self.dirichlet_priors,
									 slice_info=slice_info,
									 num_of_slice_info=slice_num,
									 config=self.config,
									 epoch=0,
									 epsilon=1e-6)
			if self.config.USE_DIRICHLET_MIX:
				conc = d1.reshape(1, self.config.GMM_NUM, self.config.IMG_SIZE, self.config.IMG_SIZE)
				pi_expect = conc / (conc.sum(dim=1, keepdim=True) + 1e-8)
				K = self.config.GMM_NUM; C = self.config.FEATURE_NUM
				mu_ = mu.reshape(1, K, C, self.config.IMG_SIZE, self.config.IMG_SIZE)
				var_ = var.reshape(1, K, C, self.config.IMG_SIZE, self.config.IMG_SIZE)
				x = features.unsqueeze(1)
				r = posterior_from_params(x=x, mu=mu_, var=var_, pi=pi_expect)
				prob = r[0].cpu().numpy()
			else:
				prob = pi[0].cpu().numpy()
			pred = np.argmax(prob, axis=0).astype(np.uint8)
		return pred, prob

	def predict_image(self, image_path: str, save: bool = True):
		from PIL import Image
		img = Image.open(image_path).convert('L')
		img_t = self.img_transform(img).unsqueeze(0).to(self.device)  # [1,1,H,W]
		pred, prob = self._infer_tensor(img_t)
		color_pred = label_to_color(pred)
		# 原图（已经标准化过，反标准化显示）
		vis_img = img_t[0, 0].cpu().numpy()
		vis_img = (vis_img * 0.5 + 0.5)  # 反 Normalize 回 [0,1]
		overlay = self._blend_overlay(vis_img, color_pred)
		fig = self._plot_triplet(vis_img, color_pred, overlay,
		                         os.path.basename(image_path))
		if save:
			os.makedirs('results', exist_ok=True)
			out_path = os.path.join(
			    'results', f"pred_{os.path.splitext(os.path.basename(image_path))[0]}.png")
			fig.savefig(out_path, dpi=150, bbox_inches='tight')
			plt.close(fig)
			return out_path
		return fig

	def predict_folder(self, folder_path: str):
		exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
		files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[
		                               1].lower() in exts]
		outputs = []
		for f in files:
			try:
				path = os.path.join(folder_path, f)
				out = self.predict_image(path, save=True)
				outputs.append(out)
			except Exception:
				continue
		return outputs

	@staticmethod
	def _blend_overlay(gray_img: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5):
		g = np.stack([gray_img]*3, axis=-1)
		g = (g * 255).astype(np.uint8)
		blended = (alpha * color_mask + (1-alpha) * g).astype(np.uint8)
		return blended

	@staticmethod
	def _plot_triplet(gray_img: np.ndarray, color_pred: np.ndarray, overlay: np.ndarray, title: str):
		fig, axes = plt.subplots(1, 3, figsize=(9, 3))
		axes[0].imshow(gray_img, cmap='gray'); axes[0].set_title(
		    'Image'); axes[0].axis('off')
		axes[1].imshow(color_pred); axes[1].set_title(
		    'Segmentation'); axes[1].axis('off')
		axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
		fig.suptitle(title)
		fig.tight_layout()
		return fig


if __name__ == '__main__':
	config = get_config()
	predictor = Predictor(config)
	if config.PREDICT_IMAGE:
		predictor.predict_image(config.PREDICT_IMAGE)
	if config.PREDICT_DIR:
		predictor.predict_folder(config.PREDICT_DIR)