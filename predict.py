#!/usr/bin/env python3
"""
GmmSeg预测脚本
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from config import Config
from models.unet import UNet
from models.regnet import RR_ResNet
from utils.train_utils import forward_pass, standardize_features
from utils.dataloader import get_dirichlet_priors
from utils.calc_post_probs import calculate_posterior_probs
from utils.metrics import dice_coefficient, iou_score, pixel_error, detailed_metrics
from data.transform import get_image_transform
import matplotlib.pyplot as plt


class GmmSegPredictor:
    def __init__(self, config: Config, model_dir: str = "checkpoints"):
        self.config = config
        self.model_dir = model_dir
        self.device = config.DEVICE
        
        # 初始化模型
        self.setup_models()
        
        # 加载预训练权重
        self.load_pretrained_weights()
        
        # 加载Dirichlet先验
        self.dirichlet_priors = get_dirichlet_priors(config)
        
        # 图像变换
        self.transform = get_image_transform(config.IMG_SIZE)
    
    def setup_models(self):
        """初始化所有模型"""
        self.unet = UNet(self.config.IN_CHANNELS, self.config.FEATURE_NUM).to(self.device)
        self.x_net = UNet(self.config.FEATURE_NUM, self.config.FEATURE_NUM*self.config.GMM_NUM*2).to(self.device)
        self.z_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.device)
        self.o_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.device)
        # 修正: reg_net初始化时input_channels=4，内部会自动*2得到8个输入通道
        self.reg_net = RR_ResNet(input_channels=self.config.GMM_NUM).to(self.device)
        
        # 设置为评估模式
        self.unet.eval()
        self.x_net.eval()
        self.z_net.eval()
        self.o_net.eval()
        self.reg_net.eval()
    
    def load_pretrained_weights(self):
        """加载预训练权重"""
        try:
            # 定义权重文件路径
            weight_files = {
                'unet': os.path.join(self.model_dir, 'unet', 'best.pth'),
                'x_net': os.path.join(self.model_dir, 'PRIOR', 'x_train_4.pth'),
                'z_net': os.path.join(self.model_dir, 'PRIOR', 'z_train_4.pth'),
                'o_net': os.path.join(self.model_dir, 'PRIOR', 'o_train_4.pth'),
                'reg_net': os.path.join(self.model_dir, 'reg_prior.pth')
            }
            
            # 加载权重
            for model_name, weight_path in weight_files.items():
                if os.path.exists(weight_path):
                    weights = torch.load(weight_path, map_location=self.device, weights_only=True)
                    getattr(self, model_name).load_state_dict(weights)
                    print(f"✅ 成功加载 {model_name} 权重")
                else:
                    print(f"⚠️  未找到 {model_name} 权重文件: {weight_path}")
                    
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """预处理输入图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('L')  # 转换为灰度图
        else:
            image = image_path
        
        # 应用变换
        image_tensor = self.transform(image)  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        print(f"预处理后的图像形状: {image_tensor.shape}")
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def predict_single_image(self, image_tensor, slice_info=None, num_of_slice_info=None, epoch=100):
        """对单张图像进行预测"""
        with torch.no_grad():
            batch_size = image_tensor.shape[0]
            
            # 如果没有提供slice信息，使用默认值
            if slice_info is None:
                slice_info = torch.tensor([5], device=self.device)  # 默认slice
            if num_of_slice_info is None:
                num_of_slice_info = torch.tensor([10], device=self.device)  # 默认总slice数
            
            # 确保slice信息与batch_size匹配
            if len(slice_info) != batch_size:
                slice_info = slice_info.repeat(batch_size)
            if len(num_of_slice_info) != batch_size:
                num_of_slice_info = num_of_slice_info.repeat(batch_size)
            
            # 前向传播
            image_4_features, mu, var, pi, d1, d0 = forward_pass(
                image_tensor, self.unet, self.x_net, self.z_net, self.o_net, 
                self.reg_net, self.dirichlet_priors, slice_info, num_of_slice_info, 
                self.config, epoch, epsilon=1e-6
            )
            
            # 计算后验概率
            posterior_probs = calculate_posterior_probs(image_4_features, mu, var, pi)
            
            # 获取预测结果
            predictions = torch.argmax(posterior_probs, dim=1)  # [B, H, W]
            
            return {
                'predictions': predictions.cpu().numpy(),
                'posterior_probs': posterior_probs.cpu().numpy(),
                'features': image_4_features.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                'var': var.cpu().numpy(),
                'pi': pi.cpu().numpy(),
                'd1': d1.cpu().numpy(),
                'd0': d0.cpu().numpy()
            }
    
    def predict_and_evaluate(self, image_path, ground_truth_path=None, 
                           slice_info=None, num_of_slice_info=None):
        """预测并评估结果"""
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        
        # 进行预测
        result = self.predict_single_image(image_tensor, slice_info, num_of_slice_info)
        predictions = result['predictions'][0]  # 取第一个batch的结果
        
        # 评估（如果提供了ground truth）
        metrics = {}
        if ground_truth_path is not None:
            # 读取ground truth
            gt_image = Image.open(ground_truth_path).convert('L')
            gt_array = np.array(gt_image.resize((self.config.IMG_SIZE, self.config.IMG_SIZE)))
            
            # 计算指标
            dice = dice_coefficient(predictions.flatten(), gt_array.flatten(), 
                                  self.config.CLASS_NUM, self.config.DATASET, background=True)
            iou = iou_score(predictions.flatten(), gt_array.flatten(), 
                          self.config.CLASS_NUM, self.config.DATASET, background=True)
            pixel_err = pixel_error(predictions.flatten(), gt_array.flatten(), 
                                  self.config.CLASS_NUM, self.config.DATASET, background=True)
            detailed = detailed_metrics(predictions.flatten(), gt_array.flatten(), 
                                      self.config.CLASS_NUM, self.config.DATASET, background=True)
            
            metrics = {
                'dice': dice,
                'iou': iou,
                'pixel_error': pixel_err,
                'detailed': detailed
            }
        
        return predictions, result, metrics
    
    def visualize_results(self, image_path, predictions, result, save_path=None):
        """可视化预测结果"""
        # 读取原始图像
        original_image = Image.open(image_path).convert('L')
        original_array = np.array(original_image.resize((self.config.IMG_SIZE, self.config.IMG_SIZE)))
        
        # 创建一行三列的布局
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 左侧：变换后的原图
        axes[0].imshow(original_array, cmap='gray')
        axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=20)
        axes[0].axis('off')
        
        # 中间：分割结果（不同颜色区分各组织）
        # 创建自定义颜色映射 - 背景黑色，RV红色，MYO绿色，LV蓝色
        from matplotlib.colors import ListedColormap
        colors = ['black', 'red', 'lime', 'cyan']  # 0:背景(黑), 1:RV(红), 2:MYO(绿), 3:LV(青)
        cmap_segmentation = ListedColormap(colors[:self.config.CLASS_NUM])
        
        im_seg = axes[1].imshow(predictions, cmap=cmap_segmentation, vmin=0, vmax=self.config.CLASS_NUM-1)
        axes[1].set_title('Segmentation Result\n(Red:RV, Green:MYO, Cyan:LV)', fontsize=16, fontweight='bold', pad=20)
        axes[1].axis('off')
        
        # 右侧：分割结果叠加在原图上（前景为分割结果，背景为原图）
        # 将原图归一化到0-1
        normalized_original = (original_array - original_array.min()) / (original_array.max() - original_array.min())
        
        # 创建RGB图像用于叠加显示
        overlay_image = np.stack([normalized_original] * 3, axis=2)  # [H, W, 3]
        
        # 为每个类别添加颜色叠加（只在分割区域）
        alpha = 0.7  # 增加透明度使效果更明显
        for class_id in range(1, self.config.CLASS_NUM):  # 跳过背景
            mask = (predictions == class_id)
            if np.any(mask):  # 如果该类别存在
                if class_id == 1:  # RV - 红色
                    overlay_image[mask, 0] = alpha * 1.0 + (1-alpha) * overlay_image[mask, 0]  # 红色通道
                    overlay_image[mask, 1] = (1-alpha) * overlay_image[mask, 1]  # 绿色通道
                    overlay_image[mask, 2] = (1-alpha) * overlay_image[mask, 2]  # 蓝色通道
                elif class_id == 2:  # MYO - 绿色
                    overlay_image[mask, 0] = (1-alpha) * overlay_image[mask, 0]  # 红色通道
                    overlay_image[mask, 1] = alpha * 1.0 + (1-alpha) * overlay_image[mask, 1]  # 绿色通道
                    overlay_image[mask, 2] = (1-alpha) * overlay_image[mask, 2]  # 蓝色通道
                elif class_id == 3:  # LV - 青色
                    overlay_image[mask, 0] = (1-alpha) * overlay_image[mask, 0]  # 红色通道
                    overlay_image[mask, 1] = alpha * 1.0 + (1-alpha) * overlay_image[mask, 1]  # 绿色通道
                    overlay_image[mask, 2] = alpha * 1.0 + (1-alpha) * overlay_image[mask, 2]  # 蓝色通道
        
        axes[2].imshow(np.clip(overlay_image, 0, 1))
        axes[2].set_title('Overlay Visualization\n(ROI Overlay on Original)', fontsize=16, fontweight='bold', pad=20)
        axes[2].axis('off')
        
        # 添加类别说明
        legend_text = "Classes: Background(Black), RV(Red), MYO(Green), LV(Cyan)"
        fig.suptitle(legend_text, fontsize=12, y=0.02)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def batch_predict(self, image_dir, output_dir, gt_dir=None):
        """批量预测"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        results = []
        
        for image_file in image_files:
            print(f"处理图像: {image_file}")
            
            image_path = os.path.join(image_dir, image_file)
            gt_path = None
            if gt_dir:
                gt_path = os.path.join(gt_dir, image_file)
                if not os.path.exists(gt_path):
                    gt_path = None
            
            # 预测
            predictions, result, metrics = self.predict_and_evaluate(image_path, gt_path)
            
            # 保存预测结果
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            pred_image = Image.fromarray(predictions.astype(np.uint8))
            pred_image.save(output_path)
            
            # 保存可视化
            vis_path = os.path.join(output_dir, f"vis_{image_file}")
            self.visualize_results(image_path, predictions, result, vis_path)
            
            # 记录结果
            result_info = {
                'image_file': image_file,
                'metrics': metrics,
                'output_path': output_path
            }
            results.append(result_info)
            
            # 打印指标
            if metrics:
                print(f"  Dice: {metrics['dice']:.4f}")
                print(f"  IoU: {metrics['iou']:.4f}")
                print(f"  Pixel Error: {metrics['pixel_error']:.4f}")
            print()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='GmmSeg预测脚本')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth目录（可选，用于评估）')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='模型权重目录')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 创建预测器
    predictor = GmmSegPredictor(config, args.model_dir)
    
    if os.path.isfile(args.image):
        # 单张图像预测
        print(f"预测单张图像: {args.image}")
        predictions, result, metrics = predictor.predict_and_evaluate(args.image, args.gt)
        
        # 保存预测结果
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "prediction.png")
        pred_image = Image.fromarray(predictions.astype(np.uint8))
        pred_image.save(output_path)
        
        # 可视化
        if args.visualize:
            vis_path = os.path.join(args.output, "visualization.png")
            predictor.visualize_results(args.image, predictions, result, vis_path)
        
        # 打印指标
        if metrics:
            print("\n=== 评估指标 ===")
            print(f"Dice Coefficient: {metrics['dice']:.4f}")
            print(f"IoU Score: {metrics['iou']:.4f}")
            print(f"Pixel Error: {metrics['pixel_error']:.4f}")
            
            if 'detailed' in metrics:
                print("\n详细指标:")
                for metric_name, class_values in metrics['detailed'].items():
                    if isinstance(class_values, dict):
                        print(f"{metric_name.upper()}:")
                        for class_name, value in class_values.items():
                            print(f"  {class_name}: {value:.4f}")
    
    elif os.path.isdir(args.image):
        # 批量预测
        print(f"批量预测目录: {args.image}")
        results = predictor.batch_predict(args.image, args.output, args.gt)
        
        # 统计总体指标
        if results and all(r['metrics'] for r in results):
            total_dice = np.mean([r['metrics']['dice'] for r in results])
            total_iou = np.mean([r['metrics']['iou'] for r in results])
            total_pixel_err = np.mean([r['metrics']['pixel_error'] for r in results])
            
            print("=== 总体指标 ===")
            print(f"平均 Dice Coefficient: {total_dice:.4f}")
            print(f"平均 IoU Score: {total_iou:.4f}")
            print(f"平均 Pixel Error: {total_pixel_err:.4f}")
    
    else:
        print(f"错误：路径不存在 {args.image}")


if __name__ == "__main__":
    main()
