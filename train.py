import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.loss import GmmLoss
from utils.train_utils import forward_pass
from utils.visualizer import create_visualization
from utils.metrics import dice_coefficient, iou_score, pixel_error
from utils.dataloader import get_loaders
from utils.train_utils import standardize_features
from models.unet import UNet
from models.regnet import RR_ResNet
from config import Config, get_config
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("Train GMM Segmentation")

    def setup(self, train=True):
        # 设置日志记录器
        self.logger.setLevel(logging.INFO)
        # 创建控制台处理器
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
        # 创建文件处理器
        logs_dir = os.path.join(self.config.LOGS_DIR, "train_logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.file_handler = logging.FileHandler(f"{logs_dir}/{self.time_str}.log", encoding="utf-8", mode='a')
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        # 添加处理器
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

        self.logger.info("Starting training...")
        self.logger.info(f"Using configuration: {self.config}")

        # 加载数据集
        self.train_loader, self.valid_loader, self.test_loader = get_loaders(self.config)
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Valid samples: {len(self.valid_loader.dataset)}")
        self.logger.info(f"Test  samples: {len(self.test_loader.dataset)}")

        # 定义模型
        self.unet = UNet(self.config.IN_CHANNELS, self.config.FEATURE_NUM).to(self.config.DEVICE)  # UNet 用于提取特征
        self.x_net = UNet(self.config.FEATURE_NUM, self.config.FEATURE_NUM * self.config.GMM_NUM * 2).to(self.config.DEVICE)  # mu, var
        self.z_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # pi
        self.o_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # d
        self.reg_net = RR_ResNet(input_channels=self.config.GMM_NUM * 2).to(self.config.DEVICE)


        # 加载预训练权重 (修复: 启用所有预训练权重的加载)
        try:
            x_net_weights = torch.load("checkpoints/unet/x_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            z_net_weights = torch.load("checkpoints/unet/z_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            o_net_weights = torch.load("checkpoints/unet/o_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            reg_net_weights = torch.load("checkpoints/regnet/regnet_prior_4chs.pth", 
                                         map_location=self.config.DEVICE, 
                                          weights_only=True)
            unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                      map_location=self.config.DEVICE, 
                                      weights_only=True)
            
            # 加载权重到模型
            # self.x_net.load_state_dict(x_net_weights)
            # self.z_net.load_state_dict(z_net_weights)
            # self.o_net.load_state_dict(o_net_weights)
            self.reg_net.load_state_dict(reg_net_weights)
            self.unet.load_state_dict(unet_weights)
            
            # 固定预训练模型的参数
            self.reg_net.eval()
            self.unet.eval()
            
            # 冻结UNet和reg_net的参数以避免过拟合
            for param in self.reg_net.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = False

        except Exception as e:
            self.logger.error(f"加载预训练权重时出错: {e}")

        # 定义损失函数和优化器
        self.criterion = GmmLoss(config=self.config).to(self.config.DEVICE)

        # 使用不同的学习率策略
        opt_params = [
            {'params': self.x_net.parameters(), 'lr': self.config.LEARNING_RATE, 'weight_decay': 1e-5},
            {'params': self.z_net.parameters(), 'lr': self.config.LEARNING_RATE, 'weight_decay': 1e-5},
            {'params': self.o_net.parameters(), 'lr': self.config.LEARNING_RATE, 'weight_decay': 1e-5},
        ]
        self.optimizer = optim.Adam(opt_params, eps=1e-8, betas=(0.9, 0.999))

        # 改进的学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode='min', 
                                                              factor=0.7,    # 更温和的衰减
                                                              patience=5,    # 更长的等待时间
                                                              min_lr=1e-7    # 最小学习率
                                                              )
        # Warmup 基础学习率缓存
        self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]

        if train:
            # 创建日志记录器
            log_dir = os.path.join(self.config.LOGS_DIR, "tensorboard", self.time_str)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

            # 创建输出目录
            self.output_dir = os.path.join(self.config.OUTPUT_DIR, self.time_str)
            os.makedirs(self.output_dir, exist_ok=True)


    def train_one_epoch(self, epoch, epsilon=1e-6):
        self.x_net.train()
        self.z_net.train()
        self.o_net.train()

        # 学习率 warmup（按 epoch 线性从 start_factor -> 1.0）
        if self.config.WARMUP_EPOCHS > 0 and epoch < self.config.WARMUP_EPOCHS:
            ratio = (epoch + 1) / self.config.WARMUP_EPOCHS
            scale = self.config.WARMUP_START_FACTOR + (1.0 - self.config.WARMUP_START_FACTOR) * ratio
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group['lr'] = base_lr * scale
        elif self.config.WARMUP_EPOCHS > 0 and epoch == self.config.WARMUP_EPOCHS:
            # 确保 warmup 结束后恢复精确基准 lr（避免累计误差）
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
                group['lr'] = base_lr


        train_loss = 0.0
        train_loss_1, train_loss_2, train_loss_3 = 0.0, 0.0, 0.0
        train_loss_mse = 0.0
        current_weight_3 = 0.0  # 记录当前的先验权重

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[train]", ncols=100):
            # 数据加载
            image = batch["image"].to(device=self.config.DEVICE, dtype=torch.float32)
            label = batch["label"].to(device=self.config.DEVICE, dtype=torch.float32)
            prior = batch["prior"].to(device=self.config.DEVICE, dtype=torch.float32)

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            img_4_chs, mu, var, pi, d1, d0 = forward_pass(image=image,
                                                          label=label,
                                                          prior=prior,
                                                          unet=self.unet,
                                                          x_net=self.x_net,
                                                          z_net=self.z_net,
                                                          o_net=self.o_net,
                                                          reg_net=self.reg_net,
                                                          config=self.config, 
                                                          epoch=epoch, 
                                                          epsilon=epsilon)
            # 计算损失 (添加异常处理)
            try:
                out = self.criterion(input=img_4_chs,
                                     mu=mu,
                                     var=var,
                                     pi=pi,
                                     alpha=d1,
                                     prior=d0,
                                     epoch=epoch,
                                     total_epochs=self.config.EPOCHS)
                loss = out['total']
                loss_1 = out['recon']
                loss_2 = out['kl_pi']
                loss_3 = out['kl_dir']
                loss_mse = out['loss_mse']
                self.current_weight_2 = out['weight_2']
                self.current_weight_3 = out['weight_3']

                if not torch.isfinite(loss):
                    if self.logger:
                        self.logger.warning(f"训练中检测到无效损失值: {loss.item()}, 跳过此batch")
                    continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"计算损失时发生错误: {e}")
                continue

            # 累加损失
            train_loss_1 += loss_1.item()
            train_loss_2 += loss_2.item()
            train_loss_3 += loss_3.item()
            train_loss_mse += loss_mse.item()
            train_loss += loss.item()

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.x_net.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.z_net.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.o_net.parameters(), max_norm=5.0)

            # 更新参数
            self.optimizer.step()


        # 计算平均损失值
        self.train_loss_1 = train_loss_1 / len(self.train_loader)
        self.train_loss_2 = train_loss_2 / len(self.train_loader)
        self.train_loss_3 = train_loss_3 / len(self.train_loader)
        self.train_loss_mse = train_loss_mse / len(self.train_loader)
        self.avg_train_loss = train_loss / len(self.train_loader)
        self.current_weight_3 = current_weight_3  # 存储当前先验权重

        # ---- 方差统计 (训练) 使用最后一个 batch 的 var ----
        try:
            with torch.no_grad():
                v = var.detach()  # 最后一个 batch 的 var
                v_min_cfg = self.config.VAR_MIN
                v_max_cfg = self.config.VAR_MAX
                eps_edge = 0.1
                ratio_at_min = (v <= v_min_cfg + eps_edge).float().mean().item()
                ratio_at_max = (v >= v_max_cfg - eps_edge).float().mean().item()
                self.var_stats_train = {
                    'min': float(v.min().item()),
                    'max': float(v.max().item()),
                    'mean': float(v.mean().item()),
                    'ratio_min': ratio_at_min,
                    'ratio_max': ratio_at_max,
                    'ratio_low': (v <= v_min_cfg + 0.1 * (v_max_cfg - v_min_cfg)).float().mean().item(),
                }
                # 记录方差直方图 (flatten 后) 到 tensorboard (每 epoch)
                try:
                    self.writer.add_histogram('Var/train', v.flatten(), epoch+1)
                except Exception as _e:
                    if self.logger:
                        self.logger.warning(f"写入 Var/train 直方图失败: {_e}")
                # μ 统计（训练）
                m = mu.detach()
                mu_range = float(self.config.MU_RANGE)
                edge_thr = 0.95 * mu_range
                ratio_edge = (m.abs() >= edge_thr).float().mean().item()
                self.mu_stats_train = {
                    'min': float(m.min().item()),
                    'max': float(m.max().item()),
                    'mean': float(m.mean().item()),
                    'ratio_edge': ratio_edge,
                }
        except Exception:
            self.var_stats_train = None
            self.mu_stats_train = None

        
    def validate(self, epoch, epsilon=1e-6):
        self.x_net.eval()
        self.z_net.eval()
        self.o_net.eval()

        valid_loss = 0
        valid_loss_1, valid_loss_2, valid_loss_3 = 0.0, 0.0, 0.0
        valid_loss_mse = 0.0
        dice_total = 0
        iou_total = 0
        pixel_err_total = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[valid]", ncols=100):
                # 数据加载
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)
                prior = batch['prior'].to(self.config.DEVICE, dtype=torch.float32)

                # 前向传播
                img_4_chs, mu, var, pi, d1, d0 = forward_pass(image=image,
                                                              label=label,
                                                              prior=prior,
                                                              unet=self.unet,
                                                              x_net=self.x_net,
                                                              z_net=self.z_net,
                                                              o_net=self.o_net,
                                                              reg_net=self.reg_net, 
                                                              config=self.config, 
                                                              epoch=epoch, 
                                                              epsilon=epsilon)
                # 计算损失 (添加异常处理)
                try:
                    out = self.criterion(input=img_4_chs,
                                         mu=mu,
                                         var=var,
                                         pi=pi,
                                         alpha=d1,
                                         prior=d0,
                                         epoch=epoch,
                                         total_epochs=self.config.EPOCHS)
                    loss = out['total']
                    loss_1 = out['recon']
                    loss_2 = out['kl_pi']
                    loss_3 = out['kl_dir']
                    loss_mse = out['loss_mse']
                    self.current_weight_2 = out['weight_2']
                    self.current_weight_3 = out['weight_3']

                    if not torch.isfinite(loss):
                        if self.logger:
                            self.logger.warning(f"验证中检测到无效损失值: {loss.item()}, 跳过此batch")
                        continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"验证计算损失时发生错误: {e}")
                    continue

                # 累加损失
                valid_loss_1 += loss_1.item()
                valid_loss_2 += loss_2.item()
                valid_loss_3 += loss_3.item()
                valid_loss_mse += loss_mse.item()
                valid_loss += loss.item()

                # 计算评价指标
                pred_cls = torch.argmax(pi, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()

                dice_total += dice_coefficient(pred_cls, label_np, self.config.GMM_NUM, background=True)
                iou_total += iou_score(pred_cls, label_np, self.config.GMM_NUM, background=True)
                pixel_err_total += pixel_error(pred_cls, label_np, self.config.GMM_NUM, background=True)

    
        self.valid_loss_1 = valid_loss_1 / len(self.valid_loader)
        self.valid_loss_2 = valid_loss_2 / len(self.valid_loader)
        self.valid_loss_3 = valid_loss_3 / len(self.valid_loader)
        self.valid_loss_mse = valid_loss_mse / len(self.valid_loader)
        self.avg_valid_loss = valid_loss / len(self.valid_loader)
        self.avg_dice = dice_total / len(self.valid_loader)
        self.avg_iou = iou_total / len(self.valid_loader)
        self.avg_pixel_err = pixel_err_total / len(self.valid_loader)

        # 可视化一个batch的结果
        i = 0
        self.mu_show = mu.detach().cpu().numpy()[i]
        self.var_show = var.detach().cpu().numpy()[i]
        self.pi_show = pi.detach().cpu().numpy()[i]
        self.d1_show = d1.detach().cpu().numpy()[i]
        self.d0_show = d0.detach().cpu().numpy()[i]
        self.image_show = image[i].squeeze().detach().cpu().numpy()
        self.label_show = label_np[i]
        self.pred_show = pred_cls[i]

        # ---- 方差统计 (验证) 使用最后一个 batch 的 var ----
        try:
            with torch.no_grad():
                v = var.detach()
                v_min_cfg = self.config.VAR_MIN
                v_max_cfg = self.config.VAR_MAX
                eps_edge = 0.1
                ratio_at_min = (v <= v_min_cfg + eps_edge).float().mean().item()
                ratio_at_max = (v >= v_max_cfg - eps_edge).float().mean().item()
                self.var_stats_test = {
                    'min': float(v.min().item()),
                    'max': float(v.max().item()),
                    'mean': float(v.mean().item()),
                    'ratio_min': ratio_at_min,
                    'ratio_max': ratio_at_max,
                    'ratio_low': (v <= v_min_cfg + 0.1 * (v_max_cfg - v_min_cfg)).float().mean().item(),
                }
                try:
                    self.writer.add_histogram('Var/valid', v.flatten(), epoch+1)
                except Exception as _e:
                    if self.logger:
                        self.logger.warning(f"写入 Var/valid直方图失败: {_e}")
                # μ 统计（验证）
                m = mu.detach()
                mu_range = float(self.config.MU_RANGE)
                edge_thr = 0.95 * mu_range
                ratio_edge = (m.abs() >= edge_thr).float().mean().item()
                self.mu_stats_test = {
                    'min': float(m.min().item()),
                    'max': float(m.max().item()),
                    'mean': float(m.mean().item()),
                    'ratio_edge': ratio_edge,
                }
        except Exception:
            self.var_stats_test = None
            self.mu_stats_test = None


    def test(self):
        z_net_weights = torch.load("checkpoints/unet/z_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        self.z_net.load_state_dict(z_net_weights)
        self.unet.load_state_dict(unet_weights)
        self.z_net.eval()
        self.unet.eval()

        test_dice = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[test]", ncols=100):
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)

                img_4_chs = self.unet(image)
                img_4_chs = standardize_features(img_4_chs)
                post = self.z_net(img_4_chs)
                pred = torch.argmax(post, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()
                test_dice += dice_coefficient(pred, label_np, self.config.GMM_NUM, background=True)

        test_dice /= len(self.test_loader)
        self.logger.info(f"[Test] Dice: {test_dice:.4f}")


    def print_log_info(self, epoch):
        if self.logger:
            self.logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}]")
            self.logger.info(
                f"[Train] Loss 1: {self.train_loss_1:.4f}  "
                f"Loss 2: {self.train_loss_2:.4f}  "
                f"Loss 3: {self.train_loss_3:.4f}  "
                f"Loss Total: {self.avg_train_loss:.4f}  "
                f"Loss MSE: {self.train_loss_mse:.4f}  "
                f"Weight_2: {self.current_weight_2:.4f}  "
                f"Weight_3: {self.current_weight_3:.4f}"
            )

            self.logger.info(
                f"[Valid] Loss 1: {self.valid_loss_1:.4f}  "
                f"Loss 2: {self.valid_loss_2:.4f}  "
                f"Loss 3: {self.valid_loss_3:.4f}  "
                f"Loss Total: {self.avg_valid_loss:.4f}  "
                f"Loss MSE: {self.valid_loss_mse:.4f}"
            )

            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"IoU: {self.avg_iou:.4f}  "
                f"Pixel Error: {self.avg_pixel_err:.4f}"
            )

            self.logger.info(f"Current best dice: {self.config.BEST_DICE:.4f}")

            # 打印方差统计
            if getattr(self, 'var_stats_train', None):
                vs_tr = self.var_stats_train
                self.logger.info(
                    f"Var[train] min={vs_tr['min']:.4f} max={vs_tr['max']:.4f} mean={vs_tr['mean']:.4f} "
                    f"ratio_min={vs_tr['ratio_min']:.3f} ratio_low={vs_tr['ratio_low']:.3f} ratio_max={vs_tr['ratio_max']:.3f}"
                )
            if getattr(self, 'var_stats_test', None):
                vs_te = self.var_stats_test
                self.logger.info(
                    f"Var[valid] min={vs_te['min']:.4f} max={vs_te['max']:.4f} mean={vs_te['mean']:.4f} "
                    f"ratio_min={vs_te['ratio_min']:.3f} ratio_low={vs_te['ratio_low']:.3f} ratio_max={vs_te['ratio_max']:.3f}"
                )
            if getattr(self, 'mu_stats_train', None):
                ms_tr = self.mu_stats_train
                self.logger.info(
                    f"Mu[train] min={ms_tr['min']:.4f} max={ms_tr['max']:.4f} mean={ms_tr['mean']:.4f} "
                    f"ratio_edge(|mu|>{0.95:.2f}*R)={ms_tr['ratio_edge']:.3f}"
                )
            if getattr(self, 'mu_stats_test', None):
                ms_te = self.mu_stats_test
                self.logger.info(
                    f"Mu[valid] min={ms_te['min']:.4f} max={ms_te['max']:.4f} mean={ms_te['mean']:.4f} "
                    f"ratio_edge(|mu|>{0.95:.2f}*R)={ms_te['ratio_edge']:.3f}"
                )


    def save_checkpoints(self, epoch):
        checkpoints_dir = os.path.join(self.config.CHECKPOINTS_DIR, "unet")
        os.makedirs(checkpoints_dir, exist_ok=True)
        if self.avg_dice > self.config.BEST_DICE:
            self.config.BEST_DICE = self.avg_dice
            self.config.BEST_EPOCH = epoch + 1
            torch.save(self.x_net.state_dict(), os.path.join(checkpoints_dir, f"x_best.pth"))
            torch.save(self.z_net.state_dict(), os.path.join(checkpoints_dir, f"z_best.pth"))
            torch.save(self.o_net.state_dict(), os.path.join(checkpoints_dir, f"o_best.pth"))
            self.logger.info(f"Saved best model at epoch {epoch+1} with Dice: {self.avg_dice:.4f}")


    def visualize(self, epoch):
        create_visualization(
            image_show=getattr(self, 'image_show', None),
            label_show=getattr(self, 'label_show', None),
            mu_show=getattr(self, 'mu_show', None),
            var_show=getattr(self, 'var_show', None),
            pi_show=getattr(self, 'pi_show', None),
            d1_show=getattr(self, 'd1_show', None),
            d0_show=getattr(self, 'd0_show', None),
            pred_show=getattr(self, 'pred_show', None),
            output_dir=self.output_dir,
            epoch=epoch,
            logger=self.logger,
        )

    def add_tensorboard_scalars(self, epoch):
        self.writer.add_scalars("loss1", {
            "train": self.train_loss_1,
            "valid": self.valid_loss_1
        }, epoch+1)
        self.writer.add_scalars("loss2", {
            "train": self.train_loss_2,
            "valid": self.valid_loss_2
        }, epoch+1)
        self.writer.add_scalars("loss3", {
            "train": self.train_loss_3,
            "valid": self.valid_loss_3
        }, epoch+1)
        self.writer.add_scalars("loss_total", {
            "train": self.avg_train_loss,
            "valid": self.avg_valid_loss
        }, epoch+1)
        self.writer.add_scalars("Metrics", {
            "dice": self.avg_dice, 
            "iou": self.avg_iou, 
            "pixel_err": self.avg_pixel_err
        }, epoch+1)

        # 记录当前学习率（主分组）
        if len(self.optimizer.param_groups) > 0:
            self.writer.add_scalar("LR/current_group0", self.optimizer.param_groups[0]['lr'], epoch+1)


    def cleanup(self, train=True):
        if train:
            self.writer.close()
            self.logger.info("Best model saved at epoch {}, with Dice: {:.4f}".format(self.config.BEST_EPOCH, self.config.BEST_DICE))
            self.logger.info("✅Training completed.")
        else:
            self.logger.info("✅Testing completed.")
        self.logger.removeHandler(self.console_handler)
        self.logger.removeHandler(self.file_handler)


    def run(self, train=True):
        self.setup(train=train)
        if train:
            for epoch in range(self.config.EPOCHS):
                self.train_one_epoch(epoch)
                self.validate(epoch)
                self.print_log_info(epoch)
                # 只有在 warmup 完成后再使用调度器
                if epoch >= self.config.WARMUP_EPOCHS:
                    self.scheduler.step(self.avg_valid_loss)
                self.save_checkpoints(epoch)
                self.visualize(epoch)
                self.add_tensorboard_scalars(epoch)
        self.test()
        self.cleanup(train=train)



if __name__ == "__main__":
    # 获取配置
    config = get_config()
    trainer = Trainer(config)
    trainer.run(train=False)

    
