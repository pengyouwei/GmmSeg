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
from utils.dataloader import get_loaders, get_dirichlet_priors
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

    def setup(self):
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
        self.train_loader, self.test_loader = get_loaders(self.config)

        # 加载Dirichlet先验分布
        self.dirichlet_priors = get_dirichlet_priors(self.config)

        # 定义模型
        self.unet = UNet(self.config.IN_CHANNELS, self.config.FEATURE_NUM).to(self.config.DEVICE)  # UNet 用于提取特征
        self.x_net = UNet(self.config.FEATURE_NUM, self.config.FEATURE_NUM*self.config.GMM_NUM*2).to(self.config.DEVICE)  # mu, var
        self.z_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # pi
        self.o_net = UNet(self.config.FEATURE_NUM, self.config.GMM_NUM).to(self.config.DEVICE)  # d
        self.reg_net = RR_ResNet(input_channels=self.config.GMM_NUM).to(self.config.DEVICE)


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
            reg_net_weights = torch.load("checkpoints/regnet/dirichlet_registration.pth", 
                                         map_location=self.config.DEVICE, 
                                          weights_only=True)
            unet_weights = torch.load("checkpoints/unet/feature_extraction.pth", 
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
                
            if self.logger:
                self.logger.info("✅ 成功加载所有预训练权重")
                
        except FileNotFoundError as e:
            if self.logger:
                self.logger.warning(f"⚠️ 预训练权重文件未找到: {e}")
                self.logger.warning("将使用随机初始化的权重进行训练")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ 加载预训练权重失败: {e}")
                self.logger.warning("将使用随机初始化的权重进行训练")

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
                                                              min_lr=1e-7,   # 最小学习率
                                                              )
        # Warmup 基础学习率缓存
        self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]

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
        train_loss1, train_loss2, train_loss3 = 0.0, 0.0, 0.0
        train_loss_mse = 0.0
        current_weight_3 = 0.0  # 记录当前的先验权重

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[train]", ncols=100):
            # 数据加载
            image = batch["image"]["data"].to(device=self.config.DEVICE, dtype=torch.float32)
            label = batch["label"]["data"].to(device=self.config.DEVICE, dtype=torch.float32)
            prior = batch["prior"]["data"].to(device=self.config.DEVICE, dtype=torch.float32)
            slice_info = batch["image"]["slice"]
            num_of_slice_info = batch["image"]["slice_num"]

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            image_4_features, mu, var, pi, d1, d0 = forward_pass(image=image, 
                                                                 unet=self.unet, 
                                                                 x_net=self.x_net, 
                                                                 z_net=self.z_net, 
                                                                 o_net=self.o_net, 
                                                                 reg_net=self.reg_net,
                                                                 dirichlet_priors=self.dirichlet_priors, 
                                                                 slice_info=slice_info, 
                                                                 num_of_slice_info=num_of_slice_info, 
                                                                 config=self.config, 
                                                                 epoch=epoch, 
                                                                 epsilon=epsilon)
            # d0 = prior
            # 计算损失 (添加异常处理)
            try:
                out = self.criterion(input=image_4_features,
                                      mu=mu,
                                      var=var,
                                      pi=pi,
                                      d=d1,
                                      d0=d0,
                                      epoch=epoch,
                                      total_epochs=self.config.EPOCHS)
                loss = out['total']
                loss_1 = out['recon']
                loss_2 = out.get('kl_pi', torch.tensor(0.0, device=loss.device))
                loss_3 = out.get('kl_dir', torch.tensor(0.0, device=loss.device))
                loss_mse = out.get('loss_mse', torch.tensor(0.0, device=loss.device))
                current_weight_3 = out['w_dir'].item() if torch.is_tensor(out['w_dir']) else float(out['w_dir'])
                self.current_weight_3 = current_weight_3

                if not torch.isfinite(loss):
                    if self.logger:
                        self.logger.warning(f"训练中检测到无效损失值: {loss.item()}, 跳过此batch")
                    continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"计算损失时发生错误: {e}")
                continue

            # 累加损失
            train_loss1 += loss_1.item()
            train_loss2 += loss_2.item()
            train_loss3 += loss_3.item()
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

        # 记录梯度范数（最后一个 step 后的梯度）
        def grad_norm(module):
            total = 0.0
            for p in module.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    total += p.grad.data.norm(2).item() ** 2
            return math.sqrt(total) if total > 0 else 0.0
        self.grad_norms = {
            'x_net': grad_norm(self.x_net),
            'z_net': grad_norm(self.z_net),
            'o_net': grad_norm(self.o_net)
        }

        # 计算平均损失值
        self.train_loss1 = train_loss1 / len(self.train_loader)
        self.train_loss2 = train_loss2 / len(self.train_loader)
        self.train_loss3 = train_loss3 / len(self.train_loader)
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

        test_loss = 0
        test_loss1, test_loss2, test_loss3 = 0.0, 0.0, 0.0
        test_loss_mse = 0.0
        dice_total = 0
        iou_total = 0
        pixel_err_total = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[test ]", ncols=100):
                # 数据加载
                image = batch['image']["data"].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label']["data"].to(self.config.DEVICE, dtype=torch.float32)
                prior = batch['prior']["data"].to(self.config.DEVICE, dtype=torch.float32)
                slice_info = batch["image"]["slice"]
                num_of_slice_info = batch["image"]["slice_num"]

                # 前向传播
                image_4_features, mu, var, pi, d1, d0 = forward_pass(image=image, 
                                                                     unet=self.unet, 
                                                                     x_net=self.x_net, 
                                                                     z_net=self.z_net, 
                                                                     o_net=self.o_net, 
                                                                     reg_net=self.reg_net,
                                                                     dirichlet_priors=self.dirichlet_priors, 
                                                                     slice_info=slice_info, 
                                                                     num_of_slice_info=num_of_slice_info, 
                                                                     config=self.config, 
                                                                     epoch=epoch, 
                                                                     epsilon=epsilon)
                # d0 = prior
                # 计算损失 (添加异常处理)
                try:
                    out = self.criterion(input=image_4_features,
                                          mu=mu,
                                          var=var,
                                          pi=pi,
                                          d=d1,
                                          d0=d0,
                                          epoch=epoch,
                                          total_epochs=self.config.EPOCHS)
                    loss = out['total']
                    loss_1 = out['recon']
                    loss_2 = out.get('kl_pi', torch.tensor(0.0, device=loss.device))
                    loss_3 = out.get('kl_dir', torch.tensor(0.0, device=loss.device))
                    loss_mse = out.get('loss_mse', torch.tensor(0.0, device=loss.device))
                    weight_3 = out['w_dir']
                    self.current_weight_3 = weight_3.item() if torch.is_tensor(weight_3) else float(weight_3)
                    pred_basis = out['pi']  # 变分后验概率π_{ik}

                    if not torch.isfinite(loss):
                        if self.logger:
                            self.logger.warning(f"验证中检测到无效损失值: {loss.item()}, 跳过此batch")
                        continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"验证计算损失时发生错误: {e}")
                    continue

                # 累加损失
                test_loss1 += loss_1.item()
                test_loss2 += loss_2.item()
                test_loss3 += loss_3.item()
                test_loss_mse += loss_mse.item()
                test_loss += loss.item()

                # 计算评价指标
                pred_cls = torch.argmax(pred_basis, dim=1).cpu().numpy()
                label_np = torch.squeeze(label, dim=1).cpu().numpy()

                dataset_name = os.path.basename(self.config.DATASET)
                dice_total += dice_coefficient(pred_cls, label_np, self.config.GMM_NUM, dataset_name, background=True)
                iou_total += iou_score(pred_cls, label_np, self.config.GMM_NUM, dataset_name, background=True)
                pixel_err_total += pixel_error(pred_cls, label_np, self.config.GMM_NUM, dataset_name, background=True)


        self.test_loss1 = test_loss1 / len(self.test_loader)
        self.test_loss2 = test_loss2 / len(self.test_loader)
        self.test_loss3 = test_loss3 / len(self.test_loader)
        self.test_loss_mse = test_loss_mse / len(self.test_loader)
        self.avg_test_loss = test_loss / len(self.test_loader)
        self.avg_dice = dice_total / len(self.test_loader)
        self.avg_iou = iou_total / len(self.test_loader)
        self.avg_pixel_err = pixel_err_total / len(self.test_loader)

        # 可视化最后一个batch的结果
        i = 0
        self.mu_show = mu.cpu().detach().numpy()[i]
        self.var_show = var.cpu().detach().numpy()[i]
        self.pi_show = pi.cpu().detach().numpy()[i]
        self.d0_show = d0.cpu().detach().numpy()[i]
        self.d1_show = d1.cpu().detach().numpy()[i]
        self.image_show = image[i].squeeze().cpu().detach().numpy()
        self.feature_show = image_4_features[i].squeeze().cpu().detach().numpy()
        self.label_show = label_np[i]
        self.pred_show = pred_cls[i]
        self.slice_id = slice_info[i].item()

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
                    self.writer.add_histogram('Var/test', v.flatten(), epoch+1)
                except Exception as _e:
                    if self.logger:
                        self.logger.warning(f"写入 Var/test 直方图失败: {_e}")
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


    def print_log_info(self, epoch):
        if self.logger:
            self.logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}]")
            self.logger.info(
                f"[Train] Loss 1: {self.train_loss1:.4f}  "
                f"Loss 2: {self.train_loss2:.4f}  "
                f"Loss 3: {self.train_loss3:.4f}  "
                f"Loss Total: {self.avg_train_loss:.4f}  "
                f"Loss MSE: {self.train_loss_mse:.4f}  "
                f"Weight_of_loss3: {self.current_weight_3:.4f}"
            )

            self.logger.info(
                f"[Test ] Loss 1: {self.test_loss1:.4f}  "
                f"Loss 2: {self.test_loss2:.4f}  "
                f"Loss 3: {self.test_loss3:.4f}  "
                f"Loss Total: {self.avg_test_loss:.4f}  "
                f"Loss MSE: {self.test_loss_mse:.4f}"
            )

            extra = ''
            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"IoU: {self.avg_iou:.4f}  "
                f"Pixel Error: {self.avg_pixel_err:.4f}"
            )

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
                    f"Var[test ] min={vs_te['min']:.4f} max={vs_te['max']:.4f} mean={vs_te['mean']:.4f} "
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
                    f"Mu[test ] min={ms_te['min']:.4f} max={ms_te['max']:.4f} mean={ms_te['mean']:.4f} "
                    f"ratio_edge(|mu|>{0.95:.2f}*R)={ms_te['ratio_edge']:.3f}"
                )


    def save_checkpoints(self, epoch):
        checkpoints_dir = os.path.join(self.config.CHECKPOINTS_DIR, "unet")
        os.makedirs(checkpoints_dir, exist_ok=True)
        if self.avg_dice > self.config.BEST_DICE:
            self.config.BEST_DICE = self.avg_dice
            torch.save(self.x_net.state_dict(), os.path.join(checkpoints_dir, f"x_best.pth"))
            torch.save(self.z_net.state_dict(), os.path.join(checkpoints_dir, f"z_best.pth"))
            torch.save(self.o_net.state_dict(), os.path.join(checkpoints_dir, f"o_best.pth"))
            self.logger.info(f"Saved best model at epoch {epoch+1} with Dice: {self.avg_dice:.4f}")


    def visualize(self, epoch):
        create_visualization(
            image_show=getattr(self, 'image_show', None),
            feature_show=getattr(self, 'feature_show', None),
            mu_show=getattr(self, 'mu_show', None),
            var_show=getattr(self, 'var_show', None),
            pi_show=getattr(self, 'pi_show', None),
            d0_show=getattr(self, 'd0_show', None),
            d1_show=getattr(self, 'd1_show', None),
            label_show=getattr(self, 'label_show', None),
            pred_show=getattr(self, 'pred_show', None),
            slice_id=getattr(self, 'slice_id', -1),
            output_dir=self.output_dir,
            epoch=epoch,
            logger=self.logger,
        )

    def add_tensorboard_scalars(self, epoch):
        self.writer.add_scalars("Train", {
            "loss1": self.train_loss1, 
            "loss2": self.train_loss2, 
            "loss3": self.train_loss3, 
            "total": self.avg_train_loss,
            "weight_of_loss3": self.current_weight_3,
            "entropy_pi": getattr(self.criterion, 'last_entropy', float('nan')) if hasattr(self.criterion, 'last_entropy') else float('nan'),
        }, epoch+1)
        self.writer.add_scalars("Test", {
            "loss1": self.test_loss1, 
            "loss2": self.test_loss2, 
            "loss3": self.test_loss3, 
            "total": self.avg_test_loss
        }, epoch+1)
        self.writer.add_scalars("Metrics", {
            "dice": self.avg_dice, 
            "iou": self.avg_iou, 
            "pixel_err": self.avg_pixel_err
        }, epoch+1)
        # 记录当前学习率（主分组）
        if len(self.optimizer.param_groups) > 0:
            self.writer.add_scalar("LR/current_group0", self.optimizer.param_groups[0]['lr'], epoch+1)


    def cleanup(self):
        self.writer.close()
        self.logger.removeHandler(self.console_handler)
        self.logger.removeHandler(self.file_handler)
        self.logger.info("Training completed.")


    def run(self):
        self.setup()
        for epoch in range(self.config.EPOCHS):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            self.print_log_info(epoch)
            # 只有在 warmup 完成后再使用调度器
            if epoch >= self.config.WARMUP_EPOCHS:
                self.scheduler.step(self.avg_test_loss)
            self.save_checkpoints(epoch)
            self.visualize(epoch)
            self.add_tensorboard_scalars(epoch)
        self.cleanup()



if __name__ == "__main__":
    # 获取配置
    config = get_config()
    trainer = Trainer(config)
    trainer.run()

    
