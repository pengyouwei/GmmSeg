import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
from data.transform import apply_rotate_transform
from utils.loss import GmmLoss
from utils.train_utils import forward_pass, compute_responsibilities
from utils.visualizer import create_visualization
from utils.metrics import dice_coefficient, iou_score, pixel_error
from utils.dataloader import get_loaders
from utils.train_utils import standardize_features, process_gmm_parameters
from models.unet import UNet
from models.regnet import RR_ResNet
from models.align_net import Align_ResNet
from config import Config, get_config
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("Train GMM Segmentation")

    def setup(self, train=True):
        random.seed(42)
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
        self.align_net = Align_ResNet(input_channels=1).to(self.config.DEVICE)


        # 加载预训练权重 (修复: 启用所有预训练权重的加载)
        try:
            x_net_weights = torch.load("checkpoints/unet/x_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            z_net_weights = torch.load("checkpoints/unet/z_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            o_net_weights = torch.load("checkpoints/unet/o_pretrained_acdc.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            reg_net_weights = torch.load("checkpoints/regnet/regnet_prior_4chs.pth", 
                                         map_location=self.config.DEVICE, 
                                          weights_only=True)
            align_net_weights = torch.load("checkpoints/align_net/align_best.pth",
                                           map_location=self.config.DEVICE,
                                           weights_only=True)
            unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                      map_location=self.config.DEVICE, 
                                      weights_only=True)
            
            # 加载权重到模型
            # self.x_net.load_state_dict(x_net_weights)
            # self.z_net.load_state_dict(z_net_weights)
            self.o_net.load_state_dict(o_net_weights)
            self.reg_net.load_state_dict(reg_net_weights)
            self.unet.load_state_dict(unet_weights)
            self.align_net.load_state_dict(align_net_weights)
            
            # 固定预训练模型的参数
            self.reg_net.eval()
            self.unet.eval()
            self.align_net.eval()
            
            # 冻结UNet和reg_net的参数以避免过拟合
            for param in self.reg_net.parameters():
                param.requires_grad = False
            for param in self.unet.parameters():
                param.requires_grad = False
            for param in self.align_net.parameters():
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

        train_loss = 0.0
        train_loss_1, train_loss_2, train_loss_3 = 0.0, 0.0, 0.0
        train_loss_mu = 0.0
        train_loss_var = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[train]", ncols=100):
            # 数据加载
            image = batch["image"].to(device=self.config.DEVICE, dtype=torch.float32)
            label = batch["label"].to(device=self.config.DEVICE, dtype=torch.float32)
            prior = batch["prior"].to(device=self.config.DEVICE, dtype=torch.float32)
            label_prior = batch["label_prior"].to(device=self.config.DEVICE, dtype=torch.float32)

            if self.config.USE_LABEL_PRIOR:
                prior = label_prior  # 直接使用标签先验进行训练

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            forward_pass_result = forward_pass(image=image,
                                               label=label,
                                               prior=prior,
                                               unet=self.unet,
                                               x_net=self.x_net,
                                               z_net=self.z_net,
                                               o_net=self.o_net,
                                               reg_net=self.reg_net,
                                               align_net=self.align_net,
                                               config=self.config, 
                                               epoch=epoch, 
                                               epsilon=epsilon)
            img_4_chs = forward_pass_result["img_4_chs"]
            label = forward_pass_result["label"]
            mu = forward_pass_result["mu"]
            var = forward_pass_result["var"]
            pi = forward_pass_result["pi"]
            d1 = forward_pass_result["d1"]
            d0 = forward_pass_result["d0"]

            # 计算损失 (添加异常处理)
            try:
                loss_out = self.criterion(input=img_4_chs,
                                          mu=mu,
                                          var=var,
                                          pi=pi,
                                          alpha=d1,
                                          prior=d0,
                                          epoch=epoch,
                                          total_epochs=self.config.EPOCHS)
                loss = loss_out['total']
                loss_1 = loss_out['recon']
                mu_component = loss_out['recon_mu']
                var_component = loss_out['recon_var']
                loss_2 = loss_out['kl_pi']
                loss_3 = loss_out['kl_dir']

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
            train_loss_mu += mu_component.item()
            train_loss_var += var_component.item()
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
        self.avg_train_loss = train_loss / len(self.train_loader)
        self.train_loss_mu = train_loss_mu / len(self.train_loader)
        self.train_loss_var = train_loss_var / len(self.train_loader)

    def validate(self, epoch, epsilon=1e-6):
        self.x_net.eval()
        self.z_net.eval()
        self.o_net.eval()

        valid_loss = 0
        valid_loss_1, valid_loss_2, valid_loss_3 = 0.0, 0.0, 0.0
        valid_loss_mu, valid_loss_var = 0.0, 0.0
        dice_total = 0
        iou_total = 0
        pixel_err_total = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[valid]", ncols=100):
                # 数据加载
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)
                prior = batch['prior'].to(self.config.DEVICE, dtype=torch.float32)
                label_prior = batch['label_prior'].to(self.config.DEVICE, dtype=torch.float32)

                if self.config.USE_LABEL_PRIOR:
                    prior = label_prior  # 直接使用标签先验进行验证

                # 前向传播
                forward_pass_result = forward_pass(image=image,
                                                    label=label,
                                                    prior=prior,
                                                    unet=self.unet,
                                                    x_net=self.x_net,
                                                    z_net=self.z_net,
                                                    o_net=self.o_net,
                                                    reg_net=self.reg_net, 
                                                    align_net=self.align_net,
                                                    config=self.config, 
                                                    epoch=epoch, 
                                                    epsilon=epsilon)
                img_4_chs = forward_pass_result["img_4_chs"]
                label = forward_pass_result["label"]
                mu = forward_pass_result["mu"]
                var = forward_pass_result["var"]
                pi = forward_pass_result["pi"]
                d1 = forward_pass_result["d1"]
                d0 = forward_pass_result["d0"]
                scale_pred = forward_pass_result["scale_pred"]
                tx_pred = forward_pass_result["tx_pred"]
                ty_pred = forward_pass_result["ty_pred"]


                # 计算损失 (添加异常处理)
                try:
                    loss_out = self.criterion(input=img_4_chs,
                                         mu=mu,
                                         var=var,
                                         pi=pi,
                                         alpha=d1,
                                         prior=d0,
                                         epoch=epoch,
                                         total_epochs=self.config.EPOCHS)
                    loss = loss_out['total']
                    loss_1 = loss_out['recon']
                    mu_component = loss_out['recon_mu']
                    var_component = loss_out['recon_var']
                    loss_2 = loss_out['kl_pi']
                    loss_3 = loss_out['kl_dir']
                    self.current_weight_1 = loss_out['weight_1']
                    self.current_weight_2 = loss_out['weight_2']
                    self.current_weight_3 = loss_out['weight_3']

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
                valid_loss_mu += mu_component.item()
                valid_loss_var += var_component.item()
                valid_loss += loss.item()

                # 计算评价指标
                # pred_cls = torch.argmax(pi, dim=1).detach().cpu().numpy()
                r = compute_responsibilities(img_4_chs, mu, var, alpha=d1)
                pred_cls = torch.argmax(r, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()

                dice_total += dice_coefficient(pred_cls, label_np, self.config.CLASS_NUM, background=False)
                iou_total += iou_score(pred_cls, label_np, self.config.CLASS_NUM, background=False)
                pixel_err_total += pixel_error(pred_cls, label_np, self.config.CLASS_NUM, background=False)

        self.valid_loss_1 = valid_loss_1 / len(self.valid_loader)
        self.valid_loss_2 = valid_loss_2 / len(self.valid_loader)
        self.valid_loss_3 = valid_loss_3 / len(self.valid_loader)
        self.valid_loss_mu = valid_loss_mu / len(self.valid_loader)
        self.valid_loss_var = valid_loss_var / len(self.valid_loader)
        self.avg_valid_loss = valid_loss / len(self.valid_loader)
        self.avg_dice = dice_total / len(self.valid_loader)
        self.avg_iou = iou_total / len(self.valid_loader)
        self.avg_pixel_err = pixel_err_total / len(self.valid_loader)

        # 随机可视化最后一个batch中的一个样本
        i = random.randint(0, image.shape[0]-1)
        self.pi_show = pi.detach().cpu().numpy()[i]
        self.d1_show = d1.detach().cpu().numpy()[i]
        self.d0_show = d0.detach().cpu().numpy()[i]
        self.image_show = image[i].squeeze().detach().cpu().numpy()
        self.label_show = label_np[i]
        self.pred_show = pred_cls[i]
        # 记录配准参数
        self.scale_pred = scale_pred[i].item()
        self.tx_pred = tx_pred[i].item()
        self.ty_pred = ty_pred[i].item()
        # 记录高斯参数范围
        self.mu_min = mu.min().item()
        self.mu_max = mu.max().item()
        self.mu_mean = mu.mean().item()
        self.var_min = var.min().item()
        self.var_max = var.max().item()
        self.var_mean = var.mean().item()
        # 记录浓度参数范围
        self.d1_min = d1.min().item()
        self.d1_max = d1.max().item()
        self.d0_min = d0.min().item()
        self.d0_max = d0.max().item()

    def test(self):
        if len(self.test_loader) == 0:
            return

        x_net_weights = torch.load("checkpoints/unet/x_best.pth",
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        z_net_weights = torch.load("checkpoints/unet/z_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        o_net_weights = torch.load("checkpoints/unet/o_best.pth",
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        align_net_weights = torch.load("checkpoints/align_net/align_best.pth", 
                                        map_location=self.config.DEVICE, 
                                        weights_only=True)
        self.x_net.load_state_dict(x_net_weights)
        self.z_net.load_state_dict(z_net_weights)
        self.o_net.load_state_dict(o_net_weights)
        self.unet.load_state_dict(unet_weights)
        self.align_net.load_state_dict(align_net_weights)
        self.x_net.eval()
        self.z_net.eval()
        self.o_net.eval()
        self.unet.eval()
        self.align_net.eval()

        test_dice = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[test]", ncols=100):
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)

                if self.config.DATASET == "SCD":
                    angle = self.align_net(image)  # 预测旋转角度
                    angle = torch.clamp(angle, config.ROTATE_RANGE[0], config.ROTATE_RANGE[1])
                    image = apply_rotate_transform(image, -angle[:, 0])  # 逆向旋转对齐
                    label = apply_rotate_transform(label.float(), -angle[:, 0], mode='nearest').long()

                img_4_chs = self.unet(image)
                img_4_chs = standardize_features(img_4_chs)
                output_x = self.x_net(img_4_chs)  # mu, var
                output_z = self.z_net(img_4_chs)  # pi
                output_o = self.o_net(img_4_chs)  # d
                # pred = torch.argmax(output_z, dim=1).detach().cpu().numpy()
                mu, var, pi, alpha = process_gmm_parameters(output_x=output_x, 
                                                            output_z=output_z, 
                                                            output_o=output_o, 
                                                            config=self.config, 
                                                            epsilon=1e-6)
                r = compute_responsibilities(x=img_4_chs, mu=mu, var=var, alpha=alpha)
                pred = torch.argmax(r, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()
                test_dice += dice_coefficient(pred, label_np, self.config.CLASS_NUM, background=False)

        
        test_dice /= len(self.test_loader)
        self.logger.info(f"[Test] Dice: {test_dice:.4f}")
    


    def print_log_info(self, epoch):
        if self.logger:
            self.logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}]")

            self.logger.info(
                f"[Train] Total: {self.avg_train_loss:.4f}  "
                f"Loss1: {self.train_loss_1:.4f}  (mu: {self.train_loss_mu:.4f}  var: {self.train_loss_var:.4f})  "
                f"Loss2: {self.train_loss_2:.4f}  "
                f"Loss3: {self.train_loss_3:.4f}  "
                f"(w1: {self.current_weight_1:.4f}, w2: {self.current_weight_2:.4f}, w3: {self.current_weight_3:.4f})"
            )
            self.logger.info(
                f"[Valid] Total: {self.avg_valid_loss:.4f}  "
                f"Loss1: {self.valid_loss_1:.4f}  (mu: {self.valid_loss_mu:.4f}  var: {self.valid_loss_var:.4f})  "
                f"Loss2: {self.valid_loss_2:.4f}  "
                f"Loss3: {self.valid_loss_3:.4f}  "
            )

            self.logger.info(
                f"mu range: [{self.mu_min:.4f}, {self.mu_max:.4f}]  (mean: {self.mu_mean:.4f})  "
                f"var range: [{self.var_min:.4f}, {self.var_max:.4f}]  (mean: {self.var_mean:.4f})  "
                f"d1 range: [{self.d1_min:.4f}, {self.d1_max:.4f}]  "
                f"d0 range: [{self.d0_min:.4f}, {self.d0_max:.4f}]  "
            )

            self.logger.info(
                f"scale: {self.scale_pred:.2f}  "
                f"tx: {self.tx_pred:.2f}  "
                f"ty: {self.ty_pred:.2f}"
            )

            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"IoU: {self.avg_iou:.4f}  "
                f"Pixel Error: {self.avg_pixel_err:.4f}"
            )

            self.logger.info(f"Current best dice: {self.config.BEST_DICE:.4f}, at epoch {self.config.BEST_EPOCH}")


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
            pi_show=getattr(self, 'pi_show', None),
            d1_show=getattr(self, 'd1_show', None),
            d0_show=getattr(self, 'd0_show', None),
            pred_show=getattr(self, 'pred_show', None),
            output_dir=self.output_dir,
            epoch=epoch,
            logger=self.logger,
        )

    def add_tensorboard_scalars(self, epoch):
        # loss1 各子项
        self.writer.add_scalars("loss1", {
            "train/total": self.train_loss_1,
            "train/mu": self.train_loss_mu,
            "train/var": self.train_loss_var,
            "valid/total": self.valid_loss_1,
            "valid/mu": self.valid_loss_mu,
            "valid/var": self.valid_loss_var,
        }, epoch+1)

        # loss2
        self.writer.add_scalars("loss2", {
            "train": self.train_loss_2,
            "valid": self.valid_loss_2
        }, epoch+1)

        # loss3
        self.writer.add_scalars("loss3", {
            "train": self.train_loss_3,
            "valid": self.valid_loss_3,
        }, epoch+1)

        # total_loss
        self.writer.add_scalars("total_loss", {
            "train": self.avg_train_loss,
            "valid": self.avg_valid_loss
        }, epoch+1)

        # metrics
        self.writer.add_scalar("metrics/dice", self.avg_dice, epoch+1)
        self.writer.add_scalar("metrics/iou", self.avg_iou, epoch+1)
        self.writer.add_scalar("metrics/pixel_err", self.avg_pixel_err, epoch+1)

        # 学习率
        if len(self.optimizer.param_groups) > 0:
            self.writer.add_scalar("learning_rate/group0", self.optimizer.param_groups[0]['lr'], epoch+1)



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
                self.scheduler.step(self.avg_valid_loss)
                self.save_checkpoints(epoch)
                self.print_log_info(epoch)
                self.visualize(epoch)
                self.add_tensorboard_scalars(epoch)
        self.test()
        self.cleanup(train=train)



if __name__ == "__main__":
    # 获取配置
    config = get_config()
    trainer = Trainer(config)
    trainer.run(train=True)

