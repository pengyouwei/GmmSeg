import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.loss import GmmLoss, DirichletGmmLoss
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
        self.unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(config.DEVICE)  # UNet 用于提取特征
        self.x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM*config.GMM_NUM*2).to(config.DEVICE)  # mu, var
        # Dirichlet 模式下无需 z_net
        if not config.USE_DIRICHLET_MIX:
            self.z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(config.DEVICE)  # pi
        else:
            self.z_net = None
        self.o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(config.DEVICE)  # d
        self.reg_net = RR_ResNet(input_channels=config.GMM_NUM).to(config.DEVICE)

        # 加载预训练权重 (修复: 启用所有预训练权重的加载)
        try:
            x_net_weights = torch.load("checkpoints/unet/x_pretrained.pth", 
                                       map_location=config.DEVICE, 
                                       weights_only=True)
            z_net_weights = torch.load("checkpoints/unet/z_pretrained.pth", 
                                       map_location=config.DEVICE, 
                                       weights_only=True)
            o_net_weights = torch.load("checkpoints/unet/o_pretrained.pth", 
                                       map_location=config.DEVICE, 
                                       weights_only=True)
            reg_net_weights = torch.load("checkpoints/regnet/dirichlet_registration.pth", 
                                         map_location=config.DEVICE, 
                                         weights_only=True)
            unet_weights = torch.load("checkpoints/unet/feature_extraction.pth", 
                                      map_location=config.DEVICE, 
                                      weights_only=True)
            
            # 加载权重到模型
            # self.x_net.load_state_dict(x_net_weights)
            # self.z_net.load_state_dict(z_net_weights)
            self.o_net.load_state_dict(o_net_weights)
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
        if self.config.USE_DIRICHLET_MIX:
            self.criterion = DirichletGmmLoss(GMM_NUM=config.GMM_NUM, nll_warmup_epochs=self.config.NLL_WARMUP_EPOCHS).to(config.DEVICE)
        else:
            self.criterion = GmmLoss(GMM_NUM=config.GMM_NUM).to(config.DEVICE)
        
        # 使用不同的学习率策略
        opt_params = [
            {'params': self.x_net.parameters(), 'lr': config.LEARNING_RATE, 'weight_decay': 1e-5},
            {'params': self.o_net.parameters(), 'lr': config.LEARNING_RATE*0.5, 'weight_decay': 1e-5},
        ]
        if self.z_net is not None:
            opt_params.insert(1, {'params': self.z_net.parameters(), 'lr': config.LEARNING_RATE, 'weight_decay': 1e-5})
        self.optimizer = optim.Adam(opt_params, eps=1e-8, betas=(0.9, 0.999))

        # 改进的学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode='min', 
                                                              factor=0.7,    # 更温和的衰减
                                                              patience=5,    # 更长的等待时间
                                                              min_lr=1e-7,   # 最小学习率
                                                              )

        # 创建日志记录器
        log_dir = os.path.join(self.config.LOGS_DIR, "tensorboard", self.time_str)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 创建输出目录
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, self.time_str)
        os.makedirs(self.output_dir, exist_ok=True)


    def train_one_epoch(self, epoch, epsilon=1e-6):
        self.x_net.train()
        if self.z_net is not None:
            self.z_net.train()
        self.o_net.train()

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
                if self.config.USE_DIRICHLET_MIX:
                    out = self.criterion(input=image_4_features,
                                          mu=mu,
                                          var=var,
                                          d=d1,
                                          d0=d0,
                                          epoch=epoch,
                                          total_epochs=self.config.EPOCHS)
                else:
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
            if self.z_net is not None:
                torch.nn.utils.clip_grad_norm_(self.z_net.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.o_net.parameters(), max_norm=5.0)

            # 更新参数
            self.optimizer.step()

        # 计算平均损失值
        self.train_loss1 = train_loss1 / len(self.train_loader)
        self.train_loss2 = train_loss2 / len(self.train_loader)
        self.train_loss3 = train_loss3 / len(self.train_loader)
        self.train_loss_mse = train_loss_mse / len(self.train_loader)
        self.avg_train_loss = train_loss / len(self.train_loader)
        self.current_weight_3 = current_weight_3  # 存储当前先验权重

        
    def validate(self, epoch, epsilon=1e-6):
        self.x_net.eval()
        if self.z_net is not None:
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
                    if self.config.USE_DIRICHLET_MIX:
                        out = self.criterion(input=image_4_features,
                                              mu=mu,
                                              var=var,
                                              d=d1,
                                              d0=d0,
                                              epoch=epoch,
                                              total_epochs=self.config.EPOCHS)
                    else:
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
                    pred_basis = out['post_prob']
                    used_nll = out.get('used_nll', None)
                    if used_nll is not None:
                        # 计算 posterior 熵 (用于日志与监控组件利用率)
                        with torch.no_grad():
                            pb = pred_basis.clamp_min(1e-8)
                            ent = -(pb * pb.log()).sum(dim=1).mean().item()
                        self.last_posterior_entropy = ent
                        self.last_used_nll = bool(used_nll)

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
        i = self.config.BATCH_SIZE - 1
        try:
            self.mu_show = mu.cpu().detach().numpy()[i]
            self.var_show = var.cpu().detach().numpy()[i]
            # 保存用于展示的概率图：Dirichlet 模式下为 posterior r；传统模式下为 gating π
            self.pi_show = pred_basis.cpu().detach().numpy()[i]
            self.d0_show = d0.cpu().detach().numpy()[i]
            self.d1_show = d1.cpu().detach().numpy()[i]
            self.image_show = image[i].squeeze().cpu().detach().numpy()
            self.feature_show = image_4_features[i].squeeze().cpu().detach().numpy()
            self.label_show = label_np[i]
            self.pred_show = pred_cls[i]
            self.slice_id = slice_info[i].item()
        except Exception:
            # 可能 batch 不满或索引越界，忽略可视化
            pass


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
            if self.config.USE_DIRICHLET_MIX:
                extra = f" | used_nll={getattr(self, 'last_used_nll', False)} | post_entropy={getattr(self, 'last_posterior_entropy', 0.0):.3f}"
            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"IoU: {self.avg_iou:.4f}  "
                f"Pixel Error: {self.avg_pixel_err:.4f}{extra}"
            )


    def save_checkpoints(self, epoch):
        checkpoints_dir = os.path.join(self.config.CHECKPOINTS_DIR, "unet")
        os.makedirs(checkpoints_dir, exist_ok=True)
        if self.avg_dice > self.config.BEST_DICE:
            self.config.BEST_DICE = self.avg_dice
            torch.save(self.x_net.state_dict(), os.path.join(checkpoints_dir, f"x_best.pth"))
            if self.z_net is not None:
                torch.save(self.z_net.state_dict(), os.path.join(checkpoints_dir, f"z_best.pth"))
            torch.save(self.o_net.state_dict(), os.path.join(checkpoints_dir, f"o_best.pth"))
            self.logger.info(f"Saved best model at epoch {epoch+1} with Dice: {self.avg_dice:.4f}")


    def visualize(self, epoch):
        create_visualization(image_show=self.image_show,
                             feature_show=self.feature_show,
                             mu_show=self.mu_show,
                             var_show=self.var_show,
                             pi_show=self.pi_show,
                             d0_show=self.d0_show,
                             d1_show=self.d1_show,
                             label_show=self.label_show,
                             pred_show=self.pred_show,
                             slice_id=self.slice_id,
                             output_dir=self.output_dir,
                             epoch=epoch, 
                             logger=self.logger)

    def add_tensorboard_scalars(self, epoch):
        self.writer.add_scalars("Train", {
            "loss1": self.train_loss1, 
            "loss2": self.train_loss2, 
            "loss3": self.train_loss3, 
            "total": self.avg_train_loss,
            "weight_of_loss3": self.current_weight_3
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
        if self.config.USE_DIRICHLET_MIX and hasattr(self, 'last_posterior_entropy'):
            self.writer.add_scalar("Dirichlet/posterior_entropy", self.last_posterior_entropy, epoch+1)
            self.writer.add_scalar("Dirichlet/used_nll", int(getattr(self, 'last_used_nll', False)), epoch+1)


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

    
