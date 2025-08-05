import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.loss import GmmSegLoss
from utils.train_utils import forward_pass
from utils.visualizer import create_visualization
from utils.calc_post_probs import calculate_posterior_probs
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
    def __init__(self, config: Config, logger: logging.Logger = None):
        self.config = config
        self.logger = logger

    def setup(self):
        # 加载数据集
        self.train_loader, self.test_loader = get_loaders(self.config)

        # 加载Dirichlet先验分布
        self.dirichlet_priors = get_dirichlet_priors(self.config)

        # 定义模型
        self.unet = UNet(config.IN_CHANNELS, config.FEATURE_NUM).to(config.DEVICE)  # UNet 用于提取特征
        self.x_net = UNet(config.FEATURE_NUM, config.FEATURE_NUM *config.GMM_NUM*2).to(config.DEVICE)  # mu, var
        self.z_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(config.DEVICE)  # pi
        self.o_net = UNet(config.FEATURE_NUM, config.GMM_NUM).to(config.DEVICE)  # d
        self.reg_net = RR_ResNet(input_channels=config.GMM_NUM).to(config.DEVICE)

        # 加载预训练权重
        # x_net_weights = torch.load("checkpoints/PRIOR/x_train_4.pth", map_location=config.DEVICE, weights_only=True)
        # z_net_weights = torch.load("checkpoints/PRIOR/z_train_4.pth", map_location=config.DEVICE, weights_only=True)
        o_net_weights = torch.load("checkpoints/PRIOR/o_train_4.pth", map_location=config.DEVICE, weights_only=True)
        reg_net_weights = torch.load("checkpoints/reg_prior.pth", map_location=config.DEVICE, weights_only=True)
        unet_weights = torch.load("checkpoints/unet/best.pth", map_location=config.DEVICE, weights_only=True)
        # 加载权重到模型
        # self.x_net.load_state_dict(x_net_weights)
        # self.z_net.load_state_dict(z_net_weights)
        self.o_net.load_state_dict(o_net_weights)
        self.reg_net.load_state_dict(reg_net_weights)
        self.unet.load_state_dict(unet_weights)
        self.reg_net.eval()
        self.unet.eval()

        # 定义损失函数和优化器
        self.criterion = GmmSegLoss(GMM_NUM=config.GMM_NUM).to(config.DEVICE)
        self.optimizer = optim.Adam([
            {'params': self.x_net.parameters(), 'lr': config.LEARNING_RATE, 'weight_decay': 1e-5},
            {'params': self.z_net.parameters(), 'lr': config.LEARNING_RATE, 'weight_decay': 1e-5},
            {'params': self.o_net.parameters(), 'lr': config.LEARNING_RATE*0.1, 'weight_decay': 1e-5},
        ])

        # 定义学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # 创建日志记录器
        time_str = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.config.LOGS_DIR, "tensorboard", time_str)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 创建输出目录
        self.output_dir = os.path.join(self.config.OUTPUT_DIR, time_str)
        os.makedirs(self.output_dir, exist_ok=True)


    def train_one_epoch(self, epoch, epsilon=1e-6):
        self.x_net.train()
        self.z_net.train()
        self.o_net.train()

        train_loss = 0.0
        train_loss1, train_loss2, train_loss3 = 0.0, 0.0, 0.0
        train_loss_mse = 0.0

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
            image_4_features, mu, var, pi, d1, d0 = forward_pass(
                image, self.unet, self.x_net, self.z_net, self.o_net, self.reg_net,
                self.dirichlet_priors, slice_info, num_of_slice_info, self.config, epoch, epsilon
            )

            # 计算损失
            loss, loss_1, loss_2, loss_3, loss_mse = self.criterion(
                image_4_features, mu, var, pi, d=d1, d0=d0, epoch=epoch, total_epochs=self.config.EPOCHS
            )

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

        # 计算平均损失值
        self.train_loss1 = train_loss1 / len(self.train_loader)
        self.train_loss2 = train_loss2 / len(self.train_loader)
        self.train_loss3 = train_loss3 / len(self.train_loader)
        self.train_loss_mse = train_loss_mse / len(self.train_loader)
        self.avg_train_loss = train_loss / len(self.train_loader)

        
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
                image_4_features, mu, var, pi, d1, d0 = forward_pass(
                    image, self.unet, self.x_net, self.z_net, self.o_net, self.reg_net,
                    self.dirichlet_priors, slice_info, num_of_slice_info, self.config, epoch, epsilon=epsilon
                )

                # 计算损失
                loss, loss_1, loss_2, loss_3, loss_mse = self.criterion(
                    image_4_features, mu, var, pi, d=d1, d0=d0, epoch=epoch, total_epochs=self.config.EPOCHS
                )

                # 累加损失
                test_loss1 += loss_1.item()
                test_loss2 += loss_2.item()
                test_loss3 += loss_3.item()
                test_loss_mse += loss_mse.item()
                test_loss += loss.item()

                # 计算评价指标
                posterior_probs = calculate_posterior_probs(image_4_features, mu, var, pi, self.config.GMM_NUM)
                pred_cls = torch.argmax(posterior_probs, dim=1).cpu().numpy()
                label_np = torch.squeeze(label, dim=1).cpu().numpy()

                dataset_name = os.path.basename(self.config.DATASET)
                dice_total += dice_coefficient(pred_cls, label_np, config.GMM_NUM, dataset_name, background=True)
                iou_total += iou_score(pred_cls, label_np, config.GMM_NUM, dataset_name, background=True)
                pixel_err_total += pixel_error(pred_cls, label_np, config.GMM_NUM, dataset_name, background=True)

        self.test_loss1 = test_loss1 / len(self.test_loader)
        self.test_loss2 = test_loss2 / len(self.test_loader)
        self.test_loss3 = test_loss3 / len(self.test_loader)
        self.test_loss_mse = test_loss_mse / len(self.test_loader)
        self.avg_test_loss = test_loss / len(self.test_loader)
        self.avg_dice = dice_total / len(self.test_loader)
        self.avg_iou = iou_total / len(self.test_loader)
        self.avg_pixel_err = pixel_err_total / len(self.test_loader)

        # 可视化最后一个batch的结果
        i = config.BATCH_SIZE - 1
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


    def print_log_info(self, epoch):
        if self.logger:
            self.logger.info(f"Epoch [{epoch+1}/{self.config.EPOCHS}]")
            self.logger.info(
                f"[Train] Loss 1: {self.train_loss1:.4f}  "
                f"Loss 2: {self.train_loss2:.4f}  "
                f"Loss 3: {self.train_loss3:.4f}  "
                f"Loss Total: {self.avg_train_loss:.4f}  "
                f"Loss MSE: {self.train_loss_mse:.4f}"
            )

            self.logger.info(
                f"[Test ] Loss 1: {self.test_loss1:.4f}  "
                f"Loss 2: {self.test_loss2:.4f}  "
                f"Loss 3: {self.test_loss3:.4f}  "
                f"Loss Total: {self.avg_test_loss:.4f}  "
                f"Loss MSE: {self.test_loss_mse:.4f}"
            )

            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"IoU: {self.avg_iou:.4f}  "
                f"Pixel Error: {self.avg_pixel_err:.4f}"
            )


    def save_checkpoints(self, epoch):
        checkpoints_dir = os.path.join(self.config.CHECKPOINTS_DIR, self.config.DATASET)
        os.makedirs(checkpoints_dir, exist_ok=True)
        if self.avg_dice > self.config.BEST_DICE:
            self.config.BEST_DICE = self.avg_dice
            torch.save(self.x_net.state_dict(), os.path.join(checkpoints_dir, f"x_train_best.pth"))
            torch.save(self.z_net.state_dict(), os.path.join(checkpoints_dir, f"z_train_best.pth"))
            torch.save(self.o_net.state_dict(), os.path.join(checkpoints_dir, f"o_train_best.pth"))
            print(f"Saved best model at epoch {epoch+1} with Dice: {self.avg_dice:.4f}")


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
            "total": self.avg_train_loss
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


    def cleanup(self):
        self.writer.close()


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

    # 设置日志记录器
    logger = logging.getLogger("Train GMM Segmentation")
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))

    # 创建文件处理器
    logs_dir = os.path.join(config.LOGS_DIR, "train_logs")
    os.makedirs(logs_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_handler = logging.FileHandler(f"{logs_dir}/{time_str}.log", encoding="utf-8", mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Starting training...")

    
    logger.info(f"Using configuration: {config}")
    trainer = Trainer(config, logger)
    trainer.run()

    logger.info("Training completed.")

    # 关闭处理器
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
