import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.loss import GmmLoss
from utils.train_utils import forward_pass, compute_responsibilities
from utils.visualizer import create_visualization
from utils.metrics import evaluate_segmentation
from utils.dataloader import get_loaders
from utils.postprocess import comprehensive_postprocess
from models.unet import UNet
from models.regnet import RR_ResNet
from models.align_net import Align_ResNet
from models.scale_net import Scale_ResNet
from config import Config, get_config, format_config
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    

class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def get_logger(self, train: bool = True):
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)

        # === 控制台 handler（简洁） ===
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
        logger.addHandler(console_handler)

        # 控制台简洁打印
        logger.info(f"Using configuration: {self.config}")

        # === 文件 handler（详细） ===
        if train:
            logs_dir = os.path.join(self.config.LOGS_DIR, "train_logs", self.config.DATASET)
        else:
            logs_dir = os.path.join(self.config.LOGS_DIR, "test_logs", self.config.DATASET)
        os.makedirs(logs_dir, exist_ok=True)

        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_path = os.path.join(logs_dir, f"{time_str}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
        logger.addHandler(file_handler)

        # === 写入启动信息 ===
        logger.info("Starting training..." if train else "Starting testing...")

        # 文件详细打印
        file_handler.stream.write("Full configuration:\n" + format_config(self.config) + "\n\n")
        file_handler.flush()

        return logger, console_handler, file_handler, time_str


    def setup(self, train=True):
        set_seed(self.config.SEED)
        self.logger, self.console_handler, self.file_handler, self.time_str = self.get_logger(train=train)
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
        self.reg_net = RR_ResNet(input_channels=2).to(self.config.DEVICE)
        self.align_net = Align_ResNet(input_channels=1).to(self.config.DEVICE)
        self.scale_net = Scale_ResNet(input_channels=2).to(self.config.DEVICE)


        # 加载预训练权重 (修复: 启用所有预训练权重的加载)
        try:
            # x_net_weights = torch.load("checkpoints/unet/x_pretrained.pth", 
            #                            map_location=self.config.DEVICE, 
            #                            weights_only=True)
            # z_net_weights = torch.load("checkpoints/unet/z_pretrained.pth", 
            #                            map_location=self.config.DEVICE, 
            #                            weights_only=True)
            o_net_weights = torch.load("checkpoints/unet/o_pretrained.pth", 
                                       map_location=self.config.DEVICE, 
                                       weights_only=True)
            unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                      map_location=self.config.DEVICE, 
                                      weights_only=True)
            reg_net_weights = torch.load("checkpoints/regnet/regnet_prior_2chs.pth", 
                                         map_location=self.config.DEVICE, 
                                         weights_only=True)
            align_net_weights = torch.load("checkpoints/align_net/align_best.pth",
                                           map_location=self.config.DEVICE,
                                           weights_only=True)
            scale_net_weights = torch.load("checkpoints/scale_net/scale_best.pth",
                                           map_location=self.config.DEVICE,
                                           weights_only=True)
            
            # 加载权重到模型
            # self.x_net.load_state_dict(x_net_weights)
            # self.z_net.load_state_dict(o_net_weights)
            self.o_net.load_state_dict(o_net_weights)
            self.unet.load_state_dict(unet_weights)
            self.reg_net.load_state_dict(reg_net_weights)
            self.align_net.load_state_dict(align_net_weights)
            self.scale_net.load_state_dict(scale_net_weights)

            # 固定预训练模型的参数
            self.unet.eval()
            self.reg_net.eval()
            self.align_net.eval()
            self.scale_net.eval()
            
            # 冻结UNet和reg_net的参数以避免过拟合
            for param in self.unet.parameters():
                param.requires_grad = False
            for param in self.reg_net.parameters():
                param.requires_grad = False
            for param in self.align_net.parameters():
                param.requires_grad = False
            for param in self.scale_net.parameters():
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
            if self.config.ADD_TENSORBOARD:
                # 创建日志记录器
                log_dir = os.path.join(self.config.LOGS_DIR, "tensorboard", self.config.DATASET, self.time_str)
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            if self.config.VISUALIZE:
                # 创建输出目录
                self.output_dir = os.path.join(self.config.OUTPUT_DIR, self.config.DATASET, self.time_str)
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
            ds = batch["ds"][0]  # 当前批次数据集标识
            forward_pass_result = forward_pass(image=image,
                                               label=label,
                                               prior=prior,
                                               unet=self.unet,
                                               x_net=self.x_net,
                                               z_net=self.z_net,
                                               o_net=self.o_net,
                                               reg_net=self.reg_net,
                                               align_net=self.align_net,
                                               scale_net=self.scale_net,
                                               ds=ds,
                                               config=self.config, 
                                               epoch=epoch, 
                                               epsilon=epsilon)
            feature_4chs = forward_pass_result["feature_4chs"]
            mu = forward_pass_result["mu"]
            var = forward_pass_result["var"]
            pi = forward_pass_result["pi"]
            d1 = forward_pass_result["d1"]
            d0 = forward_pass_result["d0"]

            # 计算损失 (添加异常处理)
            try:
                loss_out = self.criterion(input=feature_4chs,
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
        dice_total, dice_lv_total, dice_myo_total, dice_rv_total = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch+1:3}/{self.config.EPOCHS}[valid]", ncols=100):
                # 数据加载
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)
                prior = batch['prior'].to(self.config.DEVICE, dtype=torch.float32)
                label_prior = batch['label_prior'].to(self.config.DEVICE, dtype=torch.float32)

                if self.config.USE_LABEL_PRIOR:
                    prior = label_prior  # 直接使用标签先验进行验证

                ds = batch["ds"][0]  # 当前批次数据集标识
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
                                                   scale_net=self.scale_net,
                                                   ds=ds,
                                                   config=self.config, 
                                                   epoch=epoch, 
                                                   epsilon=epsilon)
                feature_4chs = forward_pass_result["feature_4chs"]
                scaled_image = forward_pass_result["scaled_image"]
                scaled_label = forward_pass_result["scaled_label"]
                mu = forward_pass_result["mu"]
                var = forward_pass_result["var"]
                pi = forward_pass_result["pi"]
                d1 = forward_pass_result["d1"]
                d0 = forward_pass_result["d0"]
                image_scale = forward_pass_result["image_scale"]
                prior_scale = forward_pass_result["prior_scale"]
                prior_tx = forward_pass_result["prior_tx"]
                prior_ty = forward_pass_result["prior_ty"]
                angle = forward_pass_result["angle"]


                # 计算损失 (添加异常处理)
                try:
                    loss_out = self.criterion(input=feature_4chs,
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
                r = compute_responsibilities(feature_4chs, mu, var, alpha=d1)
                pred_cls = torch.argmax(r, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()

                ds = batch["ds"][0]  # 当前批次数据集标识, 同一批次的数据不一定都来自同一数据集！
                class_num = batch["class_num"][0]  # 当前批次数据集类别数

                if ds == "SCD":
                    # SCD: 将预测中的类别2、3视为背景，避免无效类别拉低指标
                    pred_cls[pred_cls == 2] = 0
                    pred_cls[pred_cls == 3] = 0
                if ds == "YORK":
                    # YORK: 网络输出有4通道，但标注通常仅含0/1/2，将预测的3类归为背景
                    pred_cls[pred_cls == 3] = 0

                # 指标与测试保持一致：不计入背景（更符合常见分割评估）
                results = evaluate_segmentation(pred_cls, label_np, class_num, background=self.config.METRIC_WITH_BACKGROUND)
                dice_total += results["Dice"]["mean"]
                dice_lv_total += results["Dice"]["per_class"].get(1, 0.0)
                dice_myo_total += results["Dice"]["per_class"].get(2, 0.0)
                dice_rv_total += results["Dice"]["per_class"].get(3, 0.0)

        self.valid_loss_1 = valid_loss_1 / len(self.valid_loader)
        self.valid_loss_2 = valid_loss_2 / len(self.valid_loader)
        self.valid_loss_3 = valid_loss_3 / len(self.valid_loader)
        self.valid_loss_mu = valid_loss_mu / len(self.valid_loader)
        self.valid_loss_var = valid_loss_var / len(self.valid_loader)
        self.avg_valid_loss = valid_loss / len(self.valid_loader)
        self.avg_dice = dice_total / len(self.valid_loader)
        self.avg_dice_lv = dice_lv_total / len(self.valid_loader)
        self.avg_dice_myo = dice_myo_total / len(self.valid_loader)
        self.avg_dice_rv = dice_rv_total / len(self.valid_loader)

        # 随机可视化最后一个batch中的一个样本
        i = random.randint(0, image.shape[0]-1)
        self.pi_show = pi.detach().cpu().numpy()[i]
        self.d1_show = d1.detach().cpu().numpy()[i]
        self.d0_show = d0.detach().cpu().numpy()[i]
        self.image_show = image[i].squeeze().detach().cpu().numpy()
        self.label_show = label_np[i]
        self.pred_show = pred_cls[i]
        # 记录配准参数
        self.image_scale = image_scale[i].item() if image_scale is not None else 0
        self.prior_scale = prior_scale[i].item()
        self.prior_tx = prior_tx[i].item()
        self.prior_ty = prior_ty[i].item()
        # 角度单位提示：align_net 输出为“弧度”，日志中转为“度”展示
        self.angle_rad = angle[i].item() if angle is not None else None
        self.angle_deg = math.degrees(self.angle_rad) if self.angle_rad is not None else None
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

        x_net_weights = torch.load(f"checkpoints/unet/{self.config.DATASET}/x_best.pth",
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        z_net_weights = torch.load(f"checkpoints/unet/{self.config.DATASET}/z_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        o_net_weights = torch.load(f"checkpoints/unet/{self.config.DATASET}/o_best.pth",
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        unet_weights = torch.load("checkpoints/unet/unet_best.pth", 
                                    map_location=self.config.DEVICE, 
                                    weights_only=True)
        align_net_weights = torch.load("checkpoints/align_net/align_best.pth", 
                                        map_location=self.config.DEVICE, 
                                        weights_only=True)
        scale_net_weights = torch.load("checkpoints/scale_net/scale_best.pth",
                                        map_location=self.config.DEVICE,
                                        weights_only=True)
        self.x_net.load_state_dict(x_net_weights)
        self.z_net.load_state_dict(z_net_weights)
        self.o_net.load_state_dict(o_net_weights)
        self.unet.load_state_dict(unet_weights)
        self.align_net.load_state_dict(align_net_weights)
        self.scale_net.load_state_dict(scale_net_weights)
        self.x_net.eval()
        self.z_net.eval()
        self.o_net.eval()
        self.unet.eval()
        self.align_net.eval()
        self.scale_net.eval()

        test_dice, test_dice_lv, test_dice_myo, test_dice_rv = 0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"[test]", ncols=100):
                image = batch['image'].to(self.config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(self.config.DEVICE, dtype=torch.float32)
                prior = batch['prior'].to(self.config.DEVICE, dtype=torch.float32)

                ds = batch["ds"][0]  # 当前批次数据集标识

                forward_pass_result = forward_pass(image=image,
                                                   label=label,
                                                   prior=prior,
                                                   unet=self.unet,
                                                   x_net=self.x_net,
                                                   z_net=self.z_net,
                                                   o_net=self.o_net,
                                                   reg_net=None, 
                                                   align_net=self.align_net,
                                                   scale_net=self.scale_net,
                                                   ds=ds,
                                                   config=self.config, 
                                                   epoch=None,
                                                   epsilon=1e-6)
                feature_4chs = forward_pass_result["feature_4chs"]
                scaled_label = forward_pass_result["scaled_label"]
                mu = forward_pass_result["mu"]
                var = forward_pass_result["var"]
                pi = forward_pass_result["pi"]
                alpha = forward_pass_result["d1"]
                
                # pred_cls = torch.argmax(pi, dim=1).detach().cpu().numpy()
                r = compute_responsibilities(x=feature_4chs, mu=mu, var=var, alpha=alpha)
                pred_cls = torch.argmax(r, dim=1).detach().cpu().numpy()
                label_np = torch.squeeze(label, dim=1).detach().cpu().numpy()

                ds = batch["ds"][0]  # 当前批次数据集标识
                class_num = batch["class_num"][0]  # 当前批次数据集类别数
                # 与验证一致的类别后处理：
                if ds == "SCD":
                    pred_cls[pred_cls == 2] = 0
                    pred_cls[pred_cls == 3] = 0
                if ds == "YORK":
                    pred_cls[pred_cls == 3] = 0

                # 评估指标：不计背景
                test_dice += evaluate_segmentation(pred_cls, label_np, class_num, background=self.config.METRIC_WITH_BACKGROUND)["Dice"]["mean"]
                test_dice_lv += evaluate_segmentation(pred_cls, label_np, class_num, background=self.config.METRIC_WITH_BACKGROUND)["Dice"]["per_class"].get(1, 0.0)
                test_dice_myo += evaluate_segmentation(pred_cls, label_np, class_num, background=self.config.METRIC_WITH_BACKGROUND)["Dice"]["per_class"].get(2, 0.0)
                test_dice_rv += evaluate_segmentation(pred_cls, label_np, class_num, background=self.config.METRIC_WITH_BACKGROUND)["Dice"]["per_class"].get(3, 0.0)

        test_dice /= len(self.test_loader)
        test_dice_lv /= len(self.test_loader)
        test_dice_myo /= len(self.test_loader)
        test_dice_rv /= len(self.test_loader)
        self.logger.info(f"[Test] Dice: {test_dice:.4f}")
        self.logger.info(f"[Test] Dice LV: {test_dice_lv:.4f}")
        self.logger.info(f"[Test] Dice MYO: {test_dice_myo:.4f}")
        self.logger.info(f"[Test] Dice RV: {test_dice_rv:.4f}")


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

            # 日志展示：角度以“度”为单位（内部计算均为弧度）
            angle_str = "N/A" if not hasattr(self, "angle_deg") or self.angle_deg is None else f"{self.angle_deg:.2f} deg"
            self.logger.info(
                f"image_scale: {self.image_scale:.2f}  "
                f"image_rotate: {angle_str}  "
                f"prior_scale: {self.prior_scale:.2f}  "
                f"prior_tx: {self.prior_tx:.2f}  "
                f"prior_ty: {self.prior_ty:.2f}  "
            )

            self.logger.info(
                f"Dice: {self.avg_dice:.4f}  "
                f"(LV: {self.avg_dice_lv:.4f}, MYO: {self.avg_dice_myo:.4f}, RV: {self.avg_dice_rv:.4f})"
            )

            self.logger.info(f"Current best dice: {self.config.BEST_DICE:.4f}, at epoch {self.config.BEST_EPOCH}")


    def save_checkpoints(self, epoch):
        checkpoints_dir = os.path.join(self.config.CHECKPOINTS_DIR, "unet", self.config.DATASET)
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

        # 学习率
        if len(self.optimizer.param_groups) > 0:
            self.writer.add_scalar("learning_rate/group0", self.optimizer.param_groups[0]['lr'], epoch+1)



    def cleanup(self, train=True):
        if train:
            if self.config.ADD_TENSORBOARD: self.writer.close()
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
                if self.config.VISUALIZE:
                    self.visualize(epoch)
                if self.config.ADD_TENSORBOARD:
                    self.add_tensorboard_scalars(epoch)
        self.test()
        self.cleanup(train=train)



if __name__ == "__main__":
    # 获取配置
    config = get_config()
    trainer = Trainer(config)
    trainer.run(train=(config.MODE == 'train'))

