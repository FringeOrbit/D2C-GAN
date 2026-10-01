import os
import sys
from pathlib import Path

# Support the documented `python training/train_d2cgan.py` entry point.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib


from config import OptimizedConfig
import joblib
# 👇 新增这一行：导入你刚才改好的可视化评估类
from evaluate_d2cgan import PaperStyleEvaluator
from config import OptimizedConfig
import random

def seed_everything(seed=42):
    """固定所有随机种子，确保训练具有可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 针对 CuDNN 的确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_detailed_loss_curves(loss_history, save_path):
    """
    绘制并保存所有的细分 Loss 曲线
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # 创建一个 3x3 的画布
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    # 颜色配置 (可以按照三大联合约束分类上色)
    colors = {
        'D_Total': 'black', 'G_Total_Weighted': 'gray',
        'G_Adv_Struct': '#d62728', 'G_Adv_Texture': '#ff9896',  # 对抗约束 (红色系)
        'Recon_Total': '#1f77b4', 'Recon_Missing': '#aec7e8',  # 重构约束 (蓝色系)
        'Geo_Petro': '#2ca02c', 'Geo_Boundary': '#98df8a', 'Geo_TV': '#8c564b'  # 地质约束 (绿色/大地色系)
    }

    # x 轴是记录次数 (每次代表 50 个 batch)
    x = range(len(next(iter(loss_history.values()))))

    for i, (loss_name, values) in enumerate(loss_history.items()):
        ax = axes[i]
        # 使用一点简单的平滑让曲线更清晰 (Alpha=0.3画原图，实线画平滑)
        ax.plot(x, values, color=colors.get(loss_name, 'blue'), alpha=0.3, linewidth=1)

        # 计算滑动平均 (Window=5) 使趋势更明显
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, min_periods=1).mean()
            ax.plot(x, smooth_values, color=colors.get(loss_name, 'blue'), linewidth=2)

        ax.set_title(loss_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)  # 🌟 修改横轴名称
        ax.set_ylabel('Loss Value', fontsize=12)
        # 对抗损失用对数坐标轴有时更好看，这里保持线性，可根据实际图表微调

    plt.suptitle("Detailed Training Dynamics of Joint Loss Functions", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 详细 Loss 曲线已保存至: {save_path}")
# --- 数据集类 (已升级为序列滑动窗口版本) ---
class WellDataset(Dataset):
    def __init__(self, df, feature_names, config):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.seq_len = config.seq_len
        self.missing_rate_range = config.missing_rate_range

        self.windows = []
        self.original_missing_list = []

        # 🎯 核心修复：按井分组 (Groupby)
        # 确保滑动窗口绝对不会跨越两口不同的井！
        for well, group in df.groupby(config.well_column):
            # 获取单口井的数据
            well_data = group[feature_names].values.astype(np.float32)

            # 记录这口井的原始天然黑洞
            orig_missing = np.isnan(well_data) | np.isinf(well_data)

            # 将 NaN 替换为 0.0，以供网络计算
            well_data = np.nan_to_num(well_data, nan=0.0)

            n_samples = len(well_data)

            # 只有当这口井的长度大于一个窗口时，才进行截取
            if n_samples >= self.seq_len:
                for i in range(n_samples - self.seq_len + 1):
                    self.windows.append(well_data[i: i + self.seq_len])
                    self.original_missing_list.append(orig_missing[i: i + self.seq_len])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 直接从已经切好的、绝对属于同一口井的窗口池中提取
        full_window = self.windows[idx].copy()
        orig_missing_window = self.original_missing_list[idx]

        valid_gt_mask = ~orig_missing_window
        mask = np.ones_like(full_window, dtype=bool)

        for feat_idx in range(self.n_features):
            rand_val = np.random.rand()
            if rand_val < 0.45:
                rate = np.random.uniform(0.2, 0.85)
                mask[:, feat_idx] = np.random.rand(self.seq_len) > rate
            elif rand_val < 0.90:
                rate = np.random.uniform(*self.missing_rate_range)
                n_missing = int(self.seq_len * rate)
                if n_missing > 0:
                    start_idx = np.random.randint(0, self.seq_len - n_missing + 1)
                    mask[start_idx: start_idx + n_missing, feat_idx] = False

        mask[orig_missing_window] = False
        incomplete_window = full_window.copy()

        noise = np.random.normal(0, 0.02, incomplete_window.shape)
        incomplete_window = incomplete_window + noise * mask
        incomplete_window[~mask] = 0.0
        '''
        incomplete_curve：人为缺失后的不完整序列（缺失位置为 0）

        full_curve：完整的原始序列（缺失值已用 0 填充，但记录了掩码）
        
        missing_mask：人为制造的缺失掩码（True 表示该位置被“人为缺失”）
        
        valid_gt_mask：原始数据中非缺失的位置掩码（True 表示该位置本来就有真实值）
        '''
        return {
            'incomplete_curve': torch.FloatTensor(incomplete_window.T),
            'full_curve': torch.FloatTensor(full_window.T),
            'missing_mask': torch.BoolTensor(~mask.T),
            'valid_gt_mask': torch.BoolTensor(valid_gt_mask.T)
        }

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def export_discriminator_analysis_assets(batch, generator, discriminator, save_dir='./discriminator_assets'):
    """
    专为双流判别器 (Discriminator) 设计的可视化资产导出函数。
    严格按照 1:4 宽高比，生成 8 张图片（4 张连续输入曲线 + 4 张带置信度背景的输出曲线）。
    🎯 优化：去除了所有外围边框，实现极简悬浮效果。
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(generator.parameters()).device

    inc_tensor = batch['incomplete_curve'].to(device)

    # 1. 运行生成器和判别器
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        # 获取送入判别器的完整连续序列 (Fake样本)
        fake_tensor = generator(inc_tensor)

        # 运行双流判别器
        d_struct_patch, d_texture_patch = discriminator(fake_tensor)

        # 将 Patch 级别的得分插值到全序列长度
        seq_len = fake_tensor.shape[2]
        conf_struct = torch.nn.functional.interpolate(d_struct_patch, size=seq_len, mode='linear', align_corners=False)
        conf_texture = torch.nn.functional.interpolate(d_texture_patch, size=seq_len, mode='linear',
                                                       align_corners=False)

    generator.train()
    discriminator.train()

    # 2. 转换为 Numpy 进行绘图
    fake_np = fake_tensor[0].cpu().numpy()
    conf_struct_np = conf_struct[0, 0].cpu().numpy()
    conf_texture_np = conf_texture[0, 0].cpu().numpy()

    depth_x = np.arange(seq_len)
    feature_names = ['GR', 'RHOB', 'NPHI', 'DTC']
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#9467bd']

    # 3. 辅助函数：格式化 1:4 的子图 (无边框版)
    def setup_1_to_4_axes(fig):
        ax = fig.add_subplot(111)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(seq_len - 1, 0)  # 反转 Y 轴，深度向下
        ax.set_xticks([])
        ax.set_yticks([])

        # 🎯 核心修改：关闭所有外部边框 (spines)
        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax

    # 4. 循环生成 8 张图片
    for i, feat in enumerate(feature_names):
        # 归一化当前曲线
        y = fake_np[i]
        y_min, y_max = y.min(), y.max()
        if y_max == y_min: y_max = y_min + 1e-5
        y_norm = (y - y_min) / (y_max - y_min)

        # 区分双流！GR(0)使用纹理分支，其他使用结构分支
        if i == 0:
            current_conf_score = conf_texture_np
        else:
            current_conf_score = conf_struct_np

        # ==========================================================
        # 图 A: 判别器输入可视化 (纯净连续曲线，无缺失)
        # ==========================================================
        fig_in = plt.figure(figsize=(1.5, 6))  # 严格 1:4 比例
        ax_in = setup_1_to_4_axes(fig_in)

        # 绘制连续曲线
        ax_in.plot(y_norm, depth_x, color=colors[i], linewidth=2.5)

        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(save_dir, f'D_Input_{feat}.png'), dpi=400, transparent=True)
        plt.close(fig_in)

        # ==========================================================
        # 图 B: 判别器输出可视化 (携带 0-1 置信度背景)
        # ==========================================================
        fig_out = plt.figure(figsize=(1.5, 6))
        ax_out = setup_1_to_4_axes(fig_out)

        # 绘制置信度背景色 (RdYlGn 色带)
        cmap = plt.get_cmap('RdYlGn')
        imshow_data = np.broadcast_to(current_conf_score.reshape(seq_len, 1), (seq_len, 10))

        # 填充背景
        ax_out.imshow(imshow_data, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0,
                      extent=[-0.2, 1.2, seq_len - 1, 0], origin='upper', alpha=0.35)

        # 绘制曲线覆盖在背景上
        ax_out.plot(y_norm, depth_x, color=colors[i], linewidth=2.5)

        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(save_dir, f'D_Output_Conf_{feat}.png'), dpi=400, transparent=True)
        plt.close(fig_out)

    print(f"\n✅ 成功！已去除所有边框，8 张 (1:4) 判别器悬浮图片导出至: {os.path.abspath(save_dir)}")# --- 训练核心类 ---


class CGANTrainer:
    def __init__(self, config, generator, discriminator):
        self.config = config
        self.device = config.device
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)

        self.g_opt = torch.optim.AdamW(self.G.parameters(), lr=config.learning_rate_g,
                                       betas=(config.beta1, config.beta2), weight_decay=1e-4)
        self.d_opt = torch.optim.AdamW(self.D.parameters(), lr=config.learning_rate_d,
                                       betas=(config.beta1, config.beta2), weight_decay=1e-4)

        self.bce = nn.BCELoss()

        # ================= 🌟 核心修改: 动态损失函数分配 =================
        # 兼容 config 中没有 loss_type 的情况，默认使用 huber
        loss_type = getattr(self.config, 'loss_type', 'huber')

        if loss_type == 'mse':
            self.criterion_recon = nn.MSELoss()
            print("⚠️ 警告: 当前正在使用 [MSE Loss] 进行重构消融训练")
        elif loss_type == 'l1':
            self.criterion_recon = nn.L1Loss()
            print("⚠️ 警告: 当前正在使用 [L1 Loss] 进行重构消融训练")
        else:
            self.criterion_recon = nn.SmoothL1Loss(beta=0.5)
            print("✅ 正常: 当前正在使用 [Huber (Smooth L1) Loss] 进行训练")
        # =================================================================

        # 保留原本的 MSE 仅用于计算协方差矩阵 (物理约束需要严格的数学定义)
        self.mse_for_cov = nn.MSELoss()
        self.l1_for_boundary = nn.L1Loss()

    def petrophysical_consistency_loss(self, fake, real, missing_mask_float):
        """Historical pair-overlap matching, not common-four sample covariance.

        Curve-specific means and pairwise counts (+ epsilon) are deliberately
        retained to match the recovered pre-Wyllie trainer bytecode.
        """
        if not missing_mask_float.any():
            return torch.tensor(0.0).to(self.device)

        valid_count = missing_mask_float.sum(dim=2, keepdim=True) + 1e-8
        f_mean = (fake * missing_mask_float).sum(dim=2, keepdim=True) / valid_count
        r_mean = (real * missing_mask_float).sum(dim=2, keepdim=True) / valid_count

        f_centered = (fake - f_mean) * missing_mask_float
        r_centered = (real - r_mean) * missing_mask_float

        joint_valid_count = torch.bmm(missing_mask_float, missing_mask_float.transpose(1, 2)) + 1e-8

        f_cov = torch.bmm(f_centered, f_centered.transpose(1, 2)) / joint_valid_count
        r_cov = torch.bmm(r_centered, r_centered.transpose(1, 2)) / joint_valid_count

        return self.mse_for_cov(f_cov, r_cov)

    def boundary_awareness_loss(self, fake, real, valid_gt):
        valid_grad_mask = valid_gt[:, :, 1:] & valid_gt[:, :, :-1]
        fake_grad = fake[:, :, 1:] - fake[:, :, :-1]
        real_grad = real[:, :, 1:] - real[:, :, :-1]

        if valid_grad_mask.any():
            return self.l1_for_boundary(fake_grad[valid_grad_mask], real_grad[valid_grad_mask])
        return torch.tensor(0.0).to(self.device)

    def train_step(self, batch): 
        inc = batch['incomplete_curve'].to(self.device)
        full = batch['full_curve'].to(self.device)
        mask = batch['missing_mask'].to(self.device)
        valid_gt = batch['valid_gt_mask'].to(self.device)

        batch_size = inc.size(0)
        valid_mask = (~mask).float()

        # --------- 训练判别器 D ---------
        self.d_opt.zero_grad()
        with torch.no_grad():
            fake = self.G(inc)
        full_for_D = full.clone()
        full_for_D[~valid_gt] = fake[~valid_gt]

        d_real_struct, d_real_texture = self.D(full_for_D)
        d_fake_struct, d_fake_texture = self.D(fake.detach())

        real_labels_s = torch.full_like(d_real_struct, 0.9, device=self.device)
        fake_labels_s = torch.full_like(d_fake_struct, 0.1, device=self.device)
        real_labels_t = torch.full_like(d_real_texture, 0.9, device=self.device)
        fake_labels_t = torch.full_like(d_fake_texture, 0.1, device=self.device)

        loss_d_struct = (self.bce(d_real_struct, real_labels_s) + self.bce(d_fake_struct, fake_labels_s)) / 2
        loss_d_texture = (self.bce(d_real_texture, real_labels_t) + self.bce(d_fake_texture, fake_labels_t)) / 2
        loss_d = loss_d_struct + loss_d_texture
        loss_d.backward()
        self.d_opt.step()

        # --------- 训练生成器 G ---------
        self.g_opt.zero_grad()
        fake = self.G(inc)
        d_fake_struct_for_g, d_fake_texture_for_g = self.D(fake)
        loss_g_adv_struct = self.bce(d_fake_struct_for_g, real_labels_s)
        loss_g_adv_texture = self.bce(d_fake_texture_for_g, real_labels_t)

        # 🌟 核心替换：使用动态 criterion_recon 代替写死的 huber_loss
        if valid_gt.any():
            loss_recon_total = self.criterion_recon(fake[valid_gt], full[valid_gt])
        else:
            loss_recon_total = torch.tensor(0.0).to(self.device)

        artificial_missing = mask & valid_gt
        if artificial_missing.any():
            # 🌟 核心替换：使用动态 criterion_recon
            loss_recon_missing = self.criterion_recon(fake[artificial_missing], full[artificial_missing])
            loss_petro = self.petrophysical_consistency_loss(fake, full, artificial_missing.float())
        else:
            loss_recon_missing = torch.tensor(0.0).to(self.device)
            loss_petro = torch.tensor(0.0).to(self.device)

        loss_boundary = self.boundary_awareness_loss(fake, full, valid_gt)
        loss_smoothness = self.tv_loss(fake)

        total_g_loss = (self.config.weight_adv_struct * loss_g_adv_struct) + \
                       (self.config.weight_adv_texture * loss_g_adv_texture) + \
                       (self.config.l1_loss_weight * loss_recon_total) + \
                       (self.config.weight_recon_missing * loss_recon_missing) + \
                       (self.config.weight_petro * loss_petro) + \
                       (self.config.weight_boundary * loss_boundary) + \
                       (self.config.weight_tv * loss_smoothness)

        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.g_opt.step()

        return {
            'D_Total': loss_d.item(),
            'G_Total_Weighted': total_g_loss.item(),
            'G_Adv_Struct': loss_g_adv_struct.item(),
            'G_Adv_Texture': loss_g_adv_texture.item(),
            'Recon_Total': loss_recon_total.item(),
            'Recon_Missing': loss_recon_missing.item(),
            'Geo_Petro': loss_petro.item(),
            'Geo_Boundary': loss_boundary.item(),
            'Geo_TV': loss_smoothness.item()
        }

    def val_step(self, batch):
        inc = batch['incomplete_curve'].to(self.device)
        full = batch['full_curve'].to(self.device)
        valid_gt = batch['valid_gt_mask'].to(self.device)

        self.G.eval()
        with torch.no_grad():
            fake = self.G(inc)
            # 🌟 核心替换：使用动态 criterion_recon
            if valid_gt.any():
                loss_recon_val = self.criterion_recon(fake[valid_gt], full[valid_gt])
            else:
                loss_recon_val = torch.tensor(0.0).to(self.device)

        self.G.train()
        return loss_recon_val.item()

    def tv_loss(self, x):
        return torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))


def main():
    # 👇 在一切开始之前，锁定全局随机种子
    seed_everything(seed=42)
    config = OptimizedConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ================= 🌟 核心修改 1: 直接加载已经分好的数据集 =================
    # 确保 config.data_path 在 config.py 中已指向 "./data/train_processed.csv"
    print(f"正在加载预处理完成的训练集: {config.data_path}")
    train_df = pd.read_csv(config.data_path, delimiter=config.processed_delimiter)

    # 验证集路径 (如果 config 里没配，就使用默认的 "./data/val_processed.csv")
    val_path = config.val_path
    print(f"正在加载预处理完成的验证集: {val_path}")
    val_df = pd.read_csv(val_path, delimiter=config.processed_delimiter)
    # =========================================================================

    # 2. 直接加载预处理阶段保存好的全局 Scaler
    if not os.path.exists(config.scaler_path):
        raise FileNotFoundError(f"找不到 Scaler 文件: {config.scaler_path}，请先运行 data_preprocess.py")

    scaler = joblib.load(config.scaler_path)
    joblib.dump(scaler, os.path.join(config.checkpoint_dir, 'std_scaler.pkl'))
    print("全局 Scaler 加载成功并已同步至 checkpoint 目录。")
    # =================================================================

    # 3. 实例化模型
    from models.generator import AdvancedSeqGenerator
    gen = AdvancedSeqGenerator(config)
    from models.discriminator import PatchHybridSeqDiscriminator
    disc = PatchHybridSeqDiscriminator(config)

    # ================= 🌟 核心修改 2: 实例化双路 DataLoader =================
    # 训练集 DataLoader (开启 shuffle)
    train_dataset = WellDataset(train_df, config.feature_names, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    # 验证集 DataLoader (关闭 shuffle，batch_size 可以适当加大以加速验证过程)
    val_dataset = WellDataset(val_df, config.feature_names, config)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=1)
    # =========================================================================

    trainer = CGANTrainer(config, gen, disc)

    # 🌟 修改为监控验证集 Loss
    best_val_recon = float('inf')
    patience_counter = 0

    print(f"开始训练，设备: {config.device}")
    # try:
    #     sample_batch = next(iter(train_loader))
    #     # 传入 batch 并且传入你实例化的生成器 gen
    #     # export_three_independent_G_assets(sample_batch, gen, save_dir='./paper_model_assets')
    #     export_discriminator_analysis_assets(sample_batch, gen, disc, save_dir='./model_diagram_v9_assets')
    # except Exception as e:
    #     print(f"⚠️ 生成架构图素材失败: {e}")
    for epoch in range(config.epochs):
        # ------------------- A. 训练阶段 -------------------
        epoch_train_recon = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in pbar_train:
            # 🌟 核心修复：接收字典，而不是强行解包成三个变量
            loss_dict = trainer.train_step(batch)

            # 从字典中提取你想要显示在进度条上的关键指标
            recon = loss_dict['Recon_Missing']
            g_adv = loss_dict['G_Adv_Struct'] + loss_dict['G_Adv_Texture']
            d_l = loss_dict['D_Total']

            epoch_train_recon += recon
            pbar_train.set_postfix({'G_Adv': f'{g_adv:.2f}', 'Recon': f'{recon:.4f}', 'D': f'{d_l:.2f}'})

        avg_train_recon = epoch_train_recon / len(train_loader)

        # ------------------- B. 验证阶段 -------------------
        epoch_val_recon = 0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Valid]", leave=False)
        with torch.no_grad():  # 验证阶段不计算梯度
            for val_batch in pbar_val:
                val_recon = trainer.val_step(val_batch)
                epoch_val_recon += val_recon
                pbar_val.set_postfix({'Val_Recon': f'{val_recon:.4f}'})

        avg_val_recon = epoch_val_recon / len(val_loader)

        # 打印当前 Epoch 的双重对比总结
        print(f"Epoch {epoch + 1} 完成 | Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}")

        # ------------------- C. 模型保存与可视化触发 -------------------
        if avg_val_recon < best_val_recon:
            best_val_recon = avg_val_recon
            patience_counter = 0
            model_save_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            torch.save({'generator': gen.state_dict(), 'config': config}, model_save_path)

            print(f"\n🌟 发现当前最佳模型 (Val Recon: {best_val_recon:.4f})，正在生成盲测可视化报告...")

            original_result_dir = config.result_dir
            try:
                vis_dir = os.path.join(original_result_dir, 'training_process')
                os.makedirs(vis_dir, exist_ok=True)
                config.result_dir = vis_dir
                scaler_path = os.path.join(config.checkpoint_dir, 'std_scaler.pkl')

                evaluator = PaperStyleEvaluator(config, model_save_path, scaler_path)
                evaluator.run_ablation_study(config.test_path, current_epoch=epoch + 1)

            except Exception as e:
                print(f"\n⚠️ 可视化过程出现异常，但不影响训练继续: {e}")
            finally:
                config.result_dir = original_result_dir
                if 'evaluator' in locals():
                    del evaluator
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            patience_counter += 1

        if config.use_early_stopping and patience_counter >= config.early_stopping_patience:
            print(f"触发早停，最佳 Val Recon Loss: {best_val_recon:.6f}")
            break


if __name__ == "__main__":
    main()
