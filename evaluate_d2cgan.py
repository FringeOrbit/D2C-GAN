# evaluate_d2cgan.py
"""Legacy whole-table, pointwise-random evaluator; not a continuous-block protocol."""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

from config import OptimizedConfig

# 动态导入生成器类
AdvancedSeqGenerator = None

try:
    from models.generator import AdvancedSeqGenerator

    print("✅ 成功导入生成器类")
except ImportError:
    # 如果导入失败，使用占位符
    print("⚠️ 警告: 无法从 models.generator 导入生成器类，将尝试使用其他方式")

plt.style.use('seaborn-v0_8-whitegrid')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PaperStyleEvaluator:
    """
    论文风格评估器：用于评估地质序列生成模型的性能

    支持的功能：
    - 多缺失率下的盲测（20%, 40%, 60%, 80%）
    - R² 和 RMSE 指标计算
    - 深度曲线可视化
    - 趋势图生成（Figure 13 风格）
    """

    def __init__(self, config: OptimizedConfig, model_path: str, scaler_path: str,
                 generator_class: Optional[Any] = None):
        """
        初始化评估器

        Args:
            config: 配置对象
            model_path: 模型权重文件路径
            scaler_path: 标准化器文件路径
            generator_class: 生成器类（可选，如果不指定则使用 AdvancedSeqGenerator）
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 加载标准化器
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"找不到标准化器文件: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # 确定生成器类
        if generator_class is None:
            generator_class = self._get_default_generator_class()

        # 加载模型
        self.model = self._load_model(model_path, generator_class)

        # 缺失率列表（严格对应论文）
        self.missing_rates = [0.2, 0.4, 0.6, 0.8]

        # 颜色配置（论文风格）
        self.colors = {
            'true': '#000000',  # 黑色
            'pred': '#C44E52',  # 红色
            'mask': '#FFF9C4',  # 浅黄色
            'trend': '#C44E52'  # 趋势线颜色
        }

        print(f"✅ 评估器初始化完成 | 设备: {self.device} | 模型类型: {generator_class.__name__}")

    def _get_default_generator_class(self):
        """获取默认生成器类"""
        global AdvancedSeqGenerator

        if AdvancedSeqGenerator is not None:
            return AdvancedSeqGenerator
        else:
            # 尝试直接导入
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from models.generator import AdvancedSeqGenerator
                return AdvancedSeqGenerator
            except ImportError as e:
                # 尝试从当前目录导入
                try:
                    from generator import AdvancedSeqGenerator
                    return AdvancedSeqGenerator
                except ImportError:
                    raise ImportError(f"无法导入 AdvancedSeqGenerator: {e}")

    def _load_model(self, model_path: str, generator_class: Any) -> torch.nn.Module:
        """
        加载模型权重

        Args:
            model_path: 模型路径
            generator_class: 生成器类

        Returns:
            加载好权重的模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        # 实例化模型
        model = generator_class(self.config).to(self.device)

        # 加载权重
        from checkpoint_io import load_generator_state
        state_dict = load_generator_state(model_path, map_location=self.device)

        # 兼容不同的保存格式
        # Never silently evaluate a partly loaded, incompatible architecture.
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

    def generate_random_mask(self, tensor_shape: Tuple, missing_rate: float) -> torch.Tensor:
        """
        生成随机掩码（确保可重复性）

        使用独立的 Generator，确保相同缺失率下生成的掩码一致，
        保证高缺失率的掩码包含低缺失率的掩码。

        Args:
            tensor_shape: 张量形状
            missing_rate: 缺失率 (0-1)

        Returns:
            布尔掩码张量 (True 表示保留，False 表示缺失)
        """
        g = torch.Generator()
        g.manual_seed(42)
        random_tensor = torch.rand(tensor_shape, generator=g)
        mask = random_tensor > missing_rate
        return mask.to(self.device)

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        准备数据：处理NaN、归一化、分块

        Args:
            df: 原始数据DataFrame

        Returns:
            (raw_true, scaled_input_filled, input_tensor)
        """
        # 提取特征数据
        raw_scaled_data = df[self.config.feature_names].values

        # 记录真实数据中的天然黑洞 (NaN)
        original_nan_mask = np.isnan(raw_scaled_data)

        # 将 NaN 替换为 0.0，防止NaN污染
        scaled_input_filled = np.nan_to_num(raw_scaled_data, nan=0.0)

        # 反归一化并还原 NaN，保证评分公正
        raw_true = self.scaler.inverse_transform(scaled_input_filled)
        raw_true[original_nan_mask] = np.nan

        # 序列分块
        N = len(scaled_input_filled)
        seq_len = self.config.seq_len
        pad_len = (seq_len - (N % seq_len)) % seq_len

        if pad_len > 0:
            padded_input = np.pad(scaled_input_filled, ((0, pad_len), (0, 0)), mode='edge')
        else:
            padded_input = scaled_input_filled

        batched_input = padded_input.reshape(-1, seq_len, self.config.num_features).transpose(0, 2, 1)
        input_tensor = torch.FloatTensor(batched_input).to(self.device)

        return raw_true, scaled_input_filled, input_tensor

    @torch.no_grad()
    def _evaluate_rate(self, input_tensor: torch.Tensor, raw_true: np.ndarray,
                       rate: float):
        """
        评估单个缺失率

        Args:
            input_tensor: 输入张量
            raw_true: 真实值（已反归一化）
            rate: 缺失率

        Returns:
            评估结果字典
        """
        N = len(raw_true)

        # 生成掩码
        mask = self.generate_random_mask(input_tensor.shape, rate)
        incomplete_tensor = input_tensor.clone()
        incomplete_tensor[~mask] = 0.0

        # 模型预测
        valid_mask_tensor = mask.float()
        pred_tensor = self.model(incomplete_tensor, mask=valid_mask_tensor)

        # 转换回原始空间
        scaled_pred = pred_tensor.cpu().numpy().transpose(0, 2, 1).reshape(-1, self.config.num_features)[:N, :]
        raw_pred = self.scaler.inverse_transform(scaled_pred)
        flat_mask = mask.cpu().numpy().transpose(0, 2, 1).reshape(-1, self.config.num_features)[:N, :]

        # 计算每个特征的指标
        results = {feat: {'RMSE': [], 'R2': []} for feat in self.config.feature_names}

        for i, feat in enumerate(self.config.feature_names):
            # 只计算人为缺失的位置
            missing_positions = ~flat_mask[:, i]
            t = raw_true[missing_positions, i]
            p = raw_pred[missing_positions, i]

            # 剔除天然黑洞
            valid = ~np.isnan(t)
            t, p = t[valid], p[valid]

            if len(t) > 0:
                rmse = np.sqrt(mean_squared_error(t, p))
                r2 = r2_score(t, p)
                results[feat]['RMSE'].append(rmse)
                results[feat]['R2'].append(r2)
                print(f"  [{feat}] 盲测点数: {len(t):6d} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
            else:
                results[feat]['RMSE'].append(0.0)
                results[feat]['R2'].append(0.0)

        return results, raw_pred, flat_mask

    def plot_depth_logs(self, y_true: np.ndarray, y_pred: np.ndarray,
                        flat_mask: np.ndarray, depth: np.ndarray,
                        wells: np.ndarray, prefix: str):
        """
        绘制深度曲线日志（供外部调用）

        Args:
            y_true: 真实值
            y_pred: 预测值
            flat_mask: 掩码
            depth: 深度数组
            wells: 井名数组
            prefix: 文件名前缀
        """
        unique_wells = np.unique(wells)
        target_wells = self._get_target_wells(unique_wells)

        # 合并预测结果（掩码区域使用真实值，缺失区域使用预测值）
        combined_pred = np.where(flat_mask, y_true, y_pred)

        for well in target_wells:
            well_idx = (wells == well)
            w_true = y_true[well_idx]
            w_pred = combined_pred[well_idx]
            w_depth = depth[well_idx]
            w_mask = flat_mask[well_idx]

            self._plot_single_well(w_true, w_pred, w_depth, w_mask, well, prefix)

    def run_ablation_study(self, test_file_path: str, current_epoch: Optional[int] = None,
                           save_visualization: bool = True):
        """
        运行消融研究：测试不同缺失率下的模型性能

        Args:
            test_file_path: 测试集文件路径
            current_epoch: 当前epoch（用于保存文件名）
            save_visualization: 是否保存可视化结果
        """
        print(f"\n{'=' * 60}")
        print(f"🚀 开始消融研究")
        print(f"📁 测试集: {test_file_path}")
        print(f"{'=' * 60}\n")

        # 加载数据
        df = pd.read_csv(test_file_path, delimiter=self.config.processed_delimiter)
        df.columns = [col.lstrip(';') for col in df.columns]

        # 准备数据
        raw_true, _, input_tensor = self._prepare_data(df)

        # 存储所有结果
        all_results = {feat: {'R2': [], 'RMSE': []} for feat in self.config.feature_names}

        # 对每个缺失率进行评估
        for rate in self.missing_rates:
            print(f"\n{'=' * 40}")
            print(f"📊 测试缺失率: {int(rate * 100)}%")
            print(f"{'=' * 40}")

            results, raw_pred, flat_mask = self._evaluate_rate(input_tensor, raw_true, rate)

            # 汇总结果
            for feat in self.config.feature_names:
                all_results[feat]['R2'].append(results[feat]['R2'][0])
                all_results[feat]['RMSE'].append(results[feat]['RMSE'][0])

            # 保存40%缺失率的深度图
            if rate == 0.4 and save_visualization:
                depth = df[self.config.depth_column].values if self.config.depth_column in df.columns else np.arange(
                    len(raw_true))
                wells = df[self.config.well_column].values if self.config.well_column in df.columns else np.array(
                    ['Unknown'] * len(raw_true))
                self._save_depth_visualization(df, raw_true, raw_pred, flat_mask, depth, wells,
                                               "Missing_40_Percent", current_epoch)

            # 清理内存
            del raw_pred, flat_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 保存趋势图
        if save_visualization:
            self._save_trend_visualization(all_results, current_epoch)

        # 打印汇总结果
        self._print_summary(all_results)

        return all_results

    def _save_trend_visualization(self, results: Dict, current_epoch: Optional[int] = None):
        """
        保存R²趋势图（Figure 13风格）

        Args:
            results: 评估结果
            current_epoch: 当前epoch
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        x_labels = ['20%', '40%', '60%', '80%']
        x_pos = np.arange(len(x_labels))

        for i, feat in enumerate(self.config.feature_names):
            ax = axes[i]
            r2_values = results[feat]['R2']

            # 绘制趋势线
            ax.plot(x_pos, r2_values, marker='o', markersize=8, linestyle='-',
                    linewidth=2, color=self.colors['trend'], label='Proposed Method')

            # 添加数据点标签
            for x, y in zip(x_pos, r2_values):
                ax.text(x, y + 0.02, f"{y:.3f}", ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

            ax.set_title(f"({chr(97 + i)}) {feat}", fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=12)
            ax.set_xlabel("Missing rate", fontsize=12)
            ax.set_ylabel(r"$R^2$", fontsize=12)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower left')

        # 添加总标题
        title_suffix = f" (Epoch {current_epoch})" if current_epoch else ""
        plt.suptitle(f"Comparison of R² with different missing rates{title_suffix}",
                     fontsize=16, y=1.02)
        plt.tight_layout()

        # 保存图片
        file_name = f"evaluation_r2_trend_epoch_{current_epoch}.png" if current_epoch else "evaluation_r2_trend.png"
        save_path = os.path.join(self.config.result_dir, file_name)
        os.makedirs(self.config.result_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"📈 趋势图已保存: {save_path}")

    def _save_depth_visualization(self, df: pd.DataFrame, y_true: np.ndarray,
                                  y_pred: np.ndarray, flat_mask: np.ndarray,
                                  depth: np.ndarray, wells: np.ndarray,
                                  prefix: str, current_epoch: Optional[int] = None):
        """
        保存深度曲线可视化

        Args:
            df: 原始数据
            y_true: 真实值
            y_pred: 预测值
            flat_mask: 掩码
            depth: 深度数组
            wells: 井名数组
            prefix: 文件名前缀
            current_epoch: 当前epoch
        """
        unique_wells = np.unique(wells)

        # 选择要绘制的井
        target_wells = self._get_target_wells(unique_wells)

        # 合并预测结果（掩码区域使用真实值，缺失区域使用预测值）
        combined_pred = np.where(flat_mask, y_true, y_pred)

        for well in target_wells:
            well_idx = (wells == well)
            w_true = y_true[well_idx]
            w_pred = combined_pred[well_idx]
            w_depth = depth[well_idx]
            w_mask = flat_mask[well_idx]

            self._plot_single_well(w_true, w_pred, w_depth, w_mask, well, prefix, current_epoch)

    def _get_target_wells(self, unique_wells: np.ndarray) -> List:
        """获取要绘制的目标井"""
        if hasattr(self.config, 'target_plot_well') and self.config.target_plot_well is not None:
            if self.config.target_plot_well in unique_wells:
                return [self.config.target_plot_well]
            else:
                print(f"⚠️ 警告：配置的靶向井 '{self.config.target_plot_well}' 不在当前测试集中！将使用第一口井。")
                return [unique_wells[0]]
        return [unique_wells[0]]

    def _plot_single_well(self, y_true: np.ndarray, y_pred: np.ndarray,
                          depth: np.ndarray, mask: np.ndarray,
                          well: str, prefix: str, current_epoch: Optional[int] = None):
        """
        绘制单口井的深度曲线

        Args:
            y_true: 真实值
            y_pred: 预测值
            depth: 深度
            mask: 掩码
            well: 井名
            prefix: 文件名前缀
            current_epoch: 当前epoch
        """
        fig, axes = plt.subplots(1, 4, figsize=(14, 18), sharey=True)

        for i, col in enumerate(self.config.feature_names):
            ax = axes[i]

            # 绘制真实曲线
            ax.plot(y_true[:, i], depth, label='True Log',
                    color=self.colors['true'], linewidth=1.5)

            # 绘制预测曲线
            ax.plot(y_pred[:, i], depth, label='Reconstructed',
                    color=self.colors['pred'], linestyle='--', linewidth=1.5)

            # 填充缺失区域
            ax.fill_betweenx(depth, ax.get_xlim()[0], ax.get_xlim()[1],
                             where=~mask[:, i], color=self.colors['mask'], alpha=0.5)

            ax.set_title(col, fontsize=14, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5)

            if i == 0:
                ax.invert_yaxis()
                ax.set_ylabel("Depth (m)", fontsize=14)
                ax.legend(loc='upper right', fontsize=10)

        # 添加总标题
        title_suffix = f" (Epoch {current_epoch})" if current_epoch else ""
        plt.suptitle(f"Interpolation results ({prefix}) - {well}{title_suffix}",
                     fontsize=16, y=1.01)
        plt.tight_layout()

        # 保存图片
        safe_well_name = str(well).replace("/", "_").replace("\\", "_")
        file_name = f"Fig_{prefix}_Well_{safe_well_name}"
        if current_epoch:
            file_name += f"_Epoch_{current_epoch}"
        file_name += ".png"

        out_path = os.path.join(self.config.result_dir, file_name)
        os.makedirs(self.config.result_dir, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"📊 深度曲线图已保存: {out_path}")

    def _print_summary(self, results: Dict):
        """打印评估结果汇总"""
        print(f"\n{'=' * 60}")
        print(f"📊 评估结果汇总")
        print(f"{'=' * 60}")

        # 创建结果表格
        summary_data = []
        for feat in self.config.feature_names:
            for i, rate in enumerate(self.missing_rates):
                summary_data.append({
                    'Feature': feat,
                    'Missing Rate': f"{int(rate * 100)}%",
                    'R²': results[feat]['R2'][i],
                    'RMSE': results[feat]['RMSE'][i]
                })

        df_summary = pd.DataFrame(summary_data)

        # 打印表格
        print("\n" + df_summary.to_string(index=False))

        # 计算平均性能
        avg_r2 = np.mean([results[feat]['R2'] for feat in self.config.feature_names])
        avg_rmse = np.mean([results[feat]['RMSE'] for feat in self.config.feature_names])

        print(f"\n{'=' * 60}")
        print(f"📈 平均性能 (所有缺失率 × 所有特征):")
        print(f"   R²:   {np.mean(avg_r2):.4f} ± {np.std(avg_r2):.4f}")
        print(f"   RMSE: {np.mean(avg_rmse):.4f} ± {np.std(avg_rmse):.4f}")
        print(f"{'=' * 60}\n")



def main():
    """主函数"""
    print("Protocol: legacy whole-table reshape + pointwise random missingness; not per-well block testing.")
    config = OptimizedConfig()

    # 创建结果目录
    os.makedirs(config.result_dir, exist_ok=True)

    # 模型路径
    model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    scaler_path = os.path.join(config.checkpoint_dir, 'std_scaler.pkl')

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    if not os.path.exists(scaler_path):
        print(f"❌ 错误: 找不到标准化器文件 {scaler_path}")
        return

    # 创建评估器并运行
    try:
        evaluator = PaperStyleEvaluator(config, model_path, scaler_path)
        evaluator.run_ablation_study(config.test_path)
        print("✅ 评估完成！")
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
