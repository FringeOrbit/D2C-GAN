import torch
import torch.nn as nn

import torch
import torch.nn as nn


class PatchHybridSeqDiscriminator(nn.Module):
    def __init__(self, config):
        super(PatchHybridSeqDiscriminator, self).__init__()
        self.config = config
        dims = config.disc_hidden_dims  # e.g., [128, 64, 32]

        # 获取开关状态（如果 config 里没有，给个默认值）
        self.use_dual_branch = getattr(config, 'disc_dual_branch', True)
        self.is_asymmetric = getattr(config, 'disc_asymmetric', True)
        self.use_patch = getattr(config, 'disc_use_patch', True)

        if not self.use_dual_branch:
            # === 单分支模式 ===
            self.main_branch = self._build_branch(in_channels=4, dims=dims, use_patch=self.use_patch)
        else:
            # === 双分支模式 ===
            self.struct_branch = self._build_branch(in_channels=3, dims=dims, use_patch=self.use_patch)

            # 对称 vs 非对称
            if self.is_asymmetric:
                tex_dims = [max(1, d // 2) for d in dims]  # 通道减半
            else:
                tex_dims = dims  # 对称：通道数一样

            self.texture_branch = self._build_branch(in_channels=1, dims=tex_dims, use_patch=self.use_patch)

    def _build_branch(self, in_channels, dims, use_patch):
        layers = [
            nn.Conv1d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(dims[0], dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dims[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(dims[1], dims[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(dims[2]),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        if use_patch:
            # PatchGAN: 输出序列
            layers.append(nn.Conv1d(dims[2], 1, kernel_size=3, stride=1, padding=1))
        else:
            # Global GAN: 展平后输出一个值 (假设序列长度被前面的卷积压缩到了 L_out)
            # 为了简单，这里直接用 AdaptiveAvgPool1d 汇聚成全局特征
            layers.append(nn.AdaptiveAvgPool1d(1))
            layers.append(nn.Conv1d(dims[2], 1, kernel_size=1))

        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.use_dual_branch:
            # 单分支：4条线一起进
            out = self.main_branch(x)
            # 为了兼容 trainer 的双输出逻辑，单分支复制一份返回
            return out, out
        else:
            # 双分支：拆分 GR 和结构
            x_texture = x[:, 0:1, :]
            x_struct = x[:, 1:, :]
            out_struct = self.struct_branch(x_struct)
            out_texture = self.texture_branch(x_texture)
            return out_struct, out_texture