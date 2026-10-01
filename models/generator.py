"""Parallel generator recovered from the audited server working tree.

Matches the parameter layout of the original Ours_Full checkpoints.
Input is four zero-filled measurement channels, NOT measurements concatenated
with availability masks. The optional mask argument is retained for caller
compatibility and is unused by this historical architecture.
"""

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, max(1, in_planes // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(max(1, in_planes // reduction), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(spatial_kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class FrequencyFeatureBlock(nn.Module):
    def __init__(self, channels):
        super(FrequencyFeatureBlock, self).__init__()
        self.complex_conv = nn.Conv1d(channels * 2, channels * 2, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, L = x.shape
        x_fft = torch.fft.rfft(x, norm='ortho')
        x_real = x_fft.real
        x_imag = x_fft.imag
        x_complex_stacked = torch.cat([x_real, x_imag], dim=1)

        out_complex = self.complex_conv(x_complex_stacked)
        out_complex = self.bn(out_complex)
        out_complex = self.relu(out_complex)

        out_real, out_imag = torch.chunk(out_complex, 2, dim=1)
        out_fft = torch.complex(out_real, out_imag)
        x_time = torch.fft.irfft(out_fft, n=L, norm='ortho')

        gate_weight = self.sigmoid(self.gate(x))
        return x + x_time * gate_weight


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, config=None):
        super(DilatedResidualBlock, self).__init__()
        actual_dilation = dilation if (config and config.use_dilation) else 1

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=actual_dilation, dilation=actual_dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout1d(p=0.2)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.use_cbam = config.use_cbam if config else True
        if self.use_cbam:
            self.cbam = CBAM1D(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        return self.relu(out + residual)


class ParallelTimeFrequencyBlock(nn.Module):
    """
    时域-频域并行分支 + 自适应门控融合
    """
    def __init__(self, channels, dilation, use_fft=True, config=None):
        super(ParallelTimeFrequencyBlock, self).__init__()
        self.use_fft = use_fft
        self.use_learned_weights = getattr(config, 'use_learned_branch_weights', False)

        # 时域分支
        self.time_branch = DilatedResidualBlock(channels, dilation=dilation, config=config)

        # 频域分支
        if self.use_fft:
            self.freq_branch = FrequencyFeatureBlock(channels)

        # 自适应权重生成器
        if self.use_learned_weights and self.use_fft:
            self.weight_net = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // 4, 4)),
                nn.ReLU(inplace=True),
                nn.Linear(max(channels // 4, 4), 2),
                nn.Softmax(dim=1)
            )
        elif self.use_fft:
            # 可学习标量权重 (初始均为0.5)
            self.weight_time = nn.Parameter(torch.tensor(0.5))
            self.weight_freq = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out_time = self.time_branch(x)

        if not self.use_fft:
            return out_time

        out_freq = self.freq_branch(x)

        if self.use_learned_weights:
            weights = self.weight_net(x)          # [B, 2]
            w_time = weights[:, 0:1].unsqueeze(-1)  # [B,1,1]
            w_freq = weights[:, 1:2].unsqueeze(-1)
            out = w_time * out_time + w_freq * out_freq
        else:
            # 可学习标量融合
            out = self.weight_time * out_time + self.weight_freq * out_freq

        return out


class AdvancedSeqGenerator(nn.Module):
    def __init__(self, config):
        super(AdvancedSeqGenerator, self).__init__()
        self.config = config
        dims = config.generator_dims
        input_channels = config.num_features

        # 动态组装 feature_extractor
        layers = [nn.Conv1d(input_channels, dims[1], kernel_size=1)]

        dilations = [1, 2, 4, 8, 16]
        for i, d in enumerate(dilations):
            has_fft = config.use_fft
            layers.append(
                ParallelTimeFrequencyBlock(dims[1], dilation=d, use_fft=has_fft, config=config)
            )

        layers.extend([
            nn.Conv1d(dims[1], dims[2], kernel_size=1),
            nn.BatchNorm1d(dims[2]),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        self.feature_extractor = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Conv1d(dims[2], config.num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(config.num_features, config.num_features, kernel_size=5, padding=2),
            nn.Identity()
        )

    def forward(self, x, mask=None):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        x = torch.clamp(x, min=-4.0, max=4.0)
        return x
