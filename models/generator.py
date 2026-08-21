import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# ================= 1D 通道注意力模块 (CAM) =================
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden_planes = max(1, in_planes // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# ================= 1D 空间注意力模块 (SAM) =================
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


# ================= 1D CBAM 综合模块 =================
class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(spatial_kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -----------------------------------------------------------------------------
# 频域模块 v1：原版（Ours_Full）
# -----------------------------------------------------------------------------
class FrequencyFeatureBlock(nn.Module):
    """频域特征提取模块 (Global Filter)"""

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


# -----------------------------------------------------------------------------
# 频域模块 v2：振幅-相位解耦版（Ours_Full_v2）
# -----------------------------------------------------------------------------
class DAPA_FrequencyBlock(nn.Module):
    """支持动态序列长度的 振幅-相位解耦频域模块 (DAPA-FFT)"""

    def __init__(self, channels):
        super(DAPA_FrequencyBlock, self).__init__()
        self.channels = channels

        self.base_freq_weight_mag = nn.Parameter(torch.ones(1, channels, 256))
        self.base_freq_weight_phase = nn.Parameter(torch.zeros(1, channels, 256))

        self.mag_net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )

        self.phase_gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, L = x.shape
        num_freqs = L // 2 + 1

        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        weight_mag = F.interpolate(self.base_freq_weight_mag, size=num_freqs, mode='linear', align_corners=False)
        weight_phase = F.interpolate(self.base_freq_weight_phase, size=num_freqs, mode='linear', align_corners=False)

        magnitude = magnitude * weight_mag
        phase = phase + weight_phase

        mag_out = self.mag_net(magnitude)
        phase_out = phase * self.phase_gate(phase)

        complex_out = torch.complex(mag_out * torch.cos(phase_out),
                                    mag_out * torch.sin(phase_out))

        x_reconstructed = torch.fft.irfft(complex_out, n=L, dim=-1, norm='ortho')
        return x + x_reconstructed


# ================= 空洞残差块（共用） =================
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


# -----------------------------------------------------------------------------
# 模型 1：原版 Ours_Full
# -----------------------------------------------------------------------------
class AdvancedSeqGenerator(nn.Module):
    def __init__(self, config):
        super(AdvancedSeqGenerator, self).__init__()
        self.config = config
        dims = config.generator_dims
        input_channels = config.num_features

        layers = [nn.Conv1d(input_channels, dims[1], kernel_size=1)]

        dilations = [1, 2, 4, 8, 16]
        for i, d in enumerate(dilations):
            layers.append(DilatedResidualBlock(dims[1], dilation=d, config=config))
            if config.use_fft and i < 4:
                layers.append(FrequencyFeatureBlock(dims[1]))  # 原版频域

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


# -----------------------------------------------------------------------------
# 模型 2：升级版 Ours_Full_v2
# -----------------------------------------------------------------------------
class AdvancedSeqGenerator_V2(nn.Module):
    def __init__(self, config):
        super(AdvancedSeqGenerator_V2, self).__init__()
        self.config = config
        dims = config.generator_dims
        input_channels = config.num_features

        layers = [nn.Conv1d(input_channels, dims[1], kernel_size=1)]

        dilations = [1, 2, 4, 8, 16]
        for i, d in enumerate(dilations):
            layers.append(DilatedResidualBlock(dims[1], dilation=d, config=config))
            if config.use_fft and i < 4:
                layers.append(DAPA_FrequencyBlock(dims[1]))  # v2 解耦频域

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