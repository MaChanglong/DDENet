from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSK(nn.Module):
    def __init__(self, channel=32, kernels=[3, 5, 7], reduction=4, L=32, base_atrous_rate=[4, 6, 8]):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList()
        self.dconvs = nn.ModuleList()

        for i, k in enumerate(kernels):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            ))
            self.dconvs.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=base_atrous_rate[i], dilation=base_atrous_rate[i], bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            ))


        self.cat_conv = Cat_conv(2 * channel, channel)


        self.gate_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def gated_fusion(self, outs1, outs2):
        sum1 = sum(outs1)
        sum2 = sum(outs2)
        gate1 = self.gate_conv(sum1)
        gate2 = self.gate_conv(sum2)
        fused = gate1 * sum1 + gate2 * sum2
        return fused

    def sk_process(self, x, convs):
        return [conv(x) for conv in convs]

    def forward(self, x):
        V1_list = self.sk_process(x, self.convs)
        V2_list = self.sk_process(x, self.dconvs)
        V = self.gated_fusion(V1_list, V2_list)
        V = self.cat_conv(V, x)
        V = V + x
        return V


class Conv(nn.Sequential):
    def  __init__(self, in_channels, out_channels):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class BC(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(BC, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x = self.relu(x1 + x2)
        return x


class GatedFusion(nn.Module):
    def __init__(self, in_channels, n_paths=3, reduction=16):
        super(GatedFusion, self).__init__()
        hidden_dim = max(1, in_channels // reduction)
        self.n_paths = n_paths

        self.weight_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, n_paths, 1, bias=False)
        )

    def forward(self, feats):  # feats: list of [B, C, H, W]
        assert len(feats) == self.n_paths, f"Expected {self.n_paths} paths, got {len(feats)}"


        x_sum = sum(feats) / self.n_paths  # [B, C, H, W]


        weights = self.weight_fc(x_sum)  # [B, n_paths, 1, 1]
        weights = F.softmax(weights, dim=1)
        weights = weights.unbind(dim=1)  # 拆成 n_paths 个 [B, 1, 1, 1]


        weighted_feats = []
        for i in range(self.n_paths):
            w = weights[i].unsqueeze(1)    # [B, 1, 1, 1]
            f = feats[i]    # [B, C, H, W]
            w_expanded = w.expand_as(f)
            weighted_f = f * w_expanded
            weighted_feats.append(weighted_f)


        out = sum(weighted_feats)  # [B, C, H, W]
        return out

class Cat_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cat_conv, self).__init__()
        self.conv = Conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2 ], dim=1)
        x = self.conv(x)
        return x

class MBD(nn.Module):
    def __init__(self, in_channels, out_channels=None, atrous_rate=8, base_atrous_rate=[4, 6, 8]):
        super(MBD, self).__init__()
        if out_channels is None:
            out_channels = 2 * in_channels

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)


        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gated_fusion = GatedFusion(in_channels)
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.msk = MSK(out_channels, base_atrous_rate=base_atrous_rate)

    def forward(self, x):
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        x_conv = self.convs2(self.convs1(x))

        # x_fused = x_max+x_avg+x_conv
        x_fused = self.gated_fusion([x_max, x_avg, x_conv])
        x_proj = self.conv_proj(x_fused)
        x_out = self.msk(x_proj)
        return x_out


