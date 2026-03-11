import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(SE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )

        self.gpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x = x * se

        return x

class MultiOrderDWConv(nn.Module):
    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, 4],
                 channel_split=[1, 2, 2, 3]):
        super(MultiOrderDWConv, self).__init__()

        assert len(dw_dilation) == len(channel_split) == 4
        assert embed_dims % sum(channel_split) == 0

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.split_dims = [int(r * embed_dims) for r in self.split_ratio]
        self.split_dims[-1] = embed_dims - sum(self.split_dims[:-1])
        kernel_sizes = [3, 5, 7, 9]
        self.dw_convs = nn.ModuleList()
        self.weight_gens = nn.ModuleList()

        for i in range(4):

            padding = (kernel_sizes[i] - 1) // 2 * dw_dilation[i]
            self.dw_convs.append(
                nn.Conv2d(
                    in_channels=self.split_dims[i],
                    out_channels=self.split_dims[i],
                    kernel_size=kernel_sizes[i],
                    padding=padding,
                    dilation=dw_dilation[i],
                    groups=self.split_dims[i],
                )
            )


            self.weight_gens.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.split_dims[i], self.split_dims[i], kernel_size=1),
                    nn.Sigmoid()
                )
            )

        self.conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x):
        x_splits = torch.split(x, self.split_dims, dim=1)
        branch_outputs = []
        for conv, branch in zip(self.dw_convs, x_splits):
            branch_outputs.append(conv(branch))


        weighted_outputs = []
        for branch_out, weight_gen in zip(branch_outputs, self.weight_gens):
            weight = weight_gen(branch_out)             # [B, C_i, 1, 1]
            weighted_outputs.append(branch_out * weight)

        x_fused = torch.cat(weighted_outputs, dim=1)
        out = self.conv(x_fused)
        return out


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MSG(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3, 4],
                 attn_channel_split=[1, 2, 2, 3],
                 attn_act_type='SiLU'):
        super(MSG, self).__init__()

        self.embed_dims = embed_dims
        self.group_num = embed_dims // 4

        self.gate = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

        self.v = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )

        self.g = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1),
            nn.Sigmoid()
        )

        self.out = Conv1x1(inplanes=embed_dims, planes=embed_dims)
        self.act_gate = nn.Sigmoid()
        self.se = SE(embed_dims,decay=4)

    def forward(self, x):
        g = self.gate(x)
        v = self.v(x)
        xg = self.act_gate(g) * v

        xl = self.g(x.mean(dim=(2, 3), keepdim=True))
        xl = xl * v
        out = self.se(xg + xl)
        # out = self.se(x)
        x_out = self.out(out)
        return x_out
