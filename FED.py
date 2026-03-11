import torch
import torch.nn as nn
import copy
from mmcv.ops import DeformConv2dPack as DCN

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


class Spectra(nn.Module):
    def __init__(self, in_depth, AF='prelu'):
        super().__init__()

        # Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth // 2 if in_depth >= 2 else self.in_depth

        # Layers
        self.AF1 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.AF2 = nn.ReLU if AF == 'relu' else nn.PReLU(self.inter_depth)
        self.inConv = nn.Sequential(nn.Conv2d(self.in_depth, self.inter_depth, 1),
                                    nn.BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = nn.Sequential(nn.Conv2d(self.inter_depth, self.inter_depth, 1),
                                     nn.BatchNorm2d(self.inter_depth),
                                     self.AF2)
        self.outConv = nn.Conv2d(self.inter_depth, self.in_depth, 1)

    def forward(self, x):
        x = self.inConv(x)
        _, _, H, W = x.shape
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)
        real_rfft = torch.real(rfft)
        imag_rfft = torch.imag(rfft)
        cat_rfft = torch.cat((real_rfft, imag_rfft), dim=-1)
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1] // 2
        real_rfft = cat_rfft[..., :mid]
        imag_rfft = cat_rfft[..., mid:]
        rfft = torch.complex(real_rfft, imag_rfft)
        spect = torch.fft.irfft2(rfft, s=(H, W))
        out = self.outConv(spect + skip)
        return out


class FED(nn.Module):
    def __init__(self, in_depth, AF='prelu'):
        super().__init__()
        # Params
        self.in_depth = in_depth

        # Layers
        self.AF1 = nn.ReLU if AF == 'relu' else nn.PReLU(self.in_depth)
        self.AF2 = nn.ReLU if AF == 'relu' else nn.PReLU(self.in_depth)
        self.conv_ll = nn.Conv2d(self.in_depth, self.in_depth, 3, padding='same')
        self.conv_lg = nn.Conv2d(self.in_depth, self.in_depth, 3, padding='same')
        self.conv_gl = nn.Conv2d(self.in_depth, self.in_depth, 3, padding='same')
        self.conv_gg = Spectra(self.in_depth, AF)
        self.bnaf1 = nn.Sequential(nn.BatchNorm2d(self.in_depth), self.AF1)
        self.bnaf2 = nn.Sequential(nn.BatchNorm2d(self.in_depth), self.AF2)
        self.se = SE(in_depth)
        self.deform = DCN(in_depth, in_depth, kernel_size=3, stride=1, padding=1)
        self.deform1 = DCN(in_depth*2, in_depth, kernel_size=3, stride=1, padding=1)

    def forward(self, x_loc,x_glo):
        # mid = x.shape[1] // 2
        # x_loc = x[:, :mid, :, :]
        # x_glo = x[:, mid:, :, :]
        # x_ll = self.conv_ll(x_loc)
        # print("x_loc size:", x_loc.shape)
        x_ll = self.deform(x_loc)

        x_ll = self.se(x_ll) * x_ll

        x_lg = self.conv_lg(x_loc)
        x_gl = self.conv_gl(x_glo)
        x_gg = self.conv_gg(x_glo)
        # x_gg = x_gl
        # print("x_ll size:",x_ll.shape)
        # print("x_gl size:", x_gl.shape)
        out_loc = torch.add((self.bnaf1(x_ll + x_gl)), x_loc)
        out_glo = torch.add((self.bnaf2(x_gg + x_lg)), x_glo)
        out = torch.cat((out_loc, out_glo), dim=1)
        return self.deform1(out)

