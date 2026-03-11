import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from MSG import MSG
from FED import FED
from pvtv2 import pvt_v2_b2
from MBD import MBD

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


class AMCM(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(AMCM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x_1 = self.avg_pool(input)
        x_2 = self.max_pool(input)
        x1 = self.conv1(x_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x_2).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)

        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = out1 + out2
        out = self.sigmoid(out)
        return out * input


class DDE_blocks(nn.Module):
    def __init__(self, in_channel, out_channel,  decay=2):
        super(DDE_blocks, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, 2, stride=2)
        self.conv = nn.Conv2d(out_channel*2, out_channel, 3, padding=1)
        self.msg = MSG(out_channel)
        self.CBR = nn.Sequential(
            nn.Conv2d(out_channel , out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        satt = self.msg(point)
        att = self.CBR(satt)
        return att

class RAF(nn.Module):
    def __init__(self, in_channels):
        super(RAF, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.se = SE(in_channels)

        self.conv1 = nn.Conv2d(64, 512, 1, padding=1)
        self.conv2 = nn.Conv2d(512, 1, 1, padding=1)
        self.conv3 = nn.Conv2d(1024, 512, 1)

    def forward(self, x , y):
        pred = torch.sigmoid(x)
        pred = self.conv1(pred)
        pred = F.interpolate(pred, (y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)
        FOR = y * pred
        background_att = 1 - pred
        B = y * background_att
        fusion_feature = torch.cat([FOR, B], dim=1)
        input_feature = self.fusion_conv(fusion_feature)
        out = input_feature + y

        return out


class DDENet(nn.Module):
    def __init__(self, n_class=1, decay=2):
        super(DDENet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.pool = nn.MaxPool2d(2)

        self.down_conv5 = AMCM(512)
        self.up_conv5 = DDE_blocks(512, 320, decay)
        self.up_conv4 = DDE_blocks(320, 128, decay)
        self.up_conv3 = DDE_blocks(128, 64, decay)
        self.up_conv2 = DDE_blocks(64, 32, decay)
        self.up_conv1 = DDE_blocks(64, 32, decay)

        self.dp5 = nn.Conv2d(512, 1, 1)
        self.dp4 = nn.Conv2d(320, 1, 1)
        self.dp3 = nn.Conv2d(128, 1, 1)
        self.dp2 = nn.Conv2d(64, 1, 1)

        self.center5 = nn.Conv2d(1024, 512, 1)
        self.decodeup4 = nn.Conv2d(512, 256, 1)
        self.decodeup3 = nn.Conv2d(256, 128, 1)
        self.decodeup2 = nn.Conv2d(128, 64, 1)

        self.raf = RAF(512)

        self.f1 = FED(64)
        self.f2 = FED(128)
        self.f3 = FED(320)
        self.f4 = FED(512)

        self.down1 = MBD(3, 64)
        self.down2 = MBD(64,128)
        self.down3 = MBD(128,320)
        self.down4 = MBD(320,512)

    def forward(self, inputs):
        pvt = self.backbone(inputs)
        down1 = pvt[0]
        down2 = pvt[1]
        down3 = pvt[2]
        down4 = pvt[3]

        inputs = self.pool(inputs)
        cnn_down1 = self.down1(inputs)
        cnn_down1 = self.f1(cnn_down1,down1)
        cnn_down2 = self.down2(cnn_down1)
        cnn_down2 = self.f2(cnn_down2,down2)
        cnn_down3 = self.down3(cnn_down2)
        cnn_down3 = self.f3(cnn_down3, down3)

        center = self.down4(cnn_down3)
        center = self.f4(center,down4)
        center = self.down_conv5(center)
        center = self.raf(cnn_down1,center)
        out5 = self.dp5(center)
        deco4 = self.up_conv5(center, down3)
        out4 = self.dp4(deco4)
        deco3 = self.up_conv4(deco4, down2)
        out3 = self.dp3(deco3)
        deco2 = self.up_conv3(deco3, down1)
        out2 = self.dp2(deco2)

        return out2, out3, out4, out5