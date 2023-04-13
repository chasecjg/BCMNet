import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.pvtv2_bacobone import pvt_v2_b2_backbone
from lib.pvtv2_one_layer import pvt_v2_b2_one_layer
from utils.Attention import ChannelAttention, SpatialAttention
from utils.tensor_ops import cus_sample

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# part of SAM
class SAM_L(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SAM_L, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# MSCA attention
class MSCA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


# DEM modlue
class DEM(nn.Module):
    def __init__(self, channel=64):
        super(DEM, self).__init__()

        self.msca = MSCA()
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        xy = x + y
        wei = self.msca(xy)
        xo = x * wei + y * (1 - wei)
        xo = self.conv(xo)

        return xo




class BCMNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(BCMNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.pvt = pvt_v2_b2_one_layer()
        self.tfaux = pvt_v2_b2_backbone()
        self.dem = DEM()

        self.rfb1_1 = SAM_L(64, channel)
        self.rfb2_1 = SAM_L(128, channel)
        self.rfb3_1 = SAM_L(64, channel)
        self.rfb4_1 = SAM_L(64, channel)

        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()

        self.inplanes0 = 64
        self.inplanes1 = 64
        self.ca2 = ChannelAttention(self.inplanes0)
        self.sa2 = SpatialAttention()
        self.ca1 = ChannelAttention(self.inplanes1)
        self.sa1 = SpatialAttention()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv = nn.Conv2d(128, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(1024, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(192, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(2048, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(1088, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, 3, 3, 1, 1)
        self.conv6 = nn.Conv2d(6, channel, 3, 1, 1)

        self.out234 = nn.Conv2d(channel, 1, 1)


    def forward(self, x):
        y = self.tfaux(x)
        y1 = y[0]
        y2 = y[1]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x3_1 = self.upsample2(x3)
        x3_2 = self.conv1(x3_1)
        x3_3 = self.relu(x3_2)
        x3_4 = torch.cat((y2, x3_3), 1)
        x3_5 = self.conv2(x3_4)
        x3_6 = self.relu(x3_5)
        # print(x3_6.shape)
        x4_1 = self.upsample2(x4)
        x4_2 = self.conv3(x4_1)
        x4_3 = self.relu(x4_2)
        x4_4 = torch.cat((x3, x4_3), 1)
        x4_5 = self.conv4(x4_4)
        x4_6 = self.relu(x4_5)
        # print(x4_6.shape)
        x4_1_1 = self.upsample2(x4_6)
        x4_2_1 = self.conv0(x4_1_1)
        x4_3_1 = self.relu(x4_2_1)
        x4_4_1 = torch.cat((x4_3_1, y2), dim=1)
        x4_5_1 = self.conv2(x4_4_1)
        x4_6_1 = self.relu(x4_5_1)
        # print(x4_6_1.shape)
        x1_rfb = self.rfb1_1(y1)  # channel -> 64
        x2_rfb = self.rfb2_1(y2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3_6)  # channel -> 64
        x4_rfb = self.rfb4_1(x4_6_1)  # channel -> 64

        x_ca1 = self.ca1(x1_rfb) * x1_rfb
        x_sa1 = self.sa1(x_ca1) * x_ca1
        x_ca2 = self.ca2(x2_rfb) * x2_rfb
        x_sa2 = self.sa2(x_ca2) * x_ca2
        x_sa2_u = self.upsample2(x_sa2)

        x3_rfb_1 = self.conv5(x3_rfb)
        x3_rfb_2 = self.relu(x3_rfb_1)
        x4_rfb_1 = self.conv5(x4_rfb)
        x4_rfb_2 = self.relu(x4_rfb_1)

        feature3, x3_t = self.pvt(x3_rfb_2)
        feature4, x4_t = self.pvt(x4_rfb_2)

        out3_1 = torch.cat((feature3, x3_t), dim=1)
        out3_2 = self.conv6(out3_1)
        out3 = self.relu(out3_2)
        out4_1 = torch.cat((feature4, x4_t), dim=1)
        out4_2 = self.conv6(out4_1)
        out4 = self.relu(out4_2)
        out = out3 + out4

        x_dem1 = self.dem(x_sa1, out)
        x_dem2 = self.dem(x_sa2_u, out)
        x_dff = x_dem1 + x_dem2

        P1_1 = self.out234(out3)
        P2_1 = self.out234(out4)
        P3_1 = self.out234(out)
        P4_1 = self.out234(x_dem1)
        P5_1 = self.out234(x_dem2)
        P6_1 = self.out234(x_dff)

        P1 = F.interpolate(P1_1, scale_factor=8, mode='bilinear')
        P2 = F.interpolate(P2_1, scale_factor=8, mode='bilinear')
        P3 = F.interpolate(P3_1, scale_factor=8, mode='bilinear')
        P4 = F.interpolate(P4_1, scale_factor=4, mode='bilinear')
        P5 = F.interpolate(P5_1, scale_factor=4, mode='bilinear')
        P6 = F.interpolate(P6_1, scale_factor=4, mode='bilinear')

        return P1, P2, P3, P4, P5, P6




if __name__ == '__main__':

    model = BCMNet(channel=64)
    data = torch.randn(3, 3, 352, 352)
    out = model(data)
    for i in range(6):
        print(out[i].shape)



