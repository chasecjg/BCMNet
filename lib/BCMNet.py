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


# 特征增强模块RFB
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
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


# 注意力模块MSCA
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


# AFF模块，特征融合
class ACFM(nn.Module):
    def __init__(self, channel=64):
        super(ACFM, self).__init__()

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



class SAM(nn.Module):
    def __init__(self, num_in=64, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

# 利用transformer补充纹理信息，resnet50提取语义信息。transformer参与了解码和编码
class BCMNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(BCMNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.pvt = pvt_v2_b2_one_layer()
        self.tfaux = pvt_v2_b2_backbone()
        self.acfm = ACFM()

        self.rfb0_1 = RFB_modified(64, channel)
        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(64, channel)
        self.rfb4_1 = RFB_modified(64, channel)

        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()

        self.SAM = SAM()
        self.inplanes0 = 64
        self.inplanes1 = 64
        self.ca2 = ChannelAttention(self.inplanes0)
        self.sa2 = SpatialAttention()
        self.ca1 = ChannelAttention(self.inplanes1)
        self.sa1 = SpatialAttention()
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_double_f = nn.Conv2d(channel, 1, 1)
        self.out_cbma = nn.Conv2d(channel, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv = nn.Conv2d(128, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(1024, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(192, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(2048, channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(1088, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, 3, 3, 1, 1)
        self.conv6 = nn.Conv2d(6, channel, 3, 1, 1)
        self.conv7 = nn.Conv2d(6, 1, 3, 1, 1)
        self.conv8 = nn.Conv2d(2 * channel, channel, 3, 1, 1)

        self.out_double_f = nn.Conv2d(channel, 1, 1)
        self.conv9 = nn.Conv2d(2 * channel, 1, 1)
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

        x_aff1 = self.acfm(x_sa1, out)
        x_aff2 = self.acfm(x_sa2_u, out)
        x_dff = x_aff1 + x_aff2

        P1_1 = self.out234(out3)
        P2_1 = self.out234(out4)
        P3_1 = self.out234(out)
        P4_1 = self.out234(x_aff1)
        P5_1 = self.out234(x_aff2)
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



