import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
from STANet.models.backbones import ResNet_1_DCNV2_CA
from models.convlstm import ConvLSTM

from STANet.models.base import BaseModel

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)


class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lateral_conv = FSM(c1, c2)
        self.offset = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        return feat_align + feat_arm


class FaPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]

        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])
        # self.temp_att=nn.ModuleList([LTAE2d(in_channels[0])])
        self.temp_att=nn.ModuleList([])
        self.temp_att.append(ConvLSTM(input_size=(8, 8), input_dim=channel, hidden_dim=channel, kernel_size=(3, 3)))
        self.temp_att.append(ConvLSTM(input_size=(16, 16), input_dim=channel, hidden_dim=channel, kernel_size=(3, 3)))
        self.temp_att.append(ConvLSTM(input_size=(32, 32), input_dim=channel, hidden_dim=channel, kernel_size=(3, 3)))
        self.temp_att.append(ConvLSTM(input_size=(64, 64), input_dim=channel, hidden_dim=channel, kernel_size=(3, 3)))

        for ch in in_channels[0:]:
            self.align_modules.append(FAM(ch, channel))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        # self.dropout = nn.Dropout2d(0.1)




    def forward(self, features, x_b, x_t):# -> Tensor:
        features = features[::-1] #逆序序列

        # aa=self.temp_att[0](features[0].view(x_b, x_t, *features[0].shape[-3:]))[0]#.sum(dim=1)
        feat = features[0].view(x_b, x_t, *features[0].shape[-3:]).permute(1, 0, 2, 3, 4)
        out = self.align_modules[0](feat[2])




        iiiii=0
        for feat, align_module, output_conv,temp_conv in zip(features, self.align_modules[1:], self.output_convs,self.temp_att):

            iiiii+=1
            feat=feat.view(x_b, x_t, *feat.shape[-3:]).permute(1,0,2,3,4)


            out_1=align_module(feat[0], out)
            out_2=align_module(feat[1], out)
            out_3=align_module(feat[2], out)
            out=torch.stack([out_1,out_2,out_3], dim=1)
            _,out=temp_conv(out)
            out = output_conv(out[0][0])

        out = self.conv_seg(out)


        return out


class TPN(BaseModel):
    """Image Cascade Network"""

    def __init__(self, nclass, backbone='resnet101', pretrained_base=False, **kwargs):
        super(TPN, self).__init__()
        # self.backbone = SwinTransformer(in_channels=3,pretrain_img_size=256)
        self.backbone=ResNet_1_DCNV2_CA('101')
        # self.backbone=ResNet_1_DCNV2_CACAn('101')


        self.head = FaPNHead([256, 512, 1024, 2048], 128, nclass)
        # self.head = FaPNHead([64, 128, 256, 512], 128, nclass)


        # self.head = FaPNHead([96, 192, 384, 768], 128, nclass)


    def forward(self, x):

        x_b, x_t, x_c, x_h, x_w = x.size(0), x.size(1),x.size(2), x.size(3), x.size(4)

        x_view=x.view(-1,*x.shape[-3:])
        feature = self.backbone(x_view)


        out = self.head(feature, x_b, x_t)

        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # plt.matshow(out[0], cmap='viridis')


        # plt.imshow(out[0][1].detach().numpy(), cmap='viridis',interpolation= 'nearest')
        # plt.imshow(torch.max(out[0]).numpy(), cmap='viridis',interpolation= 'nearest')
        #
        # plt.colorbar()
        # plt.show()
        # import torchvision.transforms as transforms
        # out_path=r"C:\Users\lsc\Desktop\南农\_lsc_paper\Paper_1\pic\可视化\\"
        #
        # for i in range(x_b):
        #     pre = transforms.ToPILImage()(out[i].type(torch.float32))
        #     pre.save(out_path + str(i) + "_pre.png")

        return out

# 每次都加注意力，att_group组合，原始分割conv2d
if __name__ == '__main__':

    #解码时 全转无time
    model = TPN(2, backbone="ResNet", pretrained_base=False)

    x = torch.randn(4,3, 3, 256, 256)

    aa=nn.Conv2d(3,18,1,1,0)(x.view(-1,*x.shape[-3:]))
    aa=aa.view(4,3,*aa.shape[-3:])
    out = model(x)
    print(model)
    print(out.shape)