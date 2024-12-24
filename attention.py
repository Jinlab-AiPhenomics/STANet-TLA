import torch
import torch.nn as nn

#时序注意力机制
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim_size):
        """
            Assumes input will be in the form (batch, time_steps, hidden_dim_size, height, width)
            Returns reweighted timestamps.
            Implementation based on the following blog post:
            https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69
            我们是如何获得注意力权重的？图的（b）部分从上到下进行了说明。从隐藏状态的集合开始，它乘以权重矩阵，
            然后是用于非线性变换的tanh层。然后用另一个权重矩阵将另一个线性变换应用于输出，以获得预注意矩阵。
            一个 softmax 层，它按行方向应用于预注意矩阵，使其权重看起来像隐藏状态上的概率分布。
        """
        super(TemporalAttention, self).__init__()
        self.w_s1 = nn.Linear(in_features=hidden_dim_size, out_features=32, bias=False)
        self.w_s2 = nn.Linear(in_features=32, out_features=1, bias=False)
        nn.init.constant_(self.w_s1.weight, 1)
        nn.init.constant_(self.w_s2.weight, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2).contiguous()  #输入(batch, time_steps, hidden_dim_size, height, width)
        # x = x.permute(0, 2,3,4,1).contiguous() #输入(batch,hidden_dim_size， time_steps,, height, width)
        z1 = self.tanh(self.w_s1(x))
        attn_weights = self.softmax(self.w_s2(z1))

        # attn_weights = attn_weights.permute(0, 1, 4, 2, 3).contiguous()
        # attn_weights = attn_weights.permute(0,4,1, 2, 3).contiguous()
        # attn_weights = attn_weights.permute(0, 1, 4, 2, 3).contiguous()
        reweighted = attn_weights * x
        reweighted = reweighted.permute(0, 1, 4, 2, 3).contiguous()
        return reweighted#,attn_weights



# -----------------------------------------------------




#边缘注意力机制
#使用方法：特征提取后 x = x * edge
class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = self.conv(x)
        edge = self.sigmoid(edge)

        return edge
# -----------------------------------------------------



#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# -----------------------------------------------------


#空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# -----------------------------------------------------


class CBAM_block(nn.Module):
    def __init__(self, in_channels):
        super(CBAM_block, self).__init__()
        self.ca_att=ChannelAttention(in_channels)
        self.sa_att=SpatialAttention()

    def forward(self,x):
        ca=self.ca_att(x)*x
        output=self.sa_att(ca)*ca

        return output



#-----------------------------------------------------


#SE 模块
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


#-----------------------------------------------------

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        #out = a_h.expand_as(x) * a_w.expand_as(x) * identity
        return out


#-----------------------------------------------------+

class CoordAtt_3d(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_3d, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((None,None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((None,1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c,t, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1,2, 4, 3)

        y = torch.cat([x_h, x_w], dim=3)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=3)
        x_w = x_w.permute(0, 1,2, 4, 3)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        #out = a_h.expand_as(x) * a_w.expand_as(x) * identity
        return out


if __name__ == '__main__':
    input_tensor = torch.randn(2,2048,128,128)
    model = CoordAtt(2048,2048)
    ca = model(input_tensor)
    print(ca.size())  # 输出大小为(batch_size, input_size)

    # # # # 使用示例
    # input_size = (256,256)
    # hidden_size = 16  # 与input_size保持一致
    # batch_size = 2
    # time_steps = 18
    #
    # input_tensor = torch.randn(batch_size, time_steps,hidden_size, input_size[0],input_size[1])
    #
    # attention = TemporalAttention(hidden_size)
    # attended_output = attention(input_tensor)
    #
    # print(attended_output.size())  # 输出大小为(batch_size, input_size)
    # # print(attended_output)