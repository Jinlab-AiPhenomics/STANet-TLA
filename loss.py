import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def compute_edges(image):
    # 计算边缘图的方法可以根据具体需求进行选择
    # 这里以 Sobel 滤波器为例进行示范
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()

    # image = image.unsqueeze(1)  # 添加 batch 维度和通道维度

    edges_x = nn.functional.conv2d(image, sobel_x, padding=1)
    edges_y = nn.functional.conv2d(image, sobel_y, padding=1)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)  # 计算边缘强度

    # 对边缘强度进行归一化处理（可选）
    # edges = (edges - edges.min()) / (edges.max() - edges.min())


    # edges_a=[]
    # for i in range(edges.size(0)):
    #     aaa = transforms.ToPILImage()(edges[0].type(torch.float32))
    #     edges_a.append(torch.from_numpy(np.array(aaa)))
    # edges=torch.stack(edges_a)

    # return edges # 移除通道维度，并返回边缘图
    return edges#.squeeze(1)  # 移除通道维度，并返回边缘图


def loss_edge(data_gen,data_real ):
    data_gen.requires_grad = True


    edge_real = compute_edges(data_real)
    edge_gen = compute_edges(data_gen)
    loss = nn.L1Loss(edge_real, edge_gen)

    return loss


class EdgeDistanceLoss(nn.Module):
    def __init__(self):
        super(EdgeDistanceLoss, self).__init__()

    def forward(self, pred, target):
        # 计算边缘图像
        edge_pred = F.canny(pred, threshold=0.5)
        edge_target = F.canny(target, threshold=0.5)

        # 计算边缘距离
        dist = torch.abs(edge_pred - edge_target)

        # 计算平均距离
        loss = torch.mean(dist)

        return loss


#
# # 使用示例
# edge_loss = EdgeLoss()
#
# # 假设 prediction 和 target 是形状为 (batch_size, 1, height, width) 的张量
# prediction = torch.randn(10, 2, 256, 256)
# target = torch.randn(10, 1, 256, 256)
#
# loss = edge_loss(prediction, target)
#
# print(loss)



class Edge_Loss_V1(nn.Module):

    def __init__(self):
        super(Edge_Loss_V1, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred.requires_grad = True

        # ed_true = y_true.float()
        # ed_pred = y_pred.float()
        ed_true = y_true
        ed_pred = y_pred

        ed_true = compute_edges(ed_true).type(torch.float32)
        ed_pred = compute_edges(ed_pred).type(torch.float32)

        #

        # 提取建筑物边缘，作为正样本
        # build_edge_bin = torch.mul(torch.where(ed_true < 0.5, 1.0, 0.0), torch.where(ed_true > 0.0, 1.0, 0.0))
        build_edge_bin = torch.where(ed_true > 150, 1.0, 0.0)#, torch.where(ed_true < 0, 1.0, 0.0))
        build_edge_true = torch.mul(build_edge_bin, ed_true)
        edge_pixel_num_true = torch.sum(build_edge_bin)
        edge_pixel_num_true = edge_pixel_num_true.float()

        # 提取背景边缘，作为负样本
        # background_edge_bin = torch.where(ed_true > 0.5, 0.1, 0.0)  # 把负样本中的30，40，50 变为3，4，5
        background_edge_bin = torch.where(ed_true < 150, 1.0, 0.0)
        background_edge_true = torch.mul(background_edge_bin, ed_true)
        edge_pixel_num_false = torch.sum(background_edge_bin) * 10.0
        edge_pixel_num_false = edge_pixel_num_false.float()


        # aaa = transforms.ToPILImage()(background_edge_bin[0].type(torch.float32))
        # ss=np.array(aaa)
        # aaa.show()
        #
        # bbb = transforms.ToPILImage()(ed_true[0].type(torch.float32))
        # sss=np.array(bbb)
        # bbb.show()

        # 正样本的权重 β， 负样本权重 1-β
        edge_pixel_num = edge_pixel_num_true + edge_pixel_num_false  # + 1.0
        beta = edge_pixel_num_true / edge_pixel_num

        # edge loss
        '''
        -y*log(1/(1+e^(-x))) = y*log(1+e^(-x))  当x为较大的负数时，损失会出现Nan，对此进一步化简
        y*log(1+e^(-x)) = y*log(1+e^(-x))+x*y - x*y = y*log(1-e^x) - x*y  
        合并：
        y*log(1+e^(-|x|)) - min(x, 0)
        '''
        build_loss = torch.sub(
            torch.mul(build_edge_bin,
                      torch.log1p(torch.exp(-torch.abs(ed_pred)))),
            torch.clamp(ed_pred, min=0)
        )
        build_loss = torch.mul(build_edge_true.float(), build_loss)

        background_loss = torch.add(
            torch.mul(background_edge_bin,
                      torch.log1p(torch.exp(-torch.abs(ed_pred)))),
            torch.clamp(ed_pred, max=0)
        )
        background_loss = torch.mul(background_edge_true.float(), background_loss)

        '''
        loss = β * build_edge_loss / edge_pixel_num_true 
        + (1-β) * background_edge_loss / edge_pixel_num_false
        '''
        edge_loss = torch.add(torch.mul(beta, torch.divide(torch.sum(build_loss), edge_pixel_num_true)),
                              torch.mul(torch.sub(1.0, beta),
                                        torch.divide(torch.sum(background_loss), edge_pixel_num_false)))

        # edge_loss = torch.add(torch.mul(beta, build_loss),
        #                    torch.mul((1.0 - beta), background_loss))

        # torch.print(torch.reshape(background_loss, (128,128)), summarize=-1)
        return edge_loss


# 3333333333333333333333333333333
def active_contour_loss(y_pred, y_true, weight=10):
    '''
    :  Learning Active Contour Models for Medical Image Segmentation
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''

    y_pred.requires_grad = True

    # y_true = y_true.float()
    # y_pred = y_pred.float()

    # y_true = compute_edges(y_true).unsqueeze(1)
    # y_pred = compute_edges(y_pred).unsqueeze(1)
    # plt.plot(transforms.ToPILImage()(y_true[0][0].type(torch.float32)))
    # plt.show()
    # aaa=transforms.ToPILImage()(y_pred[0][0].type(torch.float32))
    # aaa.show()


    # length term
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # region term
    c_in = torch.ones_like(y_pred)
    c_out = torch.zeros_like(y_pred)

    region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
    region = region_in + region_out



    # https://zhuanlan.zhihu.com/p/77551832
    loss = lenth + weight * region
    # loss = weight *lenth +  region


    return loss



import torch.nn.functional as F

def combined_loss(output, target):
    output.requires_grad = True


    # 计算边缘定位损失
    # sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    # sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    # gx = F.conv2d(output, sobel_x)
    # gy = F.conv2d(output, sobel_y)
    # grad = torch.sqrt(gx * gx + gy * gy)
    grad=compute_edges(output)
    # edge_weight = torch.where(grad > 0, torch.tensor(10.0).cuda(), torch.tensor(1.0).cuda())
    edge_weight = target.cuda()

    edge_loc_loss = F.binary_cross_entropy_with_logits(grad, edge_weight)

    # 计算边缘平滑损失
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).cuda()
    laplacian = nn.functional.conv2d(output, laplacian_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    edge_smooth_loss = F.binary_cross_entropy_with_logits(laplacian, edge_weight)

    # 组合边缘定位损失和边缘平滑损失
    edge_loss = edge_loc_loss + edge_smooth_loss

    # 组合综合损失
    # loss = ce_weight * ce_loss + (1 - ce_weight) * edge_loss

    return edge_loss




#4444444444444444444444444
def edge_lossnn(output, target):
    output.requires_grad = True


    # 计算边缘定位损失

    grad=compute_edges(output)
    edge_weight = compute_edges(target)

    edge_loss = F.mse_loss(grad, edge_weight)



    return edge_loss


# 555555555555555555555      ×××××××
def edge_lossnnn(output, target):
    output.requires_grad = True


    # 计算边缘定位损失

    grad=compute_edges(output)
    edge_weight = compute_edges(target)

    edge_loc_loss = F.mse_loss(grad, edge_weight)

    # 计算边缘平滑损失
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).cuda()
    laplacian = nn.functional.conv2d(output, laplacian_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    edge_smooth_loss = F.mse_loss(laplacian, edge_weight)

    # 组合边缘定位损失和边缘平滑损失
    edge_loss = (edge_loc_loss + edge_smooth_loss)/2


    return edge_loss

import torch
import torch.nn as nn

#6666666666666666666666666666666666666
class ActiveContourLoss(nn.Module):
    #在模块的前向函数中，我们首先对输入和目标进行了sigmoid变换。然后，我们计算了输入和目标的边缘梯度，并使用它们计算了边缘势能。接着，
    # #我们计算了输入和目标的线段势能，并使用它们计算了总势能。最后，我们使用总势能计算了损失函数。
    def __init__(self, alpha=0.1, beta=0.2, w_line=1.0, w_edge=1.0):
        super(ActiveContourLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.w_line = w_line
        self.w_edge = w_edge

    def forward(self, input, target):
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)

        # 计算边缘梯度
        dx_input = torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])
        dy_input = torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :])
        dx_target = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        dy_target = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        # 计算边缘势能
        C_in = torch.sum(dx_input ** 2 + dy_input ** 2)
        C_tar = torch.sum(dx_target ** 2 + dy_target ** 2)

        # 计算线段势能
        L_in = torch.sum((input - target) ** 2)
        L_tar = torch.sum((target - 0.5) ** 2)

        # 计算总势能
        E_in = self.alpha * L_in + self.beta * C_in
        E_tar = self.alpha * L_tar + self.beta * C_tar

        # 计算损失函数
        loss = self.w_line * (E_in - E_tar) + self.w_edge * C_in

        return loss.mean()


class SoftDiceLoss(nn.Module): #×
    def __init__(self, smooth=1., dims=(-2,-1)):
        super(SoftDiceLoss,self).__init__()
        self.smooth = smooth
        self.dims = dims
    def forward(self,x,y):
        x.requires_grad = True

        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

def dice_loss(pred, target):
    pred.requires_grad = True

    smooth = 1.
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    loss = 1 - dice
    return loss

if __name__ == '__main__':
    edge_loss = Edge_Loss_V1()

    # 假设 prediction 和 target 是形状为 (batch_size, 1, height, width) 的张量
    prediction = torch.randn(10, 2, 256, 256)
    target = torch.randn(10, 1, 256, 256)

    loss = active_contour_loss(prediction, target)
    # loss = dice_loss(prediction, target)

    #
    # loss = SoftDiceLoss()(prediction, target)



    print(loss)
