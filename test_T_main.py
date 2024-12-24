#所有输入  小图总精度

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import numpy as np
from STANet.src import VSegDataset_N


from models.cc_model2_H import TPN



# c_H 48 f1  E:\Soybean_Data\out_n\1.19\c_H\\ iou0.8510   img=float32  T=3
# all_Eval on test set:epoch:5 acc:0.9453，miou:0.8649，class_1_acc:0.9453，class_2_acc:0.9453，class_1_pre:0.9517，class_2_pre:0.9087，class_1_iou:0.8789，
# class_2_iou:0.8510，class_1_rec:0.9164，class_2_rec:0.9275，class_1_f1:0.9316，class_2_f1:0.9166
#all_Eval on test set:epoch:5 acc:0.9440，miou:0.8775，class_1_acc:0.9440，class_2_acc:0.9440，class_1_pre:0.9526，class_2_pre:0.9188，class_1_iou:0.8860，
# class_2_iou:0.8690，class_1_rec:0.9249，class_2_rec:0.9406，class_1_f1:0.9374，class_2_f1:0.9291


# new_test
#all_Eval on test set:epoch:5 acc:0.9446，miou:0.8641，class_1_acc:0.9446，class_2_acc:0.9446，class_1_pre:0.9508，class_2_pre:0.9098，class_1_iou:0.8757，
# class_2_iou:0.8525，class_1_rec:0.9136，class_2_rec:0.9287，class_1_f1:0.9296，class_2_f1:0.9177
#all_Eval on test set:epoch:5 acc:0.9446，miou:0.8495，class_1_acc:0.9446，class_2_acc:0.9446，class_1_pre:0.9490，class_2_pre:0.8962，class_1_iou:0.8707，
# class_2_iou:0.8283，class_1_rec:0.9100，class_2_rec:0.9106，class_1_f1:0.9249，class_2_f1:0.8951

#fapn
#all_Eval on test set:epoch:5 acc:0.9413，miou:0.8581，class_1_acc:0.9629，class_2_acc:0.8981，class_1_pre:0.9563，class_2_pre:0.8914，class_1_iou:0.8690，
# class_2_iou:0.8473，class_1_rec:0.9008，class_2_rec:0.9421，class_1_f1:0.9256，class_2_f1:0.9144
#all_Eval on test set:epoch:5 acc:0.9413，miou:0.8436，class_1_acc:0.9608，class_2_acc:0.8882，class_1_pre:0.9543，class_2_pre:0.8755，class_1_iou:0.8639，
# class_2_iou:0.8232，class_1_rec:0.8976，class_2_rec:0.9257，class_1_f1:0.9205，class_2_f1:0.8912




## 3.12new 39 aall_Eval on test set:epoch:5 acc:0.9447，miou:0.8642，class_1_acc:0.9447，class_2_acc:0.9447，class_1_pre:0.9539，class_2_pre:0.9075，
# class_1_iou:0.8742，class_2_iou:0.8543，class_1_rec:0.9092，class_2_rec:0.9334，class_1_f1:0.9282，class_2_f1:0.9189



#qy bm
# 23 all_Eval on test set:epoch:5 acc:0.9520，miou:0.8950，class_1_acc:0.9520，class_2_acc:0.9520，class_1_pre:0.9476，class_2_pre:0.9420，
# class_1_iou:0.8947，class_2_iou:0.8952，class_1_rec:0.9392，class_2_rec:0.9471，class_1_f1:0.9432，class_2_f1:0.9443

#49 all_Eval on test set:epoch:5 acc:0.9520，miou:0.8950，class_1_acc:0.9520，class_2_acc:0.9520，class_1_pre:0.9492，class_2_pre:0.9386，
# class_1_iou:0.8947，class_2_iou:0.8953，class_1_rec:0.9375，class_2_rec:0.9507，class_1_f1:0.9432，class_2_f1:0.9444

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_=False
# data_path_txt = r'E:\Soybean_Data\ybj_256\3.12_xz'  # xz'
# data_path_txt = r'E:\Soybean_Data\transfer_2023\ybj_256'  # xz'
data_path_txt = r'E:\Soybean_Data\chm_2022\ybj256'
# data_path_txt = r'E:\Soybean_Data\bm_2022\ybj256'

# data_path=r'E:\Soybean_Data\ybj_256'
# data_path=r'E:\Soybean_Data\transfer_2023\ybj_256'
data_path=r'E:\Soybean_Data\chm_2022\ybj256'
# data_path=r'E:\Soybean_Data\bm_2022\ybj256'



out_path=r"E:\Soybean_Data\out_n\3.12\\"   #epoch_test_best_39_model_iou
# out_path=r"E:\Soybean_Data\transfer_2023\ybj_256\qy\\"
# out_path=r"F:\DL_data\DL_data_chm_256\qy\\"   # epoch_test_best_29_model_f1 f1 epoch_best_46_model_f1 epoch_best_52_model_loss
# out_path=r"E:\Soybean_Data\bm_2022\ybj256\qy\\"    #  epoch_test_best_49_model_f1


pth_path=out_path+"epoch_test_best_39_model_iou.pth"


record_log=open(out_path+"test_acc1_etoz48f11.txt","w")
# record_logo=open(out_path+"out\\"+"test_tjz52f11.txt","w")
# record_log_f1_low=open(out_path+"test_acc1snn_f1_low.txt","w")

ep_size=5
ep_sav=5
best_epoch=39
modellx="f1"
from torchvision.ops import DeformConv2d
from torch import nn

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
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


model = TPN(2, backbone="ResNet", pretrained_base=False).cuda()

# mmodel_path = out_path+'epoch_'+str(ep_i*ep_sav)+'_model.pth'
# model_path = out_path+'epoch_best_'+str(best_epoch)+'_model_'+modellx+'.pth'
model_path = pth_path

model.load_state_dict(torch.load(model_path))

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.35,0.35,0.35], std=[0.21,0.21,0.21])
    transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

])

# if "chm" in data_path:
#     print("ssssssssssssssssssss")
#     model.backbone.conv1 = DCNv2(1, 64, 7, 2, 3).cuda()
#     model.backbone.offset1= nn.Conv2d(1, 64, 1, bias=False).cuda()
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.35], std=[0.21])
#         # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#
#     ])
# else:
#     transform = transforms.Compose([
#         # transforms.Resize((512, 512)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#     ])


test_dataset = VSegDataset_N(data_path,data_path_txt,'test',transform)
test_dataloader = DataLoader(test_dataset, batch_size=4)


print(device)


#测试
model.eval()
for ep_i in range(1,int(ep_size/ep_sav)+1):



    with torch.no_grad():
        acc_a = 0
        miou_a=0
        class_iou_a=[0,0]
        class_acc_a=[0,0]
        class_pre_a=[0,0]
        class_rec_a=[0,0]
        class_f1_a =[0,0]
        total = 0

        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)


            _, predicted = torch.max(outputs.data, 1)
            labels = torch.squeeze(labels, dim=1)


            if save_==True:
                # predicted=predicted.transpose(0,1).contiguous()
                # labels=labels.transpose(0,1).contiguous()

                for j in range(predicted.shape[0]):
                    pre = transforms.ToPILImage()(predicted[j].type(torch.float32))
                    pre.save(out_path+"\\out\\" +str(i)+"_"+str(j)+"_pre.png")
                    lab = transforms.ToPILImage()(labels[j].type(torch.float32))
                    lab.save(out_path+"\\out\\" + str(i) + "_" + str(j) + "_lab.png")

                    tp1 = ((predicted[j] == 1) & (labels[j] == 1)).sum().item()
                    tn1 = ((predicted[j] != 1) & (labels[j] != 1)).sum().item()
                    fp1 = ((predicted[j] == 1) & (labels[j] != 1)).sum().item()
                    fn1 = ((predicted[j] != 1) & (labels[j] == 1)).sum().item()
                    iou_c1 = tp1 / (tp1 + fp1 + fn1 + 1e-6)  # 交并比
                    record_logo.write(str(i)+ " " +str(j)+ " " +str(tp1) + " " + str(tn1) + " " + str(fp1) + " " + str(fn1) + " " + str(iou_c1)+"\n")

            # 计算各种评价指标
            accuracy = (predicted == labels).float().mean()
            class_iou = []
            class_acc = []
            class_pre = []
            class_rec = []
            class_f1 = []
            # for j in range(predicted.shape[0]):
            for c in range(2):
                tp = ((predicted == c) & (labels == c)).sum().item()
                tn = ((predicted != c) & (labels != c)).sum().item()
                fp = ((predicted == c) & (labels != c)).sum().item()
                fn = ((predicted != c) & (labels == c)).sum().item()

                acc_c = (tp + tn) / (tp+tn + fn + fp+ 1e-6)
                class_acc.append(acc_c)

                pre_c = tp / (tp + fp + 1e-6)  # 查准
                class_pre.append(pre_c)

                iou_c = tp / (tp + fp + fn + 1e-6)  # 交并比
                class_iou.append(iou_c)

                rec_c = tp / (tp + fn + 1e-6)  # 查全
                class_rec.append(rec_c)

                f1_c = 2 * pre_c * rec_c / (pre_c + rec_c+ 1e-6)  # 查准与查全调和平均
                class_f1.append(f1_c)


                # ap = 0
                # for t in torch.arange(0, 1.1, 0.1):
                #     mask = rec_c >= t
                #     if mask.any():
                #         ap += torch.max(pre_c[mask])
                # ap /= 11

            miou = np.mean(class_iou)

            total += 1
            acc_a += accuracy.item()
            miou_a += miou
            class_iou_a = [class_iou[0] + class_iou_a[0], class_iou[1] + class_iou_a[1]]
            class_acc_a = [class_acc[0] + class_acc_a[0], class_acc[1] + class_acc_a[1]]
            class_pre_a = [class_pre[0] + class_pre_a[0], class_pre[1] + class_pre_a[1]]
            class_rec_a = [class_rec[0] + class_rec_a[0], class_rec[1] + class_rec_a[1]]
            class_f1_a = [class_f1[0] + class_f1_a[0], class_f1[1] + class_f1_a[1]]

            # if f1_low>class_f1[1] and class_f1[1]>0.1:
            #     f1_low=class_f1[1]
            #     print("------------------"+str(i)+"  "+str(f1_low))
            #     record_log.write("------------------"+str(i)+"  "+str(f1_low))
            # 输出结果
            # print("test_"+str(i),'accuracy:', accuracy.item(),'，miou:', miou,'，class accuracy:', class_pre,'，class iou:', class_iou)

    print(
        'all_Eval on test set:epoch:{} acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，'
        'class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            ep_i * ep_sav,
            acc_a / total, miou_a / total, class_acc_a[0] / total, class_acc_a[1] / total,
            class_pre_a[0] / total, class_pre_a[1] / total, class_iou_a[0] / total, class_iou_a[1] / total,
            class_rec_a[0] / total, class_rec_a[1] / total, class_f1_a[0] / total, class_f1_a[1] / total))

    record_log.write(
        'all_Eval on test set:epoch:{} acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，'
        'class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            ep_i * ep_sav,
            acc_a / total, miou_a / total, class_acc_a[0] / total, class_acc_a[1] / total,
            class_pre_a[0] / total, class_pre_a[1] / total, class_iou_a[0] / total, class_iou_a[1] / total,
            class_rec_a[0] / total, class_rec_a[1] / total, class_f1_a[0] / total,
            class_f1_a[1] / total) + "\n")
    record_log.flush()
