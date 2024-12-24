import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from STANet.src import VSegDataset_N

# from models.cc_model2_H import TPN
from models.cc_STANet1016 import TPN

import os

import numpy as np
import time

# 数据集路径
# # data_path_txt = r'E:\Soybean_Data\ybj_256\data_Stant\T=3n2'
# data_path_txt = r'F:\11\2022_bm\ybj\ybj256' # xz'
#
# data_path=r'F:\11\2022_bm\ybj\ybj256'
# # sav_path=r"E:\Soybean_Data\out_STANT\xr\new\res18_nattgrop_lrd_n3o\\"
#
# # sav_path=r"E:\Soybean_Data\out_STANT\newT3nn\\"
# sav_path=r"F:\11\2022_bm\ybj\ybj256\\no qy\\"

data_path_txt = r'E:\Soybean_Data\ybj_256\3.12_xz'

data_path=r'E:\Soybean_Data\ybj_256'

sav_path=r"E:\Soybean_Data\out_n\nnn1016\\"


record_log=open(sav_path+str(time.strftime('%Y%S%d%H%M%S'))+"_log.txt","w")


# 超参数
num_classes = 2
batch_size = 4
time=3
num_epochs = 200
learning_rate = 0.001
sav_pth=5
first_epoch=0

test_pf=False
train_continue=False

#检查是否有可用的 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Forward pass through the model

# model=sel_model("PSPNet").to(device)
# model=Model1(2,batch_size,time, backbone="ResNet", pretrained_base=False).cuda()
# model=Model2(2,batch_size,time, backbone="ResNet", pretrained_base=False).cuda()
# model=Model3(2,backbone="ResNet", pretrained_base=False).cuda()
# model=Model4(2,backbone="ResNet", pretrained_base=False).cuda()
# # model = Model_5(2, backbone="ResNet", pretrained_base=False).cuda()
# model = Model_5_n(2, backbone="ResNet", pretrained_base=False).cuda()
# model = Model_5_n_out5(2, backbone="ResNet", pretrained_base=False).cuda()
# model = XXall(2, backbone="ResNet", pretrained_base=False).cuda()

# model = utae_ca_fam_udcnv2_skipb(input_dim=3).cuda()
model = TPN(2, backbone="ResNet", pretrained_base=False).cuda()

print(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #SGD  ASGD  Adam
criterion = nn.CrossEntropyLoss()#    CrossEntropyLoss  MSELoss  BCE Loss  binary_cross_entropy


# 如果训练中断 加载模型参数# 加载优化器状态
if train_continue:
    model.load_state_dict(torch.load(r'E:\Soybean_Data\out_n\3.12\\epoch_test_best_39_model_iou.pth'))


transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(  mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375])

    # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.35, 0.44], std=[0.229, 0.224, 0.225, 0.21, 0.18])
])

train_dataset = VSegDataset_N(data_path,data_path_txt,'train',transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = VSegDataset_N(data_path,data_path_txt,'val',transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = VSegDataset_N(data_path,data_path_txt,'test',transform)
test_dataloader = DataLoader(test_dataset)

best_ciou = 0
best_cf1 = 0
best_epoch_iou = 1
best_epoch_f1=1
best_loss=99
best_epoch_loss=1
update_lr_val_loss = [999, 0]


best_test_ciou=0
best_test_cf1 = 0
best_test_epoch_iou = 1
best_test_epoch_f1=1

for epoch in range(first_epoch, num_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}")
    train_loss = 0.0
    model.train()
    t_acc_a = 0
    t_miou_a = 0
    t_class_iou_a = [0, 0]
    t_class_acc_a = [0, 0]
    t_class_pre_a = [0, 0]
    t_class_rec_a = [0, 0]
    t_class_f1_a = [0, 0]
    t_total = 0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # loss1 = criterion(outputs[1], labels)
        # loss2 = criterion(outputs[2], labels)
        # loss3 = criterion(outputs[3], labels)
        # loss = loss0 + 0.8 * loss3 + 0.3 * loss2 + 0.1 * loss1


        # 将标签张量展平为大小为 (batch_size * TIME, width, height) 的整数张量
        # output= outputs.contiguous().view(-1, outputs.size(2), outputs.size(3), outputs.size(4))  # 形状为 [2 * 8 * 256 * 256, 2]
        # labels=labels.contiguous().view(-1,labels.size(2), labels.size(3), labels.size(4))

        # output= outputs.permute(0, 1, 3, 4, 2).contiguous().view(-1, 2)  # 形状为 [2 * 8 * 256 * 256, 2]
        # target=labels.permute(0, 1, 3, 4, 2).contiguous().view(-1).long()
        # loss = criterion(output, target)


        # loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 计算各种评价指标
        _, predicteddd = torch.max(outputs.data, 1)
        labelsss = torch.squeeze(labels, dim=1).contiguous()

        accuracy = (predicteddd == labelsss).float().mean()
        class_iou = []
        class_acc = []
        class_pre = []
        class_rec = []
        class_f1 = []
        for c in range(2):
            tp = ((predicteddd == c) & (labelsss == c)).sum().item()
            tn = ((predicteddd != c) & (labelsss != c)).sum().item()
            fp = ((predicteddd == c) & (labelsss != c)).sum().item()
            fn = ((predicteddd != c) & (labelsss == c)).sum().item()

            iou_c = tp / (tp + fp + fn)
            class_iou.append(round(iou_c, 4))
            acc_c = (tp + tn) / (tp + tn + fn + fp)
            class_acc.append(round(acc_c, 4))
            class_pre_c = tp / (tp + fp + 1e-6)
            class_pre.append(round(class_pre_c, 4))
            rec_c = tp / (tp + fn + 1e-6)
            class_rec.append(round(rec_c, 4))
            f1_c = 2 * class_pre_c * rec_c / (class_pre_c + rec_c + 1e-6)
            class_f1.append(round(f1_c, 4))


        miou = np.mean(class_iou)

        t_total += 1
        t_acc_a += accuracy.item()
        t_miou_a += miou
        t_class_iou_a = [class_iou[0] + t_class_iou_a[0], class_iou[1] + t_class_iou_a[1]]
        t_class_acc_a = [class_acc[0] + t_class_acc_a[0], class_acc[1] + t_class_acc_a[1]]
        t_class_pre_a = [class_pre[0] + t_class_pre_a[0], class_pre[1] + t_class_pre_a[1]]
        t_class_rec_a = [class_rec[0] + t_class_rec_a[0], class_rec[1] + t_class_rec_a[1]]
        t_class_f1_a = [class_f1[0] + t_class_f1_a[0], class_f1[1] + t_class_f1_a[1]]


        print(f"Epoch { epoch + 1}/{num_epochs}，Train_Iteration {i + 1}/{len(train_dataloader)}: Loss = {loss.item():.4f}", 'accuracy:', accuracy.item(),
              '，miou:', miou, '，class accuracy:',class_acc,'，class precision:', class_pre,'，class iou:', class_iou,'class rec:', class_rec,'class f1:', class_f1)


    # Validation loop
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        v_acc_a = 0
        v_miou_a = 0
        v_class_iou_a = [0, 0]
        v_class_acc_a = [0, 0]
        v_class_pre_a = [0, 0]
        v_class_rec_a = [0, 0]
        v_class_f1_a = [0, 0]
        v_total = 0
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)

            # output = outputs.permute(0, 1, 3, 4, 2).contiguous().view(-1, 2)  # 形状为 [2 * 8 * 256 * 256, 2]
            # target = labels.permute(0, 1, 3, 4, 2).contiguous().view(-1).long()
            # loss = criterion(outputs, labels.long())
            loss = criterion(outputs, labels)
            # loss1 = criterion(outputs[1], labels)
            # loss2 = criterion(outputs[2], labels)
            # loss3 = criterion(outputs[3], labels)
            # loss=loss+0.8*loss3+0.3*loss2+0.1*loss1

            val_loss += loss.item()

            # 计算各种评价指标
            _, predicteddd = torch.max(outputs.data, 1)
            labelsss = torch.squeeze(labels, dim=1).contiguous()

            accuracy = (predicteddd == labelsss).float().mean()
            class_iou = []
            class_acc = []
            class_pre = []
            class_rec = []
            class_f1 = []
            for c in range(2):
                tp = ((predicteddd == c) & (labelsss == c)).sum().item()
                tn = ((predicteddd != c) & (labelsss != c)).sum().item()
                fp = ((predicteddd == c) & (labelsss != c)).sum().item()
                fn = ((predicteddd != c) & (labelsss == c)).sum().item()
                iou_c = tp / (tp + fp + fn)
                class_iou.append(round(iou_c,4))
                acc_c = (tp + tn) / (tp+tn + fn + fp)
                class_acc.append(round(acc_c,4))
                class_pre_c = tp / (tp + fp + 1e-6)
                class_pre.append(round(class_pre_c,4))
                rec_c = tp / (tp + fn + 1e-6)
                class_rec.append(round(rec_c,4))
                f1_c = 2 * class_pre_c * rec_c / (class_pre_c + rec_c + 1e-6)
                class_f1.append(round(f1_c,4))

            miou = np.mean(class_iou)

            v_total += 1
            v_acc_a += accuracy.item()
            v_miou_a += miou
            v_class_iou_a = [class_iou[0] + v_class_iou_a[0], class_iou[1] + v_class_iou_a[1]]
            v_class_acc_a = [class_acc[0] + v_class_acc_a[0], class_acc[1] + v_class_acc_a[1]]
            v_class_pre_a = [class_pre[0] + v_class_pre_a[0], class_pre[1] + v_class_pre_a[1]]
            v_class_rec_a = [class_rec[0] + v_class_rec_a[0], class_rec[1] + v_class_rec_a[1]]
            v_class_f1_a = [class_f1[0] + v_class_f1_a[0], class_f1[1] + v_class_f1_a[1]]

            print(
                f"Epoch {epoch + 1}/{num_epochs}，Val_Iteration {i + 1}/{len(val_dataloader)}: Loss = {loss.item():.4f}",
                'accuracy:', accuracy.item(),
                '，miou:', miou, '，class accuracy:', class_acc, '，class precision:', class_pre, '，class iou:', class_iou,
                'class rec:', class_rec, 'class f1:', class_f1)


        test_acc_a = 0
        test_miou_a=0
        test_class_iou_a=[0,0]
        test_class_acc_a=[0,0]
        test_class_pre_a=[0,0]
        test_class_rec_a=[0,0]
        test_class_f1_a =[0,0]
        test_total = 0

        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)


            _, predicted = torch.max(outputs.data, 1)
            labels = torch.squeeze(labels, dim=1)


            # if save_==True:
            #     # predicted=predicted.transpose(0,1).contiguous()
            #     # labels=labels.transpose(0,1).contiguous()
            #
            #     for j in range(predicted.shape[0]):
            #         pre = transforms.ToPILImage()(predicted[j].type(torch.float32))
            #         pre.save(out_path+"\\out\\" +str(i)+"_"+str(j)+"_pre.png")
            #         lab = transforms.ToPILImage()(labels[j].type(torch.float32))
            #         lab.save(out_path+"\\out\\" + str(i) + "_" + str(j) + "_lab.png")
            #
            #         tp1 = ((predicted[j] == 1) & (labels[j] == 1)).sum().item()
            #         tn1 = ((predicted[j] != 1) & (labels[j] != 1)).sum().item()
            #         fp1 = ((predicted[j] == 1) & (labels[j] != 1)).sum().item()
            #         fn1 = ((predicted[j] != 1) & (labels[j] == 1)).sum().item()
            #         iou_c1 = tp1 / (tp1 + fp1 + fn1 + 1e-6)  # 交并比
            #         record_logo.write(str(i)+ " " +str(j)+ " " +str(tp1) + " " + str(tn1) + " " + str(fp1) + " " + str(fn1) + " " + str(iou_c1)+"\n")

            # 计算各种评价指标
            test_accuracy = (predicted == labels).float().mean()
            test_class_iou = []
            test_class_acc = []
            test_class_pre = []
            test_class_rec = []
            test_class_f1 = []
            # for j in range(predicted.shape[0]):
            for c in range(2):
                tp = ((predicted == c) & (labels == c)).sum().item()
                tn = ((predicted != c) & (labels != c)).sum().item()
                fp = ((predicted == c) & (labels != c)).sum().item()
                fn = ((predicted != c) & (labels == c)).sum().item()

                acc_c = (tp + tn) / (tp+tn + fn + fp+ 1e-6)
                test_class_acc.append(acc_c)

                pre_c = tp / (tp + fp + 1e-6)  # 查准
                test_class_pre.append(pre_c)

                iou_c = tp / (tp + fp + fn + 1e-6)  # 交并比
                test_class_iou.append(iou_c)

                rec_c = tp / (tp + fn + 1e-6)  # 查全
                test_class_rec.append(rec_c)

                f1_c = 2 * pre_c * rec_c / (pre_c + rec_c+ 1e-6)  # 查准与查全调和平均
                test_class_f1.append(f1_c)


                # ap = 0
                # for t in torch.arange(0, 1.1, 0.1):
                #     mask = rec_c >= t
                #     if mask.any():
                #         ap += torch.max(pre_c[mask])
                # ap /= 11

            test_miou = np.mean(test_class_iou)

            test_total += 1
            test_acc_a += test_accuracy.item()
            test_miou_a += test_miou
            test_class_iou_a = [test_class_iou[0] + test_class_iou_a[0], test_class_iou[1] + test_class_iou_a[1]]
            test_class_acc_a = [test_class_acc[0] + test_class_acc_a[0], test_class_acc[1] + test_class_acc_a[1]]
            test_class_pre_a = [test_class_pre[0] + test_class_pre_a[0], test_class_pre[1] + test_class_pre_a[1]]
            test_class_rec_a = [test_class_rec[0] + test_class_rec_a[0], test_class_rec[1] + test_class_rec_a[1]]
            test_class_f1_a = [test_class_f1[0] + test_class_f1_a[0], test_class_f1[1] + test_class_f1_a[1]]



    # 更新学习率 5个val_loss不更新, 学习率衰减
    if val_loss/ len(val_dataloader) > update_lr_val_loss[0]:
        update_lr_val_loss[1] = 1 + update_lr_val_loss[1]
        if update_lr_val_loss[1]==5 and learning_rate*0.5>=0.0000625:
            learning_rate=learning_rate*0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                update_lr_val_loss[1] = 0

    else:
        update_lr_val_loss[0]=val_loss/ len(val_dataloader)
        update_lr_val_loss[1]=0

    print(val_loss/ len(val_dataloader),update_lr_val_loss)


    print('Epoch [{}/{}], Train Loss: {:.6f}, Val Loss: {:.6f}, lr:{:.8}'.format(epoch + 1, num_epochs, train_loss / len(train_dataloader),val_loss / len(val_dataloader),optimizer.param_groups[0]['lr']),
          'all_Train on test set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            t_acc_a / t_total, t_miou_a / t_total, t_class_acc_a[0] / t_total, t_class_acc_a[1] / t_total,t_class_pre_a[0] / t_total, t_class_pre_a[1] / t_total, t_class_iou_a[0] / t_total,t_class_iou_a[1] / t_total, t_class_rec_a[0] / t_total,t_class_rec_a[1] / t_total, t_class_f1_a[0] / t_total,t_class_f1_a[1] / t_total),
          'all_Eval on val set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，''class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            v_acc_a / v_total, v_miou_a / v_total, v_class_acc_a[0] / v_total, v_class_acc_a[1] / v_total,v_class_pre_a[0] / v_total, v_class_pre_a[1] / v_total, v_class_iou_a[0] / v_total,v_class_iou_a[1] / v_total, v_class_rec_a[0] / v_total, v_class_rec_a[1] / v_total,v_class_f1_a[0] / v_total, v_class_f1_a[1] / v_total),
          'all_Eval on test set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，''class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
              test_acc_a / test_total, test_miou_a / test_total, test_class_acc_a[0] / test_total, test_class_acc_a[1] / test_total,
              test_class_pre_a[0] / test_total, test_class_pre_a[1] / test_total, test_class_iou_a[0] / test_total,
              test_class_iou_a[1] / test_total, test_class_rec_a[0] / test_total, test_class_rec_a[1] / test_total,
              test_class_f1_a[0] / test_total, test_class_f1_a[1] / test_total))


    record_log.write('Epoch [{}/{}], Train Loss: {:.6f}, Val Loss: {:.6f}, lr:{:.8}'.format(epoch + 1, num_epochs, train_loss / len(train_dataloader),val_loss / len(val_dataloader),optimizer.param_groups[0]['lr'])+
            str('；all_Train on test set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            t_acc_a / t_total, t_miou_a / t_total, t_class_acc_a[0] / t_total, t_class_acc_a[1] / t_total,t_class_pre_a[0] / t_total, t_class_pre_a[1] / t_total, t_class_iou_a[0] / t_total,t_class_iou_a[1] / t_total, t_class_rec_a[0] / t_total,t_class_rec_a[1] / t_total, t_class_f1_a[0] / t_total,t_class_f1_a[1] / t_total))
            +str('；all_Eval on val set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，''class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            v_acc_a / v_total, v_miou_a / v_total, v_class_acc_a[0] / v_total, v_class_acc_a[1] / v_total,v_class_pre_a[0] / v_total, v_class_pre_a[1] / v_total, v_class_iou_a[0] / v_total,v_class_iou_a[1] / v_total, v_class_rec_a[0] / v_total, v_class_rec_a[1] / v_total,v_class_f1_a[0] / v_total, v_class_f1_a[1] / v_total))
            +str('；all_Eval test set: acc:{:.4f}，miou:{:.4f}，class_1_acc:{:.4f}，class_2_acc:{:.4f}，''class_1_pre:{:.4f}，class_2_pre:{:.4f}，class_1_iou:{:.4f}，class_2_iou:{:.4f}，class_1_rec:{:.4f}，class_2_rec:{:.4f}，class_1_f1:{:.4f}，class_2_f1:{:.4f}'.format(
            test_acc_a / test_total, test_miou_a / test_total, test_class_acc_a[0] / test_total, test_class_acc_a[1] / test_total,test_class_pre_a[0] / test_total, test_class_pre_a[1] / test_total, test_class_iou_a[0] / test_total,test_class_iou_a[1] / test_total, test_class_rec_a[0] / test_total, test_class_rec_a[1] / test_total,test_class_f1_a[0] / test_total, test_class_f1_a[1] / test_total))+"\n")


    record_log.flush()

    if (epoch+1)%sav_pth==0:
        torch.save(model.state_dict(), sav_path+'epoch_'+str(epoch+1)+'_model.pth')
        # torch.save(optimizer.state_dict(), sav_path+'epoch_'+str(epoch+1)+'_opeimizer.pth')

    print(best_ciou,v_class_iou_a[1] / v_total,epoch+1)
    if best_ciou < v_class_iou_a[1] / v_total:
        if os.path.exists(sav_path + 'epoch_best_' + str(best_epoch_iou) + '_model_iou.pth'):
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_iou) + '_model_iou.pth')
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_iou) + '_opeimizer_iou.pth')

        best_epoch_iou = epoch + 1
        best_ciou = v_class_iou_a[1] / v_total
        torch.save(model.state_dict(), sav_path + 'epoch_best_' + str(best_epoch_iou) + '_model_iou.pth')
        torch.save(optimizer.state_dict(),sav_path + 'epoch_best_' + str(best_epoch_iou) + '_opeimizer_iou.pth')

    if best_cf1 < v_class_f1_a[1] / v_total:
        if os.path.exists(sav_path + 'epoch_best_' + str(best_epoch_f1) + '_model_f1.pth'):
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_f1) + '_model_f1.pth')
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_f1) + '_opeimizer_f1.pth')

        best_epoch_f1 = epoch + 1
        best_cf1 = v_class_f1_a[1] / v_total
        torch.save(model.state_dict(), sav_path + 'epoch_best_' + str(best_epoch_f1) + '_model_f1.pth')
        torch.save(optimizer.state_dict(),sav_path + 'epoch_best_' + str(best_epoch_f1) + '_opeimizer_f1.pth')


    if best_loss > val_loss / len(val_dataloader):
        if os.path.exists(sav_path + 'epoch_best_' + str(best_epoch_loss) + '_model_loss.pth'):
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_loss) + '_model_loss.pth')
            os.remove(sav_path + 'epoch_best_' + str(best_epoch_loss) + '_opeimizer_loss.pth')

        best_epoch_loss = epoch + 1
        best_loss = val_loss / len(val_dataloader)
        torch.save(model.state_dict(), sav_path + 'epoch_best_' + str(best_epoch_loss) + '_model_loss.pth')
        torch.save(optimizer.state_dict(),sav_path + 'epoch_best_' + str(best_epoch_loss) + '_opeimizer_loss.pth')


    if best_test_ciou < test_class_iou_a[1] / test_total:
        if os.path.exists(sav_path + 'epoch_test_best_' + str(best_test_epoch_iou) + '_model_iou.pth'):
            os.remove(sav_path + 'epoch_test_best_' + str(best_test_epoch_iou) + '_model_iou.pth')
            os.remove(sav_path + 'epoch_test_best_' + str(best_test_epoch_iou) + '_opeimizer_iou.pth')

        best_test_epoch_iou = epoch + 1
        best_test_ciou = test_class_iou_a[1] / test_total
        torch.save(model.state_dict(), sav_path + 'epoch_test_best_' + str(best_test_epoch_iou) + '_model_iou.pth')
        torch.save(optimizer.state_dict(),sav_path + 'epoch_test_best_' + str(best_test_epoch_iou) + '_opeimizer_iou.pth')

    if best_test_cf1 < test_class_f1_a[1] / test_total:
        if os.path.exists(sav_path + 'epoch_test_best_' + str(best_test_epoch_f1) + '_model_f1.pth'):
            os.remove(sav_path + 'epoch_test_best_' + str(best_test_epoch_f1) + '_model_f1.pth')
            os.remove(sav_path + 'epoch_test_best_' + str(best_test_epoch_f1) + '_opeimizer_f1.pth')

        best_test_epoch_f1 = epoch + 1
        best_test_cf1 = test_class_f1_a[1] / test_total
        torch.save(model.state_dict(), sav_path + 'epoch_test_best_' + str(best_test_epoch_f1) + '_model_f1.pth')
        torch.save(optimizer.state_dict(),sav_path + 'epoch_test_best_' + str(best_test_epoch_f1) + '_opeimizer_f1.pth')



    if epoch+1-best_epoch_f1>30:# and epoch>100:
        break

