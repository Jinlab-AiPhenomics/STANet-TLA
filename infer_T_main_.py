#所有输入  小图总精度  按文件夹内文件搜索test

import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


from models.cc_model2_H import TPN
import os
from osgeo import gdal

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name = ["20220727","20220802","20220810","20220815","20220822",
#         "20220830","20220906","20220912","20220919","20220927","20221013"]

# name =["20231217","20231217","20231219","20231223","20231225","20231227","20231230","20240102","20240105","20240108",
#        "20240110","20240115","20240118","20240121","20240124","20240127","20240130","20240202","20240205","20240208",
#        "20240211","20240214",]

folder_path=r"E:\Data\BM_2024\No2\3.mask\1.clip_rgb\\"
all_items = os.listdir(folder_path) # 获取文件夹内的所有文件和文件夹

name = [f for f in all_items if os.path.isdir(os.path.join(folder_path, f))] # 遍历所有项目,找出文件夹
save_ = True

for iiii in range(len(name)):

    IMAGE_INPUT_PATH_n=r'E:\Data\BM_2024\No2\3.mask\1.clip_rgb\\'+name[iiii]+'\\images\\'
    out_path=r'E:\Data\BM_2024\No2\3.mask\2.clip_mask\\'+name[iiii]+'_label'
    model_path = r"E:\Soybean_Data\transfer_2023\ybj_256\qy\\epoch_test_best_20_model_f1.pth"

    if not os.path.exists(out_path):
        os.mkdir(out_path)


    model = TPN(2, backbone="ResNet", pretrained_base=False).cuda()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    ])


    print(device)



    model.eval()
    model.load_state_dict(torch.load(model_path))

    for each_image in os.listdir(IMAGE_INPUT_PATH_n):
        if each_image.endswith(".tif") == False:
            continue

        frames_new = []
        aaa=[]
        for i in range(3):
            ssssss=3-i-1
            image_filename = IMAGE_INPUT_PATH_n + '/' + each_image
            if iiii-ssssss<0:
                image_filename = image_filename.replace(str(name[iiii]),str(name[0]))
            else:
                image_filename = image_filename.replace(str(name[iiii]),str(name[iiii-ssssss]))
            # print(image_filename)
            aaa.append(os.path.basename(image_filename).split(".")[0])
            frame = gdal.Open(image_filename)
            frame = frame.ReadAsArray(0, 0, frame.RasterXSize, frame.RasterYSize)
            pil_image1 = Image.fromarray(np.uint8(frame).transpose(1, 2, 0))  # 将NumPy数组转换为PIL图像
            aaaaa = transform(np.float32((np.array(pil_image1).transpose(2, 0, 1))))
            frames_new.append(aaaaa[0])

        frames_new = torch.stack(frames_new)
        frames_new=torch.unsqueeze(frames_new, 0).to(device)

        # frames_new.to(device)
        outputs = model(frames_new)


        _, predicted = torch.max(outputs.data, 1)



        # for j in range(predicted.shape[0]):
        pre = transforms.ToPILImage()(predicted.type(torch.float32))
        # pre=pre.resize((512, 512))
        pre.save(out_path+"\\" +each_image.replace(".tif",".png"))
        # print(out_path+"\\" +each_image.replace(".tif",".png"))
        print(aaa)


