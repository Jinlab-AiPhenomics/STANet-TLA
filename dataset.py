import torch
from PIL import Image
import cv2
import os
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
from osgeo import gdal

# Get item from dataset

class PSegDataset(Dataset):
    def __init__(self, data_dir,data_txt_dir, image_set, transform=None,data_i=None):
        super(PSegDataset).__init__()


        self.data_dir = data_dir
        self.data_txt_dir=data_txt_dir
        self.transform = transform

        if image_set == 'train':
            split_f = os.path.join(self.data_txt_dir, 'train.txt')
        elif image_set == 'val':
            split_f = os.path.join(self.data_txt_dir, 'val.txt')
        elif image_set == 'test':
            split_f = os.path.join(self.data_txt_dir, 'test.txt')
            # split_f=r"E:\Soybean_Data\ybj_256\test_pre\test_yb\test_cc\\test_tje.txt"

        self.tif_list = []

        with open(split_f, 'r') as f:
            for line in f:
                if data_i is not None:
                    if data_i in line:
                        self.tif_list.append((line.strip()))  # 去除每行末尾的换行符
                else:
                    self.tif_list.append((line.strip()))  # 去除每行末尾的换行符

    def __len__(self):
        return len(self.tif_list)

    def __getitem__(self, idx):

        frame_path = os.path.join(self.data_dir, "images",  self.tif_list[idx] + ".tif")
        label_path = os.path.join(self.data_dir, "labels",  self.tif_list[idx] + ".png")

        #frame = np.array(Image.open(frame_path))#.convert('RGB')

        frame = gdal.Open(frame_path)
        frame = frame.ReadAsArray(0, 0, frame.RasterXSize, frame.RasterYSize)

        label = Image.open(label_path)  # .convert('L')


        # Apply transformations if specified
        if self.transform:
            frames = self.transform(frame)
            # labels = torch.tensor(labels)


        # label = torch.tensor(label).unsqueeze(1)
        label = torch.tensor(np.array(label))
        return frames[0], label


#
# class VSegDataset_N(Dataset):
#     def __init__(self, data_dir,data_txt,image_set, transform=None,data_i=None):
#         super(VSegDataset_N).__init__()
#
#         self.data_dir = data_dir
#         self.data_txt_dir = data_txt
#
#         self.transform = transform
#
#         if image_set=='train':
#             split_f = os.path.join(self.data_txt_dir, 'train.txt') #train_s
#         elif image_set=='val':
#             split_f = os.path.join(self.data_txt_dir, 'val.txt')
#         elif image_set=='test':
#             split_f = os.path.join(self.data_txt_dir, 'test.txt')
#         elif image_set == 'test1':
#             split_f = os.path.join(self.data_txt_dir, 'test1.txt')
#             # split_f = os.path.join(r"E:\Soybean_Data\ybj_256\test_pre\test_yb\test_data", 'test_tje_t.txt')
#
#
#         self.video_list = []
#
#         with open(split_f, 'r') as f:
#             for line in f:
#                 if data_i is not None:
#                     aaa=line.strip().split("  ")
#                     if data_i in aaa[len(aaa)-1]:
#                         self.video_list.append((line.strip()))  # 去除每行末尾的换行符
#                 else:
#                     self.video_list.append((line.strip()))  # 去除每行末尾的换行符
#
#
#     def __len__(self):
#         return len(self.video_list)
#
#     def __getitem__(self, idx):
#         video_name = self.video_list[idx]
#
#         # Load video frames and labels
#         frames = []
#
#         aaa = self.video_list[idx].split("  ")
#         for iii in range (len(aaa)):
#             frame_path = os.path.join(self.data_dir,"images\\1111",aaa[iii]+".tif")
#             # frame_path = os.path.join(self.data_dir,"images",aaa[iii]+".tif")
#
#             # frame_temp = Image.open(frame_path)#.convert('RGB')
#             # frame = gdal.Open(frame_path)
#             # frame=frame.ReadAsArray(0,0,frame.RasterXSize,frame.RasterYSize)
#             #
#             # data_scaled = np.clip((frame - frame.min()) * (255 / (frame.max() - frame.min())), 0, 255)
#             # data_final = np.where(frame > 2, 255, np.where(frame < 0, 0, data_scaled))
#             #
#             # r = Image.fromarray(data_final.astype(np.uint8))  # 获取第1波段)
#             # frame_temp = Image.merge('RGB', (r, r, r))
#
#
#             # frame.save("output_image.jpg")
#
#             # frames.append(np.transpose(np.array(frame_temp), (2, 0, 1)))
#
#             frame = gdal.Open(frame_path)
#             frame=frame.ReadAsArray(0,0,frame.RasterXSize,frame.RasterYSize)
#             frames.append(frame)
#
#         frames=np.array(frames)
#         label_path = os.path.join(self.data_dir,"labels",aaa[len(aaa)-1]+".png")
#         labels = Image.open(label_path)  # .convert('L')
#
#
#         # Apply transformations if specified
#         if self.transform:
#             # frames = [self.transform(np.float32(framea)) for framea in frames]
#             frames = [self.transform(framea) for framea in frames]
#
#             # labels = torch.tensor(labels)
#
#
#         frames_new=[]
#         for i in range(len(aaa)):
#             frames_new.append(frames[i][0])
#         frames_new=torch.stack(frames_new)#.permute(0, 3, 1, 2)
#         labels = torch.tensor(np.array(labels))#.unsqueeze(1)
#
#
#         return frames_new, labels
#

class VSegDataset_N(Dataset):
    def __init__(self, data_dir, data_txt, image_set, transform=None, data_i=None):
        super(VSegDataset_N).__init__()

        self.data_dir = data_dir
        self.data_txt_dir = data_txt

        self.transform = transform

        if image_set == 'train':
            split_f = os.path.join(self.data_txt_dir, 'train.txt')  # train_s
        elif image_set == 'val':
            split_f = os.path.join(self.data_txt_dir, 'val.txt')
        elif image_set == 'test':
            split_f = os.path.join(self.data_txt_dir, 'test.txt')
            # split_f = os.path.join(r"E:\Soybean_Data\ybj_256\test_pre\test_yb\test_data", 'test_tje_t.txt')

        self.video_list = []

        with open(split_f, 'r') as f:
            for line in f:
                if data_i is not None:
                    aaa = line.strip().split("  ")
                    if data_i in aaa[len(aaa) - 1]:
                        self.video_list.append((line.strip()))  # 去除每行末尾的换行符
                else:
                    self.video_list.append((line.strip()))  # 去除每行末尾的换行符

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        # Load video frames and labels
        frames = []

        aaa = self.video_list[idx].split("  ")
        for iii in range(len(aaa)):
            frame_path = os.path.join(self.data_dir, "images\\1111", aaa[iii] + ".tif")

            # frame = np.array(Image.open(frame_path))#.convert('RGB')
            frame = gdal.Open(frame_path)
            frame = frame.ReadAsArray(0, 0, frame.RasterXSize, frame.RasterYSize)
            frames.append(frame)

        frames = np.array(frames)
        label_path = os.path.join(self.data_dir, "labels", aaa[len(aaa) - 1] + ".png")
        labels = Image.open(label_path)  # .convert('L')

        # Apply transformations if specified
        if self.transform:
            # frames = [self.transform(framea.transpose(1, 2, 0)) for framea in frames]
            # labels = torch.tensor(labels)
            frames = [self.transform(framea) for framea in frames]

        frames_new = []
        for i in range(len(aaa)):
            frames_new.append(frames[i][0])
        frames_new = torch.stack(frames_new)

        labels = torch.tensor(np.array(labels))  # .unsqueeze(1)

        return frames_new, labels


class VSegDataset_test(Dataset):
    def __init__(self, data_dir, image_set, transform=None, data_i=None):
        super(VSegDataset_test).__init__()

        self.data_dir = data_dir
        self.transform = transform

        if image_set == 'train':
            split_f = os.path.join(self.data_dir, 'train_t.txt')  # train_s
        elif image_set == 'val':
            split_f = os.path.join(self.data_dir, 'val_t.txt')
        elif image_set == 'test':
            split_f = os.path.join(self.data_dir, 'test_t.txt')
            # split_f = os.path.join(r"E:\Soybean_Data\ybj_256\test_pre\test_yb\test_cc", 'test_tj100_t.txt')


        self.video_list = []


        with open(split_f, 'r') as f:
            for line in f:
                if data_i is not None:
                    aaa = line.strip().split("  ")
                    if data_i in aaa[len(aaa) - 1]:
                        self.video_list.append((line.strip()))  # 去除每行末尾的换行符
                else:
                    self.video_list.append((line.strip()))  # 去除每行末尾的换行符

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        # Load video frames and labels
        frames = []

        aaa = self.video_list[idx].split("  ")
        for iii in range(len(aaa)):
            frame_path = os.path.join(self.data_dir, "images", aaa[iii] + ".tif")

            # frame = np.array(Image.open(frame_path))#.convert('RGB')
            frame = gdal.Open(frame_path)
            frame = frame.ReadAsArray(0, 0, frame.RasterXSize, frame.RasterYSize)
            frames.append(frame)

        frames = np.array(frames)
        label_path = os.path.join(self.data_dir, "labels", aaa[len(aaa) - 1] + ".png")
        labels = Image.open(label_path)  # .convert('L')


        date=aaa[len(aaa)-1].split("_")[0]

        # Apply transformations if specified
        if self.transform:
            frames = [self.transform(framea) for framea in frames]
            # labels = torch.tensor(labels)

        frames_new = []
        for i in range(len(aaa)):
            frames_new.append(frames[i][0])
        frames_new = torch.stack(frames_new)

        labels = torch.tensor(np.array(labels))  # .unsqueeze(1)

        return frames_new, labels,date




class VSegDataset_Nttt(Dataset):
    def __init__(self, data_dir,data_txt,image_set, transform=None,data_i=None):
        super(VSegDataset_Nttt).__init__()

        self.data_dir = data_dir
        self.data_txt_dir = data_txt

        self.transform = transform

        if image_set=='train':
            split_f = os.path.join(self.data_txt_dir, 'train.txt') #train_s
        elif image_set=='val':
            split_f = os.path.join(self.data_txt_dir, 'val.txt')
        elif image_set=='test':
            split_f = os.path.join(self.data_txt_dir, 'test.txt')
            # split_f = os.path.join(r"E:\Soybean_Data\ybj_256\test_pre\test_yb\test_data", 'test_tje_t.txt')


        self.video_list = []

        with open(split_f, 'r') as f:
            for line in f:
                if data_i is not None:
                    aaa=line.strip().split("  ")
                    if data_i in aaa[len(aaa)-1]:
                        self.video_list.append((line.strip()))  # 去除每行末尾的换行符
                else:
                    self.video_list.append((line.strip()))  # 去除每行末尾的换行符


    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        # Load video frames and labels
        frames = []

        aaa = self.video_list[idx].split("  ")
        for iii in range (len(aaa)):
            frame_path = os.path.join(self.data_dir,"images",aaa[iii])+".tif"

            # frame = np.array(Image.open(frame_path))#.convert('RGB')
            frame = gdal.Open(frame_path)
            frame=frame.ReadAsArray(0,0,frame.RasterXSize,frame.RasterYSize)
            frames.append(frame)

        frames=np.array(frames)


        # Apply transformations if specified
        if self.transform:
            frames = [self.transform(framea) for framea in frames]

        frames_new=[]
        for i in range(len(aaa)):
            frames_new.append(frames[i][0])
        frames_new=torch.stack(frames_new)



        return frames_new