import time
import cv2 as cv
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy
from sklearn.preprocessing import MinMaxScaler
import torch
from PIL import Image
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from network_unet2 import *
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.preprocessing import MinMaxScaler
model = smp.MAnet(
    encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
class MyData(Dataset):
    # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 路径的拼接
        self.img_path = os.listdir(self.path)  # 获得图片所有地址

    ## 获取所有图片的地址列表
    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取图片名称  self.全局的
        img_item_path = os.path.join(self.path, img_name)  # 获取每个图片的地址(相对路径)
        img = Image.open(img_item_path)
        return img,img_name

    def __len__(self):
        return len(self.img_path)
mode = "dir_predict"
root_dir1 = r'D:\python_new\ggb\best_new'
M1 = "test_image"
tu = MyData(root_dir1, M1)
ecoph = 0
save_path = r'D:\python_new\ggb\best_new\MANet'
now=time.time()
for i in tu:  # tu，tu_traget均为数据集
    i,name=i
    tran = transforms.ToTensor()
    ecoph += 1
    img1=tran(i)#图片转化位矩阵
    img1=img1[0]
    img5=img1
    img1 = torch.unsqueeze(img1, 0)
    img1 = torch.unsqueeze(img1, 0)
    model.load_state_dict(torch.load('D:\python_new\ggb\MANet——2024-5-7_15-point_best_model.pth'))#这里的模型和小土堆的不同，保存的是模型的参数
    ouput = model(img1)
    save_image(ouput, f'{save_path}/{name}')
    now1=time.time()
    print(f'第{ecoph}图','传出成功','花费时间',now1-now,'s')
