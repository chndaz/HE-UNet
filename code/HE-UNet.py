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
model = smp.Unet(
    encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
def up1_improve1(img1):
    new_list=[]
    img1 = chn(img1)
    img1 = img1.numpy()
    features1 = pd.DataFrame(img1)
    train_value1 = features1.iloc[0]
    train_value = list(train_value1)
    for op in train_value:
        if op <0.4:
            op = 0
            new_list.append(op)

        else:
            op=1
            new_list.append(op)
    list_result=torch.tensor(new_list,dtype=float)
    lis=torch.reshape(list_result, (1024, 1024))
    return lis
def improve(biaoji, img):
    lis3 = []
    img = torch.reshape(img, (1024, 1024))
    with torch.no_grad():
        img = numpy.array(img)
        for i, ip in zip(biaoji, img):
            for j, jp in zip(i, ip):
                if j[0] == 0:
                    lis3.append(jp)
                else:
                    lis3.append(j[0])
        lis3 = torch.tensor(lis3, dtype=float)
        lis3 = torch.reshape(lis3, (1024, 1024))
    return lis3


def prodect(img):
    list = []
    img = img[0]
    img = numpy.array(img)
    for j in img:
        for i in j:
            if i==1:
                i =1
                list.append(i)
            else:
                i =0
                list.append(i)
    list = torch.tensor(list, dtype=float)
    list = torch.reshape(list, (1024, 1024))
    return list


def biaoji(img1):
    lis2 = []
    img1 = numpy.array(img1)
    for i, ip in zip(img1, range(1024)):
        for j, jp in zip(i, range(1024)):
            lis1 = [j, ip, jp]
            lis2.append(lis1)
    lis2 = torch.tensor(lis2)
    lis2 = torch.reshape(lis2, (1024, 1024, 3))
    return lis2
def bianjie(img):
    h,w=img.shape
    with torch.no_grad():
        img=np.array(img)
        lis=[]
        for i,j in zip(img,range(h)):
            if j==0:
                for ip in i:
                    lis.append(ip)
            elif j!=0 and j!=h-1:
                lis.append(i[0])
                lis.append(i[w-1])
            else:
                for jp in i:
                    lis.append(jp)

    return lis

def beaut(img,remove1,remove2):
    h,w=img.shape
    # img=np.array(img)
    img1=np.array(img)
    mat2 = np.zeros((h, w))
    with torch.no_grad():
        for i in range(0, h,2):
            for j in range(0, w,2):
                mat = np.zeros((remove1, remove2))
                t = img[i:i + remove1, j: j + remove2]
                lis = bianjie(t)
                if 0 in lis:
                    if 1024-j>remove2+1 and 1024-i>remove1+1:
                        img1[i:i + remove1, j: j + remove2]=t
                else:
                    if 1024-j>remove2+1 and 1024-i>remove1+1:
                        mat.fill(1)
                        mat2[i:i + remove1, j: j + remove2]=mat

        mat2=np.array(mat2)
        mat2 = torch.tensor(mat2)
        mat2=biaoji(mat2)
        img1=improve(mat2,img)
        return img1
class Chn(nn.Module):
    def __init__(self):
        super(Chn,self).__init__()
        self.flatten=Flatten()#平铺了
    def forward(self,x):
        x=self.flatten(x)
        return x
chn=Chn()
def img_contrast_bright(img,a,b,g):
    # h,w,c=img.shape
    blank=np.zeros([1024,1024,1],img.dtype)
    dst=cv2.addWeighted(img,a,blank,b,g)
    return dst
def circle_detection(image,list):
    list_empty1 = []
    circles_value1=list[0][2]
    if circles_value1>=9:
        a=45
    else:
        a=30
    # print('circles_value1',circles_value1)
    for i in list:
        list_empty = []
        x1=int(i[0])-a
        x2=int(i[0])+a
        y1=int(i[1])-a
        y2=int(i[1])+a
        if x1<0:
            x1=0
        if y1<0:
            y1=0
        if x2<0:
            x2=0
        if y2<0:
            y2=0
        for j in list:
            a1=int(j[0])
            b1=int(j[1])
            c1=int(j[2])
            if x1<a1<x2 and y1<b1<y2:
                list_empty.append(a1)
        if len(list_empty)>=9:
            list_empty1.append([int(i[0]),int(i[1]),int(i[2])])
        # print(list_empty1)
    #删除列表中的相同元素
    list_ygls=[]
    for ip in list_empty1:
        if ip in list_ygls:
            pass
        else:
            list_ygls.append(ip)
    return list_ygls

def cv_circles_dispose(image):
    # image = cv2.imread('./best/1_7.png')
    # image_gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
    a = 1.2
    b = 1 - a
    g = 50
    # 增加亮度
    image_gray = img_contrast_bright(image, a, b, g)
    # 中值模糊
    # image_gray=cv.medianBlur(image_gray,7),这里滤波器不适用
    # 下列函数的值的选择，dp默认1，misDist表示<该值的两个圆的圆心半径值，则两个圆为同一个圆，param2表示半径大于该值才视为圆
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1, 15, param1=200, param2=10, minRadius=5, maxRadius=10)
    circles1=str(circles)
    if circles1!='None':#这里做的判断是针对没有过孔的图像的情况
        circles = circles[0]
        len1 = circles.shape[0]
        circles = circle_detection(image_gray, circles)
        circles = numpy.array(circles)
        circles = numpy.array(circles)
        for i in circles:  # circles前两个是圆心，最后一个是半径
            cv.circle(image, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 40)
        image = Image.fromarray(image)  # 输出的是4通道图
        image = image.convert('L')
    else:
        image = Image.fromarray(image)  # 输出的是4通道图
        image = image.convert('L')
    return image

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
save_path = r'D:\python_new\ggb\best_new\HE-UNet1'
now=time.time()
for j,i in enumerate(tu):  # tu，tu_traget均为数据集
    i,name=i
    tran = transforms.ToTensor()
    ecoph += 1
    img1=tran(i)#图片转化位矩阵
    img1=img1[0]
    img5=img1
    img1 = torch.unsqueeze(img1, 0)
    img1 = torch.unsqueeze(img1, 0)
    model.load_state_dict(torch.load(r'D:\python_new\ggb\UNET_6——2024-5-7_16-point_best_model.pth'))#这里的模型和小土堆的不同，保存的是模型的参数
    ouput = model(img1)
    ############################################霍夫圆检测
    # image =numpy.array(i)
    image=numpy.asarray(i)
    img_feng=cv_circles_dispose(image)
    tran = transforms.ToTensor()
    img_feng = tran(img_feng)
    img_feng= prodect(img_feng)
    img_feng= biaoji(img_feng)
    ###########################################边缘检测池化
    img2=improve(img_feng,ouput)
    save_image(img2, f'{save_path}/77778888--.png')
    path11=f'{save_path}/77778888--.png'
    img2=Image.open(path11)
    img2=tran(img2)
    img2=img2[0]
    img2=beaut(img2,40,40)
    img2=beaut(img2,115,115)
    img2=torch.unsqueeze(img2,0)
    save_image(img2, f'{save_path}/{name}')
    now1=time.time()
    print(f'第{ecoph}图','传出成功','花费时间',now1-now,'s')
