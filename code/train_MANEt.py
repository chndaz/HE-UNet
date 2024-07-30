import copy
import os
import time
import pandas as pd
import numpy
import torch.cuda
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Dropout
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

model = smp.MAnet(
    encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
tran=transforms.ToTensor()
# data_train = torchvision.datasets.CIFAR10(r'D:\联想电脑\python新代码\ggb\ppic', train=True, transform=transforms.ToTensor(),
#                                           download=True)
# data_train, data_label = random_split(data_train, [int(len(data_train) * 0.8), int(len(data_train) * 0.2)])
# data_train = DataLoader(data_train, batch_size=64, shuffle=True)
# data_test = DataLoader(data_label, batch_size=64, shuffle=True)
class Dataset1(Dataset):
    # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
    def __init__(self, root_dir,value_dir,label_dir):
        self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
        self.label_dir = label_dir
        self.value_dir=value_dir
        self.path2= os.path.join(self.root_dir, self.label_dir) # 路径的拼接
        self.path1= os.path.join(self.root_dir, self.value_dir) # 路径的拼接
        self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
        self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址
    ## 获取所有图片的地址列表
    def __getitem__(self, idx):
        img_name1 = self.img_path1[idx] #获取图片名称  self.全局的
        img_name2= self.img_path2[idx] #获取图片名称  self.全局的
        img_item_path = os.path.join(self.path1, img_name1) # 获取每个图片的地址(相对路径)
        smg_item_path = os.path.join(self.path2, img_name2) # 获取每个图片的地址(相对路径)
        img1= Image.open(img_item_path)
        img2=Image.open(smg_item_path)
        img1=tran(img1)
        img2=tran(img2)
        img1=img1[0]
        img2= img2[0]
        img1=torch.reshape(img1,(1,256,256))
        img2=torch.reshape(img2,(1,256,256))
        return img1,img2

    def __len__(self):
        return len(self.img_path1)#这里返回一个就行
class Dataset2(Dataset):
    # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
    def __init__(self, root_dir,value_dir,label_dir):
        self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
        self.label_dir = label_dir
        self.value_dir=value_dir
        self.path2= os.path.join(self.root_dir, self.label_dir) # 路径的拼接
        self.path1= os.path.join(self.root_dir, self.value_dir) # 路径的拼接
        self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
        self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址
    ## 获取所有图片的地址列表
    def __getitem__(self, idx):
        img_name1 = self.img_path1[idx] #获取图片名称  self.全局的
        img_name2= self.img_path2[idx] #获取图片名称  self.全局的
        img_item_path = os.path.join(self.path1, img_name1) # 获取每个图片的地址(相对路径)
        smg_item_path = os.path.join(self.path2, img_name2) # 获取每个图片的地址(相对路径)
        img1= Image.open(img_item_path)
        img2=Image.open(smg_item_path)
        img1=tran(img1)
        img2=tran(img2)
        img2= img2[0]
        img2= torch.reshape(img2, (1, 1024,1024))
        return img1,img2
    def __len__(self):
        return len(self.img_path1)#这里返回一个就行
save_path='/kaggle/input/myset111/new'
data=Dataset1(r"D:\python_new\ggb\新图像_交稿件\train_image\data_enhancement\image","train_image_clip_image","train_image_clip_label")
data2=Dataset2(r'D:\python_new\ggb\新图像_交稿件\train_image\data_enhancement\image','val_image','val_label')
data2_loader=DataLoader(data2,batch_size=1,shuffle=True)
data_loader=DataLoader(data,batch_size=1,shuffle=True)

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self,x:torch.Tensor):
        return self.pool(x)
class UpSample(nn.Module):
    def __init__(self,input_channals:int,output_channals:int):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channals,output_channals,kernel_size=2,stride=2)
        #看效果，不好试试UpsamplingBilinear2d(scale_factor=2)
    def forward(self,x:torch.Tensor):
        return self.up(x)
class CropAndConcat(nn.Module):

    def forward(self,x:torch.Tensor,contracting_x:torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x,[x.shape[2],x.shape[3]])
        x = torch.cat([x,contracting_x],dim=1)
        return x

def train_process(model, data_train, data_test,num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    # 复制当前模型的参数
    best_model = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    train_loss_acc=[]
    test_loss_acc=[]
    # 训练集损失函数的列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 计时(当前时间)
    since = time.time()
    k=1
    #squeeze函数会把前面四则（1，*，*）变为（*，*）
    for i in range(num_epoch):
        print(f'第{k}轮开始,总共{num_epoch}轮')
        #初始化值
        train_loss=0
        #训练集准确度
        train_correct=0
        val_loss=0
        #验证集的准确度
        val_correct=0
        train_num=0
        val_num=0
        plt.figure(figsize=(12,5))
        for j in data_train:
            imge,label=j
            imge=imge.to(device)
            label=label.to(device)
            #训练模式
            model.train()
            output=model(imge)
            # pre_label=torch.argmax(output,dim=1)
            loss_train=loss(output,label)
            optim.zero_grad()
            #这里的loss_train为64个样本的平均值
            loss_train.backward()
            optim.step()
            train_loss+=loss_train.item()*imge.size(0)#总的样本loss的累加
            train_correct+=torch.sum(output==label)
            train_num+=imge.size(0)
        for jp in data_test:
            imge1,label1=jp
            imge1=imge1.to(device)
            label1=label1.to(device)
            #评估模式
            model.eval()
            output=model(imge)
            pre_label_test=torch.argmax(output,dim=1)
            loss_test=loss(output,label)
            #对损失函数进行累加
            val_loss+=loss_test.item()*imge.size(0)#这里乘以64了
            val_correct+=torch.sum(output==label)
            val_num+=imge.size(0)
        #该轮次平均的loss
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        #正确率
        train_loss_acc.append(train_correct.item()/(train_num*256*256))
        test_loss_acc.append(val_correct.item()/(val_num*1024*1024))
        print(f'训练集的损失值{train_loss_all[-1]}--测试集的损失值={val_loss_all[-1]}————测试集的正确率{test_loss_acc[-1]}————训练集的正确率{train_loss_acc[-1]}')#负一为取列表的最后一位
        #寻找最高准确度
        if val_loss_all[-1]<best_loss:
            best_loss=val_loss_all[-1]
            #保存参数
            best_acc_wts=copy.deepcopy(model.state_dict())

        #时间
        time_use=time.time()-since
        print(f'训练总耗费时间{time_use//60}m,{time_use%60}s')
        k+=1
    #选择最优参数
    #选择最高精确度的模型参数
    torch.save(best_acc_wts,'MANet——2024-5-7_15-point_best_model.pth')
    train_process_all=pd.DataFrame(data={'epoch':range(num_epoch),'train_loss_all':train_loss_all,'val_loss_all':val_loss_all,
                                         'train_loss_acc':train_loss_acc,'test_acc_acc':test_loss_acc})
    return train_process_all
def matplot_acc_loss(train_process_all):
    #注意这里用到了上述字典
    # plt.rcParams['figure.figsize']=(12,4)
    # plt.figure(figsize=(12,4))#图的大小
    plt.subplot(1,2,1)
    plt.plot(train_process_all['epoch'],train_process_all.train_loss_all,'ro-',label='train_loss')
    plt.plot(train_process_all['epoch'],train_process_all.val_loss_all,'bs-',label='test_loss')#是否考虑去掉【'换成】.epoch'
    plt.legend()#图例
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(train_process_all['epoch'],train_process_all.train_loss_acc,'ro-',label='train_acc')
    plt.plot(train_process_all['epoch'],train_process_all.test_acc_acc,'bs-',label='test_acc')#是否考虑去掉【'换成】.epoch'
    plt.legend()#图例
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('kkkk11.jpg')


if __name__ == '__main__':
    #模型实例化
    train_process=train_process(model,data_loader,data2_loader,num_epoch=11)
    matplot_acc_loss(train_process)
