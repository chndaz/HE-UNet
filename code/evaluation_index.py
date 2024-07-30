import numpy as np
import numpy
import torchvision
from PIL import Image
from datasets import Dataset
import os

from tqdm import tqdm

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


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
        return img

    def __len__(self):
        return len(self.img_path)
tran=torchvision.transforms.ToTensor()
mode = "dir_predict"
root_dir1 = r'C:\联想电脑\python新代码\ggb\pitu'
M1 = "m2_ps"
tu = MyData(root_dir1, M1)
ecoph = 0
save_path = r'C:\联想电脑\python新代码\ggb\best\附件\数据最新cv，除去deeplab'
len1=len(tu)
miou=[]
MPA=[]
for i in tqdm(range(len1)):
    i=i+1
    dir1=fr'C:\联想电脑\python新代码\ggb\best\图片\111 ({i}).png'
    dir_label=fr'C:\联想电脑\python新代码\ggb\pitu\m3_ps\111 ({i}).png'
    img1 = Image.open(dir1)
    img1 = tran(img1)
    img1 = img1[0]
    img2 = Image.open(dir_label)
    img2 = tran(img2)
    img2 = img2[0]
    imgPredict = np.array(img1,dtype=int)  # 可直接换成预测图片
    imgLabel = np.array(img2,dtype=int)  # 可直接换成标注图片
    metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    MPA.append(mpa)
    miou.append(mIoU)
print('mpa=',sum(MPA)/len1)
print('miou=',sum(miou)/len1)




