# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# from . import configs as configs
import configs
from block import ResNetV2
from block import DANetHead
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

from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        
        self.DAblock1 = DANetHead(768, 768)


        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
#             print(patch_size_real[0])
#             print(patch_size_real[1])
#             print(patch_size)
#             print(img_size[0])
#             print(img_size[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        
        


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = self.DAblock1(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.da = DANetHead(64, 64)
        self.da2 = DANetHead(256, 256)
        self.da3 = DANetHead(512, 512)
            
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if skip.size(1) and x.size(1) == 64:
                skip = self.da(skip) 
            
            if skip.size(1) and x.size(1) == 256:
                skip = self.da2(skip)
                
            if skip.size(1) and x.size(1) == 512:
                skip = self.da3(skip)
                
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class DA_Transformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(DA_Transformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
if __name__ == "__main__":
    vit_name="R50-ViT-B_16"
    config_vit = CONFIGS["R50-ViT-B_16"]
    config_vit.n_classes =1
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(256 / 16), int(256/ 16))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    model=DA_Transformer(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    # model = AttentionUNet(in_channels=3, out_channels=1)
    # input_tensor = torch.randn(1, 3, 256, 256).cuda()
    # output = model(input_tensor)
    # print(output.shape)
    tran = transforms.ToTensor()


    class Dataset1(Dataset):
        # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
        def __init__(self, root_dir, value_dir, label_dir):
            self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
            self.label_dir = label_dir
            self.value_dir = value_dir
            self.path2 = os.path.join(self.root_dir, self.label_dir)  # 路径的拼接
            self.path1 = os.path.join(self.root_dir, self.value_dir)  # 路径的拼接
            self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
            self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址

        ## 获取所有图片的地址列表
        def __getitem__(self, idx):
            img_name1 = self.img_path1[idx]  # 获取图片名称  self.全局的
            img_name2 = self.img_path2[idx]  # 获取图片名称  self.全局的
            img_item_path = os.path.join(self.path1, img_name1)  # 获取每个图片的地址(相对路径)
            smg_item_path = os.path.join(self.path2, img_name2)  # 获取每个图片的地址(相对路径)
            img1 = Image.open(img_item_path)
            img2 = Image.open(smg_item_path)
            img1 = tran(img1)
            img2 = tran(img2)
            img1 = img1[0]
            img2 = img2[0]
            img1 = torch.reshape(img1, (1, 256, 256))
            img2 = torch.reshape(img2, (1, 256, 256))
            return img1, img2

        def __len__(self):
            return len(self.img_path1)  # 这里返回一个就行


    class Dataset2(Dataset):
        # 初始化类 根据类创建实例时要运行函数，为整个class提供全局变量
        def __init__(self, root_dir, value_dir, label_dir):
            self.root_dir = root_dir  # 函数的变量不能传递给另外一个变量，而self能够把指定变量给别的函数使用，全局变量
            self.label_dir = label_dir
            self.value_dir = value_dir
            self.path2 = os.path.join(self.root_dir, self.label_dir)  # 路径的拼接
            self.path1 = os.path.join(self.root_dir, self.value_dir)  # 路径的拼接
            self.img_path1 = os.listdir(self.path1)  # 获得图片所有地址
            self.img_path2 = os.listdir(self.path1)  # 获得图片所有地址

        ## 获取所有图片的地址列表
        def __getitem__(self, idx):
            img_name1 = self.img_path1[idx]  # 获取图片名称  self.全局的
            img_name2 = self.img_path2[idx]  # 获取图片名称  self.全局的
            img_item_path = os.path.join(self.path1, img_name1)  # 获取每个图片的地址(相对路径)
            smg_item_path = os.path.join(self.path2, img_name2)  # 获取每个图片的地址(相对路径)
            img1 = Image.open(img_item_path)
            img2 = Image.open(smg_item_path)
            img1 = tran(img1)
            img2 = tran(img2)
            img2 = img2[0]
            img2 = torch.reshape(img2, (1, 1024, 1024))
            return img1, img2

        def __len__(self):
            return len(self.img_path1)  # 这里返回一个就行


    save_path = '/kaggle/input/myset111/new'
    data = Dataset1(r"D:\python_new\ggb\新图像_交稿件\train_image\data_enhancement\image", "train_image_clip_image",
                    "train_image_clip_label")
    data2 = Dataset2(r'D:\python_new\ggb\新图像_交稿件\train_image\data_enhancement\image', 'val_image', 'val_label')
    data2_loader = DataLoader(data2, batch_size=1, shuffle=True)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)


    def train_process(model, data_train, data_test, num_epoch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = nn.BCEWithLogitsLoss()
        model = model.to(device)
        # 复制当前模型的参数
        best_model = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        train_loss_acc = []
        test_loss_acc = []
        # 训练集损失函数的列表
        train_loss_all = []
        # 验证集损失函数列表
        val_loss_all = []
        # 计时(当前时间)
        since = time.time()
        k = 1
        # squeeze函数会把前面四则（1，*，*）变为（*，*）
        for i in range(num_epoch):
            print(f'第{k}轮开始,总共{num_epoch}轮')
            # 初始化值
            train_loss = 0
            # 训练集准确度
            train_correct = 0
            val_loss = 0
            # 验证集的准确度
            val_correct = 0
            train_num = 0
            val_num = 0
            plt.figure(figsize=(12, 5))
            for j in data_train:
                imge, label = j
                imge = imge.to(device)
                label = label.to(device)
                # 训练模式
                model.train()
                output = model(imge)
                # pre_label=torch.argmax(output,dim=1)
                loss_train = loss(output, label)
                optim.zero_grad()
                # 这里的loss_train为64个样本的平均值
                loss_train.backward()
                optim.step()
                train_loss += loss_train.item() * imge.size(0)  # 总的样本loss的累加
                train_correct += torch.sum(output == label)
                train_num += imge.size(0)
            for jp in data_test:
                imge1, label1 = jp
                imge1 = imge1.to(device)
                label1 = label1.to(device)
                # 评估模式
                model.eval()
                output = model(imge)
                pre_label_test = torch.argmax(output, dim=1)
                loss_test = loss(output, label)
                # 对损失函数进行累加
                val_loss += loss_test.item() * imge.size(0)  # 这里乘以64了
                val_correct += torch.sum(output == label)
                val_num += imge.size(0)
            # 该轮次平均的loss
            train_loss_all.append(train_loss / train_num)
            val_loss_all.append(val_loss / val_num)
            # 正确率
            train_loss_acc.append(train_correct.item() / (train_num * 256 * 256))
            test_loss_acc.append(val_correct.item() / (val_num * 1024 * 1024))
            print(
                f'训练集的损失值{train_loss_all[-1]}--测试集的损失值={val_loss_all[-1]}————测试集的正确率{test_loss_acc[-1]}————训练集的正确率{train_loss_acc[-1]}')  # 负一为取列表的最后一位
            # 寻找最高准确度
            print(best_loss)
            if val_loss_all[-1] < best_loss:
                best_loss = val_loss_all[-1]
                # 保存参数
                best_acc_wts = copy.deepcopy(model.state_dict())
            # 时间
            time_use = time.time() - since
            print(f'训练总耗费时间{time_use // 60}m,{time_use % 60}s')
            k += 1
        # 选择最优参数
        # 选择最高精确度的模型参数
        torch.save(best_acc_wts, 'DAtransUNet——2024-7_28-point_best_model.pth')
        train_process_all = pd.DataFrame(
            data={'epoch': range(num_epoch), 'train_loss_all': train_loss_all, 'val_loss_all': val_loss_all,
                  'train_loss_acc': train_loss_acc, 'test_acc_acc': test_loss_acc})
        return train_process_all


    def matplot_acc_loss(train_process_all):
        # 注意这里用到了上述字典
        # plt.rcParams['figure.figsize']=(12,4)
        # plt.figure(figsize=(12,4))#图的大小
        plt.subplot(1, 2, 1)
        plt.plot(train_process_all['epoch'], train_process_all.train_loss_all, 'ro-', label='train_loss')
        plt.plot(train_process_all['epoch'], train_process_all.val_loss_all, 'bs-',
                 label='test_loss')  # 是否考虑去掉【'换成】.epoch'
        plt.legend()  # 图例
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(1, 2, 2)
        plt.plot(train_process_all['epoch'], train_process_all.train_loss_acc, 'ro-', label='train_acc')
        plt.plot(train_process_all['epoch'], train_process_all.test_acc_acc, 'bs-',
                 label='test_acc')  # 是否考虑去掉【'换成】.epoch'
        plt.legend()  # 图例
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig('kkkk11.jpg')
    train_process=train_process(model,data_loader,data2_loader,num_epoch=5)
    matplot_acc_loss(train_process)



