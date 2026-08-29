# 定义模块的公开接口，仅暴露ConvEncoder和ConvDecoder类
__all__ = ["ConvDenoiser"]

import torch.nn as nn
import torch.nn.functional as F


# 定义神经网络架构
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## 编码器层 ##
        # 卷积层 (输入通道数从1变为32), 3x3卷积核
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 卷积层 (输入通道数从32变为16), 3x3卷积核
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # 卷积层 (输入通道数从16变为8), 3x3卷积核
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # 池化层，用于将x-y维度减少一半；卷积核和步幅均为2
        self.pool = nn.MaxPool2d(2, 2)

        ## 解码器层 ##
        # 转置卷积层，卷积核为2，步幅为2，将空间维度增加2倍
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # 最后一个普通的卷积层，用于减少通道数
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    # forward函数用于向前传播，该函数接受一个输入张量x，并返回一个输出张量，该函数的实现真正决定了神经网络的架构
    def forward(self, x):
        ## 编码 ##
        # 添加带有ReLU激活函数的隐藏层
        # 并在之后进行最大池化
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # 添加第二个隐藏层
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 添加第三个隐藏层
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 压缩表示

        ## 解码 ##
        # 添加转置卷积层，带有ReLU激活函数
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # 再次转置卷积，输出应应用sigmoid函数
        x = F.sigmoid(self.conv_out(x))

        return x


import torch

if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 3, 64, 64)

    # 创建一个卷积编码器模型
    denoiser = ConvDenoiser()

    # 编码输入张量
    encoded_tensor = denoiser(input_tensor)
