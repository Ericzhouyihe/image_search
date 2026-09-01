# 定义模块的公开接口，仅暴露ConvEncoder和ConvDecoder类
__all__ = ["Classifier"]

import torch.nn as nn  # 导入 PyTorch 的神经网络模块，用于构建神经网络
import torch.nn.functional as F  # 导入 PyTorch 的函数模块，包含激活函数、损失函数等


class Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 定义第一个卷积层，输入通道数为3，输出通道数为8，卷积核大小为3x3，步幅为1，填充为1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 使用填充为1的卷积操作，输出特征图的尺寸与输入相同（Same convolutions）

        # 定义最大池化层，池化核大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # 定义第二个卷积层，输入通道数为8，输出通道数为16，卷积核大小为3x3，步幅为1，填充为1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 上一层的输出通道数为8，因此这一层的输入通道数为8

        # 定义全连接层，输入大小为16*16*16，输出大小为num_classes（分类数），全连接层的输入的尺寸计算在forward函数中解释
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        # 第一层卷积 + ReLU激活函数
        x = F.relu(self.conv1(x))
        # print("conv1 shape: ", x.shape)
        # 池化操作
        x = self.pool(x)
        # print("pool1 shape: ", x.shape)
        # 第二层卷积 + ReLU激活函数
        x = F.relu(self.conv2(x))
        # print("conv2 shape: ", x.shape)
        # 池化操作
        x = self.pool(x)
        # print("pool2 shape: ", x.shape)
        # 将特征图展平为一维向量，保留batch_size维度，其余维度展平
        x = x.reshape(x.shape[0], -1)
        # print("reshape shape: ", x.shape)
        # 全连接层
        # 经过两次池化后的特征图尺寸为16*16*16，将其展平为一维向量，维度为16*16*16=4096，因此，输入到全连接层的维度为4096
        x = self.fc1(x)
        # print("fc1 shape: ", x.shape)
        # 返回log_softmax输出，用于多分类任务
        return F.log_softmax(x, dim=1)


import torch

if __name__ == "__main__":
    # 创建一个随机输入张量，维度为1x3x32x32，用于测试模型
    x = torch.randn(1, 3, 64, 64)
    # 创建一个Classification模型，并传入输入通道数为3，分类数为5
    model = Classifier()
    # 将随机输入张量传入模型，得到输出
    output = model(x)
    # 打印输出维度
    print("output.shape: ", output.shape)
