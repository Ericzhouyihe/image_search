# 定义模块的公开接口，仅暴露ConvEncoder和ConvDecoder类
__all__ = ["ConvEncoder", "ConvDecoder"]

# 导入神经网络模块
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    卷积编码器：通过卷积和下采样操作压缩图像为低维特征表示
    输入形状：(batch_size, 3, H, W)  # 假设输入为RGB图像
    输出形状：(batch_size, 256, H/32, W/32)  # 经过5次2x2池化，尺寸缩小32倍
    """

    def __init__(self):
        super().__init__()  # 调用父类构造函数初始化

        # 定义编码器各层（输入通道, 输出通道, 卷积核大小, 填充）
        # Block 1: 3 -> 16 channels
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))  # 保持尺寸的卷积
        self.relu1 = nn.ReLU(inplace=True)  # 原地ReLU激活，节省内存
        self.maxpool1 = nn.MaxPool2d((2, 2))  # 尺寸减半

        # Block 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        # Block 3: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        # Block 4: 64 -> 128 channels
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        # Block 5: 128 -> 256 channels
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))
        # Block 6: 256 -> 512 channels
        self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool6 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        """前向传播：按顺序通过各层实现下采样"""
        # 初始输入假设为 (batch, 3, 64, 64)
        x = self.conv1(x)  # -> (batch, 16, 64, 64)
        # print("encode conv1: ", x.shape)
        x = self.relu1(x)
        x = self.maxpool1(x)  # -> (batch, 16, 32, 32)
        # print("encode pool1: ", x.shape)

        x = self.conv2(x)  # -> (batch, 32, 32, 32)
        # print("encode conv2: ", x.shape)
        x = self.relu2(x)
        x = self.maxpool2(x)  # -> (batch, 32, 16, 16)
        # print("encode pool2: ", x.shape)

        x = self.conv3(x)  # -> (batch, 64, 16, 16)
        # print("encode conv3: ", x.shape)
        x = self.relu3(x)
        x = self.maxpool3(x)  # -> (batch, 64, 8, 8)
        # print("encode pool3: ", x.shape)

        x = self.conv4(x)  # -> (batch, 128, 8, 8)
        # print("encode conv4: ", x.shape)
        x = self.relu4(x)
        x = self.maxpool4(x)  # -> (batch, 128, 4, 4)
        # print("encode pool4: ", x.shape)

        x = self.conv5(x)  # -> (batch, 256, 4, 4)
        # print("encode conv5: ", x.shape)
        x = self.relu5(x)
        x = self.maxpool5(x)  # -> (batch, 256, 2, 2)
        # print("encode pool5: ", x.shape)

        x = self.conv6(x)  # -> (batch, 512, 2, 2)
        # print("encode conv6: ", x.shape)
        x = self.relu6(x)
        x = self.maxpool6(x)  # -> (batch, 512, 1, 1)
        # print("encode pool6: ", x.shape)
        x = x.squeeze(-1).squeeze(-1)
        # print("encode final: ", x.shape)
        return x  # 输出压缩后的特征表示


class ConvDecoder(nn.Module):
    """
    卷积解码器：通过转置卷积上采样，将低维特征重建为原始图像
    输入形状：(batch_size, 256, H/32, W/32)
    输出形状：(batch_size, 3, H, W)  # 恢复原始尺寸
    """

    def __init__(self):
        super().__init__()
        # 定义解码器各层（输入通道, 输出通道, 核大小, 步长）
        # Block 1: 512 -> 256 channels
        self.deconv1 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU(inplace=True)

        # Block 2: 256 -> 128 channels
        self.deconv2 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))  # 尺寸翻倍
        self.relu2 = nn.ReLU(inplace=True)

        # Block 3: 128 -> 64 channels
        self.deconv3 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)

        # Block 4: 64 -> 32 channels
        self.deconv4 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        self.relu4 = nn.ReLU(inplace=True)

        # Block 5: 32 -> 16 channels
        self.deconv5 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        self.relu5 = nn.ReLU(inplace=True)

        # Block 6: 16 -> 3 channels
        self.deconv6 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))  # 输出RGB图像
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播：通过转置卷积逐步上采样"""
        # 输入为 (batch, 512, 1, 1)
        # print("decoder x:", x.shape)
        x = x.unsqueeze(-1).unsqueeze(-1)
        # print("decoder unsqueeze: ", x.shape)

        x = self.deconv1(x)  # -> (batch, 256, 2, 2)
        # print("decode deconv1: ", x.shape)
        x = self.relu1(x)

        # 输入为 (batch, 256, 2, 2)
        x = self.deconv2(x)  # -> (batch, 128, 4, 4)
        # print("decode deconv2: ", x.shape)
        x = self.relu2(x)

        x = self.deconv3(x)  # -> (batch, 64, 8, 8)
        # print("decode deconv3: ", x.shape)
        x = self.relu3(x)

        x = self.deconv4(x)  # -> (batch, 32, 16, 16)
        # print("decode deconv4: ", x.shape)
        x = self.relu4(x)

        x = self.deconv5(x)  # -> (batch, 16, 32, 32)
        # print("decode deconv5: ", x.shape)
        x = self.relu5(x)

        x = self.deconv6(x)  # -> (batch, 3, 64, 64)
        # print("decode deconv6: ", x.shape)
        x = self.relu6(x)
        return x  # 输出重建后的图像


import torch

if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 3, 64, 64)

    # 创建一个卷积编码器模型
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    # 编码输入张量
    encoded_tensor = encoder(input_tensor)
    encoded_tensor = decoder(encoded_tensor)
