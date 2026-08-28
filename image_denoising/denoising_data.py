# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset"]

from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import os  # 导入os库，用于处理文件和目录路径
from torch.utils.data import Dataset  # 从PyTorch的工具库中导入Dataset类，用于自定义数据集
import denoising_config
import torch
import numpy as np

# 正则表达式相关库
import re


def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    # 定义转换函数：将数字部分转换为整数，非数字部分转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)


class ImageDataset(Dataset):  # 定义一个名为ImageDataset的类，继承自PyTorch的Dataset类
    def __init__(self, main_dir, transform=None):
        # 定义类的构造函数，接受两个参数：main_dir（主目录路径）和transform（图像预处理操作，默认为None）
        self.main_dir = main_dir  # 将主目录路径保存为类的属性
        self.transform = transform  # 将图像预处理操作保存为类的属性
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))  # 获取所有图像文件名并按字母数字顺序排序

    def __len__(self):  # 定义__len__方法，返回数据集中图像的数量
        return len(self.all_imgs)  # 返回主目录下图像文件的数量

    def __getitem__(self, idx):  # 定义__getitem__方法，根据索引idx获取图像
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])  # 根据索引idx构建图像的完整路径
        image = Image.open(img_loc).convert("RGB")  # 使用PIL库打开图像并将其转换为RGB格式

        if self.transform is not None:  # 如果定义了图像预处理操作
            tensor_image = self.transform(image)  # 对图像进行预处理，并将其转换为张量
        else:
            # 若无变换，抛出异常提示必须提供预处理
            raise ValueError("transform参数不能为None，需指定预处理方法")

        ## 向输入图像添加随机噪声
        # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
        noisy_imgs = tensor_image + denoising_config.NOISE_FACTOR * torch.randn(*tensor_image.shape)
        # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
        noisy_imgs = torch.clip(noisy_imgs, 0.0, 1.0)

        return noisy_imgs, tensor_image  # 返回预处理后的图像张量（将噪声图片作为输入，将原图作为目标）
