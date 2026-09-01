# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset"]

# 导入必要的库
from PIL import Image  # 图像处理库
import os  # 操作系统接口库
from torch.utils.data import Dataset  # PyTorch数据集基类
import pandas as pd
import classification_config

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


class ImageDataset(Dataset):
    """
    从图像文件夹创建PyTorch数据集，返回图像的张量表示

    参数:
    - main_dir : 图片存储路径（字符串）
    - transform (可选) : 图像预处理变换（如torchvision.transforms）
    """

    def __init__(self, main_dir, fashion_label_path, transform=None):
        self.main_dir = main_dir  # 图像主目录
        self.transform = transform  # 图像预处理函数
        # 获取所有图像文件名并按字母数字顺序排序
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))

        # 读取包含分类标签的 CSV 文件
        self.classifications = pd.read_csv(fashion_label_path)  # 从指定路径加载 CSV 文件到 DataFrame
        # 将数据类型转换为字典，提升查询效率
        self.label_dict = dict(zip(self.classifications["id"], self.classifications["target"]))

    def __len__(self):
        """返回数据集中图像的总数量"""
        return len(self.all_imgs)

    def __getitem__(self, idx):
        """加载并返回指定索引的图像张量（输入和目标相同，适用于自编码器）"""
        # 拼接完整图像路径
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        # 打开图像并转换为RGB格式
        image = Image.open(img_loc).convert("RGB")

        img_flag_id = self.label_dict[idx]  # 在分类数据中查找对应的标签

        # 对图像进行预处理（如果定义了 transform）
        if self.transform is not None:
            tensor_image = self.transform(image)  # 应用预处理操作，将图像转换为张量
        else:
            raise RuntimeError("transform参数不能为None，需指定预处理方法")

        return tensor_image, img_flag_id  # 返回图像张量和标签
