# 导入必要的库
import numpy as np  # 数值计算库
import os  # 操作系统接口库
import torch  # PyTorch深度学习框架
import random  # 随机数生成库

def seed_everything(seed):
    """
    为了保证训练过程可复现，使用确定的随机数种子。对 torch，numpy 和 random 都使用相同的种子。
    :param seed: 随机数种子（整数）
    :return:
    """
    random.seed(seed)  # 设置Python内置随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN操作确定性
    torch.backends.cudnn.benchmark = False  # 禁用CuDNN性能优化