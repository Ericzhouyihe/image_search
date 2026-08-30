import sys  # 导入系统模块
from pathlib import Path  # 导入路径处理模块

# 将项目根目录加入模块搜索路径，使 `common` 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # 导入numpy用于数据处理
import torch  # 导入PyTorch核心库  # noqa: I001
import torch.nn as nn  # 导入PyTorch神经网络模块  # noqa: PLR0402
import torch.optim as optim  # 导入优化器模块  # noqa: F401, PLR0402
import torchvision.transforms as T  # 导入torchvision的图像变换模块
from tqdm import tqdm  # 导入进度条工具

import denoising_config  # 导入配置文件参数
import denoising_data  # 导入自定义数据加载模块
import denoising_engine  # 导入训练引擎模块（包含train_step和val_step）
import denoising_model  # 导入自定义模型模块（包含ConvEncoder和ConvDecoder）
from common import utils  # 导入自定义工具函数（如seed_everything）


def test(denoiser, val_loader, device):
    import matplotlib.pyplot as plt

    # 从验证数据加载器中获取一个批次的测试图像
    dataiter = iter(val_loader)  # 创建验证数据加载器的迭代器
    noisy_images, original_images = next(dataiter)  # 获取下一个批次的数据（图像和标签）
    print("测试集 images 形状: ", noisy_images.shape)  # 打印图像的形状，通常是 (batch_size, channels, height, width)

    denoiser = denoiser.to(device)  # 将模型移动到指定的设备

    noisy_images = noisy_images.to(device)  # 将噪声图像移动到指定的设备
    print("测试集 noisy_images 形状: ", noisy_images.shape)  # 打印图像的形状

    # 获取模型的输出（去噪后的图像）
    output = denoiser(noisy_images)  # 将噪声图像输入模型，得到去噪后的输出
    print("测试集输出结果 output 形状: ", output.shape)  # 打印输出的形状，通常是 (batch_size, channels, height, width)

    # 准备噪声图像用于显示
    noisy_images = noisy_images.cpu().numpy()  # 将噪声图像从 PyTorch 张量转换为 NumPy 数组
    print("noisy_imgs 转换为 numpy 数组后的形状: ", noisy_images.shape)  # 打印噪声图像的形状
    noisy_images = np.moveaxis(noisy_images, 1, -1)  # 调整维度顺序，将通道维度移到最后一维，适配 matplotlib 的显示格式

    # 将输出调整为批次图像的形状
    output = output.view(
        denoising_config.TEST_BATCH_SIZE, 3, 64, 64
    )  # 将输出张量调整为 (batch_size, channels, height, width) 的形状
    # 使用 detach() 分离梯度信息，并将其转换为 NumPy 数组
    output = output.detach().cpu().numpy()  # 将输出从 PyTorch 张量转换为 NumPy 数组
    print("output 转换为 numpy 数组后的形状: ", output.shape)  # 打印输出的形状
    output = np.moveaxis(output, 1, -1)  # 调整维度顺序，将通道维度移到最后一维，适配 matplotlib 的显示格式
    print("output 的通道维度移到最后一维后的形状: ", output.shape)  # 打印调整后的输出形状

    original_images = original_images.cpu().numpy().transpose((0, 2, 3, 1))
    print("original_image shape: ", original_images.shape)

    # 绘制前 10 张输入图像和重建图像
    _, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25, 4))  # 创建 2 行 10 列的子图
    # 第一行显示噪声图像，第二行显示重建图像
    for imgs, row in zip([noisy_images, original_images, output], axes):  # 遍历噪声图像和重建图像
        for img, ax in zip(imgs, row):  # 遍历每张图像和对应的子图
            ax.imshow(np.squeeze(img))  # 显示图像，并去除多余的维度
            ax.get_xaxis().set_visible(False)  # 隐藏 x 轴
            ax.get_yaxis().set_visible(False)  # 隐藏 y 轴
    plt.show()  # 显示图像


# 主程序入口
if __name__ == "__main__":
    # 检测GPU可用性并设置设备
    if torch.cuda.is_available():
        device = "cuda"  # 优先使用GPU
    else:
        device = "cpu"  # 回退到CPU

    # 打印随机种子配置信息
    print(f"设置训练去噪模型的随机数种子, seed = {denoising_config.SEED}")

    # 调用工具函数设置全局随机种子（确保可复现性）
    utils.seed_everything(denoising_config.SEED)

    # 定义图像预处理流程
    transforms = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    # 打印日志，提示用户正在创建数据集
    print("------------ 正在创建数据集 ------------")

    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    full_dataset = denoising_data.ImageDataset(denoising_config.IMG_PATH, transforms)

    # 计算训练集和验证集大小
    train_size = int(denoising_config.TRAIN_RATIO * len(full_dataset))  # 75%训练
    val_size = len(full_dataset) - train_size  # 25%验证

    # 随机划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 数据加载器配置阶段
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    # 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=denoising_config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    # 验证数据加载器（不打乱，完整加载）
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=denoising_config.TEST_BATCH_SIZE)

    print("------------ 数据加载器创建完成 ------------")

    # 初始化自编码器用于去噪
    denoiser = denoising_model.ConvDenoiser()  # 创建自编码器实例
    # 指定损失函数
    loss_fn = nn.MSELoss()
    # 将模型移动到指定设备（GPU/CPU）
    denoiser.to(device)
    # 指定优化器
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=denoising_config.LEARNING_RATE)
    # 初始化最佳损失值为极大值
    min_loss = 9999

    # 开始训练循环
    print("------------ 开始训练 ------------")

    # 使用tqdm进度条遍历预设的epoch数量
    for epoch in tqdm(range(denoising_config.EPOCHS)):
        # 执行一个训练epoch
        train_loss = denoising_engine.train_step(denoiser, train_loader, loss_fn, optimizer, device=device)
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")

        # 执行验证步骤
        val_loss = denoising_engine.val_step(denoiser, val_loader, loss_fn, device=device)

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            min_loss = val_loss
            # 保存去噪自编码器状态字典
            torch.save(denoiser.state_dict(), denoising_config.DENOISER_MODEL_NAME)
        else:
            print("验证集的损失没有减小，不保存模型。")

        # 打印验证损失
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")

    # 训练结束提示
    print("\n==========> 训练结束 <==========\n")

    print("本次训练的去噪模型测试结果如下")
    test(denoiser, val_loader, device)

    print("================> 从磁盘加载模型 <================")

    load_denoiser = denoising_model.ConvDenoiser()
    load_denoiser.load_state_dict(torch.load(denoising_config.DENOISER_MODEL_NAME, map_location=device))
    load_denoiser.to(device)

    print("从磁盘加载的去噪模型测试结果如下")
    test(load_denoiser, val_loader, device)
