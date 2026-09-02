import numpy as np  # 导入numpy用于数据处理
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.optim as optim  # 导入优化器模块
import torchvision.transforms as T  # 导入torchvision的图像变换模块
from tqdm import tqdm  # 导入进度条工具

import similarity_config  # 导入配置文件参数
import similarity_data  # 导入自定义数据加载模块
import similarity_engine  # 导入训练引擎模块（包含train_step和val_step）
import similarity_model  # 导入自定义模型模块（包含ConvEncoder和ConvDecoder）
from common import utils  # 导入自定义工具函数（如seed_everything）

# 主程序入口
if __name__ == "__main__":
    # 检测GPU可用性并设置设备
    if torch.cuda.is_available():
        device = "cuda"  # 优先使用GPU
    else:
        device = "cpu"  # 回退到CPU

    # 打印随机种子配置信息
    print("设置训练模型的随机数种子, seed = {}".format(similarity_config.SEED))

    # 调用工具函数设置全局随机种子（确保可复现性）
    utils.seed_everything(similarity_config.SEED)

    # 定义图像预处理流程
    transforms = T.Compose(
        [T.Resize((64, 64)), T.ToTensor()]  # 统一缩放到64x64分辨率  # 转换为PyTorch张量（范围[0,1]）
    )
    # 数据集创建阶段
    print("------------ 正在创建数据集 ------------")
    # 实例化完整数据集（输入和目标均为同一图像，自监督学习）
    full_dataset = similarity_data.ImageDataset(similarity_config.IMG_PATH, transforms)

    # 计算训练集和验证集大小
    train_size = int(similarity_config.TRAIN_RATIO * len(full_dataset))  # 75%训练
    val_size = len(full_dataset) - train_size  # 25%验证

    # 随机划分数据集
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 数据加载器配置阶段
    print("------------ 数据集创建完成 ------------")
    print("------------ 创建数据加载器 ------------")
    # 训练数据加载器（打乱顺序，丢弃最后不完整的批次）
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=similarity_config.TRAIN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    # 验证数据加载器（不打乱，完整加载）
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=similarity_config.TEST_BATCH_SIZE)
    # 全量数据加载器（用于生成嵌入）
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=similarity_config.FULL_BATCH_SIZE)

    print("------------ 数据加载器创建完成 ------------")

    # 定义损失函数（均方误差损失）
    loss_fn = nn.MSELoss()

    # 初始化编码器和解码器
    encoder = similarity_model.ConvEncoder()  # 创建编码器实例
    decoder = similarity_model.ConvDecoder()  # 创建解码器实例

    # 将模型移动到指定设备（GPU/CPU）
    encoder.to(device)
    decoder.to(device)

    # 定义优化器（联合优化编码器和解码器参数）
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=similarity_config.LEARNING_RATE)  # 使用AdamW优化器

    # 初始化最佳损失值为极大值
    min_loss = 9999

    # 开始训练循环
    print("------------ 开始训练 ------------")

    # 使用tqdm进度条遍历预设的epoch数量
    for epoch in tqdm(range(similarity_config.EPOCHS)):
        # 执行一个训练epoch
        train_loss = similarity_engine.train_step(encoder, decoder, train_loader, loss_fn, optimizer, device=device)
        # 打印当前epoch的训练损失
        print(f"\n----------> Epochs = {epoch + 1}, Training Loss : {train_loss} <----------")

        # 执行验证步骤
        val_loss = similarity_engine.val_step(encoder, decoder, val_loader, loss_fn, device=device)

        # 模型保存逻辑：当验证损失创新低时保存模型
        if val_loss < min_loss:
            print("验证集的损失减小了，保存新的最好的模型。")
            min_loss = val_loss
            # 保存编码器和解码器状态字典
            torch.save(encoder.state_dict(), similarity_config.ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), similarity_config.DECODER_MODEL_NAME)
        else:
            print("验证集的损失没有减小，不保存模型。")

        # 打印验证损失
        print(f"Epochs = {epoch + 1}, Validation Loss : {val_loss}")

    # 训练结束提示
    print("\n==========> 训练结束 <==========\n")

    # 生成嵌入阶段
    print("---- 对整个数据集创建嵌入 ---- ")
    # 调用函数生成所有数据的嵌入表示
    embedding = similarity_engine.create_embedding(encoder, full_loader, device)

    # 数据格式转换和保存
    numpy_embedding = embedding.cpu().detach().numpy()  # 转CPU并转numpy
    num_images = numpy_embedding.shape[0]  # 获取样本数量

    # 将高维嵌入展平为二维数组（样本数 x 特征维度）
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    # 保存到npy文件供后续使用
    np.save(similarity_config.EMBEDDING_NAME, flattened_embedding)
