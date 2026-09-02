# ------------ 数据路径与预处理配置 ------------
IMG_PATH = "../common/dataset/"  # 原始图像存储根目录（需确保存在子目录）
IMG_HEIGHT = 64
IMG_WIDTH = 64

# ------------ 随机性与数据划分配置 ------------
SEED = 73  # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75  # 训练集划分比例（75%训练，25%验证）
VAL_RATIO = 1 - TRAIN_RATIO  # 验证集比例（自动计算，无需修改）
SHUFFLE_BUFFER_SIZE = 100  # 数据混洗缓冲区大小（影响数据加载顺序随机性）

# ------------ 训练超参数配置 ------------
LEARNING_RATE = 1e-3  # 初始学习率（AdamW优化器使用）
EPOCHS = 30  # 总训练轮次（需平衡过拟合与欠拟合）
TRAIN_BATCH_SIZE = 32  # 训练批次大小（GPU显存不足时可调小）
VAL_BATCH_SIZE = 32  # 验证批次大小（建议与训练批次一致）
TEST_BATCH_SIZE = 32  # 测试批次大小（建议与训练批次一致）
FULL_BATCH_SIZE = 32  # 全量数据生成嵌入时的批次大小
CHROMA_INSERT_BATCH = 5000  # Chroma 限制单次插入数据不能超过 5000

# ------------ 模型与嵌入存储配置 ------------
PACKAGE_NAME = "image_similarity"
ENCODER_MODEL_NAME = "deep_encoder.pt"  # 编码器权重保存路径（需写权限）
DECODER_MODEL_NAME = "deep_decoder.pt"  # 解码器权重保存路径（需写权限）
CHROMA_BACKEND_PATH = "chroma_backend"  # 特征嵌入存储路径
