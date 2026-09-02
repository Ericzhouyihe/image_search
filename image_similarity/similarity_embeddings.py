# 定义模块的公开接口
__all__ = ["create_embeddings", "get_embedding_collection", "search_similar_img_ids"]

# 定义模块的公开接口，仅暴露FolderDataset类
from PIL import Image  # 图像处理库
import os  # 操作系统接口库

# 正则表达式相关库
import re
import torchvision.transforms.transforms as T
import torch

import numpy

from image_similarity import similarity_torch_model
from image_similarity import similarity_config

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from math import ceil


def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    # 定义转换函数：将数字部分转换为整数，非数字部分转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)


def get_id2image(main_dir, transform):
    """获取所有图片的id和图片
    :param main_dir: 图片所在目录
    :param transform: 图片预处理
    :return: id2imgs: id和图片的映射
    """

    all_imgs = sorted_alphanumeric(os.listdir(main_dir))

    id2imgs = {}

    for i, img in enumerate(all_imgs):
        img_loc = os.path.join(main_dir, img)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = transform(image)
        id2imgs[str(i)] = tensor_image.numpy()

    return id2imgs


"""
定义一个嵌入函数，继承自EmbeddingFunction类
"""


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model
        model.to("cpu")

    def __call__(self, input: Documents) -> Embeddings:
        with torch.no_grad():
            return self.model(torch.tensor(numpy.array(input))).squeeze(0).numpy()


def create_embeddings(encoder):
    """
    将图片嵌入向量写入数据库
    :param encoder: 嵌入模型
    :return: None
    """
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    print("正在加载图片")
    id2imgs = get_id2image("../common/dataset", transform)

    ids = list(id2imgs.keys())
    imgs = list(id2imgs.values())
    print("图片加载完毕")

    print("正在向嵌入数据库中写入向量")
    chroma_client = chromadb.PersistentClient(similarity_config.CHROMA_BACKEND_PATH)
    collection = chroma_client.get_or_create_collection(
        name="image_similarity", embedding_function=MyEmbeddingFunction(encoder)
    )

    insert_batch_size = 5000
    for i in range(ceil(len(ids) / insert_batch_size)):
        collection.upsert(
            ids=ids[i * insert_batch_size : min(len(ids), (i + 1) * insert_batch_size)],
            images=imgs[i * insert_batch_size : min(len(ids), (i + 1) * insert_batch_size)],
        )
    print("向量写入完成")


def get_embedding_collection(encoder):
    """
    获取嵌入向量集合
    :return: 嵌入向量集合
    """
    chroma_client = chromadb.PersistentClient(similarity_config.CHROMA_BACKEND_PATH)
    return chroma_client.get_or_create_collection(
        name="image_similarity", embedding_function=MyEmbeddingFunction(encoder)
    )


def search_similar_img_ids(collection, image_tensor, img_cnt):
    result = collection.query(
        query_images=[image_tensor.numpy()],
        n_results=img_cnt,
        # include=["embeddings"]
    )
    # print(result)
    # print(type(result))
    # print(result["ids"])
    ids = [int(id) for id in result["ids"][0]]

    return ids


if __name__ == "__main__":
    print("正在加载嵌入模型")
    encoder = similarity_torch_model.ConvEncoder()  # 初始化编码器
    # 加载编码器的预训练权重（自动处理设备映射）
    encoder.load_state_dict(
        torch.load(os.path.join("..", similarity_config.SIMILARITY_PACKAGE_NAME, similarity_config.ENCODER_MODEL_NAME))
    )
    print("嵌入模型加载完毕")

    # 插入嵌入向量，初次测试需要插入
    # insert_embeddings(encoder)

    # 测试嵌入向量集合
    collection = get_embedding_collection(encoder)

    print(collection.peek())

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    test_img_path = os.path.join(similarity_config.IMG_PATH, "6582.jpg")
    # print(f"{test_img_path = }")
    test_img = Image.open(test_img_path).convert("RGB")
    test_img_tensor = transform(test_img)

    ids = search_similar_img_ids(collection, test_img_tensor, 5)
    print(f"{ids = }")
