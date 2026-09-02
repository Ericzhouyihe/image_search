# 导入必要的库
import base64
import sys
from io import BytesIO
from pathlib import Path

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torchvision.transforms as T  # 图像预处理工具
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image  # PIL图像处理库

# 项目根目录（本文件位于 <root>/web/web_app.py）
ROOT_DIR = Path(__file__).resolve().parent.parent
# 将项目根目录加入模块搜索路径，使 image_* 各包可被导入
sys.path.insert(0, str(ROOT_DIR))

from image_classification import classification_config, classification_model  # noqa: E402
from image_denoising import denoising_config, denoising_model  # noqa: E402
from image_similarity import similarity_config, similarity_embeddings, similarity_model  # noqa: E402

# 创建 FastAPI 应用实例
app = FastAPI(title="智图寻宝", description="智能商品识别系统")

# 静态资源挂载（按前缀路由到对应目录，HTML 中直接引用原始 URL）
app.mount("/dataset", StaticFiles(directory=ROOT_DIR / "common" / "dataset"), name="dataset")
app.mount("/pictures", StaticFiles(directory=ROOT_DIR / "web" / "pictures"), name="pictures")
app.mount("/logo", StaticFiles(directory=ROOT_DIR / "web" / "logo"), name="logo")


# 首页路由
@app.get("/")
def index():
    return FileResponse(ROOT_DIR / "web" / "templates" / "index.html")


# 设备检测与设置（优先使用GPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_state(model, package_dir: str, filename: str) -> None:
    """从指定包目录加载模型权重"""
    model.load_state_dict(torch.load(ROOT_DIR / package_dir / filename, map_location=device))


print("正在加载去噪模型")
denoiser = denoising_model.ConvDenoiser()
_load_state(denoiser, "image_denoising", denoising_config.DENOISER_MODEL_NAME)
denoiser.to(device).eval()
print("去噪模型加载完毕")

print("正在加载分类模型")
classifier = classification_model.Classifier()
_load_state(classifier, "image_classification", classification_config.CLASSIFIER_MODEL_NAME)
classifier.to(device).eval()
print("分类模型加载完毕")

print("正在加载嵌入模型")
encoder = similarity_model.ConvEncoder()
_load_state(encoder, "image_similarity", similarity_config.ENCODER_MODEL_NAME)
encoder.to(device).eval()
print("嵌入模型加载完毕")

print("正在加载向量集合")
# 只需创建一次嵌入向量集合（初次使用需先运行 similarity_embeddings.create_embeddings）
collection = similarity_embeddings.get_embedding_collection(encoder)
print("向量集合加载完毕")


def _preprocess(file: UploadFile) -> torch.Tensor:
    """将上传的图片解码为 (1, 3, 64, 64) 的归一化张量"""
    file.file.seek(0)
    image = Image.open(file.file).convert("RGB")
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    return transform(image).unsqueeze(0)


def _encode_base64(img_array: np.ndarray) -> str:
    """将 HWC、0-255 的 uint8 数组编码为前端可用的 base64 PNG 字符串"""
    img = Image.fromarray(img_array.astype("uint8"))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# 去噪路由：返回加噪图与去噪图（均为 base64 PNG）
@app.post("/denoising")
def get_denoised_image(image: UploadFile = File(...)):
    tensor = _preprocess(image).squeeze(0)

    # 向输入图像添加随机噪声，并将像素值裁剪到 [0, 1]
    noisy_img = torch.clip(tensor + denoising_config.NOISE_FACTOR * torch.randn(*tensor.shape), 0.0, 1.0)

    with torch.no_grad():
        denoised_image = denoiser(noisy_img.unsqueeze(0).to(device)).squeeze(0).cpu()

    # CHW -> HWC 并转换到 0-255 范围
    noisy_np = noisy_img.permute(1, 2, 0).numpy() * 255
    denoised_np = denoised_image.permute(1, 2, 0).numpy() * 255

    return {"noisy_img": _encode_base64(noisy_np), "denoised_image": _encode_base64(denoised_np)}


# 分类路由：返回中文商品类型文本
@app.post("/classification")
def classification(image: UploadFile = File(...)):
    with torch.no_grad():
        logits = classifier(_preprocess(image).to(device))

    pred_idx = int(np.argmax(logits.cpu().detach().numpy()))
    return "您搜索的商品类型是：" + classification_config.classification_names[pred_idx]


# 相似图像检索路由：返回相似图片索引列表
@app.post("/simimages")
def simimages(image: UploadFile = File(...)):
    tensor = _preprocess(image).squeeze(0)
    indices_list = similarity_embeddings.search_similar_img_ids(collection, tensor, img_cnt=5)
    return {"indices_list": indices_list}


# 主程序入口
if __name__ == "__main__":
    import uvicorn

    # 启动 FastAPI 应用，禁用调试模式，监听9000端口
    uvicorn.run(app, host="0.0.0.0", port=9000)
