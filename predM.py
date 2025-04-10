import torch
import json
import pickle
from torchvision import transforms
from PIL import Image
from ModelMoudle.ImageTextCaption import ImageCaptioningModel  # 替换为你的模型模块

# 加载配置文件
with open("config.json", "r") as f:
    config = json.load(f)

# 加载 transform
with open("transform.pkl", "rb") as f:
    transform = pickle.load(f)

# 加载 tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 加载模型
model = ImageCaptioningModel(
    config["embed_dim"], config["hidden_dim"], config["vocab_size"], config["num_layers"]
).to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("image_captioning_model.pth"))
model.eval()
print("Model, tokenizer, and transform loaded.")

# 图像预处理函数
def preprocess_image(image_path):
    """
    加载图像并应用预处理 transform。
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image.to("cuda" if torch.cuda.is_available() else "cpu")

# 生成描述函数
def generate_caption(model, image, max_len=20):
    """
    使用模型生成图像描述。
    """
    model.eval()
    with torch.no_grad():
        # 初始化输入
        caption = [tokenizer.start_token]  # 假设 tokenizer 有 start_token
        for _ in range(max_len):
            # 将当前 caption 转换为张量
            caption_tensor = torch.tensor([tokenizer.encode(caption)]).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            # 前向传播
            output = model(image, caption_tensor)
            # 获取预测的下一个词
            next_word_idx = output.argmax(dim=-1)[:, -1].item()
            # 将词添加到 caption 中
            caption.append(tokenizer.decode([next_word_idx]))
            # 如果遇到结束符，停止生成
            if next_word_idx == tokenizer.end_token:  # 假设 tokenizer 有 end_token
                break
    return " ".join(caption[1:-1])  # 去掉 start_token 和 end_token

# 测试推理
test_image_path = "cat0.jpg"  # 替换为你的测试图像路径
test_image = preprocess_image(test_image_path)
caption = generate_caption(model, test_image)
print(f"Generated Caption: {caption}")
