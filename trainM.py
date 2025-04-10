import torch.nn as nn
import torch
from ModelMoudle.ImageTextCaption import ImageCaptioningModel
import torchvision.transforms as transforms
from datasetUtil.dataPress import ImageCaptionDataset, SimpleTokenizer
from torch.utils.data import Dataset, DataLoader
import pickle
import json

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 示例数据
image_paths = ["cat.jpg", "dog.jpg"]
captions = ["a cute cat sitting on the floor", "a dog running in the park"]

tokenizer = SimpleTokenizer(captions)
dataset = ImageCaptionDataset(image_paths, captions, tokenizer, max_seq_len=20, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义超参数
embed_dim = 2048
hidden_dim = 2048
vocab_size = tokenizer.vocab_size
num_layers = 1

# 初始化模型、损失函数和优化器
model = ImageCaptioningModel(embed_dim, hidden_dim, vocab_size, num_layers).to(
    "cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充符
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print(model)

# 训练循环
for epoch in range(20):  # 假设训练 3 个 epoch
    model.train()
    for images, captions in dataloader:
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        captions = captions.to("cuda" if torch.cuda.is_available() else "cpu")
        # print('3',captions[:,:-1].shape)
        # 前向传播
        outputs = model(images, captions[:, :-1])  # 输入去掉最后一个词
        # print('4',outputs.shape)

        # targets = captions[:, 1:]  # 目标去掉第一个词
        # print('5',targets.shape)
        targets = captions
        print(targets.shape)
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    # 保存模型
    torch.save(model.state_dict(), "image_captioning_model.pth")

    # 保存 tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # 保存 transform
    transform_save_path = "transform.pkl"  # 文件保存路径
    with open(transform_save_path, "wb") as f:
        pickle.dump(transform, f)

    transform_config = {
        "Resize": {"size": (224, 224)},
        "ToTensor": {},
    }
    config = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "transform_config": transform_config,  # 保存 transform 配置
    }

    with open("config.json", "w") as f:
        json.dump(config, f)

