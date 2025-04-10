from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import torch.nn.functional as F
import torch


# 自定义数据集类
class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer, max_seq_len, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # 加载图像
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 分词并转换为 token IDs
        tokens = self.tokenizer(caption)
        token_ids = [self.tokenizer.vocab[token] for token in tokens]
        token_ids = token_ids[:self.max_seq_len]  # 截断
        token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))  # 填充

        return image, torch.tensor(token_ids)


# 示例分词器
# class SimpleTokenizer:
#     def __init__(self, captions):
#         all_tokens = " ".join(captions).split()
#         counter = Counter(all_tokens)
#         self.vocab = {word: idx + 1 for idx, word in enumerate(counter.keys())}
#         self.vocab["<PAD>"] = 0  # 填充符
#         self.vocab_size = len(self.vocab)
#
#     def __call__(self, text):
#         return text.split()
class SimpleTokenizer:
    def __init__(self, captions):
        # 获取所有 token
        all_tokens = " ".join(captions).split()
        counter = Counter(all_tokens)

        # 构建词汇表
        self.vocab = {word: idx + 3 for idx, word in enumerate(counter.keys())}  # 从 3 开始，预留特殊 token 的位置
        # 添加特殊 token
        self.vocab["<PAD>"] = 0  # 填充符
        self.vocab["<START>"] = 1  # 开始符
        self.vocab["<END>"] = 2  # 结束符
        self.vocab_size = len(self.vocab)

        # 定义特殊 token 属性
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.pad_token = "<PAD>"

        # 构建反向词汇表
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

    def __call__(self, text):
        """
        将文本分词为 token 列表。
        """
        return text.split()

    def encode(self, text):
        """
        将文本编码为 token ID 列表。
        """
        if isinstance(text, list):  # 如果传入的是列表，先转换为字符串
            text = " ".join(text)
        tokens = self(text)  # 使用 __call__ 方法分词
        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids):
        """
        将 token ID 列表解码为文本。
        """
        if isinstance(token_ids, str):  # 如果传入的是字符串，先转换为列表
            token_ids = [int(idx) for idx in token_ids.split()]
        return " ".join([self.idx_to_word[idx] for idx in token_ids])

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