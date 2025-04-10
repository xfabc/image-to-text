import torch
import torch.nn as nn
from torchvision import models


# 文本解码器：基于 LSTM 的序列生成
class TextDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(TextDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # captions: [batch_size, seq_len]
        # print('1',captions.shape)
        embeddings = self.embedding(captions)  # [batch_size, seq_len, embed_dim]
        # print('2',embeddings.shape)
        # 将图像特征作为初始输入
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # [batch_size, seq_len+1, embed_dim]

        outputs, _ = self.lstm(inputs)  # [batch_size, seq_len+1, hidden_dim]
        outputs = self.fc(outputs)  # [batch_size, seq_len+1, vocab_size]
        return outputs