import torch.nn as nn
from ModelMoudle.imageEncode import ImageEncoder
from ModelMoudle.textDecode import TextDecoder
# 多模态模型：图像描述生成
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_decoder = TextDecoder(embed_dim, hidden_dim, vocab_size, num_layers)

    def forward(self, images, captions):
        image_features = self.image_encoder(images)  # [batch_size, 49, 2048]

        image_features = image_features.mean(dim=1)  # [batch_size, 2048] (平均池化)

        outputs = self.text_decoder(image_features, captions)

        return outputs