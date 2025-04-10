import torch
import torch.nn as nn
from torchvision import models


# 图像编码器：基于 ResNet 提取图像特征
class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后两层

    def forward(self, images):
        features = self.feature_extractor(images)  # [batch_size, 2048, 7, 7]

        features = features.permute(0, 2, 3, 1)  # [batch_size, 7, 7, 2048]

        features = features.view(features.size(0), -1, features.size(-1))  # [batch_size, 49, 2048]

        return features