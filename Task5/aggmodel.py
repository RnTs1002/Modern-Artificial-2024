import torch
import torch.nn as nn
import torchvision as tv
import transformers

# 定义一个残差块，用于深度学习中的残差连接
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 四层全连接网络，每层使用 LeakyReLU 激活函数
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU()
        )

    def forward(self, X):
        # 输入与全连接网络输出相加，形成残差连接
        return self.fc(X) + X

# 定义跨模态注意力层
class CrossAttentionLayer(nn.Module):
    def __init__(self, img_dim, txt_dim):
        super().__init__()
        # 图像特征映射到文本特征维度
        self.img_projection = nn.Linear(img_dim, txt_dim)
        # 文本特征映射到图像特征维度
        self.text_projection = nn.Linear(txt_dim, img_dim)

    def forward(self, img_features, txt_features, txt_mask):
        # 投影图像特征和文本特征
        projected_img = self.img_projection(img_features)
        projected_txt = self.text_projection(txt_features)

        # 融合图像和文本特征
        combined_features = projected_img + projected_txt
        # 应用掩码以忽略无效的文本位置
        attention_output = combined_features * (~txt_mask).unsqueeze(dim=-1)

        return attention_output

# 定义多模态情感分析模型
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        TEXT_DIM = 768  # 文本特征维度
        IMG_DIM = 2048  # 图像特征维度

        # 加载预训练的 BERT 文本模型
        self.text_model = transformers.AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        # 加载预训练的 ResNet50 图像模型，并移除最后一层分类器
        self.img_model = nn.Sequential(*list(tv.models.resnet50(pretrained=True).children())[:-1])

        # 跨模态融合层
        self.cross_attention = CrossAttentionLayer(IMG_DIM, TEXT_DIM)

        # 文本编码器层，使用 TransformerEncoder
        self.enc_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(TEXT_DIM, 4, batch_first=True, norm_first=True), 2)

        # 最后的线性层和 Softmax 分类器
        self.fc = nn.Sequential(
            # 输出三分类
            nn.Linear(TEXT_DIM, 3),
            # 使用 Softmax 激活函数
            nn.Softmax(dim=-1)
        )

    def forward(self, img, text):
        with torch.no_grad():
            # 获取文本掩码，并提取文本和图像特征
            # 文本掩码
            mask = ~text['attention_mask'].to(torch.bool)
            # 图像特征展平
            img_features = self.img_model(img).flatten(-3, -1)
            # 文本特征
            txt_features = self.text_model(**text).last_hidden_state

        # 使用编码器处理文本特征
        txt_features = self.enc_layer(txt_features, src_key_padding_mask=mask)

        # 使用跨模态注意力层融合图像和文本特征
        fusion_output = self.cross_attention(img_features, txt_features, mask)

        # 对融合后的特征进行加权池化，忽略无效的文本位置
        feature = (fusion_output * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)

        # 添加噪声以防止过拟合（仅在推理阶段）
        if not self.training:
            feature = (feature + torch.randn_like(feature) * 0.1) / (1.0 + 0.1)

        # 使用线性层进行分类
        return self.fc(feature.squeeze(-2))

# 定义仅使用文本特征的情感分析模型
class OnlyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        TEXT_DIM = 768

        # 加载预训练的 BERT 文本模型
        self.text_model = transformers.AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

        # 最后的线性层和 Softmax 分类器
        self.fc = nn.Sequential(
            # 输出三分类
            nn.Linear(TEXT_DIM, 3),
            # 使用 Softmax 激活函数
            nn.Softmax(dim=-1)
        )

    def forward(self, img, text):
        with torch.no_grad():
            # 获取文本掩码并提取文本特征，图像特征置为零
            mask = ~text['attention_mask'].to(torch.bool)
            txt_features = self.text_model(**text).last_hidden_state

        # 对文本特征进行加权池化
        feature = (txt_features * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)

        # 添加噪声以防止过拟合（仅在推理阶段）
        if not self.training:
            feature = (feature + torch.randn_like(feature) * 0.1) / (1.0 + 0.1)

        return self.fc(feature.squeeze(-2))

# 定义仅使用图像特征的情感分析模型
class OnlyImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        IMG_DIM = 2048

        # 加载预训练的 ResNet50 图像模型，并移除最后一层分类器
        self.img_model = nn.Sequential(*list(tv.models.resnet50(pretrained=True).children())[:-1])

        # 最后的线性层和 Softmax 分类器
        self.fc = nn.Sequential(
            # 输出三分类
            nn.Linear(IMG_DIM, 3),
            # 使用 Softmax 激活函数
            nn.Softmax(dim=-1)
        )

    def forward(self, img, text):
        with torch.no_grad():
            # 提取图像特征，文本特征置为零
            img_features = self.img_model(img).flatten(-3, -1)

        # 添加噪声以防止过拟合（仅在推理阶段）
        if not self.training:
            img_features = (img_features + torch.randn_like(img_features) * 0.1) / (1.0 + 0.1)

        return self.fc(img_features.squeeze(-2))
