
import torch.nn as nn
import timm

class SpineCNN(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_s_in21ft1k", out_dim=1, pretrained=True, feat_dim=512):
        super().__init__()
        self.backbone_name = backbone

        # 创建 backbone
        self.encoder = timm.create_model(
            backbone,
            in_chans=2,              # 输入通道数=2
            num_classes=0,           # 去掉默认分类头
            pretrained=pretrained,
        )

        # 获取特征维度
        if hasattr(self.encoder, "num_features"):
            in_features = self.encoder.num_features
        else:
            raise ValueError(f"Backbone {backbone} not supported!")

        # 统一映射到 feat_dim=512
        self.feature_proj = nn.Linear(in_features, feat_dim)

        # 分类头（对每个椎骨二分类）
        self.classifier = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        """
        x: (B, 24, 2, H, W)
        return_features: 是否返回中间特征 (B, 24, 512)
        """
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)  # 合并 batch 和椎骨

        # backbone 提取特征
        feat = self.encoder(x)              # (B*V, in_features)
        feat = self.feature_proj(feat)      # (B*V, 512)

        # 分类预测
        logits = self.classifier(feat)      # (B*V, 1)

        feat = feat.view(B, V, -1)
        logits = logits.view(B, V)          # (B, 24)

        return logits, feat