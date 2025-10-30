import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch


# SalienceRefiner定义
# ----------------------------------------------------------------------
# 创新点：显著性对比增强模块 (Salience Refiner)
# 通过特征空间距离对比，增强异常（缺陷）特征的信噪比。
# ----------------------------------------------------------------------
class SalienceRefiner(nn.Module):
    """
    通过计算每个 Patch Token 与全局原型之间的距离，生成显著性权重，
    用于增强特征图中属于“异常/缺陷”的小目标特征。
    """
    def __init__(self, in_dim):
        super(SalienceRefiner, self).__init__()
        self.in_dim = in_dim
        
        # 可训练的权重生成器：将显著性分数 (1维) 映射到最终的权重。
        # 使用轻量级 MLP 赋予模型一定的可学习性。
        self.weight_generator = nn.Sequential(
            nn.Linear(1, 1), 
            nn.Sigmoid()     # 确保权重在 (0, 1) 之间
        )
        
    def forward(self, x_tokens):
        """
        Args:
            x_tokens (Tensor): 输入的 Patch Tokens (B, N, D)。
        
        Returns:
            Tensor: 增强后的 Patch Tokens (B, N, D)。
        """
        B, N, D = x_tokens.shape
        
        # 1. 计算全局原型 (Global Prototype)
        # 简单取所有 Patch Token 的均值作为全局/背景原型 P_global (B, 1, D)
        p_global = x_tokens.mean(dim=1, keepdim=True)
        
        # 2. 计算每个 Patch 与全局原型的相似度/距离 (显著性分数)
        # 标准化特征 (计算余弦相似度前必需)
        x_norm = F.normalize(x_tokens, dim=-1)
        p_global_norm = F.normalize(p_global, dim=-1)
        
        # 相似度 S: (B, N, 1). 越接近 1 越接近背景。
        similarity_score = torch.sum(x_norm * p_global_norm, dim=-1, keepdim=True)
        
        # 3. 转换为显著性权重 W_S
        # 显著性 = 1 - 相似度 (Score越小，权重越高)
        salience_score = 1.0 - similarity_score 
        
        # 经过可学习的映射，生成最终的权重 W
        W_S = self.weight_generator(salience_score)
        
        # 4. 特征增强 (逐元素相乘)
        x_enhanced_tokens = x_tokens * W_S
        
        return x_enhanced_tokens

#L-decoder头部定义，将多尺度特征进行融合并输出分割结果
class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024],
    ):
        super(DPTHead, self).__init__()

        # -------------------------------------------------------------
        # 创新点集成：定义 Salience Refiner
        # in_channels 对应 DINOv3 的 embed_dim
        self.salience_refiner = SalienceRefiner(in_dim=in_channels)
        # -------------------------------------------------------------



        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(features*4, nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        # -------------------------------------------------------------
        # 创新点集成：对最后一层特征进行显著性增强
        # DINOv3 输出的 out_features 是一个列表，最后一层 (索引 3) 是最高语义层。
        # -------------------------------------------------------------


        out_features_list = list(out_features) # <-- 新增：将元组转换为列表


        # 1. 增强最高语义层特征 (out_features[3] 或列表的最后一个元素)
        last_layer_idx = len(out_features) - 1 
        last_feature_token = out_features[last_layer_idx]
        
        # 使用 Salience Refiner 增强特征
        enhanced_last_feature_token = self.salience_refiner(last_feature_token)
        
        # 将增强后的特征替换原来的最后一层特征
        # 这确保了融合时最高语义层的特征具有最高的缺陷信噪比
        out_features_list[last_layer_idx] = enhanced_last_feature_token
        
        
        # -------------------------------------------------------------
        out = []
        for i, x in enumerate(out_features_list):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))                  #将 ViT 输出的一维 Patch Token 序列（形状B*N*D）重塑为二维特征图（形状 B*D*H*W），以便进行 2D 卷积和上采样。
            x = self.projects[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        target_hw = layer_1_rn.shape[-2:]  
        #此处对应论文中的双线性插值
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out



#encoder封装与前向传播，封装了冻结的骨干网络dinov3，并定义整个模型的前向传播过程
class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
        }#确定从dinov3的哪几层提取特征，此处为2，5，8，11
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)




    #冻结骨干网络参数
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16           #计算 ViT Patch Token 对应的空间分辨率（假设 Patch Size 为 16*16）
        features = self.backbone.get_intermediate_layers(
            x, n = self.intermediate_layer_idx[self.encoder_size]
        )
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)          #将 L-Decoder 的输出 Mask 上采样回原始输入图像的分辨率。
        return out
