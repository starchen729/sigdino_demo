import torch
import torch.nn as nn
import torch.nn.functional as F






try:
    import kornia.morphology as K_morph
    import torch.nn.functional as F
    KORNIA_AVAILABLE = True
except ImportError:
    # 如果没有 Kornia，我们使用一个警告占位符
    KORNIA_AVAILABLE = False
    print("WARNING: Kornia not found. Boundary weights will be 1.0 (no tolerance).")









# 用于防止分母为零，提高数值稳定性
SMOOTH = 1e-6 

# ---------------------------------------------------------------------
# 1. 评估指标 (Evaluation Metrics): DSC 和 IoU
#    用于 train_one_epoch 中的 with torch.no_grad() 块内
#    基于硬预测 (Hard Prediction, 需要阈值化)
# ---------------------------------------------------------------------

def dice_binary_torch(logits, targets, thresh=0.5, smooth=SMOOTH):
    """
    计算二分类分割的 Dice 系数 (DSC)。
    Args:
        logits (Tensor): 模型的原始输出 (B, 1, H, W)。
        targets (Tensor): 真实标签 (B, 1, H, W)。
        thresh (float): 阈值，用于将概率转换为硬 Mask。
    """
    # 1. 将 logits 转换为概率，并进行阈值化，得到硬预测 Mask
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float() 
    
    # 2. 确保 targets 和 pred 是相同的类型和形状 (float)
    targets = targets.float()

    # 3. 计算 TP, FP, FN
    # 展平处理，确保计算的是整个 Batch 的平均
    pred = pred.view(pred.shape[0], -1) 
    targets = targets.view(targets.shape[0], -1)

    intersection = torch.sum(pred * targets, dim=1) # TP
    union = torch.sum(pred, dim=1) + torch.sum(targets, dim=1) # (TP + FP) + (TP + FN)

    # Dice Coefficient (DSC)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    
    # 返回 Batch 中每个样本的 Dice Score (mean() 在 train_one_epoch 中被调用)
    return dice_score 

def iou_binary_torch(logits, targets, thresh=0.5, smooth=SMOOTH):
    """
    计算二分类分割的 IoU (交并比)。
    Args:
        logits (Tensor): 模型的原始输出 (B, 1, H, W)。
        targets (Tensor): 真实标签 (B, 1, H, W)。
        thresh (float): 阈值，用于将概率转换为硬 Mask。
    """
    # 1. 转换为硬预测 Mask
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float() 
    
    targets = targets.float()

    # 2. 展平处理
    pred = pred.view(pred.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = torch.sum(pred * targets, dim=1) # TP
    union = torch.sum(pred, dim=1) + torch.sum(targets, dim=1) - intersection # TP + FP + FN
    
    # IoU
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score


# ---------------------------------------------------------------------
# 2. 训练损失 (Training Loss): Dice Loss
#    用于替换 train_one_epoch 中的 PlaceholderDiceLoss
#    基于软预测 (Soft Prediction, 概率)
# ---------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss (用于二分类分割训练)
    """
    def __init__(self, smooth=SMOOTH):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): 模型的原始输出 (B, 1, H, W)。
            targets (Tensor): 真实标签 (B, 1, H, W)，必须是 float 类型。
        """
        # 1. 转换为概率 (Sigmoid 激活)
        probabilities = torch.sigmoid(logits)
        
        # 2. 展平处理 (按 Batch 维度)
        probabilities = probabilities.view(probabilities.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        
        # 3. 计算交集和并集
        # 注意：Dice Loss 通常使用 p*g 作为交集，p^2+g^2 作为并集的近似，但这里我们使用更常见的 Dice 指标公式
        intersection = torch.sum(probabilities * targets, dim=1)
        union = torch.sum(probabilities, dim=1) + torch.sum(targets, dim=1)
        
        # 4. Dice Score 和 Loss
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回平均 Dice Loss (1 - DSC)
        return torch.mean(1. - dice_score)


# ---------------------------------------------------------------------
# 3. 边界容错权重生成函数 (创新点)
# ---------------------------------------------------------------------


def get_boundary_weights_map(targets_float, core_weight=5.0, boundary_weight=1.0, 
                             bg_weight=0.1, kernel_size=3):
    """
    生成边界容错权重图 W。
    W = (核心区域 * core_weight) + (边界环 * boundary_weight) + (背景 * bg_weight)
    
    Args:
        targets_float (Tensor): 真实标签 (B, 1, H, W), float 类型 (0 或 1)。
        core_weight (float): 缺陷核心区域的权重（高惩罚）。
        boundary_weight (float): 缺陷边界环形区域的权重（标准惩罚）。
        bg_weight (float): 背景区域的权重（低惩罚）。
        kernel_size (int): 用于形态学操作的核尺寸。
    """
    if not KORNIA_AVAILABLE:
        # 如果 Kornia 不可用，返回一个均匀的低权重背景图（仅作占位）
        return torch.ones_like(targets_float) * bg_weight

    device = targets_float.device
    
    # Kornia 需要一个形状为 (kernel_size, kernel_size) 的结构元 (Structure Element)
    kernel = torch.ones(kernel_size, kernel_size, device=device) 
    
    # -----------------------------------------------------------
    # 1. 腐蚀 (Erosion): 找到缺陷核心区域 (Core Mask)
    #    腐蚀掉边界，只留下中心区域
    # -----------------------------------------------------------
    # Kornia 的形态学操作要求输入的形状为 (B, C, H, W)。
    # targets_float (B, 1, H, W) 符合要求
    core_mask = K_morph.erosion(targets_float, kernel)

    # -----------------------------------------------------------
    # 2. 边界环形区域 (Boundary Ring)
    #    Boundary Ring = 原始 Mask - 核心区域
    # -----------------------------------------------------------
    boundary_ring = targets_float - core_mask
    
    # -----------------------------------------------------------
    # 3. 背景区域 (Background)
    # -----------------------------------------------------------
    background_mask = 1.0 - targets_float

    # -----------------------------------------------------------
    # 4. 组合加权图 W
    # -----------------------------------------------------------
    
    # 初始化权重为 bg_weight
    weights = torch.ones_like(targets_float) * bg_weight
    
    # 在前景区域 (targets_float > 0) 覆盖 boundary_weight
    # boundary_ring 和 core_mask 加起来就是 targets_float
    
    # 将边界环的权重设置为 boundary_weight
    weights[boundary_ring > 0.5] = boundary_weight
    
    # 将核心区域的权重设置为 core_weight (覆盖边界权重)
    weights[core_mask > 0.5] = core_weight

    # ⚠️ 确保 background_mask 区域是 bg_weight（已在初始化时完成）
    # 另一种更清晰的组合方式（如代码段中）：
    # final_weights = (core_mask * core_weight) + (boundary_ring * boundary_weight) + (background_mask * bg_weight)
    
    return weights






# def get_boundary_weights_map(targets_float, core_weight=5.0, boundary_weight=1.0):
#     """
#     边界容错权重图生成函数 (目前仅为占位符)
#     """
#     # 占位符: 暂时返回权重为 1.0 的张量，即不加权
#     weights = torch.ones_like(targets_float, dtype=torch.float)
#     return weights