# train_segdinov3.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from loss_cal import dice_binary_torch, iou_binary_torch, DiceLoss





# ⚠️ 注意: 这是一个占位符。请在后续步骤中用您实际的边界容错权重生成函数替换它。
# 目标: 生成一个与 targets 相同形状的权重图 W (W[核心] > W[边界])
def get_boundary_weights_map(targets, core_weight=5.0, boundary_weight=1.0):
    # 假设这里是形态学操作，用于生成边界权重图
    #print("Warning: Using Placeholder Boundary Weight Map. Replace this function with your implementation.")
    
    # 默认返回一个权重为 1.0 的全张量（即不加权）
    weights = torch.ones_like(targets, dtype=torch.float)
    return weights





def tensor_to_rgb(img_t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img

def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        pred_gray = mask_to_gray(logits[i], thr)
        gt_gray   = mask_to_gray(targets[i], thr)
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png",  img_bgr)
        cv2.imwrite(base + "_pred.png", pred_gray)
        cv2.imwrite(base + "_gt.png",   gt_gray)

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    pred_gray = mask_to_gray(logits, thr)
    gt_gray   = mask_to_gray(targets, thr)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png",  img_bgr)
    cv2.imwrite(base + "_pred.png", pred_gray)
    cv2.imwrite(base + "_gt.png",   gt_gray)

def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)

def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice

def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, vis_dir=None, epoch=0):
    model.train()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []


    #此处为交叉熵损失函数，应该改成Dice损失和交叉熵的加权平均
    #criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss(reduction='none') 
    
    # 新增的 Dice Loss
    criterion_dice = DiceLoss() 
    
    # 组合损失的权重（可调参数）
    lambda_dice = 0.5
    lambda_bce  = 0.5

    # 确保 dice_binary_torch 和 iou_binary_torch 在全局可用
    # (假设它们是您代码中已定义的评估函数)
    from loss_cal import dice_binary_torch, iou_binary_torch


    first_batch_logged = False
    pbar = tqdm(train_loader, desc=f"[Train e{epoch}]")
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs  = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)




        # 确保 targets 转换为 float 类型，以便与 logits 进行计算
        targets_float = targets.float()




        if num_classes == 1:
            #loss_cls = criterion(logits, targets)


            # 1. Dice Loss: 解决类别不平衡（解决欠拟合的核心）
            loss_dice = criterion_dice(logits, targets_float)
            
            # 2. 边界容错加权 BCE Loss (解决边界模糊/过拟合的创新点)
            
            # 2.1 获取边界容错权重图 W (需确保 targets_float 形状和 device 匹配)
            # ⚠️ 注意: 此处使用了占位符函数
            boundary_weights = get_boundary_weights_map(targets_float).to(device)
            
            # 2.2 计算逐像素的 BCE Loss
            loss_bce_pixel = criterion_bce(logits, targets_float)
            
            # 2.3 应用边界容错权重 W 进行加权
            loss_bce_weighted = torch.mean(boundary_weights * loss_bce_pixel)
            
            # 3. 组合损失: 解决欠拟合 (Dice) 和过拟合 (加权BCE)
            loss = lambda_dice * loss_dice + lambda_bce * loss_bce_weighted

        else:
            criterion = nn.CrossEntropyLoss()
            #loss_cls = criterion(logits, targets.squeeze(1).long())
            loss = criterion(logits, targets.squeeze(1).long())
        #loss = loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets, out_dir=vis_dir, max_save=8, thr=dice_thr)
            first_batch_logged = True
    avg_loss = total_loss / max(1, len(train_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Train Epoch {epoch}] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    return avg_loss, avg_dice

@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5, vis_dir=None):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores  = []
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    idx_global = 0
    pbar = tqdm(val_loader, desc="[Eval]")
    for (inputs, targets, _) in pbar:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())
        total_loss += loss.item()
        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou  = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        dice_scores.append(dice)
        iou_scores.append(iou)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}", iou=f"{iou:.4f}")
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            B = inputs.size(0)
            for b in range(B):
                save_eval_visuals(
                    idx_global,
                    inputs[b],
                    logits[b],
                    targets[b],
                    out_dir=vis_dir,
                    thr=dice_thr,
                    fname_prefix="val"
                )
                idx_global += 1
    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou  = float(np.mean(iou_scores))  if len(iou_scores)  > 0 else 0.0
    print(f"[Eval] loss={avg_loss:.4f}  dice={avg_dice:.4f}  iou={avg_iou:.4f}")
    return avg_loss, avg_dice, avg_iou

def main():
    import argparse
    import random
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True,help="Path to the pretrained DINO checkpoint (.pth). "
                         "Use ViT-B/16 checkpoint for --dino_size b, "
                         "or ViT-S/16 checkpoint for --dino_size s.")
    parser.add_argument("--dino_size", type=str, default="b", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--last_layer_idx", type=int, default=-1)
    parser.add_argument("--vis_max_save", type=int, default=8)
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_root = f"./runs/segdino_{args.dino_size}_{args.input_h}_{args.dataset}"
    os.makedirs(save_root, exist_ok=True)
    train_vis_dir = os.path.join(save_root, "train_vis")
    val_vis_dir   = os.path.join(save_root, "val_vis")
    ckpt_dir      = os.path.join(save_root, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    from dpt import DPT
    model = DPT(nclass=1, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    from dataset import FolderDataset, ResizeAndNormalize
    root = os.path.join(args.data_dir, args.dataset)

    train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    val_transform   = ResizeAndNormalize(size=(args.input_h, args.input_w))

    train_dataset = FolderDataset(
        root=root,
        split="train",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=train_transform,
    )
    val_dataset = FolderDataset(
        root=root,
        split="val",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=val_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    best_val_dice = -1.0
    best_val_dice_epoch = -1
    best_val_iou  = -1.0
    best_val_iou_epoch  = -1

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=train_vis_dir, epoch=epoch
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, device,
            num_classes=args.num_classes, dice_thr=0.5,
            vis_dir=val_vis_dir
        )

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            latest_path
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_dice_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}_{val_iou:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best ckpt: {best_path}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch

    print("=" * 60)
    print(f"[Summary] Best Val Dice = {best_val_dice:.4f} @ epoch {best_val_dice_epoch}")
    print(f"[Summary] Best Val IoU  = {best_val_iou:.4f}  @ epoch {best_val_iou_epoch}")
    print("=" * 60)

if __name__ == "__main__":
    main()
