# inference_segdino.py (MODIFIED with DE-NORMALIZATION)
import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse 

# 假设 (!!) 您的 dataset.py 使用的是标准的 ImageNet 均值和标准差
# 这是 DINOv3 (ViT) 的标准配置。
# 如果您的 ResizeAndNormalize 中硬编码了不同的值，请在此处更新
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# -------------------- Utils (consistent with train visualization) --------------------

# ==============================================================================
#  ↓↓↓ 函数已修改 ↓↓↓
# ==============================================================================
def tensor_to_rgb(img_t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """
    (修改后)
    将 (C, H, W) 标准化 Tensor 转换回 BGR np.ndarray (0-255)
    """
    img = img_t.detach().cpu() # (C, H, W)
    
    # 1. 转换 (mean, std) 为 (C, 1, 1) Tensors 以便广播
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    
    # 2. 反标准化: (img * std) + mean
    img = img.mul_(std_t).add_(mean_t)
    
    # 3. 转换回 [0, 1] 范围
    img = img.clamp(0, 1).numpy()
    
    # 4. 转换回 [0, 255]
    img = (img * 255.0).round().astype(np.uint8)
    
    # 5. (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))
    
    # 6. RGB -> BGR (因为 OpenCV 使用 BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
# ==============================================================================
#  ↑↑↑ 函数已修改 ↑↑↑
# ==============================================================================

def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """(此函数未变)"""
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

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets=None, out_dir=None, thr=0.5, fname_prefix="test"):
    """
    (此函数未变, 它调用的 tensor_to_rgb 已经被修正)
    """
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. 获取原始图像 (H, W, 3) BGR
    #    现在 tensor_to_rgb 会自动执行反归一化
    img_bgr_orig = tensor_to_rgb(inputs)
    
    # 2. 获取预测掩码 (H, W) 0/255
    pred_gray = mask_to_gray(logits, thr)
    
    # 3. 创建预测叠加 (红色)
    pred_color_mask = np.zeros_like(img_bgr_orig)
    pred_color_mask[pred_gray == 255] = [0, 0, 255] # Red
    
    # 4. 混合
    overlay_pred = cv2.addWeighted(img_bgr_orig, 0.7, pred_color_mask, 0.3, 0)

    # 5. 保存
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_overlay.png", overlay_pred)


# -------------------- Main Inference (Modified) --------------------
@torch.no_grad()
def run_inference_and_save(model, loader, device, dice_thr=0.5, vis_dir=None):
    """(此函数未变)"""
    if vis_dir is None:
        print("[Error] 'vis_dir' must be provided to save visualization results. Exiting.")
        return

    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    
    idx_global = 0

    pbar = tqdm(loader, desc="[Inference]")
    for batch in pbar:
        if len(batch) == 3:
            inputs, targets, ids = batch
        elif len(batch) == 2:
            inputs, targets = batch
            ids = [f"case_{idx_global + i:05d}" for i in range(inputs.size(0))]
        else:
            raise ValueError(f"Unexpected batch format. Expected 2 or 3 items, got {len(batch)}")

        inputs  = inputs.to(device)
        targets = targets.to(device) 

        # 模型推理
        logits = model(inputs)
        
        B = inputs.size(0)
        for b in range(B):
            save_eval_visuals(
                idx_global, 
                inputs[b], 
                logits[b], 
                targets[b], 
                vis_dir, 
                thr=dice_thr, 
                fname_prefix="inference"
            )
            idx_global += 1

    print("=" * 60)
    print(f"[Inference Complete] All prediction overlay images saved to: {vis_dir}")
    print("=" * 60)

    return None

def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    """(此函数未变)"""
    obj = torch.load(ckpt_path, map_location=map_location) 
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

def main():
    """(此函数未变)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="tn3k")
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=1024)
    parser.add_argument("--input_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--dice_thr", type=float, default=0.5)

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the trained segmentation model checkpoint (.pth).")
    parser.add_argument("--save_root", type=str, default="./runs")
    parser.add_argument("--img_dir_name", type=str, default="Original")
    parser.add_argument("--label_dir_name", type=str, default="Ground truth")

    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"],
                        help="DINO backbone size: b=ViT-B/16, s=ViT-S/16")
    parser.add_argument("--dino_ckpt", type=str, required=True,
                        help="Path to the pretrained DINO checkpoint (.pth).")
    parser.add_argument("--repo_dir", type=str, default="./dinov3",
                        help="Local path to the DINOv3 torch.hub repo (contains hubconf.py).")

    args = parser.parse_args()

    save_root = os.path.join(args.save_root, f"segdino_{args.dino_size}_{args.input_h}_{args.dataset}")
    vis_dir   = os.path.join(save_root, "inference_vis_overlay") 
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    try:
        from dpt import DPT
        from dataset import FolderDataset, ResizeAndNormalize
    except ImportError as e:
        print(f"Error importing DPT or FolderDataset: {e}")
        print("Please ensure dpt.py and dataset.py are in the same directory or accessible in PYTHONPATH.")
        return

    model = DPT(nclass=1, backbone=backbone)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"[Load segmentation ckpt] {args.ckpt}")
    load_ckpt_flex(model, args.ckpt, map_location=device)

    # Dataset (假设 dataset.py 未修改)
    test_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
    test_dataset = FolderDataset(
        root=os.path.join(args.data_dir, args.dataset),
        split="test",
        img_dir_name=args.img_dir_name,
        label_dir_name=args.label_dir_name,
        mask_ext=args.mask_ext if hasattr(args, "mask_ext") else None,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    run_inference_and_save(
        model, 
        test_loader, 
        device, 
        dice_thr=args.dice_thr, 
        vis_dir=vis_dir
    )


if __name__ == "__main__":
    main()