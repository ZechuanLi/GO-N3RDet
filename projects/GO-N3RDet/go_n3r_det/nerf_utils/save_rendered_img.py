# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    mse = ((pred - target)**2).mean()
    return compute_psnr_from_mse(mse).cpu().numpy()


def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y:y + h, x:x + w]
        target = target[y:y + h, x:x + w]
    try:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1)
    except ValueError:
        ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), multichannel=True)
    return ssim

def save_rendered_img2(img_meta, rendered_results):
    filename = img_meta[0]['img_path']
    scenes = filename.split('/')[-2]

    for ret in rendered_results:
        depth = ret['outputs_coarse']['depth']
        rgb = ret['outputs_coarse']['rgb']
        gt = ret['gt_rgb']
        gt_depth = ret['gt_depth']
        
        # 打印图像尺寸信息
        print("depth:", depth.shape)
        print("rgb:", rgb.shape)
        print("gt:", gt.shape)
        print("gt_depth:", gt_depth.shape)

    # 保存图像
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    for v in range(gt.shape[0]):
        # 移除 batch 维度，确保图像是 [339, 460, 3] 而不是 [1, 339, 460, 3]
        rgb_v = rgb[v].squeeze(0)  # 去掉 batch size 维度
        gt_v = gt[v].squeeze(0)  # 去掉 batch size 维度
        depth_v = depth[v].squeeze(0)  # 去掉 batch size 维度
        gt_depth_v = gt_depth[v].squeeze(0)  # 去掉 batch size 维度

        # 计算 RMSE
        rsme += ((depth_v - gt_depth_v) ** 2).cpu().numpy()

        # 归一化 depth 图像并保存
        depth_ = ((depth_v - depth_v.min()) /
                  (depth_v.max() - depth_v.min() + 1e-8)).repeat(1, 1, 3)
        img_to_save = torch.cat([rgb_v, gt_v, depth_], dim=1)
        image_path = os.path.join('nerf_vs_rebuttal_2', scenes)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        save_dir = os.path.join(image_path, 'view_' + str(v) + '.png')

        # 计算 PSNR
        psnr = compute_psnr(rgb_v, gt_v, mask=None)
        psnr_total += psnr

        # 计算 SSIM
        ssim = compute_ssim(rgb_v, gt_v, mask=None)
        ssim_total += ssim

        # 在图像上添加 PSNR 文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = np.uint8(img_to_save.cpu().numpy() * 255.0)
        image = cv2.putText(
            image, 'PSNR: ' + '%.2f' % psnr, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # 保存图像
        cv2.imwrite(save_dir, image)

    return psnr_total / gt.shape[0], ssim_total / gt.shape[0], rsme / gt.shape[0]


def save_rendered_img(img_meta, rendered_results):
    #print("img_meta",img_meta)
    #print("rendered_results",rendered_results)
    filename = img_meta[0]['img_path']
    scenes = filename.split('/')[-2]

    for ret in rendered_results:
        depth = ret['outputs_coarse']['depth']
        rgb = ret['outputs_coarse']['rgb']
        gt = ret['gt_rgb']
        gt_depth = ret['gt_depth']
        print("depth:",depth.shape)
        print("rgb:",rgb.shape)
        print("gt:",gt.shape)
        print("gt_depth:",gt_depth.shape)                        
    # save images
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    for v in range(gt.shape[0]):
        rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
        depth_ = ((depth[v] - depth[v].min()) /
                  (depth[v].max() - depth[v].min() + 1e-8)).repeat(1, 1, 3)
        img_to_save = torch.cat([rgb[v], gt[v], depth_], dim=1)
        image_path = os.path.join('nerf_vs_rebuttal_v2', scenes)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        save_dir = os.path.join(image_path, 'view_' + str(v) + '.png')

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = np.uint8(img_to_save.cpu().numpy() * 255.0)
        psnr = compute_psnr(rgb[v], gt[v], mask=None)
        psnr_total += psnr
        #ssim = compute_ssim(rgb[v], gt[v], mask=None)
        #ssim_total += ssim
        # image = cv2.putText(
        #     image, 'PSNR: ' + '%.2f' % compute_psnr(rgb[v], gt[v], mask=None),
        #     org, font, fontScale, color, thickness, cv2.LINE_AA)
        print("PSNR:",psnr)
        cv2.imwrite(save_dir, image)

    return psnr_total / gt.shape[0], ssim_total / gt.shape[0], rsme / gt.shape[
        0]
