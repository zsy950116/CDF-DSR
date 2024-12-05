import os
import cv2 as cv
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage.metrics import structural_similarity as cal_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sobel_kernel_x = torch.tensor([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
sobel_kernel_y = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()


def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""

    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image


def mod_crop(img, modulo):
    if len(img.shape) == 2:
        h, w = img.shape
        return img[: h - (h % modulo), :w - (w % modulo)]
    else:
        h, w, _ = img.shape
        return img[: h - (h % modulo), :w - (w % modulo), :]


def sobel_layer(img, k=0.01):
    sobel_x = F.conv2d(img, sobel_kernel_x, stride=1, padding=1)
    sobel_y = F.conv2d(img, sobel_kernel_x, stride=1, padding=1)
    sobel = torch.sqrt(0.5 * sobel_x ** 2 + 0.5 * sobel_y ** 2)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-8)
    sobel = k * torch.exp(np.log(1 / k + 1) * sobel) - k

    return sobel


def uint8img(img):
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img


def make_coord(shape, flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):

        v0, v1 = -1, 1
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def sava_depth(depth_output, data, output_path):
    depth_up_ours = depth_output[0, :, :, 0].cpu().numpy()
    depth, depthLR = data["depth"], data["depthLR"]

    depth = depth[0, :, :, 0].cpu().numpy()
    depthLR = depthLR[0, :, :, 0].cpu().numpy()
    depth_up_bicubic = np.array(
        Image.fromarray(depthLR).resize((depth.shape[0], depth.shape[1]), Image.BICUBIC))

    mse_bicubic = np.mean((depth - depth_up_bicubic) ** 2)
    psnr_bicubic = -10 * np.log10(mse_bicubic)
    mse_ours = np.mean((depth - depth_up_ours) ** 2)
    psnr_ours = -10 * np.log10(mse_ours)

    cv.imwrite(os.path.join(output_path, 'depth.png'), uint8img(depth))
    cv.imwrite(os.path.join(output_path, 'depth_lr.png'), uint8img(depthLR))
    cv.imwrite(os.path.join(output_path, 'depth_up_bicubic_psnr={:.2f}.png'.format(psnr_bicubic)),
               uint8img(depth_up_bicubic))
    cv.imwrite(os.path.join(output_path, 'depth_up_ours_psnr={:.2f}.png'.format(psnr_ours)),
               uint8img(depth_up_ours))


sobel_kernel_x = torch.tensor([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
sobel_kernel_y = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3).cuda()
laplace = torch.tensor([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
avgpool = torch.tensor([[1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9],
                        [1 / 9, 1 / 9, 1 / 9]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)


def sobel_conv(img, k=0.01):
    sobel_x = F.conv2d(img, sobel_kernel_x, stride=1, padding=1)
    sobel_y = F.conv2d(img, sobel_kernel_x, stride=1, padding=1)
    sobel = torch.sqrt(0.5 * sobel_x ** 2 + 0.5 * sobel_y ** 2)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-8)
    sobel = k * torch.exp(np.log(1 / k + 1) * sobel) - k
    return sobel


def get_visuals(lr_gt, hr_pred, hr_gt, out_dir):
    lr_gt_visual = ((lr_gt - lr_gt.min()) / (lr_gt.max() - lr_gt.min()) * 255).astype(np.uint8)
    hr_gt_visual = ((hr_gt - hr_gt.min()) / (hr_gt.max() - hr_gt.min()) * 255).astype(np.uint8)
    hr_pred_visual = ((hr_pred - hr_pred.min()) / (hr_pred.max() - hr_pred.min()) * 255).astype(np.uint8)

    return lr_gt_visual, hr_gt_visual, hr_pred_visual


def make_top_percent_mask(tensor, percent=0.2):
    num_elements = tensor.numel()  # 张量中的元素数量
    k = int(num_elements * percent)  # 要保留的元素数量

    _, top_indices = tensor.flatten().topk(k)  # 获取前k个最大值的索引
    mask = torch.zeros_like(tensor, dtype=torch.bool)  # 创建与张量形状相同的全零掩码
    mask.view(-1)[top_indices] = True  # 将前k个最大值的索引置为True

    valid_points = torch.where(mask, tensor, torch.zeros_like(tensor))
    valid_mask = valid_points > 0

    return valid_mask  # 根据掩码选择保留的值


def cal_metrics(hr_pred, hr_gt):

    hr_pred = np.clip(hr_pred, hr_gt.min(), hr_gt.max())
    MAE = np.mean(np.abs(hr_pred - hr_gt))
    RMSE = np.sqrt(np.mean((hr_pred - hr_gt) ** 2))
    SSIM = cal_ssim(hr_pred, hr_gt, data_range=(hr_gt.max() - hr_gt.min()))

    return MAE, RMSE, SSIM


def cal_metrics_mask(hr_pred, hr_gt, mask_path):

    mask = cv.imread(mask_path, flags=cv.IMREAD_UNCHANGED).astype(np.uint8)
    mask = mask / 255
    hr_pred = hr_pred * mask
    hr_gt = hr_gt * mask

    hr_pred = np.clip(hr_pred, hr_gt.min(), hr_gt.max())
    MAE = np.mean(np.abs(hr_pred - hr_gt))
    RMSE = np.sqrt(np.mean((hr_pred - hr_gt) ** 2))
    SSIM = cal_ssim(hr_pred, hr_gt, data_range=(hr_gt.max() - hr_gt.min()))

    return MAE, RMSE, SSIM
