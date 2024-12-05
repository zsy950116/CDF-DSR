import torch
from torchvision.transforms import transforms
import cv2 as cv
import numpy as np
from PIL import Image
from utils import make_coord, device, sobel_conv

PIXEL_NORM_MEANS = [0.485, 0.456, 0.406]
PIXEL_NORM_VALS = [0.229, 0.224, 0.225]


def data_process(rgb_path=None,
                 dep_path=None,  # high-resolution data if exists else low-reoslution data
                 sr_factor=5,  # only work when there exists high-resolution depth data.
                 norm_type='MinMax'):
    # load and transform rgb-guided data
    hr_img = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)[:, :, :3]
    h_hr, w_hr = hr_img.shape[0], hr_img.shape[1]

    scale = np.random.uniform(1.0, float(sr_factor * 2), 1)
    lr_img = np.array(Image.fromarray(hr_img).resize((int(w_hr/scale), int(h_hr/scale)), Image.BICUBIC))
    hr_img = hr_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    hr_img = torch.from_numpy(hr_img)
    hr_img = transforms.Normalize(PIXEL_NORM_MEANS, PIXEL_NORM_VALS)(hr_img).unsqueeze(0).to(device)

    lr_img = lr_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    lr_img = torch.from_numpy(lr_img)
    lr_img = transforms.Normalize(PIXEL_NORM_MEANS, PIXEL_NORM_VALS)(lr_img).unsqueeze(0).to(device)


    # load hr_depth and generate lr_up, for validing.
    hr_depth = cv.imread(dep_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    tmp = np.array(Image.fromarray(hr_depth).resize((w_hr // sr_factor, h_hr // sr_factor), Image.NEAREST))
    lr_up = np.array(Image.fromarray(tmp).resize((w_hr, h_hr), Image.BICUBIC))

    # generate the normed lr_depth for inputting the network.
    if norm_type == 'ZScore':
        hr_norm = (hr_depth - hr_depth.mean()) / (hr_depth.std() + 1e-8)
    elif norm_type == 'MinMax':
        hr_norm = (hr_depth - hr_depth.min()) / (hr_depth.max() - hr_depth.min() + 1e-8)

    else:
        raise NotImplementedError

    lr_norm = np.array(
        Image.fromarray(hr_norm).resize((w_hr // sr_factor, h_hr // sr_factor), Image.NEAREST))

    hr_coords = make_coord((h_hr, w_hr), flatten=False).numpy()

    hr_coords_x, hr_coords_y = hr_coords[:, :, 0], hr_coords[:, :, 1]
    lr_coords_x = np.array(
        Image.fromarray(hr_coords_x).resize((w_hr // sr_factor, h_hr // sr_factor), Image.NEAREST))
    lr_coords_y = np.array(
        Image.fromarray(hr_coords_y).resize((w_hr // sr_factor, h_hr // sr_factor), Image.NEAREST))

    lr_coords = np.concatenate([np.expand_dims(lr_coords_x, axis=2), np.expand_dims(lr_coords_y, axis=2)], axis=2)
    h_lr, w_lr, c = lr_coords.shape

    lr_coords = torch.from_numpy(lr_coords.reshape(1, h_lr, w_lr, c)).to(device)  # [1, h_lr, w_lr, 2]
    hr_coords = torch.from_numpy(hr_coords.reshape(1, h_hr, w_hr, c)).to(device)  # [1, h_hr, w_hr, 2]

    # calculate the field
    lr_distance_h = 2 / h_lr
    lr_distance_w = 2 / w_lr
    lr_distance = torch.tensor([lr_distance_h, lr_distance_w]).cuda()

    field = torch.ones([8]).cuda()
    _, cH, cW, _ = hr_coords.shape
    ch = cH // 2
    cw = cW // 2
    f1 = abs(hr_coords[0, ch + 1, cw - 1] - hr_coords[0, ch, cw])
    field[0:2] = f1 / lr_distance
    f2 = abs(hr_coords[0, ch - 1, cw - 1] - hr_coords[0, ch, cw])
    field[2:4] = f2 / lr_distance
    f3 = abs(hr_coords[0, ch + 1, cw + 1] - hr_coords[0, ch, cw])
    field[4:6] = f3 / lr_distance
    f4 = abs(hr_coords[0, ch - 1, cw + 1] - hr_coords[0, ch, cw])
    field[6:] = f4 / lr_distance
    field = field.unsqueeze(0)

    lr_norm = torch.from_numpy(lr_norm)[None, :, :, None].to(device)  # [1, h_lr, w_lr, 1]
    hr_depth = torch.from_numpy(hr_depth)[None, :, :, None].to(device)  # [1, h_hr, w_hr, 1]
    lr_up = torch.from_numpy(lr_up)[None, :, :, None].to(device)  # [1, h_hr, w_hr, 1]
    hr_norm = torch.from_numpy(hr_norm)[None, :, :, None].to(device)  # [1, h_hr, w_hr, 1]

    depthsobel = sobel_conv(lr_norm.permute(0, 3, 1, 2), k=0.1)
    depthsobel[:, :, :1, :] = 0
    depthsobel[:, :, -1:, :] = 0
    depthsobel[:, :, :, :1] = 0
    depthsobel[:, :, :, -1:] = 0

    depth_core_values = {'min': hr_depth.min(),
                         'max': hr_depth.max(),
                         'mean': hr_depth.mean(),
                         'std': hr_depth.std()}

    return {'rgb_path': rgb_path,
            'hr_guide': hr_img,
            'lr_guide': lr_img,
            'hr_depth': hr_depth,
            'hr_norm': hr_norm,
            'lr_up': lr_up,
            'lr_depth': lr_norm,
            'hr_coords': hr_coords,
            'lr_coords': lr_coords,
            'sobel': depthsobel,
            'field': field,
            'depth_core_values': depth_core_values}

