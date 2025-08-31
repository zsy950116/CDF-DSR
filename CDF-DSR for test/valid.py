import os
import torch
import warnings
import cv2 as cv
import numpy as np
from utils import cal_metrics
from network import Model
from configs.configs import params
from singlesource import data_process

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid_model_original(model, query_coords, core_values, norm_type):
    depth_min = core_values['min']
    depth_max = core_values['max']
    depth_std = core_values['std']
    depth_mean = core_values['mean']

    model.eval()
    with torch.no_grad():
        hr_out = model(query_coords)  # [1, H, W, 1]
        if norm_type == 'MinMax':
            hr_out = hr_out * (depth_max - depth_min) + depth_min
        elif norm_type == 'ZScore':
            hr_out = hr_out * depth_std + depth_mean
    return hr_out.squeeze().detach().cpu().numpy()



if __name__ == '__main__':
    set_name = 'MiddleBury'
    sr_factor = 4

    model = Model(mlp_type='LIIFMLP',
                  pos_params=params['IREMencoder'],
                  mlp_output=1,
                  mlp_hidden_list=params['LIIFMLPnet']['mlp_hidden_list']).to(device)

    rgb_path = 'test/rgbs/MiddleBury_01.png'
    dep_path = 'test/depths/MiddleBury_01.png'
    vis_path = 'test/vis_depths/MiddleBury_01.png'
    out_path = 'visuals/MiddleBury_01_out.png'
    ckpt_path = 'test/checkpoints/MiddleBury_01.pth'

    data = data_process(rgb_path=rgb_path,
                        dep_path=dep_path,
                        sr_factor=sr_factor,
                        norm_type='MinMax')

    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        hr_gt = data['hr_depth'].squeeze().detach().cpu().numpy()
        pred = valid_model_original(model, data['hr_coords'], data['depth_core_values'], norm_type='MinMax')
        pred = np.clip(pred, hr_gt.min(), hr_gt.max())
        mae, rmse, ssim = cal_metrics(pred, hr_gt)
        print(mae, rmse, ssim)

        vis_depth = cv.imread(vis_path, cv.IMREAD_UNCHANGED).astype(np.float32)

        a, b = np.polyfit(pred.flatten(), vis_depth.flatten(), 1)
        pred_norm = pred * a + b
        pred_norm = np.clip(pred_norm, vis_depth.min(), vis_depth.max()).astype(np.uint8)

        cv.imwrite(out_path, pred_norm)


