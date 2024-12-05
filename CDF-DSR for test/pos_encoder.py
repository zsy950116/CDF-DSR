import numpy as np
import torch
import torch.nn as nn


class Positional_Encoder(nn.Module):
    def __init__(self, params):
        super(Positional_Encoder, self).__init__()
        self.mask = params['mask']
        self.scale = params['scale']
        self.output_dim = params['embedding_size'] * params['coordinates_size']
        self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
        self.B = self.B.cuda()
        self.B = nn.Parameter(self.B, requires_grad=False)

        self.r = self.B[:, 0] ** 2 + self.B[:, 1] ** 2
        self.r = nn.Parameter(self.r, requires_grad=False)
        self.itk = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.start_f = (params['freq'][0] * self.scale) ** 2
        self.end_f = (params['freq'][1] * self.scale) ** 2
        self.start_itk = params['itk'][0]
        self.end_itk = params['itk'][1]

    def forward(self, x):
        if self.mask:
            mask_f = (self.end_f - self.start_f) / (self.end_itk - self.start_itk) * (
                        self.itk - self.start_itk) + self.start_f
            mask_f = torch.clamp(mask_f, min=self.start_f, max=self.end_f)
            mask = self.r < mask_f
            mask = mask[None, None, :]
        else:
            mask = torch.tensor(1.0)

        x_embedding = (2. * np.pi * x) @ self.B.t()
        sin_x = torch.sin(x_embedding) * mask
        cos_x = torch.cos(x_embedding) * mask
        x_embedding = torch.cat([sin_x, cos_x], dim=-1)

        return x_embedding

