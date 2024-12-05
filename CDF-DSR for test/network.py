import torch
import torch.nn as nn
from pos_encoder import Positional_Encoder

class LIIFMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super(LIIFMLP, self).__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        if out_dim != 1:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x, coord_win_max=None):
        x = self.layers(x)
        if self.out_dim != 1:
            x = (x * 2 - 1) * coord_win_max
        return x


class Model(nn.Module):
    def __init__(self,
                 mlp_type='LIIFMLP',
                 rgb_feats_dim=0,
                 dep_feats_dim=0,
                 pos_params=None,
                 mlp_output=1,
                 mlp_hidden_list=[128, 128, 128, 128]):
        super(Model, self).__init__()

        self.pos_encoder = Positional_Encoder(pos_params)

        if mlp_type == 'LIIFMLP':
            mlp_input_dim = self.pos_encoder.output_dim + rgb_feats_dim + dep_feats_dim
            self.mlp = LIIFMLP(mlp_input_dim, out_dim=mlp_output, hidden_list=mlp_hidden_list)
        else:
            raise NotImplementedError

    def forward(self, query_coords, coord_win_max=None):
        B, H, W, _ = query_coords.shape
        inputs = self.pos_encoder(query_coords.view(B, H * W, -1))
        y = self.mlp(inputs, coord_win_max=coord_win_max)  # [N, 1]
        y = torch.reshape(y, (B, H, W, -1))
        return y

