import torch
import torch.nn as nn
from collections import OrderedDict

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)#Glorot
class Network(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        act=torch.nn.Tanh,
    ):
        super(Network, self).__init__()

        #input layers
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        # hidden layers
        for i in range(depth):
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))

        # output layers
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        self.layers = torch.nn.Sequential(OrderedDict(layers))
        self.apply(weights_init_)

    def forward(self, x, min_val=10, max_val_xy=110, max_val_z=110):
        output = self.layers(x)
        X_pred = output[:,0]
        Y_pred = output[:,1]
        Z_pred = output[:,2]
        # X_pred = torch.sigmoid(output[:, 0])
        # Y_pred = torch.sigmoid(output[:, 1])
        # Z_pred = torch.sigmoid(output[:, 2])
        # X_pred = X_pred * (max_val_xy - min_val) + min_val
        # Y_pred = Y_pred * (max_val_xy - min_val) + min_val
        # Z_pred = Z_pred * (max_val_z - min_val) + min_val
        X_pred = X_pred.unsqueeze(1)
        Y_pred = Y_pred.unsqueeze(1)
        Z_pred = Z_pred.unsqueeze(1)
        return X_pred, Y_pred, Z_pred

