import numpy as np
import torch
from torch import nn


class TrimRepeat(nn.Module):
    """
    Makes trim of audio and repeat if too short.
    """

    def __init__(self, target_length=78400, axis=0):
        super().__init__()
        self.target_length = target_length
        self.axis = axis

    def forward(self, x):

        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # 1D audio case
        if x.ndim == 1:
            x = x.squeeze()
            current_length = len(x)
            
            if current_length > self.target_length:
                x = x[:self.target_length]
            elif current_length < self.target_length:
                n_repeats = int(np.ceil(self.target_length / current_length))
                x = np.tile(x, n_repeats)[:self.target_length]

        elif x.ndim >= 2:
            current_length = x.shape[self.axis]
            
            if current_length > self.target_length:
                slices = [slice(None)] * x.ndim
                slices[self.axis] = slice(0, self.target_length)
                x = x[tuple(slices)]
            elif current_length < self.target_length:
                num_of_repeats = int(np.ceil(self.target_length / current_length))
                x = np.repeat(x, num_of_repeats, axis=self.axis)
                slices = [slice(None)] * x.ndim
                slices[self.axis] = slice(0, self.target_length)
                x = x[tuple(slices)]

        x = x.astype(np.float32)
        return torch.from_numpy(x)
    
    def __call__(self, x):
        return self.forward(x)
