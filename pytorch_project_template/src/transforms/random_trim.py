import numpy as np
import torch
from torch import nn


class RandomTrimRepeat(nn.Module):
    """
    Makes random trim of audio.
    """

    def __init__(self, target_length=78400):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        if x.ndim > 1:
            x = x.squeeze()
        
        current_length = len(x)
        
        if current_length > self.target_length:
            max_new_start = current_length - self.target_length
            start_idx = np.random.randint(0, max_new_start + 1)
            x = x[start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            num_of_repeats = int(np.ceil(self.target_length / current_length))
            x = np.tile(x, num_of_repeats)[:self.target_length]

        x = x.astype(np.float32)
        return torch.from_numpy(x)
    
    def __call__(self, x):
        return self.forward(x)
