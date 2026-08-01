import torch.nn as nn

class Compose(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x