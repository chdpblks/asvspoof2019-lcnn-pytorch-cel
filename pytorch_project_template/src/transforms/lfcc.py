import torch
import torch.nn as nn
import torchaudio.transforms as transforms


class LFCC(nn.Module):
    """
    LFCC frontend.
    """

    def __init__(self, **kwargs):
        super().__init__()

        valid_params = {}
        if 'sample_rate' in kwargs:
            valid_params['sample_rate'] = kwargs['sample_rate']
        if 'n_lfcc' in kwargs:
            valid_params['n_lfcc'] = kwargs['n_lfcc']

        self.lfcc_transform = transforms.LFCC(**valid_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Audio signal with (batch, samples) or (samples,) form.
        Returns:
            Tensor: LFCC coefficients with (1, n_lfcc, time) form for single sample.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        lfcc = self.lfcc_transform(x)
        
        # lfcc shape: (batch, n_lfcc, time)
        lfcc = lfcc.unsqueeze(1)
        
        return lfcc
