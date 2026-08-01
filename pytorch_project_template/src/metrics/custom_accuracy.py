import torch
import torch.nn as nn

class CustomAccuracy(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.dim() == 2:
            preds = torch.argmax(preds, dim=1)
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total

    def reset(self) -> None:
        self.register_buffer('correct', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total', torch.tensor(0, dtype=torch.long))

    def forward(self, features_or_logits, target, **batch):
        if 'logits' in batch:
            # for asoftmax tests
            logits = batch['logits']
        else:
            logits = features_or_logits 
        self.update(logits, target)
        return self.compute()