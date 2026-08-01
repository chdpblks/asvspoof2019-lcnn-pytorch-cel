import torch
from torch import nn

class ASVSpoofLoss(nn.Module):

    '''CrossEntropyLoss with weights.'''

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
            self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"class_weights: {self.class_weights}")
        else:
            self.loss = nn.CrossEntropyLoss()
            self.register_buffer('class_weights', None)
            print("class_weights")
        
        self.batch_count = 0

    def forward(self, **batch):
        logits = batch['logits']
        labels = batch.get('labels') if 'labels' in batch else batch['label']

        # to see distribution of classes
        if self.batch_count < 5:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            print(f"Batch {self.batch_count} class's distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
            self.batch_count += 1
        
        loss_value = self.loss(logits, labels)
        return {"loss": loss_value, "logits": logits}