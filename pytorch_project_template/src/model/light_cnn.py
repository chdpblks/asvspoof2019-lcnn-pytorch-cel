import torch
import torch.nn as nn
import torch.nn.functional as functional

class MFM(nn.Module):
    """Max-Feature-Map: split channels on two parts and take element-wise max."""

    def __init__(self, in_channels, out_channels, type='conv'):
        """
        Args:
            in_channels: number of input channels (for conv) or input features (for linear)
            out_channels: number of output channels after MFM (half of the internal filter count)
            type: 'conv' or 'linear'
        """
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if type == 'conv':
            self.filter = nn.Conv2d(in_channels, 
                                    2 * out_channels, 
                                    kernel_size=1, 
                                    stride=1, 
                                    padding=0)
        elif type == 'linear':
            self.filter = nn.Linear(in_channels, 2 * out_channels)
        else:
            raise ValueError('Unknown type: {}'.format(type))

    def forward(self, x):
        x = self.filter(x)
        a, b = torch.split(x, self.out_channels, dim=1)
        return torch.max(a, b)


class LightCNN_ASVspoof2019(nn.Module):
    """
    LCNN architecture described in Table 1 of:
    "STC Antispoofing Systems for the ASVspoof2019 Challenge"
    https://arxiv.org/abs/1904.05576
    """

    def __init__(self, num_classes=2, dropout_prob=0.75):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.mfm2 = MFM(64, 32)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm5 = MFM(64, 32)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm8 = MFM(96, 48)
        self.pool9 = nn.MaxPool2d(2, 2)
        self.bn10 = nn.BatchNorm2d(48)

        self.conv11 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.mfm12 = MFM(96, 48)
        self.bn13 = nn.BatchNorm2d(48)

        self.conv14 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm15 = MFM(128, 64)
        self.pool16 = nn.MaxPool2d(2, 2)

        self.conv17 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.mfm18 = MFM(128, 64)
        self.bn19 = nn.BatchNorm2d(64)

        self.conv20 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm21 = MFM(64, 32)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv23 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.mfm24 = MFM(64, 32)
        self.bn25 = nn.BatchNorm2d(32)

        self.conv26 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm27 = MFM(64, 32)
        self.pool28 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((53, 37))
        
        self.dropout_big = nn.Dropout(p=0.5)

        self.fc29 = nn.Linear(53 * 37 * 32, 160)
        self.mfm30 = MFM(160, 80, type='linear')
        self.bn31 = nn.BatchNorm1d(80)
        
        self.fc32 = nn.Linear(80, num_classes)

        self.dropout = nn.Dropout(p=dropout_prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_object, **batch):
        """
        Args:
            data_object (Tensor): input spectrogram tensor.
        Returns:
            output (dict): dict containing logits.
        """
        x = data_object
        x = self.conv1(x)
        x = self.mfm2(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.mfm5(x)
        x = self.bn6(x)

        x = self.conv7(x)
        x = self.mfm8(x)
        x = self.pool9(x)
        x = self.bn10(x)

        x = self.conv11(x)
        x = self.mfm12(x)
        x = self.bn13(x)

        x = self.conv14(x)
        x = self.mfm15(x)
        x = self.pool16(x)

        x = self.conv17(x)
        x = self.mfm18(x)
        x = self.bn19(x)

        x = self.conv20(x)
        x = self.mfm21(x)
        x = self.bn22(x)

        x = self.conv23(x)
        x = self.mfm24(x)
        x = self.bn25(x)

        x = self.conv26(x)
        x = self.mfm27(x)
        x = self.pool28(x)

        x = self.adaptive_pool(x)

        x = self.dropout_big(x)

        x = x.view(x.size(0), -1)

        x = self.fc29(x)
        x = self.mfm30(x)
        x = self.dropout(x)
        x = self.bn31(x)
        
        x = self.fc32(x)
        return {'logits': x}

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = super().__str__()
        info += f'\nTotal parameters: {total:,}'
        info += f'\nTrainable parameters: {trainable:,}'
        return info
