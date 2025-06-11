import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, num_layers=3):
        """
        in_ch:     # of input channels (e.g. 3 for RGB)
        base_ch:   # filters in first conv
        num_layers: # of downsampling conv blocks before the final stride-1 layer
        """
        super().__init__()
        layers = []
        # 1) initial conv (no normalization on first layer)
        layers.append(nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 2) downsampling blocks
        nf = base_ch
        for n in range(1, num_layers):
            nf_prev = nf
            nf = min(nf_prev * 2, 512)
            layers.append(nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(nf))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 3) one more conv with stride=1
        layers.append(nn.Conv2d(nf, min(nf * 2, 512), kernel_size=4, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(min(nf * 2, 512)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 4) final output conv → 1-channel patch of logits
        layers.append(nn.Conv2d(min(nf * 2, 512), 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, in_ch, H, W]
        returns: [B, 1, H/2^{num_layers}, W/2^{num_layers}] logits
        """
        return self.model(x)