import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two 3×3 conv residual block with GroupNorm + Swish."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, ch_mults=(1,2,4,8), n_res=2, z_dim=3):
        """
        in_ch: input channels (3 for RGB)
        base_ch: initial number of filters
        ch_mults: multipliers at each downsampling stage
        n_res: number of ResBlocks per stage
        z_dim: final latent channels (c in z∈ℝ^{h×w×c})
        """
        super().__init__()
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs = [base_ch * m for m in ch_mults]
        self.downs = nn.ModuleList()
        prev_ch = base_ch
        for ch in chs:
            layers = []
            # n_res residual blocks
            for _ in range(n_res):
                layers.append(ResBlock(prev_ch))
            # downsample
            layers.append(nn.Conv2d(prev_ch, ch, 4, stride=2, padding=1))
            prev_ch = ch
            self.downs.append(nn.Sequential(*layers))
        # project to mean and logvar
        self.to_mu = nn.Conv2d(prev_ch, z_dim, 3, padding=1)
        self.to_logvar = nn.Conv2d(prev_ch, z_dim, 3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        for stage in self.downs:
            x = stage(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, ch_mults=(1,2,4,8), n_res=2, z_dim=3):
        super().__init__()
        # project latent to the deepest feature map
        self.from_z = nn.Conv2d(z_dim, base_ch * ch_mults[-1], 3, padding=1)

        # build one up-sampling block *per* down-sampling stage
        chs = [base_ch * m for m in ch_mults[::-1]]  # e.g. [512,256,128,64]
        prev_ch = chs[0]
        self.ups = nn.ModuleList()
        for ch in chs[1:]:
            block = []
            # 1) upsample by 2
            block.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=4, stride=2, padding=1))
            # 2) residual blocks
            for _ in range(n_res):
                block.append(ResBlock(ch))
            self.ups.append(nn.Sequential(*block))
            prev_ch = ch

        # === ADD THIS EXTRA UPSAMPLE ===
        # last upsample to go from 64→64 at 16→32
        self.final_upsample = nn.ConvTranspose2d(prev_ch, prev_ch, 4, stride=2, padding=1)

        # output layers
        self.out_norm = nn.GroupNorm(8, prev_ch)
        self.out_act  = nn.SiLU(inplace=False)
        self.out_conv = nn.Conv2d(prev_ch, out_ch, 3, padding=1)

    def forward(self, z):
        x = self.from_z(z)               # starts at 2×2
        for stage in self.ups:
            x = stage(x)                 # 2→4→8→16
        x = self.final_upsample(x)       # 16→32
        x = self.out_norm(x)
        x = self.out_act(x)
        return torch.sigmoid(self.out_conv(x))  # now back to 32×32


class PerceptualVAE(nn.Module):
    """Full perceptual autoencoder with KL-regularization."""
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())