import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math, random
import torchvision.utils as vutils


# ----------------------------------------
# Utilities for continuous-time diffusion
# ----------------------------------------
def sample_log_snr(u, lambda_min=-20.0, lambda_max=20.0):
    """
    Convert uniform samples u in [0,1] to log-SNR values via
    λ = -2 * log(tan(a*u + b)), where
    b = arctan(e^{-λ_max/2}), a = arctan(e^{-λ_min/2}) - b
    """
    b = math.atan(math.exp(-lambda_max / 2.0))
    a = math.atan(math.exp(-lambda_min / 2.0)) - b
    return -2.0 * torch.log(torch.tan(a * u + b))

@torch.no_grad()
def sample_classifier_free(
    model, shape, cond, w=1.0,
    num_steps=200,
    lambda_min=-20.0, lambda_max=20.0,
    v=0.3, device='cuda'
):
    model.eval()
    B, C, H, W = shape
    z = torch.randn(shape, device=device)
    us = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    timesteps = sample_log_snr(us, lambda_min, lambda_max)

    for i in range(num_steps - 1):
        # batch log-SNRs
        lam_t = timesteps[i].unsqueeze(0).expand(B)
        lam_next = timesteps[i + 1]

        # coefficients
        alpha_t = torch.sqrt(torch.sigmoid(lam_t)).view(B,1,1,1)
        sigma_t = torch.sqrt(torch.sigmoid(-lam_t)).view(B,1,1,1)

        # predict noise
        eps_cond = model(z, lam_t, cond)
        eps_uncond = model(z, lam_t, None)
        eps_guided = (1.0 + w) * eps_cond - w * eps_uncond

        # estimate x0
        x_est = (z - sigma_t * eps_guided) / alpha_t

        # compute next noise scale
        alpha_next = torch.sqrt(torch.sigmoid(lam_next))
        sigma_interp = ((1.0 - torch.exp(lam_t - lam_next)) ** (1.0 - v)) * \
                       ((torch.sigmoid(-lam_t) - torch.sigmoid(-lam_next)) ** v)
        sigma_interp = sigma_interp.view(B,1,1,1)

        # sample next z
        noise = torch.randn_like(z)
        z = alpha_next * x_est + sigma_interp * noise

    return x_est

# ----------------------------------------
# Joint training with classifier-free guidance
# ----------------------------------------
def train_classifier_free(
    model, dataloader, optimizer,
    puncond=0.5,
    lambda_min=-20.0, lambda_max=20.0,
    device='cuda'
):
    model.train()
    train_loss = 0
    count = 0
    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        cond = labels if random.random() >= puncond else None

        # sample noise level
        u = torch.rand(x.size(0), device=device)
        lam = sample_log_snr(u, lambda_min, lambda_max)

        # corrupt
        alpha = torch.sqrt(torch.sigmoid(lam)).view(-1,1,1,1)
        sigma = torch.sqrt(torch.sigmoid(-lam)).view(-1,1,1,1)
        eps = torch.randn_like(x)
        z = alpha * x + sigma * eps

        # predict and update
        eps_pred = model(z, lam, cond)
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count+=1
        train_loss+=loss.item()
    print(f"Training loss: {train_loss/count:.4f}")

# ----------------------------------------
# Improved conditional DeepUNet for CIFAR-10
# ----------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)

class DeepUNet(nn.Module):
    def __init__(
        self, in_channels=3, base_ch=64,
        num_classes=10, time_emb_dim=128
    ):
        super().__init__()
        # time & label embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # encoder
        self.enc1 = ConvBlock(in_channels + 2*time_emb_dim, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        # bottleneck
        self.bot = ConvBlock(base_ch*4, base_ch*8)
        # decoder
        self.dec3 = ConvBlock(base_ch*8 + base_ch*4, base_ch*4)
        self.dec2 = ConvBlock(base_ch*4 + base_ch*2, base_ch*2)
        self.dec1 = ConvBlock(base_ch*2 + base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, lam, cond):
        B, C, H, W = x.shape
        # time conditioning
        t = lam.view(B,1)
        t_emb = self.time_mlp(t).view(B,-1,1,1).expand(-1,-1,H,W)
        # label conditioning
        if cond is not None:
            l_emb = self.label_emb(cond).view(B,-1,1,1).expand(-1,-1,H,W)
        else:
            l_emb = torch.zeros_like(t_emb)
        h = torch.cat([x, t_emb, l_emb], dim=1)

        # encode
        e1 = self.enc1(h)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # bottleneck
        b = self.bot(self.pool(e3))
        # decode
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out_conv(d1)

# ----------------------------------------
# Main: CIFAR-10 training & sampling
# ----------------------------------------
def main():
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    model = DeepUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    epochs = 200
    for ep in range(epochs):
        train_classifier_free(model, dl, opt, puncond=0.1, device=device)
        print(f"Epoch {ep+1}/{epochs} complete")
        if ep > 0 and ep % 10 == 0:
            # sample and save
            B = 16
            cond = torch.full((B,), 3, dtype=torch.long, device=device)
            samples = sample_classifier_free(model, (B,3,32,32), cond, w=0.5, device=device)
            imgs = (samples.clamp(-1,1) + 1) / 2
            vutils.save_image(imgs, f'samples_{ep}.png', nrow=4)

if __name__ == '__main__':
    main()