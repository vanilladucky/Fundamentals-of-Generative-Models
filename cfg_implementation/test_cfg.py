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
    Sample log SNR using u~Uniform[0,1]
    lambda = -2 * log(tan(a*u + b))
    """
    # compute scalar coefficients
    b = math.atan(math.exp(-lambda_max / 2.0))
    a = math.atan(math.exp(-lambda_min / 2.0)) - b
    # return tensor of log-SNR values
    return -2.0 * torch.log(torch.tan(a * u + b))

@torch.no_grad()
def sample_classifier_free(
    model, shape, cond, w=1.0,
    num_steps=200,
    lambda_min=-20.0, lambda_max=20.0,
    v=0.3, device='cuda:2'
):
    model.eval()
    B, C, H, W = shape
    # initialize noise
    z = torch.randn(shape, device=device)
    # precompute schedule of log-SNRs
    us = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    timesteps = sample_log_snr(us, lambda_min, lambda_max)

    for i in range(num_steps - 1):
        lam_t = timesteps[i]
        # expand scalar to batch
        lam_t = lam_t.unsqueeze(0).expand(B)
        lam_next = timesteps[i + 1]

        # compute coefficients
        alpha_t = torch.sqrt(torch.sigmoid(lam_t)).view(B, 1, 1, 1)
        sigma_t = torch.sqrt(torch.sigmoid(-lam_t)).view(B, 1, 1, 1)

        # predict noise
        eps_cond = model(z, lam_t, cond)
        eps_uncond = model(z, lam_t, None)
        eps_guided = (1.0 + w) * eps_cond - w * eps_uncond

        # estimate x
        x_est = (z - sigma_t * eps_guided) / alpha_t

        # prepare next noise scale
        alpha_next = torch.sqrt(torch.sigmoid(lam_next))
        sigma_interp = ((1.0 - torch.exp(lam_t - lam_next)) ** (1.0 - v)) * \
                       ((torch.sigmoid(-lam_t) - torch.sigmoid(-lam_next)) ** v)
        sigma_interp = sigma_interp.view(B, 1, 1, 1)

        # add noise
        noise = torch.randn_like(z)
        z = alpha_next * x_est + sigma_interp * noise

    return x_est

# ----------------------------------------
# Joint training with classifier-free guidance
# ----------------------------------------
def train_classifier_free(model, dataloader, optimizer,
                           puncond=0.1,
                           lambda_min=-20.0, lambda_max=20.0,
                           device='cuda:2'):
    model.train()
    total_loss = 0
    count = 0
    for x, labels in dataloader:
        count+=1
        x = x.to(device)
        labels = labels.to(device)
        # drop conditioning
        if random.random() < puncond:
            cond = None
        else:
            cond = labels
        # sample noise time
        u = torch.rand(x.size(0), device=device)
        lam = sample_log_snr(u, lambda_min, lambda_max)
        # noise and corrupt
        alpha = torch.sqrt(torch.sigmoid(lam)).view(-1, 1, 1, 1)
        sigma = torch.sqrt(torch.sigmoid(-lam)).view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        z = alpha * x + sigma * eps
        # predict and optimize
        eps_pred = model(z, lam, cond)
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Training Loss: {total_loss/count:.4f}")

# ----------------------------------------
# Example conditional UNet for CIFAR-10
# ----------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.layers(x)

class DeepUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, num_classes=10, time_emb_dim=128):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # label embedding
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        # encoder
        self.enc1 = ConvBlock(in_channels + time_emb_dim + time_emb_dim, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        # bottleneck
        self.bot = ConvBlock(base_ch * 4, base_ch * 8)
        # decoder
        self.dec3 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, lam, cond):
        B, C, H, W = x.shape
        # time embedding
        t = lam.view(B, 1)
        t_emb = self.time_mlp(t).view(B, -1, 1, 1).expand(-1, -1, H, W)
        # label embedding
        if cond is not None:
            l_emb = self.label_emb(cond).view(B, -1, 1, 1).expand(-1, -1, H, W)
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
        return self.out(d1)

# ----------------------------------------
# Main: CIFAR-10 training & sampling
# ----------------------------------------
def main():
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # CIFAR-10 dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    ds = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # model and optimizer
    model = DeepUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # training loop
    epochs = 200
    for ep in range(epochs):
        train_classifier_free(model, dl, opt, puncond=0.5, device=device)
        print(f"Epoch {ep+1}/{epochs} done")
        if ep % 10 and ep > 0:
            # sampling example for class 3 (e.g., cat)
            B = 16
            cond = torch.full((B,), 3, dtype=torch.long, device=device)
            samples = sample_classifier_free(model, (B, 3, 32, 32), cond, w=0.3, device=device)
            # map [-1,1] to [0,1]
            imgs = (samples.clamp(-1, 1) + 1) / 2
            # save grid
            vutils.save_image(imgs, f'samples_{ep}.png', nrow=4)

    

if __name__ == '__main__':
    main()