import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math, random

# ----------------------------------------
# Utilities for continuous-time diffusion
# ----------------------------------------
def sample_log_snr(u, lambda_min=-20.0, lambda_max=20.0):
    """
    Sample log SNR using u~Uniform[0,1]
    lambda = -2 * log(tan(a*u + b))
    """
    b = math.atan(math.exp(-lambda_max / 2))
    a = math.atan(math.exp(-lambda_min / 2)) - b
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
    # init sample
    z = torch.randn(shape, device=device)
    # time schedule
    us = torch.linspace(0, 1, steps=num_steps, device=device)
    timesteps = sample_log_snr(us, lambda_min, lambda_max)

    for i in range(num_steps-1):
        lam_t = timesteps[i]
        lam_next = timesteps[i+1]
        alpha_t = torch.sqrt(torch.sigmoid(lam_t))
        sigma_t = torch.sqrt(torch.sigmoid(-lam_t))
        # pred eps_cond and eps_uncond
        eps_cond = model(z, lam_t, cond)
        eps_uncond = model(z, lam_t, None)
        eps_guided = (1 + w) * eps_cond - w * eps_uncond
        # x estimate
        x_est = (z - sigma_t * eps_guided) / alpha_t
        # next params
        alpha_next = torch.sqrt(torch.sigmoid(lam_next))
        sigma_interp = ((1 - torch.exp(lam_t - lam_next)) ** (1 - v)) * \
                       ((torch.sigmoid(-lam_t) - torch.sigmoid(-lam_next)) ** v)
        noise = torch.randn_like(z)
        z = alpha_next * x_est + sigma_interp * noise
    return x_est

# ----------------------------------------
# Joint training with classifier-free guidance
# ----------------------------------------
def train_classifier_free(model, dataloader, optimizer,
                           puncond=0.1,
                           lambda_min=-20.0, lambda_max=20.0,
                           device='cuda'):
    model.train()
    for x, labels in dataloader:
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
        alpha = torch.sqrt(torch.sigmoid(lam)).view(-1,1,1,1)
        sigma = torch.sqrt(torch.sigmoid(-lam)).view(-1,1,1,1)
        eps = torch.randn_like(x)
        z = alpha * x + sigma * eps
        # predict
        eps_pred = model(z, lam, cond)
        loss = F.mse_loss(eps_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ----------------------------------------
# Example conditional UNet for CIFAR-10
# ----------------------------------------
class CIFARCondUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, num_classes=10, emb_dim=16):
        super().__init__()
        # embedding for labels
        self.label_emb = nn.Embedding(num_classes, emb_dim)
        # downsample blocks
        self.conv1 = nn.Conv2d(in_channels + 1 + emb_dim, base_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        # bottleneck
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1)
        # upsample blocks
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1)
        self.conv5 = nn.Conv2d(base_ch*2, base_ch, 3, padding=1)
        self.out = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, z, lam, cond):
        B, C, H, W = z.shape
        # time embed as channel
        t_emb = lam.view(B,1,1,1).expand(-1,1,H,W)
        # label embed
        if cond is not None:
            le = self.label_emb(cond).view(B,-1,1,1).expand(-1,-1,H,W)
        else:
            le = torch.zeros(B, self.label_emb.embedding_dim, H, W, device=z.device)
        # concat inputs
        x = torch.cat([z, t_emb, le], dim=1)
        # encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        xd = self.down(x2)
        # bottleneck
        xb = F.relu(self.conv3(xd))
        # decoder
        xu = self.up(xb)
        x4 = F.relu(self.conv4(xu))
        x5 = F.relu(self.conv5(x4 + x2))  # skip connection
        return self.out(x5)

# ----------------------------------------
# Main: CIFAR-10 training & sampling
# ----------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # CIFAR-10 dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # model and optimizer
    model = CIFARCondUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # training loop
    epochs = 20
    for ep in range(epochs):
        train_classifier_free(model, dl, opt, puncond=0.1, device=device)
        print(f"Epoch {ep+1}/{epochs} done")

    # sampling example for class 3 (e.g., cat)
    B = 16
    cond = torch.full((B,), 3, dtype=torch.long, device=device)
    samples = sample_classifier_free(model, (B,3,32,32), cond, w=1.5, device=device)
    # map [-1,1] to [0,1]
    imgs = (samples.clamp(-1,1) + 1) / 2
    # save grid
    import torchvision.utils as vutils
    vutils.save_image(imgs, 'samples.png', nrow=4)

if __name__ == '__main__':
    main()