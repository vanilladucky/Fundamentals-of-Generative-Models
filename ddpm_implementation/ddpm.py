import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

# -----------------------------------------------------
# Helper to extract timestep-dependent scalars
# -----------------------------------------------------
def extract(a, t, x_shape):
    # a: [T] tensor, t: [B] longs in [0,T-1], x_shape: e.g. [B,C,H,W]
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))

# -----------------------------------------------------
# Beta schedule and scheduler buffers
# -----------------------------------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionScheduler(nn.Module):
    def __init__(self, timesteps=1000, device='cuda', schedule='linear', variance='complex'):
        super().__init__()
        self.timesteps = timesteps
        if schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = sigmoid_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        # register buffers
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).to(device))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas).to(device))
        if variance=='simple':
            self.register_buffer('posterior_variance', betas) # Section 3.2 - claims to have similar results
        else:
            self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)) # Section 3.2

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.long().to(self.betas.device).clamp(0, self.timesteps - 1)
        # extract scalars and reshape for broadcasting
        sqrt_acp = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om  = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_acp * x_start + sqrt_om * noise

# -----------------------------------------------------
# Sinusoidal time embeddings
# -----------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

# -----------------------------------------------------
# Residual block with time conditioning
# -----------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Linear(time_emb_dim, out_ch)
        # dynamic group norm
        g1 = 8 if in_ch % 8 == 0 else in_ch
        g2 = 8 if out_ch % 8 == 0 else out_ch
        self.block = nn.Sequential(
            nn.GroupNorm(g1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block(x)
        emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        return h + self.res_conv(x) + emb

# -----------------------------------------------------
# U-Net with concatenation skips
# -----------------------------------------------------
class UNet(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # down
        self.res1 = ResidualBlock(img_channels, base_ch, time_emb_dim)
        self.res2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.res3 = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.res4 = ResidualBlock(base_ch*4, base_ch*8, time_emb_dim)
        # up
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.res5 = ResidualBlock(base_ch*8 + base_ch*4, base_ch*4, time_emb_dim)
        self.res6 = ResidualBlock(base_ch*4 + base_ch*2, base_ch*2, time_emb_dim)
        self.res7 = ResidualBlock(base_ch*2 + base_ch, base_ch, time_emb_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8 if base_ch % 8 == 0 else base_ch, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, 1)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.res1(x, t)
        x2 = self.res2(self.pool(x1), t)
        x3 = self.res3(self.pool(x2), t)
        x4 = self.res4(self.pool(x3), t)
        x = self.up(x4)
        x = self.res5(torch.cat([x, x3], dim=1), t)
        x = self.up(x)
        x = self.res6(torch.cat([x, x2], dim=1), t)
        x = self.up(x)
        x = self.res7(torch.cat([x, x1], dim=1), t)
        return self.out(x)

# -----------------------------------------------------
# Loss, training, sampling
# -----------------------------------------------------
def p_losses(model, scheduler, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = scheduler.q_sample(x_start, t, noise)
    pred = model(x_noisy, t)
    return F.mse_loss(pred, noise)


def train(e):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4)

    model = UNet().to(device)
    scheduler = DiffusionScheduler(timesteps=1000, device=device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(e):
        for step, (x, _) in enumerate(loader):
            x = x.to(device)
            t = torch.randint(0, scheduler.timesteps, (x.size(0),), device=device)
            loss = p_losses(model, scheduler, x, t)
            optim.zero_grad(); loss.backward(); optim.step()
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
        torch.save(model.state_dict(), f"ddpm_epoch_{epoch}.pth")

@torch.no_grad()
def sample(model, scheduler, shape=(16,3,32,32)):
    model.eval()
    x = torch.randn(shape, device=scheduler.betas.device)
    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        beta = scheduler.betas[i]
        sqrt_om = scheduler.sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip = scheduler.sqrt_recip_alphas[i]
        mean = sqrt_recip * (x - beta / sqrt_om * eps)
        if i > 0:
            noise = torch.randn_like(x)
            var = scheduler.posterior_variance[i]
            x = mean + var.sqrt() * noise
        else:
            x = mean
    return x

def sample_and_save(output_path='sample.png',
                    model_ckpt='ddpm_epoch_49.pth',  
                    device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    scheduler = DiffusionScheduler(timesteps=1000, device=device).to(device)

    with torch.no_grad():
        img = sample(model, scheduler, shape=(1,3,32,32))

    img = (img + 1) * 0.5
    img = img.clamp(0,1)

    save_image(img, output_path)
    print(f"Saved sample to {output_path}")

if __name__ == '__main__':
    epoch = 3
    train(epoch)
    sample_and_save(model_ckpt=f'ddpm_epoch_{epoch-1}.pth')