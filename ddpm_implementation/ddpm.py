import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from torch.amp import autocast

# -----------------------------------------------------
# Beta schedule and precomputed constants
# -----------------------------------------------------
def extract(a, t, x_shape): # Helper function to extract out batch sizes and format shape 
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

class DiffusionScheduler:
    def __init__(self, timesteps=1000, device='cuda', schedule='linear', variance='complex'):
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
        self.register_buffer = {}
        self.register_buffer['betas'] = betas.to(device)
        self.register_buffer['alphas'] = alphas.to(device)
        self.register_buffer['alphas_cumprod'] = alphas_cumprod.to(device)
        self.register_buffer['alphas_cumprod_prev'] = alphas_cumprod_prev.to(device)
        self.register_buffer['sqrt_alphas_cumprod'] = torch.sqrt(alphas_cumprod).to(device)
        self.register_buffer['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1 - alphas_cumprod).to(device)
        self.register_buffer['sqrt_recip_alphas'] = torch.sqrt(1.0 / alphas).to(device)
        if variance=='simple':
            self.register_buffer['posterior_variance'] = betas # Section 3.2 - claims to have similar results
        else:
            self.register_buffer['posterior_variance'] = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # Section 3.2

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps - adding noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.register_buffer['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
        sqrt_om = self.register_buffer['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
        #sqrt_acp = extract(sqrt_acp, t, x_start.shape)
        #sqrt_om = extract(sqrt_om, t, x_start.shape)
        return sqrt_acp * x_start + sqrt_om * noise

# -----------------------------------------------------
# Sinusoidal time embeddings
# -----------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# -----------------------------------------------------
# A simple U-Net backbone
# -----------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # MLP to condition on time embedding
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch)
        )
        # Choose num_groups dynamically: up to 8 if divisible, else fall back to in_ch/out_ch
        g1 = 8 if in_ch % 8 == 0 else in_ch
        g2 = 8 if out_ch % 8 == 0 else out_ch
        self.block = nn.Sequential(
            nn.GroupNorm(g1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.block(x)
        time_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        return h + self.res_conv(x) + time_emb

class UNet(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Downsampling path
        self.res1 = ResidualBlock(img_channels, base_ch, time_emb_dim)
        self.res2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.res3 = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.res4 = ResidualBlock(base_ch*4, base_ch*8, time_emb_dim)
        # Upsampling path
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.res5 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.res6 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.res7 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8 if base_ch % 8 == 0 else base_ch, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, kernel_size=1)
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
# Loss and training step
# -----------------------------------------------------
def p_losses(model, scheduler, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = scheduler.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = model(x_noisy, t)
    return F.mse_loss(predicted_noise, noise) # Equation 14 in https://arxiv.org/pdf/2006.11239

# -----------------------------------------------------
# Training loop - page 4 of https://arxiv.org/pdf/2006.11239
# -----------------------------------------------------
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # Model and scheduler
    model = UNet().to(device)
    scheduler = DiffusionScheduler(timesteps=1000, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training
    epochs = 50
    for epoch in range(epochs):
        for step, (x, _) in enumerate(loader):
            x = x.to(device)
            t = torch.randint(0, scheduler.timesteps, (x.shape[0],), device=device).long()
            loss = p_losses(model, scheduler, x, t)
            optim.zero_grad(); loss.backward(); optim.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # save model checkpoint
        torch.save(model.state_dict(), f"ddpm_epoch_{epoch}.pth")

# -----------------------------------------------------
# Sampling loop - page 4 of https://arxiv.org/pdf/2006.11239
# -----------------------------------------------------
@torch.no_grad()
def sample(model, scheduler, shape=(16, 3, 32, 32)):
    device = scheduler.register_buffer['betas'].device
    model.eval()
    x = torch.randn(shape, device=device)
    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        predicted_noise = model(x, t)
        beta = scheduler.register_buffer['betas'][i]
        sqrt_one_minus_acp = scheduler.register_buffer['sqrt_one_minus_alphas_cumprod'][i]
        sqrt_recip_alpha = scheduler.register_buffer['sqrt_recip_alphas'][i]
        x = sqrt_recip_alpha * (x - beta / sqrt_one_minus_acp * predicted_noise)
        if i > 0:
            noise = torch.randn_like(x)
            var = scheduler.register_buffer['posterior_variance'][i-1]
            x += torch.sqrt(var) * noise
    return x

if __name__ == '__main__':
    train()
