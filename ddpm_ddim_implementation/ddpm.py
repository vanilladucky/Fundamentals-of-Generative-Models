import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import math
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import argparse

def extract(a, t, x_shape):
    # a: [T] tensor, t: [B] longs in [0,T-1], x_shape: e.g. [B,C,H,W]
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
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
    t = torch.linspace(0, timesteps, steps) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionScheduler(nn.Module):
    def __init__(self, timesteps=10000, device='cuda:2', schedule='linear', variance='complex'):
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

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)

class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)

class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(3 + 16, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input, log_snrs, cond=None):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        # class_embed = expand_to_planes(self.class_embed(cond), input.shape) # CFG implemented elsewhere 
        return self.net(torch.cat([input, timestep_embed], dim=1))

def p_losses(model, scheduler, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = scheduler.q_sample(x_start, t, noise)
    pred = model(x_noisy, t)
    return F.mse_loss(pred, noise)

def evaluate(model, scheduler, loader, device, last=False, mode = 'ddpm', ddim_steps = 100):
    model.eval()
    total_loss = 0.0
    if last:
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(0, scheduler.timesteps, (x.size(0),), device=device)
            loss = p_losses(model, scheduler, x, t)
            total_loss += loss.item() * x.size(0)
            if last:
                if mode == 'ddpm':
                    fake = sample(model, scheduler, shape=x.shape)
                else:
                    fake = sample_ddim(model, scheduler, shape=x.shape, ddim_steps=ddim_steps)
                real_images = ((x + 1) * 0.5 * 255).clamp(0, 255).to(torch.uint8)
                fake_images = ((fake + 1) * 0.5 * 255).clamp(0, 255).to(torch.uint8)
                fid_metric.update(real_images, real=True)
                fid_metric.update(fake_images, real=False)
    avg_loss = total_loss / len(loader.dataset)
    if last:
        fid_score = fid_metric.compute().item()
        return avg_loss, fid_score
    else:
        return avg_loss, 0

def train_and_eval(epochs, cuda_device=0, image_size = 128, steps=1000, batch_size = 128, eta=0, mode='ddpm', ddim_steps = 100):
    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5],
            [0.5]
        )
    ])

    # train / test splits
    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=1)

    model     = Diffusion().to(device)
    scheduler = DiffusionScheduler(timesteps=steps, device=device).to(device)
    optim     = torch.optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        model.train()
        for step, (x, _) in enumerate(train_loader):
            x = x.to(device)
            t = torch.randint(0, scheduler.timesteps, (x.size(0),), device=device)
            loss = p_losses(model, scheduler, x, t)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 100 == 0:
                print(f"[Train] Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        ckpt = f"ddpm_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt)

        if epoch == epochs-1: # Last epoch then test FID metric
            if mode == 'ddpm':
                test_loss, fid_loss = evaluate(model, scheduler, test_loader, device, True)
            else:
                test_loss, fid_loss = evaluate(model, scheduler, test_loader, device, last = True, mode = mode, ddim_steps=ddim_steps)
            print(f"[Eval ] Epoch {epoch} | Avg Loss {test_loss:.4f} | Avg FID {fid_loss:.4f}")
        else:
            if mode == 'ddpm':
                test_loss, fid_loss = evaluate(model, scheduler, test_loader, device, False)
            else:
                test_loss, fid_loss = evaluate(model, scheduler, test_loader, device, last = False, mode = mode, ddim_steps=ddim_steps)
            print(f"[Eval ] Epoch {epoch} | Avg Loss {test_loss:.4f}")

        output_path = f"./figures/sample_epoch_{epoch}_{mode}.png"
        if epoch % 50 == 0:
            sample_and_save(
                output_path=output_path,
                model_ckpt=ckpt,
                device=device,
                sample_shape=image_size,
                steps=steps,
                eta=eta,
                mode = mode,
                ddim_steps = ddim_steps
            )

@torch.no_grad()
def sample_ddim(model,
                scheduler,
                shape=(16, 3, 128, 128),
                device='cuda:2',
                ddim_steps=100,
                eta=0.5):

    model.eval()
    ts = scheduler.timesteps                
    alphas   = scheduler.alphas_cumprod     
    sqrt_a   = scheduler.sqrt_alphas_cumprod
    sqrt_oma = scheduler.sqrt_one_minus_alphas_cumprod

    idx = torch.linspace(0, ts-1, ddim_steps, device=device).long()
    ddim_ts       = idx            
    desc          = ddim_ts.flip(0)
    desc_prev     = torch.cat([desc[1:], desc.new_empty(1).fill_(0)])

    x = torch.randn(shape, device=device)

    for t, t_prev in zip(desc, desc_prev):
        eps = model(x, torch.full((shape[0],), t, device=device, dtype=torch.long))

        a_t    = alphas[t]
        a_prev = alphas[t_prev]
        sa_t   = sqrt_a[t]
        sa_prev= sqrt_a[t_prev]
        som_t  = sqrt_oma[t]

        x0_pred = (x - som_t * eps) / sa_t

        sigma   = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt  = torch.sqrt(1 - a_prev - sigma**2) * eps
        noise   = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

        x = sa_prev * x0_pred + dir_xt + sigma * noise

    return x

@torch.no_grad()
def sample(model, scheduler, shape=(16, 3, 128, 128)):
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

def sample_and_save(output_path='samples_grid.png',
                    model_ckpt='ddpm_epoch_49.pth',
                    device='cuda:2' if torch.cuda.is_available() else 'cpu',
                    sample_shape=128,
                    steps=1000,
                    num_samples=50,
                    mode='ddpm',
                    eta=0,
                    ddim_steps = 100):

    model = Diffusion().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    scheduler = DiffusionScheduler(timesteps=steps, device=device).to(device)

    with torch.no_grad():
        if mode == 'ddpm':
            imgs = sample(model, scheduler, shape=(num_samples, 3, sample_shape, sample_shape))
        else:
            imgs = sample_ddim(model, scheduler, shape=(num_samples, 3, sample_shape, sample_shape), ddim_steps=ddim_steps, eta=eta, device=device)

    imgs = (imgs + 1) * 0.5
    imgs = imgs.clamp(0, 1)
    print(imgs.min(), imgs.max())

    grid = make_grid(imgs, nrow=10, padding=2)
    save_image(grid, output_path)
    print(f"Saved sample to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number (e.g., 0, 1, etc.)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=128, help='Dimension of image')
    parser.add_argument('--steps', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--mode', type=str, default='ddpm')
    parser.add_argument('--ddim_steps', type=int, default=100)
    args = parser.parse_args()

    train_and_eval(args.epochs, args.cuda, args.image_size, args.steps, args.batch_size, args.eta, args.mode, args.ddim_steps)