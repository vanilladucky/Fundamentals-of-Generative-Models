import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from diffusers import DDIMScheduler
import random
import argparse

# ----------------------------------------
# Joint training with classifier-free guidance
# ----------------------------------------
def train_classifier_free(
    model, train_loader, val_loader, optimizer,
    puncond=0.1,
    timesteps=1000,
    beta_start=0.0001, beta_end=0.02,
    device='cuda'
):
    # build scheduler for noise scales
    scheduler = DDIMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=timesteps
    )
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    one_minus_alphas_cumprod = (1 - alphas_cumprod)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            B = x.size(0)
            # sample random training timesteps
            t = torch.randint(0, timesteps, (B,), device=device)
            alpha_t = alphas_cumprod[t].view(B,1,1,1)
            sigma_t = one_minus_alphas_cumprod[t].view(B,1,1,1)
            # noise and corrupt
            noise = torch.randn_like(x)
            z = alpha_t.sqrt() * x + sigma_t.sqrt() * noise
            # classifier-free guidance dropout
            cond = labels if random.random() > puncond else None
            # predict noise
            eps_pred = model(z, t, cond)
            loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * B
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                B = x.size(0)
                t = torch.randint(0, timesteps, (B,), device=device)
                alpha_t = alphas_cumprod[t].view(B,1,1,1)
                sigma_t = one_minus_alphas_cumprod[t].view(B,1,1,1)
                noise = torch.randn_like(x)
                z = alpha_t.sqrt() * x + sigma_t.sqrt() * noise
                cond = labels if random.random() > puncond else None
                eps_pred = model(z, t, cond)
                val_loss += F.mse_loss(eps_pred, noise).item() * B
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

        # sample and save every 10 epochs
        if epoch % 10 == 0:
            B = 16
            cond = torch.full((B,), sample_class, dtype=torch.long, device=device)
            samples = sample_classifier_free(
                model,
                (B, 3, args.image_size, args.image_size),
                cond,
                w=0.3,
                num_inference_steps=200,
                beta_start=beta_start,
                beta_end=beta_end,
                device=device
            )
            print(f"Before sampling: Max: {samples.max()} | Min: {samples.min()}")
            imgs = (samples.clamp(-1,1) + 1) / 2
            import torchvision.utils as vutils
            vutils.save_image(imgs, f'samples_epoch_{epoch}.png', nrow=4)

# ----------------------------------------
# Sampling with DDIM Scheduler and CFG
# ----------------------------------------
@torch.no_grad()
def sample_classifier_free(
    model, shape, cond, w=0.3,
    num_inference_steps=200,
    beta_start=0.0001, beta_end=0.02,
    device='cuda'
):
    model.eval()
    # instantiate scheduler
    scheduler = DDIMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=1000
    )
    scheduler.set_timesteps(num_inference_steps)
    # initial noise
    z = torch.randn(shape, device=device)
    B = shape[0]
    for t in scheduler.timesteps:
        # expand and move timestep to correct device
        t_batch = t.unsqueeze(0).expand(B).to(device)
        # predict noise
        eps_cond = model(z, t_batch, cond)
        eps_uncond = model(z, t_batch, None)
        # classifier-free guidance
        eps_guided = (1 + w) * eps_cond - w * eps_uncond
        # step
        out = scheduler.step(eps_guided, t, z)
        z = out.prev_sample
    return z

# ----------------------------------------
# DeepUNet definition (unchanged)
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
    def __init__(self, in_channels=3, base_ch=256, num_classes=10, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        self.enc1 = ConvBlock(in_channels + 2 * time_emb_dim, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bot = ConvBlock(base_ch * 4, base_ch * 8)
        self.dec3 = ConvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2 + base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, cond):
        B, C, H, W = x.shape
        t_emb = self.time_mlp(t.float().view(B,1)).view(B,-1,1,1).expand(-1,-1,H,W)
        if cond is not None:
            l_emb = self.label_emb(cond).view(B,-1,1,1).expand(-1,-1,H,W)
        else:
            l_emb = torch.zeros_like(t_emb)
        h = torch.cat([x, t_emb, l_emb], dim=1)
        e1 = self.enc1(h)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out_conv(d1)

# ----------------------------------------
# Main: CIFAR-10 training & sampling
# ----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size",    type=int,   default=32,    help="Training image size")
    parser.add_argument("--batch_size",    type=int,   default=128,   help="Batch size")
    parser.add_argument("--epochs",        type=int,   default=500,   help="Number of epochs")
    parser.add_argument("--lr",            type=float, default=2e-4,  help="Base learning rate")
    parser.add_argument("--uncond_prob",   type=float, default=0.4,   help="Probability of dropping label (CFG)")
    parser.add_argument("--min_lambda",    type=float, default=-20.0, help="Min λ for sampling log‐SNR")
    parser.add_argument("--max_lambda",    type=float, default=20.0,  help="Max λ for sampling log‐SNR")
    parser.add_argument("--device",        type=str,   default="cuda",help="“cuda” or “cpu”")
    parser.add_argument("--save_every",    type=int,   default=10,    help="Save checkpoint every N epochs")
    parser.add_argument("--out_dir",       type=str,   default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--guide",         type=float, default=0.7,   help="Guidance strength for inference")
    parser.add_argument("--desired_class", type=int,   default=0,   help="Desired class for sampling")
    args = parser.parse_args()

    device = f'{args.device}' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    num_epochs = args.epochs
    sample_class = 3  # class index to sample
    train_classifier_free(
        model, train_loader, val_loader,
        optimizer,
        puncond=args.uncond_prob,
        timesteps=1000,
        beta_start=args.min_lambda,
        beta_end=args.max_lambda,
        device=device
    )