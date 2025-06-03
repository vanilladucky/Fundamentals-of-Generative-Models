# References: https://github.com/KimRass/CFG/blob/main/classifier_free_guidance.py

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import contextlib
from unet import UNet
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets, transforms
import torchvision
from torchvision.utils import save_image
import argparse
from torchsummary import summary

class SimpleDDPMScheduler(nn.Module):
    def __init__(self, timesteps: int = 1000, device = 'cuda:0'):
        super().__init__()
        self.timesteps = timesteps

        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod).to(device))

@torch.no_grad()
def sample_cfg_ddim(
    model,
    scheduler: SimpleDDPMScheduler,
    labels: torch.LongTensor,
    shape=(1, 3, 32, 32),
    device="cuda:0",
    num_ddim_steps: int = 100,
    eta: float = 0.0
) -> torch.Tensor:

    model.eval()
    batch_size = shape[0]
    T = scheduler.alphas_cumprod.shape[0]
    device = torch.device(device)

    ddim_timesteps = torch.linspace(0, T - 1, num_ddim_steps, dtype=torch.long, device=device)
    ddim_timesteps_prev = torch.cat([ddim_timesteps[1:], torch.tensor([0], device=device, dtype=torch.long)])

    x = torch.randn(shape, device=device)

    for i, t in enumerate(reversed(ddim_timesteps)):
        t_prev = ddim_timesteps_prev[num_ddim_steps - 1 - i]

        alpha_t = scheduler.alphas_cumprod[t]            
        alpha_prev = scheduler.alphas_cumprod[t_prev]        
        sqrt_alpha_t = scheduler.sqrt_alphas_cumprod[t]   
        sqrt_alpha_prev = scheduler.sqrt_alphas_cumprod[t_prev]  
        sqrt_one_minus_alpha_t = scheduler.sqrt_one_minus_alphas_cumprod[t]      

        diffusion_steps = torch.full(
            (batch_size,),
            fill_value=int(t.item()),
            dtype=torch.long,
            device=device,
        )

        eps = model.predict_noise(noisy_image=x, diffusion_step_idx=diffusion_steps, label=labels)

        x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t)) * torch.sqrt(
            torch.tensor(max((1 - alpha_t / alpha_prev).item(), 0.0), device=device)
        )

        under_sqrt = max((1 - alpha_prev - sigma_t**2).item(), 0.0)
        dir_xt = torch.sqrt(torch.tensor(under_sqrt, device=device)) * eps

        if sigma_t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = sqrt_alpha_prev * x0_pred + dir_xt + sigma_t * z

    return x

def train_and_eval(img_size, batch_size, device, timesteps, epochs = 100, base_lr = 0.01):
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root="path/to/store/cifar10",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="path/to/store/cifar10",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,       
        pin_memory=True     
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    net = UNet(n_classes=10).to(device)
    cfg = CFG(net = net, img_size=img_size, batch_size=batch_size, device=device, timesteps=timesteps)
    optim = torch.optim.Adam(net.parameters(), lr=base_lr)
    scheduler = SimpleDDPMScheduler(timesteps=timesteps, device=device).to(device)

    for epoch in range(epochs):
        net.train()
        for step, (img,labels) in enumerate(train_loader):
            images = img.to(device)        
            labels = labels.to(device) 
            loss = cfg.get_loss(ori_image=images, true_label=labels, scheduler=scheduler)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 100 == 0:
                print(f"[Train] Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        ckpt = f"ddpm_epoch_{epoch}.pth"
        torch.save(net.state_dict(), ckpt)

        with torch.no_grad():
            net.eval()
            total_loss = 0
            for step, (img, labels) in enumerate(test_loader):
                images = img.to(device)
                labels = labels.to(device)
                loss = cfg.get_loss(images, labels, scheduler=scheduler)
                total_loss+=loss.item()
            total_loss/=step
            print(f"[Eval] Epoch {epoch} | Loss {total_loss:.4f}") 

        with torch.no_grad():
            sample_batch_size = 16
            sample_labels = torch.zeros(
                sample_batch_size, dtype=torch.long, device=device
            )  # all “class 0” (airplane)
            samples = sample_cfg_ddim(
                model=cfg,                     
                scheduler=scheduler,
                labels=sample_labels,
                shape=(sample_batch_size, 3, img_size, img_size),
                device=device,
                num_ddim_steps=100,
                eta=0.5,                       
            )
            # 4) samples are in roughly [−1,1] or [0,1] depending on UNet’s output range.
            #    Denormalize back to pixel range for saving: (assuming UNet learned to output in [−1,1]):
            samples = (samples.clamp(-1, 1) + 1) / 2.0  # now in [0,1]

            grid = torchvision.utils.make_grid(samples, nrow=4)
            save_image(grid, f"./figures/samples_epoch_{epoch}.png")

            print(f"[Sample] Saved sample grid for epoch {epoch} → samples_epoch_{epoch}.png")

class CFG(nn.Module):
    def __init__(self,
        net, 
        img_size, 
        batch_size,
        device,
        min_lambda = -20,
        max_lambda = 20,
        guidance_coeff = 0.5,
        uncondition_prob = 0.3, 
        interpol_coef = 0.3,
        img_channels = 3,
        timesteps = 1000
    ):

        super().__init__()

        self.net = net.to(device)
        self.img_size = img_size
        self.batch_size = batch_size
        self.guidance_str = guidance_coeff
        self.uncondition_prob = uncondition_prob
        self.interpol_coef = interpol_coef
        self.img_channels = img_channels
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.n_classes = 10

        self.diffusion_step = torch.linspace(
            min_lambda, max_lambda, timesteps, device=device,
        )

        self.b = torch.arctan(
            torch.exp(torch.tensor(-max_lambda / 2, device=self.device))
        )
        self.a = torch.arctan(
            torch.exp(torch.tensor(-min_lambda / 2, device=self.device))
        ) - self.b

    def sample_noise(self, batch_size): 
        return torch.randn(
            size=(batch_size, self.img_channels, self.img_size, self.img_size),
            device=self.device,
        )
    
    def sample_lambda(self, batch_size):
        u = torch.rand(batch_size, device=self.device)  
        lamb = -2 * torch.log(torch.tan(self.a * u + self.b))  
        return lamb

    def lambda_to_signal_ratio(self, lamb):
        return 1 / (1 + torch.exp(-lamb))

    def signal_ratio_to_noise_ratio(self, signal_ratio):
        return 1 - signal_ratio

    def perform_diffusion_process(self, ori_image, lamb, rand_noise=None):
        signal_ratio = self.lambda_to_signal_ratio(lamb)      
        noise_ratio  = 1 - signal_ratio                        
        sqrt_sr = signal_ratio.sqrt().view(-1,1,1,1)           
        sqrt_nr = noise_ratio.sqrt().view(-1,1,1,1)            
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        return sqrt_sr * ori_image + sqrt_nr * rand_noise # Equation (6)

    def forward(self, noisy_image, diffusion_step, label='null'):
        # Feed through model
        return self.net(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )

    def predict_noise(self, noisy_image, diffusion_step_idx, label):
        # Equation (6) 
        diffusion_step = diffusion_step_idx.to(dtype=torch.long, device=self.device)
        # Run it through forward()
        pred_noise_cond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )
        # Run it through forward()
        pred_noise_uncond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=torch.full((label.shape[0],), 10, dtype=torch.long, device=self.device),
        )
        return (1 + self.guidance_str) * pred_noise_cond - self.guidance_str * pred_noise_uncond

    def get_loss(self, ori_image: torch.Tensor, true_label: torch.LongTensor, scheduler: SimpleDDPMScheduler):
        B = ori_image.size(0)
        device = ori_image.device

        # ========== 1) Undo CIFAR-normalization → [0,1], then [–1,1] ==========
        inv_std = torch.tensor((0.2470,0.2435,0.2616), device=device).view(1,3,1,1)
        inv_mean = torch.tensor((0.4914,0.4822,0.4465), device=device).view(1,3,1,1)
        x0 = ori_image * inv_std + inv_mean     # now in [0,1]
        x0 = x0.clamp(0,1)
        x0 = x0 * 2.0 - 1.0                      # now in [–1,1]
        
        # ========== 2) Sample a random integer timestep t_int ∈ {0,…,T–1} ==========
        t_int = torch.randint(low=0, high=self.timesteps, size=(B,), device=device)  # shape [B]

        # ========== 3) Look up αₜ and √(1–αₜ) from scheduler ==========
        alpha_t = scheduler.alphas_cumprod[t_int]                  # [B]
        sqrt_alpha_t = scheduler.sqrt_alphas_cumprod[t_int].view(-1,1,1,1)       # [B,1,1,1]
        sqrt_one_minus_alpha_t = scheduler.sqrt_one_minus_alphas_cumprod[t_int].view(-1,1,1,1)    # [B,1,1,1]

        # ========== 4) Sample Gaussian noise ε and build x_t ==========
        eps = self.sample_noise(B)  # [B, C, H, W]
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * eps

        # ========== 5) Sample “null” vs “cond” labels ==========
        coin = torch.rand(B, device=device)
        use_null = coin < self.uncondition_prob
        null_labels = torch.full_like(true_label, fill_value=self.n_classes)
        cond_labels = torch.where(use_null, null_labels, true_label)  # [B]

        # ========== 6) Predict noise via classifier-free guidance ==========
        pred_noise = self.predict_noise(
            noisy_image=x_t,                      # [B, C, H, W]
            diffusion_step_idx=t_int,             # integer timesteps
            label=cond_labels                     # [B]
        )

        # ========== 7) MSE loss against the true ε ==========
        loss = F.mse_loss(pred_noise, eps, reduction="mean")
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number (e.g., 0, 1, etc.)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--image_size', type=int, default=32, help='Dimension of image')
    parser.add_argument('--steps', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--base_lr', type=float, default=2e-4, help='Initial Learning Rate')
    args = parser.parse_args()

    train_and_eval(img_size = args.image_size, 
                    batch_size=args.batch_size,
                    device = args.cuda,
                    timesteps = args.steps,
                    epochs = args.epochs,
                    base_lr = args.base_lr)