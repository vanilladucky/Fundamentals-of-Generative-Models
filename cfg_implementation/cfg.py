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
def sample_ddim_cfg(
        model,
        scheduler: SimpleDDPMScheduler,  
        labels_cond: torch.LongTensor,
        shape=(16, 3, 32, 32),
        device="cuda:0",
        num_ddim_steps: int = 50,
        eta: float = 0.0,
        w: float = 1.0,
        ) -> torch.Tensor:

        model.eval()
        B = shape[0]
        device = torch.device(device)
        
        min_lambda = model.min_lambda
        max_lambda = model.max_lambda
        timesteps = model.timesteps
        
        lambdas = torch.linspace(max_lambda, min_lambda, num_ddim_steps, device=device)
        lambdas_prev = torch.cat([lambdas[1:], torch.tensor([min_lambda], device=device)])

        x = torch.randn(shape, device=device)

        null_labels = torch.full((B,), fill_value=model.n_classes, dtype=torch.long, device=device)

        for i, lamb in enumerate(reversed(lambdas)):
            lamb_prev = lambdas_prev[num_ddim_steps - 1 - i]

            signal_ratio = model.lambda_to_signal_ratio(lamb)             
            noise_ratio = model.signal_ratio_to_noise_ratio(signal_ratio)  
            signal_ratio_prev = model.lambda_to_signal_ratio(lamb_prev)   

            sqrt_sr = torch.sqrt(signal_ratio).view(-1, 1, 1, 1)         
            sqrt_nr = torch.sqrt(noise_ratio).view(-1, 1, 1, 1)           
            sqrt_sr_prev = torch.sqrt(signal_ratio_prev).view(-1, 1, 1, 1) 

            lamb_batch = torch.full((B,), fill_value=lamb.item(), device=device)

            eps_cond = model.predict_noise(x, diffusion_step_idx=lamb_batch, label=labels_cond)
            eps_uncond = model.predict_noise(x, diffusion_step_idx=lamb_batch, label=null_labels)
            eps_guided = (1.0 + w) * eps_cond - w * eps_uncond

            x0_pred = (x - sqrt_nr * eps_guided) / sqrt_sr

            sigma_coef = (1 - signal_ratio_prev) - (signal_ratio_prev / signal_ratio) * (1 - signal_ratio)
            ddim_sigma = eta * torch.sqrt(sigma_coef.clamp(min=0)).view(-1, 1, 1, 1)

            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = sqrt_sr_prev * x0_pred + ddim_sigma * noise

        samples = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        return samples

def train_and_eval(img_size, batch_size, device, timesteps, epochs = 100, base_lr = 0.01, guidance_str = 0.5):
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
            samples = sample_ddim_cfg(
                model=cfg,                     
                scheduler=scheduler,
                labels_cond=sample_labels,
                shape=(sample_batch_size, 3, img_size, img_size),
                device=device,
                num_ddim_steps=100,
                eta=0.5,       
                w = guidance_str                
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
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

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
        return sqrt_sr * ori_image + sqrt_nr * rand_noise # Equation (6) - z_lambda

    def forward(self, noisy_image, diffusion_step, label='null'):
        # Feed through model
        return self.net(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )

    def predict_noise(self, noisy_image, diffusion_step_idx, label):
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

        inv_std = torch.tensor((0.2470,0.2435,0.2616), device=device).view(1,3,1,1)
        inv_mean = torch.tensor((0.4914,0.4822,0.4465), device=device).view(1,3,1,1)
        x0 = ori_image * inv_std + inv_mean     
        x0 = x0.clamp(0,1)
        x0 = x0 * 2.0 - 1.0       

        lamb = self.sample_lambda(B) 
        eps = self.sample_noise(B)
        x_t = self.perform_diffusion_process(x0, lamb, rand_noise=eps)                
        
        """t_int = torch.randint(low=0, high=self.timesteps, size=(B,), device=device)  

        alpha_t = scheduler.alphas_cumprod[t_int]                  
        sqrt_alpha_t = scheduler.sqrt_alphas_cumprod[t_int].view(-1,1,1,1)      
        sqrt_one_minus_alpha_t = scheduler.sqrt_one_minus_alphas_cumprod[t_int].view(-1,1,1,1)   

        eps = self.sample_noise(B) 
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * eps"""

        coin = torch.rand(B, device=device)
        use_null = coin < self.uncondition_prob
        null_labels = torch.full_like(true_label, fill_value=self.n_classes)
        cond_labels = torch.where(use_null, null_labels, true_label)  

        pred_noise = self.predict_noise(
            noisy_image=x_t,                      
            diffusion_step_idx=lamb,             
            label=cond_labels                     
        )

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
    parser.add_argument('--guidance_strength', type=float, default=0.5)
    args = parser.parse_args()

    train_and_eval(img_size = args.image_size, 
                    batch_size=args.batch_size,
                    device = args.cuda,
                    timesteps = args.steps,
                    epochs = args.epochs,
                    base_lr = args.base_lr,
                    guidance_str = args.guidance_strength)