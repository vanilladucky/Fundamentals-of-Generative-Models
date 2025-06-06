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
import logging

@torch.no_grad()
def sample_ddim_cfg(
    model,
    labels_cond: torch.LongTensor,
    shape=(16, 3, 32, 32),
    device="cuda:0",
    num_ddim_steps: int = 256,
    eta: float = 0.3,
    w: float = 0.1,
) -> torch.Tensor:

    model.eval()
    B = shape[0]
    device = torch.device(device)

    min_lambda = model.min_lambda
    max_lambda = model.max_lambda

    lambdas = torch.linspace(min_lambda, max_lambda, num_ddim_steps, device=device)
    lambdas_prev = torch.cat([lambdas[1:], torch.tensor([min_lambda], device=device)])

    x = torch.randn(shape, device=device)

    null_labels = torch.full((B,), fill_value=model.n_classes, dtype=torch.long, device=device)

    eps_floor = 1e-6  

    for i, lamb in enumerate(lambdas):
        lamb_prev = lambdas_prev[i]

        alpha_t = model.lambda_to_signal_ratio(lamb).clamp(min=eps_floor, max=1.0 - eps_floor)       
        one_minus_t = (1.0 - alpha_t).clamp(min=eps_floor)                                         
        alpha_next = model.lambda_to_signal_ratio(lamb_prev).clamp(min=eps_floor, max=1.0 - eps_floor)  
        one_minus_next = (1.0 - alpha_next).clamp(min=eps_floor)                                     

        sqrt_sr      = torch.sqrt(alpha_t).view(-1, 1, 1, 1)        
        sqrt_nr      = torch.sqrt(one_minus_t).view(-1, 1, 1, 1)     
        sqrt_sr_next = torch.sqrt(alpha_next).view(-1, 1, 1, 1)      

        lamb_batch = torch.full((B,), fill_value=lamb.item(), device=device)  

        eps_cond   = model.predict_noise(x, diffusion_step_idx=lamb_batch, label=labels_cond) 
        eps_uncond = model.predict_noise(x, diffusion_step_idx=lamb_batch, label=null_labels) 
        eps_guided = (1.0 + w) * eps_cond - w * eps_uncond                         

        """if i > 0 and i % 5 == 0:
            logging.info(f"[Step {i:03d}] λ={lamb:.2f}  "
                         f"||ε_cond||={eps_cond.norm().item():.3e}, "
                         f"||ε_uncond||={eps_uncond.norm().item():.3e}, "
                         f"||ε_guided||={eps_guided.norm().item():.3e}")"""

        
        x0_pred = (x - sqrt_nr * eps_guided) / sqrt_sr  

        if i == num_ddim_steps - 1:
            x = x0_pred
            break

        termA = one_minus_next / one_minus_t                   
        termB = (alpha_next - alpha_t) / alpha_next            
        ddim_sigma_sq = (termA * termB).clamp(min=0.0)         
        ddim_sigma    = (eta * torch.sqrt(ddim_sigma_sq)).view(-1, 1, 1, 1)  

        if eta > 0.0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = sqrt_sr_next * x0_pred + ddim_sigma * noise      

    logging.info(f"RAW x before clamp: min={x.min().item():.3f}, "
                 f"max={x.max().item():.3f}, mean={x.mean().item():.3f}, std={x.std().item():.3f}")

    samples = (x.clamp(-1.0, 1.0) + 1.0) / 2.0 
    logging.info(f"After clamp: samples.min={samples.min().item():.3f}, "
                 f"max={samples.max().item():.3f}, mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")

    return samples

def train_and_eval(img_size, batch_size, device, timesteps, epochs = 100, base_lr = 0.01, guidance_str = 0.5):
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
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
    optim = torch.optim.AdamW(net.parameters(), lr=base_lr)

    for epoch in range(epochs):
        net.train()
        for step, (img,labels) in enumerate(train_loader):
            images = img.to(device)        
            labels = labels.to(device) 
            loss = cfg.get_loss(ori_image=images, true_label=labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % 200 == 0:
                logging.info(f"[Train] Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
                print(f"[Train] Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        ckpt = f"ddpm_epoch_{epoch}.pth"
        torch.save(net.state_dict(), ckpt)

        with torch.no_grad():
            net.eval()
            total_loss = 0
            for step, (img, labels) in enumerate(test_loader):
                images = img.to(device)
                labels = labels.to(device)
                loss = cfg.get_loss(images, labels)
                total_loss+=loss.item()
            total_loss/=step
            logging.info(f"[Eval] Epoch {epoch} | Loss {total_loss:.4f}")
            print(f"[Eval] Epoch {epoch} | Loss {total_loss:.4f}") 

        with torch.no_grad():
            sample_batch_size = 16
            sample_labels = torch.zeros(
                sample_batch_size, dtype=torch.long, device=device
            )  # all “class 0” (airplane)
            samples = sample_ddim_cfg(
                model=cfg,                    
                labels_cond=sample_labels,
                shape=(sample_batch_size, 3, img_size, img_size),
                device=device,
                eta=0.5,       
                w = guidance_str                
            )
            
            logging.info(f"Saving: min={samples.min().item():.3f}, "
                 f"max={samples.max().item():.3f}, mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")
            grid = torchvision.utils.make_grid(samples, nrow=4)
            save_image(grid, f"./figures/samples_epoch_{epoch}.png")

            print(f"[Sample] Saved sample grid for epoch {epoch} → samples_epoch_{epoch}.png")

class CFG(nn.Module):
    def __init__(self,
        net, 
        img_size, 
        batch_size,
        device,
        min_lambda = -12,
        max_lambda = 12,
        guidance_coeff = 0.5,
        uncondition_prob = 0.1, 
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
        # Inspired by the discrete time cosine noise schedule 
        u = torch.rand(batch_size, device=self.device)  
        lamb = -2 * torch.log(torch.tan(self.a * u + self.b))  
        return lamb

    def lambda_to_signal_ratio(self, lamb):
        return 1 / (1 + torch.exp(-lamb))

    def signal_ratio_to_noise_ratio(self, signal_ratio):
        return 1 - signal_ratio

    def perform_diffusion_process(self, ori_image, lamb, rand_noise=None):
        lamb_clamped = lamb.clamp(self.min_lambda, self.max_lambda)
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
        # Run it through forward()
        pred_noise_cond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step_idx, label=label,
        )
        # Run it through forward()
        pred_noise_uncond = self(
            noisy_image=noisy_image, diffusion_step=diffusion_step_idx, label=torch.full((label.shape[0],), 10, dtype=torch.long, device=self.device),
        )
        return (1 + self.guidance_str) * pred_noise_cond - self.guidance_str * pred_noise_uncond

    def get_loss(self, ori_image: torch.Tensor, true_label: torch.LongTensor):
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
    parser.add_argument('--guidance_strength', type=float, default=0.1)
    parser.add_argument('--file_name', type=str, default='log')
    args = parser.parse_args()

    logging.basicConfig(
        filename = f"{args.file_name}.txt",
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    train_and_eval(img_size = args.image_size, 
                    batch_size=args.batch_size,
                    device = args.cuda,
                    timesteps = args.steps,
                    epochs = args.epochs,
                    base_lr = args.base_lr,
                    guidance_str = args.guidance_strength)