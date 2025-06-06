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
from unet_v2 import Diffusion

@torch.no_grad()
def sample_ddim_cfg_from_scratch(
    unet: nn.Module, 
    labels_cond: torch.LongTensor, 
    shape=(16, 3, 32, 32),
    device="cuda:0",
    num_ddim_steps: int = 256,
    eta: float = 0.0,      # η=0 → deterministic DDIM
    w: float = 0.5,        # classifier-free guidance weight
) -> torch.Tensor:
    """
    Generates new images from pure Gaussian noise using Classifier-Free Guidance + DDIM.

    unet: your Diffusion model (an instance of Diffusion), whose forward(z, λ, cond) → ε̂
    labels_cond: (B,) tensor of desired class labels in [0..n_classes-1]
    shape: the shape of generated images = (B, C, H, W)
    device: 'cuda:0' or 'cpu'
    num_ddim_steps: how many DDIM steps to run
    eta: controls extra noise: η=0 is purely deterministic; η>0 injects noise each step
    w: guidance weight (e.g. 0.1–1.5)
    """
    unet.eval()
    B = shape[0]
    device = torch.device(device)

    # 1) Build strictly ascending λ sequence: λ₁ = min_lambda, λ_T = max_lambda
    min_lambda = unet.min_lambda
    max_lambda = unet.max_lambda
    lambdas = torch.linspace(min_lambda, max_lambda, num_ddim_steps, device=device)
    lambdas_prev = torch.cat([lambdas[1:], torch.tensor([min_lambda], device=device)])

    # 2) Initialize z₁ ← N(0,I)
    z = torch.randn(shape, device=device)

    # 3) Prepare the unconditional (“null”) label:
    #    For CIFAR-10, we used an embedding size of n_classes+1, so the “null index” = 10
    null_label = torch.full((B,), fill_value=unet.n_classes, dtype=torch.long, device=device)

    eps_floor = 1e-6

    for i, λ in enumerate(lambdas):
        λ_prev = lambdas_prev[i]

        # 4) Compute αₜ = sigmoid(λ), 1−αₜ, α_{t+1} = sigmoid(λ_prev)
        alpha_t = unet.lambda_to_signal_ratio(λ).clamp(min=eps_floor, max=1.0 - eps_floor)        # scalar
        one_minus_t = (1.0 - alpha_t).clamp(min=eps_floor)                                         # scalar
        alpha_next = unet.lambda_to_signal_ratio(λ_prev).clamp(min=eps_floor, max=1.0 - eps_floor) # scalar
        one_minus_next = (1.0 - alpha_next).clamp(min=eps_floor)                                    # scalar

        sqrt_alpha_t = torch.sqrt(alpha_t).view(1,1,1,1)          # [1,1,1,1]
        sqrt_one_m_t = torch.sqrt(one_minus_t).view(1,1,1,1)      # [1,1,1,1]
        sqrt_alpha_next = torch.sqrt(alpha_next).view(1,1,1,1)    # [1,1,1,1]

        # 5) Build a (B,) float tensor filled with current λ so the U-Net sees λₜ
        λ_batch = torch.full((B,), fill_value=λ.item(), device=device)  # [B]

        # 6) Query the U-Net for conditional + unconditional noise predictions:
        ε_cond   = unet(z, λ_batch, labels_cond)   # [B,C,H,W]
        ε_uncond = unet(z, λ_batch, null_label)     # [B,C,H,W]
        ε_guided = (1.0 + w) * ε_cond - w * ε_uncond # [B,C,H,W]

        # 7) Compute x₀_pred = (zₜ − √(1−αₜ)·ε̃ₜ) / √(αₜ)
        x0_pred = (z - sqrt_one_m_t * ε_guided) / sqrt_alpha_t  # [B,C,H,W]

        # 8) If this is the last DDIM step, set z ← x₀_pred and break
        if i == num_ddim_steps - 1:
            z = x0_pred
            break

        # 9) Otherwise, compute the DDIM σₜ term:
        termA = one_minus_next / one_minus_t   # scalar
        termB = (alpha_next - alpha_t) / alpha_next  # scalar
        σ_t_squared = (termA * termB).clamp(min=0.0)   # scalar ≥ 0
        σ_t = (eta * torch.sqrt(σ_t_squared)).view(1,1,1,1)  # [1,1,1,1]

        # 10) Sample extra noise if η > 0
        if eta > 0.0:
            noise = torch.randn_like(z)
        else:
            noise = torch.zeros_like(z)

        # 11) DDIM update: z_{t+1} = √α_{t+1} · x₀_pred + σₜ · noise
        z = sqrt_alpha_next * x0_pred + σ_t * noise

    # end for

    # 12) Clamp z to [−1, +1] and move to [0,1]
    samples = (z.clamp(-1.0, 1.0) + 1.0) / 2.0  # [B,3,H,W] in [0,1]
    return samples

class CFGTrainer:
    """
    A helper that encapsulates:
      - noise schedule (λ ↔ alpha/√(alpha))
      - sampling λ at each batch
      - performing the “diffusion forward” z_λ = √alpha(λ) · x0 + √(1-alpha(λ)) · ε
      - randomly dropping labels with probability uncond_prob
      - computing MSE between predicted ε and true ε
    """
    def __init__(
        self,
        model: nn.Module,
        img_size: int,
        n_classes: int,
        device: torch.device,
        uncond_prob: float = 0.1,
        min_lambda: float = -12.0,
        max_lambda: float = 12.0,
    ):
        super().__init__()
        self.model = model.to(device)
        self.img_size = img_size
        self.n_classes = n_classes
        self.device = device
        self.uncond_prob = uncond_prob
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # Precompute a, b constants for inverse CF‐cosine schedule if you want,
        # but here we’ll simply invert “uniform u → λ” directly.
        # (As in your earlier code, you had: u ~ Uniform(0,1), λ = -2 * log(tan(a*u + b)).
        #  But for simplicity, we’ll just sample λ ~ Uniform(min_lambda, max_lambda) each batch.)
        # If you prefer the “cosine schedule inversion,” you can replace sample_lambda() accordingly.

    def sample_lambda(self, batch_size: int) -> torch.Tensor:
        """
        Sample λ uniformly in [min_lambda, max_lambda] for each example.
        (This corresponds roughly to sampling log‐SNR from a wide range.)
        """
        return torch.rand(batch_size, device=self.device) * (self.max_lambda - self.min_lambda) + self.min_lambda

    def lambda_to_alpha(self, lamb: torch.Tensor) -> torch.Tensor:
        """
        Convert λ → alpha = sigmoid(λ). Ensures alpha ∈ (0,1).
        """
        return torch.sigmoid(lamb)

    def diffusion_forward(self, x0: torch.Tensor, lamb: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Given a clean image x0 ∈ [-1,1], a λ (log‐SNR), and noise ε ~ N(0,I),
        compute z_λ = √alpha(λ) · x0 + √(1 - alpha(λ)) · ε.

        x0:   [B, 3, H, W], in [-1,1]
        lamb: [B] float
        noise:[B, 3, H, W] normal
        """
        alpha = self.lambda_to_alpha(lamb)                 # [B]
        sigma_squared = 1.0 - alpha                                   # [B]
        sqrt_alpha = torch.sqrt(alpha).view(-1, 1, 1, 1)       # [B,1,1,1]
        sqrt_σ = torch.sqrt(sigma_squared).view(-1, 1, 1, 1)      # [B,1,1,1]
        return sqrt_alpha * x0 + sqrt_σ * noise            # [B,3,H,W]

    def compute_loss(self, clean_imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Perform one forward pass and compute the MSE loss:
        1) Sample λ, sample ε
        2) z_λ = forward diffusion
        3) With prob uncond_prob, drop label → use “null index” in model
        4) Model predicts ε̂ = model(z_λ, λ, cond_label)
        5) return F.mse_loss(ε̂, ε)
        """
        B = clean_imgs.size(0)
        clean_imgs = clean_imgs.to(self.device)       # expected ∈ [-1,1] already
        labels = labels.to(self.device)

        # 1) Sample λ and ε
        lamb = self.sample_lambda(B)                  # [B]
        eps = torch.randn_like(clean_imgs)            # [B,3,H,W]

        # 2) Compute z_λ
        z_lambda = self.diffusion_forward(clean_imgs, lamb, eps)  # [B,3,H,W]

        # 3) Randomly drop cond labels with probability uncond_prob
        coin = torch.rand(B, device=self.device)
        use_null = coin < self.uncond_prob
        null_labels = torch.full_like(labels, fill_value=self.n_classes)  # “null” index
        cond_labels = torch.where(use_null, null_labels, labels)          # [B]

        # 4) Predict noise
        eps_pred = self.model(z_lambda, lamb, cond_labels)                # [B,3,H,W]

        # 5) MSE loss against true ε
        loss = F.mse_loss(eps_pred, eps, reduction="mean")
        return loss


def train_cfg(
    img_size: int = 32,
    batch_size: int = 128,
    epochs: int = 200,
    base_lr: float = 2e-4,
    uncond_prob: float = 0.1,
    min_lambda: float = -12.0,
    max_lambda: float = 12.0,
    device: str = "cuda",
    save_every: int = 10,
    out_dir: str = "./checkpoints",
    guidance_str = 0.7, 
    desired_class = 0
):
    """
    A standalone training loop for classifier-free guidance.
    Trains on CIFAR-10. Saves model checkpoints every `save_every` epochs.
    """
    torch.manual_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1) Data (range [0,1] → normalize to [-1,1])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # [-1,1]
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2) Instantiate model + trainer
    unet = Diffusion().to(device)
    trainer = CFGTrainer(
        model=unet,
        img_size=img_size,
        n_classes=10,
        device=device,
        uncond_prob=uncond_prob,
        min_lambda=min_lambda,
        max_lambda=max_lambda,
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=base_lr)

    # 3) Logging setup
    logging.basicConfig(
        filename="cfg_training.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting CFG training")

    # 4) Training + evaluation loop
    for epoch in range(1, epochs + 1):
        unet.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch:03d}/{epochs:03d}] Train", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)    # in [-1, 1]
            labels = labels.to(device)

            loss = trainer.compute_loss(imgs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch:03d} | Train loss: {avg_train_loss:.5f}")
        print(f"[Epoch {epoch:03d}] Train loss: {avg_train_loss:.5f}")

        # ─── Simple evaluation: compute average loss on test set ────────────
        unet.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                loss_t = trainer.compute_loss(imgs, labels)
                test_loss += loss_t.item()
        avg_test_loss = test_loss / len(test_loader)
        logging.info(f"Epoch {epoch:03d} | Test  loss: {avg_test_loss:.5f}")
        print(f"[Epoch {epoch:03d}] Test  loss: {avg_test_loss:.5f}")

        if epoch % save_every == 0:

            # Sample a new random noise vector and λ, then do one forward-diffusion plus one step of denoising
            with torch.no_grad():
                B = 16
                labels_cond = torch.full((B,), fill_value=desired_class, device=device)

                new_samples = sample_ddim_cfg_from_scratch(
                    unet=trainer.model,
                    labels_cond=labels_cond,
                    shape=(B, 3, img_size, img_size),
                    device=device,
                    num_ddim_steps=256,
                    eta=0.0,          
                    w=guidance_str
                )

                grid = torchvision.utils.make_grid(new_samples, nrow=4)
                save_image(grid, f"./figures/epoch_{epoch:03d}_samples.png")
                logging.info(f"Saved sample grid at epoch {epoch:03d}")

        # ─── Save model checkpoint ───────────────────────────────────────────
        if epoch % 100 == 0:
            ckpt_path = f"{out_dir}/unet_epoch_{epoch:03d}.pth"
            torch.save(unet.state_dict(), ckpt_path)
            logging.info(f"Saved model checkpoint: {ckpt_path}")

    print("Training complete!")

# ─── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size",    type=int,   default=32,    help="Training image size")
    parser.add_argument("--batch_size",    type=int,   default=128,   help="Batch size")
    parser.add_argument("--epochs",        type=int,   default=200,   help="Number of epochs")
    parser.add_argument("--lr",            type=float, default=2e-4,  help="Base learning rate")
    parser.add_argument("--uncond_prob",   type=float, default=0.1,   help="Probability of dropping label (CFG)")
    parser.add_argument("--min_lambda",    type=float, default=-12.0, help="Min λ for sampling log‐SNR")
    parser.add_argument("--max_lambda",    type=float, default=12.0,  help="Max λ for sampling log‐SNR")
    parser.add_argument("--device",        type=str,   default="cuda",help="“cuda” or “cpu”")
    parser.add_argument("--save_every",    type=int,   default=10,    help="Save checkpoint every N epochs")
    parser.add_argument("--out_dir",       type=str,   default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--guide",         type=float, default=0.7,   help="Guidance strength for inference")
    parser.add_argument("--desired_class", type=int,   default=0,   help="Desired class for sampling")
    args = parser.parse_args()

    import os
    os.makedirs("./figures", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    train_cfg(
        img_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_lr=args.lr,
        uncond_prob=args.uncond_prob,
        min_lambda=args.min_lambda,
        max_lambda=args.max_lambda,
        device=args.device,
        save_every=args.save_every,
        out_dir=args.out_dir,
        guidance_str=args.guide,
        desired_class=args.desired_class
    )