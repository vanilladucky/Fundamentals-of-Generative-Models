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
            # Take a fixed batch of test images to visualize denoising
            sample_imgs, sample_labels = next(iter(test_loader))
            sample_labels = torch.full((sample_imgs.size(0),), fill_value = desired_class)
            sample_imgs = sample_imgs[:16].to(device)    # pick 16 images
            sample_labels = sample_labels[:16].to(device)

            # Sample a new random noise vector and λ, then do one forward-diffusion plus one step of denoising
            with torch.no_grad():
                B = sample_imgs.size(0)
                lamb = torch.full((B,), fill_value=max_lambda, device=device)  # extreme noise
                eps = torch.randn_like(sample_imgs)
                z_lambda = trainer.diffusion_forward(sample_imgs, lamb, eps)
                # Predict noise with CFG (here cond=sample_labels, uncond=“null index”)
                eps_cond   = unet(z_lambda, lamb, sample_labels)
                eps_uncond = unet(z_lambda, lamb, torch.full_like(sample_labels, 10))
                # Interpolate guided noise
                w = guidance_str  # same as uncond_prob or your chosen guidance weight
                eps_guided = (1.0 + w) * eps_cond - w * eps_uncond
                # Single DDIM-style step: reconstruct x0
                alpha = torch.sigmoid(lamb).view(-1,1,1,1)
                one_minus = 1.0 - alpha
                sqrt_alpha = torch.sqrt(alpha)
                x0_pred = (z_lambda - torch.sqrt(one_minus) * eps_guided) / sqrt_alpha
                x0_pred = x0_pred.clamp(-1,1)

                # Map back to [0,1] for saving
                samples_to_save = (x0_pred + 1.0) * 0.5
                grid = torchvision.utils.make_grid(samples_to_save, nrow=4)
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
        desired_class=args.desired_Class
    )