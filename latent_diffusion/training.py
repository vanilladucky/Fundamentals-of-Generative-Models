import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
from VAE import vae

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


def train_vae(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    vae = vae(args.latent_dim).to(device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    #os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        vae.train()
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")
        #torch.save(vae.state_dict(), os.path.join(args.output_dir, f"vae_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./vae_checkpoints")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    train_vae(args)