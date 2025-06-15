import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
from vae import VAE
from discriminator import NLayerDiscriminator
from tqdm import tqdm
import torchvision.utils as vutils
from torch.distributions import Normal, kl_divergence

class PerceptualLoss(nn.Module):
    def __init__(self, layers=('3', '8', '15', '22')):
        """
        Uses VGG16 feature maps at given layer indices to compute
        L1 loss between recon and target in feature space.
        layers: tuple of indices into vgg.features
        """
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features.eval()
        for m in vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        for p in vgg.parameters():  # freeze
            p.requires_grad = False
        self.vgg = vgg
        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, recon, target):
        x, y = recon, target
        loss = 0.0
        for idx in self.layers:
            x = self.vgg[int(idx)](x)
            y = self.vgg[int(idx)](y)
            loss = loss + self.criterion(x, y)
        return loss

adv_criterion = nn.BCEWithLogitsLoss()

def discriminator_step(D, real_imgs, fake_imgs, opt_D):
    # real_imgs, fake_imgs: [B,3,H,W]
    real_logits = D(real_imgs)                     # → [B,1,h,w]
    fake_logits = D(fake_imgs.detach())            # detach so G isn't updated here

    real_loss = adv_criterion(real_logits, torch.ones_like(real_logits))
    fake_loss = adv_criterion(fake_logits, torch.zeros_like(fake_logits))
    d_loss = 0.5 * (real_loss + fake_loss)

    opt_D.zero_grad()
    d_loss.backward()
    opt_D.step()
    return d_loss.item()

def generator_adv_loss(D, fake_imgs):
    fake_logits = D(fake_imgs)
    # generator wants discriminator to predict 1 for its outputs
    return adv_criterion(fake_logits, torch.ones_like(fake_logits))

def kl_regularization(posterior):
    mu     = posterior.mean          
    var    = posterior.var 
    logvar    = posterior.logvar       
    kl = -0.5 * (1 + logvar - mu.pow(2) - var)
    return kl.mean()

def partial_load_model(model, saved_model_path):
    """ https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/31 """
    pretrained_dict = torch.load(saved_model_path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    status = model.load_state_dict(pretrained_dict, strict=False)
    print(status)

    return model

def train_vae(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    λ_kl = 1e-6
    G = VAE().to(device)
    D = NLayerDiscriminator().to(device)
    D = partial_load_model(D, 'day2night.t7')
    rec_crit = PerceptualLoss().to(device)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
            )
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr)        
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for x, _ in tqdm(loader, leave=False):
            x = x.to(device)
            recon, posterior = G(x)
            Lrec  = rec_crit(recon, x)                        
            Ladv  = generator_adv_loss(D, recon)            
            Lkl   = kl_regularization(posterior)  

            loss_G = Lrec + Ladv + λ_kl * Lkl
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loss_D = discriminator_step(D, x, recon, opt_D)

        # Sample random latent vectors and decode
        with torch.no_grad():
            z = torch.randn(16, 4, int((args.latent_dim)**0.5), int((args.latent_dim)**0.5)).to(device)
            sampled_imgs = G.decode(z) 

        if epoch > 0 and epoch % 100 == 0:
            # Save the generated samples
            grid = vutils.make_grid(sampled_imgs, nrow=4, normalize=True)
            vutils.save_image(grid, f"sampled_epoch_{epoch+1:03d}.png")

        print(f"[Epoch {epoch+1}] G_loss: {loss_G.item():.4f} | D_loss: {loss_D:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./vae_checkpoints")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    train_vae(args)