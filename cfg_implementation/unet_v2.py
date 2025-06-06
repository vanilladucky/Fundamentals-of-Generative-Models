import torch
from torch import nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------------------------
#   Building blocks: Sinusoidal time embeddings, residual conv blocks, and skip blocks
# ------------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding in 1D:
      Given t of shape [B], returns an embedding of shape [B, dim].
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: [B], float (e.g. λ values)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )  # [half]
        args = t[:, None].float() * freqs[None, :]  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class ResidualBlock(nn.Module):
    """
    A simple residual wrapper: main = nn.Sequential([...]),
    skip = either Identity (if in_channels == out_channels) or a 1×1 conv.
    """
    def __init__(self, main: list, skip: nn.Module = None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip is not None else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.main(x) + self.skip(x)


class ResConvBlock(ResidualBlock):
    """
    Residual 2D‐conv block:
       - conv(in → mid),  ReLU, conv(mid → out), ReLU
       - uses Dropout2d(0.1) after each conv except final optionally
       - if in≠out, skip is a 1×1 conv; else Identity
    """
    def __init__(self, c_in: int, c_mid: int, c_out: int, dropout_last: bool = True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        layers = [
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
        ]
        if dropout_last:
            layers.append(nn.Dropout2d(0.1, inplace=True))
        layers.append(nn.ReLU(inplace=True))

        super().__init__(main=layers, skip=skip)


class SkipBlock(nn.Module):
    """
    A block that concatenates main branch + skip branch along the channel dimension.
      - main: some nn.Sequential([...])
      - skip: either Identity or some transformation to match shapes.
    Output channels = main_channels + skip_channels.
    """
    def __init__(self, main: list, skip: nn.Module = None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip is not None else nn.Identity()

    def forward(self, x: torch.Tensor):
        return torch.cat([self.main(x), self.skip(x)], dim=1)


def expand_to_planes(input: torch.Tensor, shape: torch.Size):
    """
    Expand a [B, C_time] (or [B, C_class]) tensor to [B, C, H, W] by repeating
    across spatial dims.  shape = (B, C, H, W).
    """
    return input[..., None, None].repeat(1, 1, shape[2], shape[3])


# ------------------------------------------------------------------------------
#   The U-Net: time‐embedding + class‐embedding + down/up blocks + final conv
# ------------------------------------------------------------------------------

class UNet(nn.Module):
    """
    A U-Net that accepts:
      - noisy image:   [B, 3,  H, W]
      - diffusion step λ: [B]   (float log‐SNR values)
      - class label:  [B] (long in [0..n_classes] ; n_classes index is “null class”)
    and outputs:
      - predicted noise ε̂ of shape [B, 3, H, W]

    Internals:
      - SinusoidalPosEmb to embed λ into a vector of size time_dim
      - nn.Embedding to embed class idx → vector of size time_dim
      - Add time_emb + class_emb → c = [B, time_dim]
      - Expand c → [B, time_dim, H, W] and concat with input
      - Down‐sampling path: stacks of ResConvBlocks + optional AvgPool
      - Bottleneck: ResConvBlock + ResConvBlock + self‐attention (optional)
      - Up‐sampling path: skip‐connections + ResConvBlocks + upsample
      - Final conv → 3 channels
    """

    def __init__(
        self,
        n_classes: int,
        base_channels: int = 64,
        channel_mults: list = [1, 2, 4, 8],
        n_res_blocks: int = 2,
        time_emb_dim: int = 128,
        class_emb_dim: int = 128,
        use_attn: list = None,
    ):
        """
        n_classes: number of real classes (e.g. 10 for CIFAR10). We'll internally add +1 for the “null” slot.
        base_channels: number of channels at the first level (32 or 64)
        channel_mults: multipliers at each resolution level (e.g. [1,2,4,8] → channel sizes [c,2c,4c,8c])
        n_res_blocks: number of ResConvBlocks per level
        time_emb_dim: dimension of λ‐embedding
        class_emb_dim: dimension of class embedding
        use_attn: a list of booleans (same length as channel_mults), whether to put a small self‐attn at that level
        """
        super().__init__()
        self.n_classes = n_classes  # real number of classes (no “null”)
        self.time_emb_dim = time_emb_dim
        self.class_emb_dim = class_emb_dim

        if use_attn is None:
            use_attn = [False] * len(channel_mults)

        # 1) Time embedding (sinusoidal → MLP → time_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 2) Class embedding: (n_classes + 1) → class_emb_dim
        #    We reserve index = n_classes for “null / unconditional” embedding
        self.class_embedding = nn.Embedding(n_classes + 1, class_emb_dim)

        # 3) Initial “stem” convolution: from (3 + time_dim + class_dim) → base_channels
        self.init_conv = nn.Conv2d(3 + time_emb_dim + class_emb_dim, base_channels, kernel_size=3, padding=1)

        # 4) Build down‐sampling blocks
        #    At each level i: we have n_res_blocks ResConv blocks (with constant channels),
        #    then optionally self‐attention, then an avg‐pool (except last).
        down_blocks = []
        in_ch = base_channels
        self.down_sizes = []  # for skip connections
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(n_res_blocks):
                down_blocks.append(ResConvBlock(in_ch, out_ch, out_ch))
                in_ch = out_ch
            if use_attn[i]:
                # A simple 1×1 self‐attention block on in_ch channels
                down_blocks.append(nn.GroupNorm(num_groups=8, num_channels=in_ch))
                down_blocks.append(nn.ReLU(inplace=True))
                down_blocks.append(
                    nn.Conv2d(in_ch, in_ch * 3, kernel_size=1)
                )  # project to Q,K,V (stacked)
                down_blocks.append(_SimpleSelfAttention(in_ch))
            if i < len(channel_mults) - 1:
                # downsample
                down_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
            # store number of channels at this level for skip
            self.down_sizes.append(in_ch)

        self.down_seq = nn.Sequential(*down_blocks)

        # 5) Bottleneck (middle) ResConv blocks
        mid_ch = in_ch
        self.mid_block1 = ResConvBlock(mid_ch, mid_ch, mid_ch)
        self.mid_block2 = ResConvBlock(mid_ch, mid_ch, mid_ch)

        # 6) Build up‐sampling blocks (reverse of down)
        up_blocks = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # For each res_block + skip, channels double for concatenation
            for _ in range(n_res_blocks):
                # input channels = (current in_ch + skip channels)
                up_blocks.append(ResConvBlock(in_ch + self.down_sizes[i], in_ch + self.down_sizes[i], out_ch))
                in_ch = out_ch
            if use_attn[i]:
                up_blocks.append(nn.GroupNorm(num_groups=8, num_channels=in_ch))
                up_blocks.append(nn.ReLU(inplace=True))
                up_blocks.append(
                    nn.Conv2d(in_ch, in_ch * 3, kernel_size=1)
                )
                up_blocks.append(_SimpleSelfAttention(in_ch))
            if i > 0:
                # upsample
                up_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.up_seq = nn.Sequential(*up_blocks)

        # 7) Final “head” conv: from in_ch → 3 output channels
        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        self.final_act = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, lamb: torch.Tensor, cond: torch.LongTensor):
        """
        x:    [B, 3, H, W]        (noisy image in [-1,1])
        lamb: [B] float           (log‐SNR values)
        cond: [B] long            (class indices 0..n_classes for conditional, or = n_classes for “null”)
        returns:
          ε̂ of shape [B, 3, H, W]
        """
        B, _, H, W = x.shape
        # 1) Build time embedding: [B, time_emb_dim]
        t_emb = self.time_mlp(lamb)  # [B, time_emb_dim]

        # 2) Build class embedding: [B, class_emb_dim]
        c_emb = self.class_embedding(cond)  # [B, class_emb_dim]

        # 3) Expand both to [B, time_emb_dim, H, W] and [B, class_emb_dim, H, W]
        t_planes = expand_to_planes(t_emb, x.shape)  # [B, time_emb_dim, H, W]
        c_planes = expand_to_planes(c_emb, x.shape)  # [B, class_emb_dim, H, W]

        # 4) Concat input + t_planes + c_planes along channels → [B, 3 + time_emb + class_emb, H, W]
        h = torch.cat([x, t_planes, c_planes], dim=1)

        # 5) Initial conv → base_channels
        h = self.init_conv(h)

        # 6) Down‐path, storing skip connections
        skips = []
        idx = 0
        for module in self.down_seq:
            h = module(h)
            # Whenever we see an AvgPool2d, store the feature just BEFORE pooling as a skip
            if isinstance(module, nn.AvgPool2d):
                # store the feature BEFORE downsampling for skip‐connection
                skips.append(prev_h)
            prev_h = h

        # If no final pooling, store last as skip too
        if len(skips) < len(self.down_sizes):
            skips.append(h)

        # 7) Bottleneck
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # 8) Up‐path: for each ResConvBlock, concat with corresponding skip
        skip_idx = len(skips) - 1
        for module in self.up_seq:
            if isinstance(module, ResConvBlock):
                # pop skip
                skip_h = skips[skip_idx]
                skip_idx -= 1
                h = torch.cat([h, skip_h], dim=1)
                h = module(h)
            else:
                h = module(h)  # e.g. Upsample or attention or norm/act

        # 9) Final conv head → 3 channels
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.final_conv(h)

        return out  # [B, 3, H, W]


# ------------------------------------------------------------------------------
#   (Optional) A tiny Self‐Attention block to place inside U-Net if use_attn=True
#   This is a very minimal “1×1 conv → split Q,K,V → scaled dot‐prod → out”
# ------------------------------------------------------------------------------

class _SimpleSelfAttention(nn.Module):
    """
    A minimal 1×1 self‐attention block that operates on [B, C, H, W]:
      - Applies GroupNorm+ReLU (handled outside), then a 1×1 conv that produces 3×C channels.
      - Splits into Q,K,V each [B, C, H*W], does scaled dot‐prod (softmax), then reprojects to [B, C, H, W].
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # After projecting to 3×C via a 1×1 conv, we'll reproject back to C:
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # compute Q,K,V via a single 1×1 conv (assumed already inserted before this block)
        qkv = x  # shape [B, 3*C, H, W] if called right after the conv
        # split channels:
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # each [B, C, H, W]

        # reshape to [B, C, H*W]
        q = q.view(B, C, H * W)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W)

        # scaled dot‐product attention:
        attn = torch.einsum("bci, bcj -> bij", q, k) * (C ** (-0.5))  # [B, H*W, H*W]
        attn = attn.softmax(dim=-1)  # [B, H*W, H*W]

        # apply to v:
        out = torch.einsum("bij, bcj -> bci", attn, v)  # [B, C, H*W]
        out = out.view(B, C, H, W)  # [B, C, H, W]

        # final 1×1 projection:
        out = self.output_proj(out)  # [B, C, H, W]
        return out