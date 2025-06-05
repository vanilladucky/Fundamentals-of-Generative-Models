# Referenced from https://github.com/KimRass/CFG/blob/main/unet.py

import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, time_channels, max_lambda=12.0, min_lambda=-12.0, num_frequencies=64):
        super().__init__()
        self.time_channels = time_channels
        self.num_frequencies = num_frequencies
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        freq_bands = 2 ** torch.arange(num_frequencies).float()  # [num_frequencies]

        self.register_buffer("freq_bands", freq_bands)  # [num_frequencies]

        # After computing sin/cos for all frequencies, we get a vector sized = 2 * num_frequencies.
        # Use an MLP to upsample to `time_channels`.
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, time_channels),
            Swish(),
            nn.Linear(time_channels, time_channels)
        )

    def forward(self, lamb: torch.Tensor):
        """
        lamb: a shape [B] float‐tensor containing real λ values (in the range [min_lambda, max_lambda]).
        Returns: a [B, time_channels] embedding.
        """
        # 1) Clamp λ to [min_lambda, max_lambda], then scale to [0, 1]:
        lamb_clamped = lamb.clamp(self.min_lambda, self.max_lambda)
        # Optionally, normalize to [0, 1]:
        lamb_norm = (lamb_clamped - self.min_lambda) / (self.max_lambda - self.min_lambda)  # [B]
        
        # 2) Expand to shape [B, num_frequencies], multiply by freq_bands to get real argument
        #    For each λ_i:  x_k = λ_norm_i * freq_bands[k].  (Choice of scaling is somewhat arbitrary, but typical.)
        #    Then compute sin(x_k) and cos(x_k).
        B = lamb_norm.shape[0]
        freqs = self.freq_bands.view(1, -1)       # [1, num_frequencies]
        x = lamb_norm.view(-1, 1) * freqs         # [B, num_frequencies]

        # 3) Build the sinusoidal vector of length 2 * num_frequencies:
        sin_emb = torch.sin(x)                    # [B, num_frequencies]
        cos_emb = torch.cos(x)                    # [B, num_frequencies]
        pe = torch.cat([sin_emb, cos_emb], dim=1) # [B, 2*num_frequencies]

        # 4) Pass through the MLP to get [B, time_channels]
        out = self.mlp(pe)                        # [B, time_channels]
        return out


class ResConvSelfAttn(nn.Module):
    def __init__(self, channels, n_groups=32):
        super().__init__()

        self.gn = nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        self.qkv_proj = nn.Conv2d(channels, channels * 3, 1, 1, 0)
        self.out_proj = nn.Conv2d(channels, channels, 1, 1, 0)
        self.scale = channels ** (-0.5)

    def forward(self, x):
        b, c, h, w = x.shape
        skip = x

        x = self.gn(x)
        x = self.qkv_proj(x)
        q, k, v = torch.chunk(x, chunks=3, dim=1)
        attn_score = torch.einsum(
            "bci,bcj->bij", q.view((b, c, -1)), k.view((b, c, -1)),
        ) * self.scale
        attn_weight = F.softmax(attn_score, dim=2)        
        x = torch.einsum("bij,bcj->bci", attn_weight, v.view((b, c, -1)))
        x = x.view(b, c, h, w)
        x = self.out_proj(x)
        return x + skip


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_channels, attn=False, n_groups=32, drop_prob=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn = attn

        self.layers1 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
        self.time_proj = nn.Sequential(
            Swish(),
            nn.Linear(time_channels, out_channels),
        )
        self.layers2 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        else:
            self.conv = nn.Identity()

        if attn:
            self.attn_block = ResConvSelfAttn(out_channels)
        else:
            self.attn_block = nn.Identity()

    def forward(self, x, t):
        skip = x
        x = self.layers1(x)
        x = x + self.time_proj(t)[:, :, None, None]
        x = self.layers2(x)
        x = x + self.conv(skip)
        return self.attn_block(x)


class Downsample(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(channels, channels, 3, 2, 1)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_classes,
        channels=128,
        channel_mults=[1, 2, 4, 8],
        attns=[False, False, True, False],
        n_res_blocks=3,
    ):
        super().__init__()

        self.n_classes = n_classes

        assert all([i < len(channel_mults) for i in attns]), "attns index out of bound"

        time_channels = channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_channels=time_channels)
        )
        self.label_embed = nn.Embedding(n_classes + 1, time_channels)

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        cxs = [channels]
        cur_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            for _ in range(n_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        attn=attns[i]
                    )
                )
                cur_channels = out_channels
                cxs.append(cur_channels)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(cur_channels))
                cxs.append(cur_channels)

        self.mid_blocks = nn.ModuleList([
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=True,
            ),
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=False,
            ),
        ])

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = channels * mult
            for _ in range(n_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(
                        in_channels=cxs.pop() + cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        attn=attns[i],
                    )
                )
                cur_channels = out_channels
            if i != 0:
                self.up_blocks.append(Upsample(cur_channels))
        assert len(cxs) == 0

        self.fin_block = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            Swish(),
            nn.Conv2d(cur_channels, 3, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step, label="null"):
        x = self.init_conv(noisy_image)
        t = self.time_embed(diffusion_step)

        B = noisy_image.shape[0]
        device = noisy_image.device

        if label is None:
            label_tensor = torch.full((B,), fill_value=self.n_classes, dtype=torch.long, device=device)
        else:
            label_tensor = label
        y = self.label_embed(label_tensor)   
        c = t + y

        xs = [x]
        for layer in self.down_blocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, c)
            xs.append(x)

        for layer in self.mid_blocks:
            x = layer(x, c)

        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, c)
        assert len(xs) == 0
        return self.fin_block(x)


if __name__ == "__main__":
    model = UNet(n_classes=10)

    noisy_image = torch.randn(4, 3, 32, 32)
    diffusion_step = torch.randint(0, 1000, size=(4,))
    label = torch.randint(0, 10, size=(4,))
    out = model(noisy_image, diffusion_step, label)
    out.shape