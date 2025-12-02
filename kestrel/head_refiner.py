"""Mask refiner head for lightweight mask refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class MaskRefinerHead(nn.Module):
    def __init__(self, enc_dim=1152, decoder_dim=128):
        super().__init__()
        self.enc_dim = enc_dim
        self.decoder_dim = decoder_dim

        self.proj = nn.Conv2d(enc_dim, decoder_dim, 1)

        self.up1 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim),
            ConvBlock(decoder_dim, decoder_dim),
        )
        self.up2 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim),
            ConvBlock(decoder_dim, decoder_dim),
        )
        self.up3 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim),
            ConvBlock(decoder_dim, decoder_dim // 2),
        )
        self.up4 = nn.Sequential(
            ConvBlock(decoder_dim // 2 + 1, decoder_dim // 2),
            ConvBlock(decoder_dim // 2, decoder_dim // 4),
        )

        self.refine = nn.Sequential(
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
        )

        self.out = nn.Conv2d(decoder_dim // 4, 1, 1)

    def forward(self, features, coarse_mask):
        B, _, H, W = coarse_mask.shape
        coarse_logits = torch.logit(coarse_mask.clamp(0.01, 0.99))

        x = features.view(B, 27, 27, self.enc_dim).permute(0, 3, 1, 2)
        x = self.proj(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_down = F.interpolate(coarse_mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, mask_down], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_down = F.interpolate(coarse_mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, mask_down], dim=1))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_down = F.interpolate(coarse_mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, mask_down], dim=1))

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, coarse_mask], dim=1))

        x = self.refine(x)

        delta_logits = self.out(x)
        return coarse_logits + delta_logits


class HeadRefiner:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.head = MaskRefinerHead(enc_dim=1152, decoder_dim=128)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.head.load_state_dict(ckpt["head"])
        self.head = self.head.to(device).to(torch.bfloat16)
        self.head.eval()

    @torch.no_grad()
    def __call__(self, features, mask, n_iters=6):
        current_mask = mask
        for _ in range(n_iters):
            logits = self.head(features, current_mask)
            current_mask = torch.sigmoid(logits)
        return current_mask

    def parameters(self):
        return self.head.parameters()
