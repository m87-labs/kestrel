"""SAM-style mask refiner head with two-way transformer."""

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


class MaskEncoder(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()
        self.down1 = nn.Sequential(
            ConvBlock(1, decoder_dim // 4, kernel_size=3),
            ConvBlock(decoder_dim // 4, decoder_dim // 4, kernel_size=3),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(decoder_dim // 4, decoder_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(32, decoder_dim // 2), decoder_dim // 2),
            nn.GELU(),
            ConvBlock(decoder_dim // 2, decoder_dim // 2, kernel_size=3),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(decoder_dim // 2, decoder_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(32, decoder_dim), decoder_dim),
            nn.GELU(),
            ConvBlock(decoder_dim, decoder_dim, kernel_size=3),
        )

    def forward(self, mask, target_size):
        start_size = target_size * 4
        x = F.interpolate(mask, size=(start_size, start_size), mode="bilinear", align_corners=False)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.mask_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mask_self_norm = nn.LayerNorm(dim)

        self.mask_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mask_cross_norm = nn.LayerNorm(dim)

        self.mask_mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.mask_mlp_norm = nn.LayerNorm(dim)

        self.img_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.img_cross_norm = nn.LayerNorm(dim)

    def forward(self, img_tokens, mask_tokens):
        mask_normed = self.mask_self_norm(mask_tokens)
        mask_tokens = mask_tokens + self.mask_self_attn(mask_normed, mask_normed, mask_normed)[0]

        mask_tokens = mask_tokens + self.mask_cross_attn(
            self.mask_cross_norm(mask_tokens),
            img_tokens,
            img_tokens,
        )[0]

        mask_tokens = mask_tokens + self.mask_mlp(self.mask_mlp_norm(mask_tokens))

        img_tokens = img_tokens + self.img_cross_attn(
            self.img_cross_norm(img_tokens),
            mask_tokens,
            mask_tokens,
        )[0]

        return img_tokens, mask_tokens


class SAMHead(nn.Module):
    def __init__(self, enc_dim=2304, decoder_dim=256, num_heads=8):
        super().__init__()
        self.enc_dim = enc_dim
        self.decoder_dim = decoder_dim

        self.proj = nn.Conv2d(enc_dim, decoder_dim, 1)

        self.mask_encoder = MaskEncoder(decoder_dim)

        self.transformer_27 = TwoWayAttentionBlock(decoder_dim, num_heads)
        self.transformer_54 = TwoWayAttentionBlock(decoder_dim, num_heads)
        self.transformer_108 = TwoWayAttentionBlock(decoder_dim, num_heads)

        self.up1 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim),
            ConvBlock(decoder_dim, decoder_dim),
            ResBlock(decoder_dim),
            ResBlock(decoder_dim),
        )
        self.up2 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim),
            ConvBlock(decoder_dim, decoder_dim),
            ResBlock(decoder_dim),
            ResBlock(decoder_dim),
        )
        self.up3 = nn.Sequential(
            ConvBlock(decoder_dim + 1, decoder_dim // 2),
            ConvBlock(decoder_dim // 2, decoder_dim // 2),
            ResBlock(decoder_dim // 2),
            ResBlock(decoder_dim // 2),
        )
        self.up4 = nn.Sequential(
            ConvBlock(decoder_dim // 2 + 1, decoder_dim // 2),
            ConvBlock(decoder_dim // 2, decoder_dim // 4),
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
        )

        self.refine = nn.Sequential(
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
            ResBlock(decoder_dim // 4),
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

        # Resolution 27x27
        img_tokens = x.flatten(2).transpose(1, 2)
        mask_enc_27 = self.mask_encoder(coarse_mask, 27)
        mask_tokens = mask_enc_27.flatten(2).transpose(1, 2)
        img_tokens, _ = self.transformer_27(img_tokens, mask_tokens)
        x = img_tokens.transpose(1, 2).view(B, -1, 27, 27)

        # Upsample 27 -> 54
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_54 = F.interpolate(coarse_mask, size=(54, 54), mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, mask_54], dim=1))

        # Resolution 54x54
        img_tokens = x.flatten(2).transpose(1, 2)
        mask_enc_54 = self.mask_encoder(coarse_mask, 54)
        mask_tokens = mask_enc_54.flatten(2).transpose(1, 2)
        img_tokens, _ = self.transformer_54(img_tokens, mask_tokens)
        x = img_tokens.transpose(1, 2).view(B, -1, 54, 54)

        # Upsample 54 -> 108
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_108 = F.interpolate(coarse_mask, size=(108, 108), mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, mask_108], dim=1))

        # Resolution 108x108
        img_tokens = x.flatten(2).transpose(1, 2)
        mask_enc_108 = self.mask_encoder(coarse_mask, 108)
        mask_tokens = mask_enc_108.flatten(2).transpose(1, 2)
        img_tokens, _ = self.transformer_108(img_tokens, mask_tokens)
        x = img_tokens.transpose(1, 2).view(B, -1, 108, 108)

        # Upsample 108 -> 216
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        mask_216 = F.interpolate(coarse_mask, size=(216, 216), mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, mask_216], dim=1))

        # Upsample 216 -> target
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, coarse_mask], dim=1))

        x = self.refine(x)
        delta_logits = self.out(x)
        return coarse_logits + delta_logits


class SAMHeadRefiner:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.head = SAMHead(enc_dim=1152, decoder_dim=128)

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
