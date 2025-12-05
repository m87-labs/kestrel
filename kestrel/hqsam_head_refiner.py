"""HQ-SAM style mask refiner with multi-scale feature fusion."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MaskEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, embed_dim // 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, embed_dim // 16),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 16, embed_dim // 8, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, embed_dim // 8), embed_dim // 8),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(16, embed_dim // 4), embed_dim // 4),
            nn.GELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
            nn.GELU(),
        )

    def forward(self, mask):
        x = self.conv1(mask)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.interpolate(x, size=(27, 27), mode="bilinear", align_corners=False)
        return x


class HQFeatureFusion(nn.Module):
    def __init__(self, enc_dim=1152, embed_dim=256):
        super().__init__()
        self.early_proj = nn.Sequential(
            nn.Linear(enc_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
        )

    def forward(self, early_feat, final_feat):
        B = early_feat.shape[0]
        early = self.early_proj(early_feat)
        early = early.transpose(1, 2).view(B, -1, 27, 27)
        fused = torch.cat([early, final_feat], dim=1)
        return self.fusion(fused)


class TwoWayBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.output_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, output_tokens, img_tokens, output_pe, img_pe):
        q = k = output_tokens + output_pe
        attn_out = self.output_self_attn(q, k, output_tokens)[0]
        output_tokens = self.norm1(output_tokens + attn_out)

        q = output_tokens + output_pe
        k = img_tokens + img_pe
        attn_out = self.cross_attn_token_to_image(q, k, img_tokens)[0]
        output_tokens = self.norm2(output_tokens + attn_out)

        output_tokens = self.norm3(output_tokens + self.mlp(output_tokens))

        q = img_tokens + img_pe
        k = output_tokens + output_pe
        attn_out = self.cross_attn_image_to_token(q, k, output_tokens)[0]
        img_tokens = self.norm4(img_tokens + attn_out)

        return output_tokens, img_tokens


class HQSAMHead(nn.Module):
    def __init__(self, enc_dim=1152, embed_dim=256, num_heads=8, num_layers=2, num_masks=4):
        super().__init__()
        self.enc_dim = enc_dim
        self.embed_dim = embed_dim
        self.num_masks = num_masks

        self.image_proj = nn.Conv2d(enc_dim, embed_dim, 1)
        self.image_pe = nn.Parameter(torch.randn(1, embed_dim, 27, 27) * 0.02)

        self.hq_fusion = HQFeatureFusion(enc_dim, embed_dim)
        self.mask_encoder = MaskEncoder(embed_dim)

        self.mask_tokens = nn.Parameter(torch.randn(num_masks, embed_dim) * 0.02)
        self.iou_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.output_pe = nn.Parameter(torch.randn(num_masks + 1, embed_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TwoWayBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.final_token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 8),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 8, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 8),
            nn.GELU(),
        )

        self.final_q_proj = nn.Linear(embed_dim, embed_dim // 8)
        self.final_attn_up = nn.MultiheadAttention(embed_dim // 8, num_heads=4, batch_first=True)
        self.final_norm_up = nn.LayerNorm(embed_dim // 8)

        self.mask_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim // 8, embed_dim // 8),
                nn.GELU(),
                nn.Linear(embed_dim // 8, embed_dim // 8),
                nn.GELU(),
                nn.Linear(embed_dim // 8, embed_dim // 8),
            )
            for _ in range(num_masks)
        ])

        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_masks),
        )

    def forward(self, final_features, early_features, coarse_mask):
        B = final_features.shape[0]
        H, W = coarse_mask.shape[-2:]

        img = final_features.view(B, 27, 27, self.enc_dim).permute(0, 3, 1, 2)
        img = self.image_proj(img)

        img = self.hq_fusion(early_features, img)

        mask_embed = self.mask_encoder(coarse_mask)
        img = img + mask_embed

        img_tokens = img.flatten(2).transpose(1, 2)
        img_pe = self.image_pe.flatten(2).transpose(1, 2).expand(B, -1, -1)

        output_tokens = torch.cat([
            self.mask_tokens.unsqueeze(0).expand(B, -1, -1),
            self.iou_token.unsqueeze(0).expand(B, -1, -1),
        ], dim=1)
        output_pe = self.output_pe.unsqueeze(0).expand(B, -1, -1)

        for block in self.transformer_blocks:
            output_tokens, img_tokens = block(output_tokens, img_tokens, output_pe, img_pe)

        q = output_tokens + output_pe
        k = img_tokens + img_pe
        attn_out = self.final_token_to_image(q, k, img_tokens)[0]
        output_tokens = self.final_norm(output_tokens + attn_out)

        img_up = img_tokens.transpose(1, 2).view(B, self.embed_dim, 27, 27)
        img_up = self.upsample(img_up)

        img_up_tokens = img_up.flatten(2).transpose(1, 2)
        mask_tokens = output_tokens[:, :self.num_masks]
        mask_tokens_proj = self.final_q_proj(mask_tokens)
        mask_tokens_refined = mask_tokens_proj + self.final_attn_up(
            self.final_norm_up(mask_tokens_proj), img_up_tokens, img_up_tokens
        )[0]

        masks = []
        for i in range(self.num_masks):
            mask_weights = self.mask_mlps[i](mask_tokens_refined[:, i])
            mask = torch.einsum("bc,bchw->bhw", mask_weights, img_up)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)

        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

        iou_token = output_tokens[:, self.num_masks]
        iou_pred = self.iou_head(iou_token)

        return masks, iou_pred


class HQSAMHeadRefiner:
    def __init__(self, device="cuda"):
        self.device = device
        self.head = HQSAMHead(enc_dim=1152, embed_dim=256, num_masks=4)

        weights_path = hf_hub_download(
            repo_id="moondream/SegHeadRefiner",
            filename="model.pt",
            token=os.environ.get("HF_TOKEN"),
        )
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        self.head.load_state_dict(ckpt["head"])
        self.head = self.head.to(device).to(torch.bfloat16)
        self.head.eval()

    @torch.no_grad()
    def __call__(self, final_features, early_features, mask, n_iters=6):
        current_mask = mask
        for _ in range(n_iters):
            all_logits, iou_pred = self.head(final_features, early_features, current_mask)
            best_idx = iou_pred.argmax(dim=1)
            batch_idx = torch.arange(all_logits.shape[0], device=all_logits.device)
            best_logits = all_logits[batch_idx, best_idx]
            current_mask = torch.sigmoid(best_logits).unsqueeze(1)
        return current_mask

    def parameters(self):
        return self.head.parameters()
