import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiScaleGradient(nn.Module):

    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales

        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[[[-1., 0., 1.],
                                                   [-2., 0., 2.],
                                                   [-1., 0., 1.]]]], dtype=torch.float32)
        self.sobel_x.weight.requires_grad = False

        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y.weight.data = torch.tensor([[[[-1., -2., -1.],
                                                   [0., 0., 0.],
                                                   [1., 2., 1.]]]], dtype=torch.float32)
        self.sobel_y.weight.requires_grad = False

    def forward(self, image):
        gradients = []
        for scale in self.scales:
            if scale > 1:
                scaled_image = F.interpolate(image, scale_factor=1 / scale, mode='bilinear', align_corners=False)
            else:
                scaled_image = image

            gx = self.sobel_x(scaled_image)
            gy = self.sobel_y(scaled_image)
            grad = torch.sqrt(gx ** 2 + gy ** 2)

            if scale > 1:
                grad = F.interpolate(grad, size=image.shape[2:], mode='bilinear', align_corners=False)

            gradients.append(grad)

        return torch.cat(gradients, dim=1)


class ImprovedCrossAttn(nn.Module):

    def __init__(self, embed_dim, num_heads=4, block_size=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query, key, value):
        seq_len = query.size(0)
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        attn_outs = []

        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq_len)

            q_block = query[start:end]
            k_block = key[start:end]
            v_block = value[start:end]
            attn_out, _ = self.attention(q_block, k_block, v_block)
            q_block = q_block + attn_out
            q_block = self.norm1(q_block)
            ffn_out = self.ffn(q_block)
            q_block = q_block + ffn_out
            q_block = self.norm2(q_block)
            attn_outs.append(q_block)
        return torch.cat(attn_outs, dim=0)


class AdaptiveFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, img_grad, v_grad):
        concat = torch.cat([img_grad, v_grad], dim=1)
        weights = self.conv_weight(concat)
        fused = weights[:, 0:1] * img_grad + weights[:, 1:2] * v_grad
        return fused

class HSVProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.learnable_k = nn.Parameter(torch.tensor([0.5]))

    def forward(self, rgb_img):
        r, g, b = rgb_img[:, 0:1], rgb_img[:, 1:2], rgb_img[:, 2:3]
        v, _ = torch.max(rgb_img, dim=1, keepdim=True)
        epsilon = 1e-8
        compressed_v = torch.tanh(self.learnable_k * v) / (torch.tanh(self.learnable_k) + epsilon)
        return compressed_v


class ImprovedGradientFusionModule(nn.Module):
    def __init__(self, img_channels=3, v_grad_channels=2, fused_channels=3):
        super().__init__()
        self.rgb_to_gray = nn.Conv2d(img_channels, 1, kernel_size=1, stride=1, bias=False)
        self.rgb_to_gray.weight.data = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        self.rgb_to_gray.weight.requires_grad = False
        self.hsv_processor = HSVProcessor()
        self.multi_scale_grad = MultiScaleGradient(scales=[1, 2])
        self.v_proj = nn.Sequential(
            nn.Conv2d(v_grad_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1)
        )
        self.cross_attn = ImprovedCrossAttn(embed_dim=3, num_heads=3, block_size=256)
        self.adaptive_fusion = AdaptiveFusion(channels=3)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, fused_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.detail_enhancement = nn.Sequential(
            nn.Conv2d(fused_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, fused_channels, kernel_size=1)
        )

    def forward(self, rgb_img, v_grad):
        B, _, H, W = rgb_img.shape
        compressed_v = self.hsv_processor(rgb_img)
        gray = self.rgb_to_gray(rgb_img)
        img_grad = self.multi_scale_grad(gray)
        img_grad = img_grad.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        v_grad_enhanced = self.v_proj(v_grad)
        img_flat = img_grad.view(B, 3, -1).permute(2, 0, 1)  # (H*W, B, 3)
        v_flat = v_grad_enhanced.view(B, 3, -1).permute(2, 0, 1)  # (H*W, B, 3)
        attn_out = self.cross_attn(img_flat, v_flat, v_flat)
        attn_out = attn_out.permute(1, 2, 0).view(B, 3, H, W)
        adaptive_fused = self.adaptive_fusion(img_grad, v_grad_enhanced)
        fused_input = torch.cat([attn_out, adaptive_fused], dim=1)
        fused = self.fuse_conv(fused_input)
        fused = fused * (1.0 + compressed_v)
        detail_enhanced = self.detail_enhancement(fused)
        fused = fused + detail_enhanced
        return fused
