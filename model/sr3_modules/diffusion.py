import math
import torch
import cv2
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import sys
from model.improved_gradient_fusion import ImprovedGradientFusionModule
from model.wight_balance import process_tensor_image

sys.path.append('model/sr3_modules')

def stretch_channel(channel):
    p_low, p_high = np.percentile(channel, 0.5), np.percentile(channel, 99.5)
    return np.clip((channel - p_low) / (p_high - p_low + 1e-6), 0, 1)


def color_balance_simplified(img_float_rgb):
    r, g, b = cv2.split(img_float_rgb)
    return cv2.merge([stretch_channel(r), stretch_channel(g), stretch_channel(b)])


def enhance_contrast_clahe(v_channel_float):
    v_uint8 = (v_channel_float * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe_uint8 = clahe.apply(v_uint8)
    return v_clahe_uint8.astype(np.float32) / 255.0


def enhance_saturation_adaptive(s_channel, v_channel, gain_factor=1.5):
    v_weight = np.exp(-((v_channel - 0.5) ** 2) / (2 * (0.25 ** 2)))
    s_enhanced = s_channel * (1 + gain_factor * v_weight)
    return np.clip(s_enhanced, 0, 1)


def apply_enhancement_pipeline(img_rgb_float_hwc, gain_factor=1.1):
    balanced_rgb = color_balance_simplified(img_rgb_float_hwc)
    balanced_hsv = cv2.cvtColor(balanced_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(balanced_hsv)
    V_enhanced = enhance_contrast_clahe(V)
    S_enhanced = enhance_saturation_adaptive(S, V_enhanced, gain_factor=gain_factor)
    final_hsv = cv2.merge([H, S_enhanced, V_enhanced])
    final_rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    final_rgb_clipped = np.clip(final_rgb, 0, 1)

    return final_rgb_clipped

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def rgb_to_hsv_torch(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    maxc = torch.max(rgb, dim=1)[0]
    minc = torch.min(rgb, dim=1)[0]
    delta = maxc - minc
    eps = 1e-6
    h = torch.zeros_like(maxc)
    mask = delta != 0
    idx = (maxc == r) & mask
    h[idx] = ((g[idx] - b[idx]) / (delta[idx] + eps)) % 6.0
    idx = (maxc == g) & mask
    h[idx] = (((b[idx] - r[idx]) / (delta[idx] + eps)) + 2.0) % 6.0
    idx = (maxc == b) & mask
    h[idx] = (((r[idx] - g[idx]) / (delta[idx] + eps)) + 4.0) % 6.0
    h = h / 6.0
    s = torch.zeros_like(maxc)
    idx = (maxc != 0)
    s[idx] = delta[idx] / (maxc[idx] + eps)
    v = maxc
    hsv = torch.stack([h, s, v], dim=1)  # [B, 3, H, W]
    return hsv


class VGradientGuidance(nn.Module):
    def __init__(self):
        super(VGradientGuidance, self).__init__()
        self.register_buffer('kernel_x',
                             torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).float().view(1, 1, 3, 3))
        self.register_buffer('kernel_y',
                             torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).float().view(1, 1, 3, 3))

    def forward(self, rgb_img):
        B, C, H, W = rgb_img.shape
        assert C == 3, "Input must have 3 channels"
        hsv = rgb_to_hsv_torch(rgb_img)
        V_channel = hsv.select(dim=1, index=2).unsqueeze(1)
        Gx = torch.nn.functional.conv2d(V_channel, self.kernel_x, padding=1)
        Gy = torch.nn.functional.conv2d(V_channel, self.kernel_y, padding=1)
        G_mag = torch.abs(Gx) + torch.abs(Gy)
        new_guidance = torch.cat([Gx, Gy, G_mag], dim=1)
        return new_guidance

class EnhancedVGradientGuidance(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        self.register_buffer('kernel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.grad_processor = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, rgb_img):
        hsv = rgb_to_hsv_torch(rgb_img)
        v_channel = hsv[:, 2:3, :, :]
        Gx = F.conv2d(v_channel, self.kernel_x, padding=1)
        Gy = F.conv2d(v_channel, self.kernel_y, padding=1)
        grad = torch.cat([Gx, Gy], dim=1)
        enhanced_grad = self.grad_processor(grad)
        return enhanced_grad

CrossAttnGradientFusion = ImprovedGradientFusionModule

class HSColorGuidance(nn.Module):
    def __init__(self, grad_scale=0.5):
        super().__init__()
        self.grad_scale = grad_scale
        self.register_buffer('kernel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.color_attn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # 输入H+S+梯度信息
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1)  # 输出H/S增强系数
        )

    def forward(self, rgb_img):
        hsv = rgb_to_hsv_torch(rgb_img)
        h_channel = hsv[:, 0:1, :, :]
        s_channel = hsv[:, 1:2, :, :]
        h_grad_x = F.conv2d(h_channel, self.kernel_x, padding=1)
        h_grad_y = F.conv2d(h_channel, self.kernel_y, padding=1)
        s_grad_x = F.conv2d(s_channel, self.kernel_x, padding=1)
        s_grad_y = F.conv2d(s_channel, self.kernel_y, padding=1)

        color_feat = torch.cat([h_channel, s_channel,
                                torch.abs(h_grad_x + h_grad_y),
                                torch.abs(s_grad_x + s_grad_y)], dim=1)

        color_weights = torch.sigmoid(self.color_attn(color_feat)) * self.grad_scale
        return color_weights

class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            dynamic_color_guidance_config={'beta': 0.0, 'k': 3.0}
    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.cond_proj = nn.Conv2d(9, 6, kernel_size=3, padding=1)
        self.guidancetrue = APGM2()
        self.dynamic_color_guidance_config = dynamic_color_guidance_config

        if schedule_opt is not None:
            pass

    def get_dynamic_color_weight(self, t, T):
        beta = self.dynamic_color_guidance_config.get('beta', 0.0)
        if beta == 0:
            return 0.0

        k = self.dynamic_color_guidance_config.get('k', 3.0)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t] if isinstance(t, int) else t, device=self.betas.device)

        t_float = t.float()
        T_float = float(T)

        sigmoid_arg = -k * (t_float / T_float - 0.5)
        weight = torch.sigmoid(sigmoid_arg)

        return beta * weight.view(-1, 1, 1, 1)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, text_proj=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(
            x.device)

        if condition_x is not None:
            fused_condition, color_guide_for_dynamic = self.guidancetrue(condition_x)

            predicted_noise = self.denoise_fn(torch.cat([fused_condition, x], dim=1), noise_level)

            x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)
        else:
            predicted_noise = self.denoise_fn(x, noise_level)
            x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        dynamic_weight = self.get_dynamic_color_weight(t, self.num_timesteps)
        if torch.is_tensor(dynamic_weight) or dynamic_weight > 0:
            x_recon = x_recon + dynamic_weight * color_guide_for_dynamic
            if clip_denoised:
                x_recon.clamp_(-1., 1.)
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, text_proj=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, text_proj=text_proj)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, condition_x=None, text_proj=None):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:

            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps , disable=False):
                img = self.p_sample(img, i, condition_x=x, text_proj=text_proj)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, text_proj=None):
        return self.p_sample_loop(x_in, continous, text_proj=text_proj)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def q_predict_start_from_noise(self, x_t, continuous_sqrt_alpha_cumprod, noise=None):
        assert x_t.shape == noise.shape, "Dimension error"
        return (
                (1. / continuous_sqrt_alpha_cumprod) * x_t - (
                    ((1 - continuous_sqrt_alpha_cumprod ** 2).sqrt()) * (1. / continuous_sqrt_alpha_cumprod) * noise)
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape

        t_int = np.random.randint(1, self.num_timesteps + 1)
        t = torch.full((b,), t_int - 1, device=x_start.device, dtype=torch.long)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t_int - 1],
                self.sqrt_alphas_cumprod_prev[t_int],
                size=b
            )
        ).to(x_start.device).view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise
        )

        if not self.conditional:
            predicted_noise = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            color_guide_for_dynamic = None
        else:
            fused_condition, color_guide_for_dynamic = self.guidancetrue(x_in['SR'])
            predicted_noise = self.denoise_fn(torch.cat([fused_condition, x_noisy], dim=1),
                                              continuous_sqrt_alpha_cumprod)

        x0_hat = self.q_predict_start_from_noise(
            x_noisy, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), predicted_noise
        )
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        dynamic_weight = self.get_dynamic_color_weight(t, self.num_timesteps)
        if (torch.is_tensor(dynamic_weight) or dynamic_weight > 0) and color_guide_for_dynamic is not None:
            x0_hat_corrected = x0_hat + dynamic_weight * color_guide_for_dynamic
            x0_hat_corrected = torch.clamp(x0_hat_corrected, -1.0, 1.0)
        else:
            x0_hat_corrected = x0_hat
        original_loss = self.loss_func(noise, predicted_noise)
        return x0_hat_corrected, original_loss

    def forward(self, x, *args, **kwargs):
            return self.p_losses(x, *args, **kwargs)

class APGM_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_gradient_guidance = EnhancedVGradientGuidance(out_channels=2)
        self.grad_fusion = ImprovedGradientFusionModule(img_channels=3, v_grad_channels=2, fused_channels=3)
        self.hs_guidance = HSColorGuidance()
        self.color_adaptor = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x_in):
        with torch.no_grad():
            batch_size = x_in.shape[0]
            device = x_in.device
            x_in_np_01 = (x_in.cpu().detach().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0
            processed_images_np = [apply_enhancement_pipeline(img_np, gain_factor=1.1) for img_np in x_in_np_01]
            corrected_img_np = np.stack(processed_images_np, axis=0)
            corrected_img_base = torch.from_numpy(corrected_img_np).permute(0, 3, 1, 2).float().to(device) * 2.0 - 1.0

        v_grad = self.v_gradient_guidance(corrected_img_base)
        structure_enhancement = self.grad_fusion(corrected_img_base, v_grad)

        color_weights = self.hs_guidance(corrected_img_base)
        color_correction = self.color_adaptor(color_weights)
        alpha = 0.1
        beta = 0.05
        fused_guide = x_in + alpha * structure_enhancement + beta * color_correction
        final_guide = torch.clamp(fused_guide, -1.0, 1.0)
        return final_guide


class APGM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_gradient_guidance = EnhancedVGradientGuidance(out_channels=2)
        self.grad_fusion = ImprovedGradientFusionModule(img_channels=3, v_grad_channels=2, fused_channels=3)
        self.hs_guidance = HSColorGuidance()
        self.color_adaptor = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x_in):
        with torch.no_grad():
            batch_size = x_in.shape[0]
            device = x_in.device
            x_in_np_01 = (x_in.cpu().detach().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0
            processed_images_np = [apply_enhancement_pipeline(img_np, gain_factor=1.1) for img_np in x_in_np_01]
            corrected_img_np = np.stack(processed_images_np, axis=0)
            corrected_img_base = torch.from_numpy(corrected_img_np).permute(0, 3, 1, 2).float().to(device) * 2.0 - 1.0
        v_grad = self.v_gradient_guidance(corrected_img_base)
        structure_guide = self.grad_fusion(corrected_img_base, v_grad)
        color_weights = self.hs_guidance(corrected_img_base)
        color_guide = self.color_adaptor(color_weights)
        alpha = 0.1
        beta = 0.01
        structurally_enhanced_img = x_in + alpha * structure_guide
        fused_condition = structurally_enhanced_img * (1 + beta * color_guide)
        fused_condition = torch.clamp(fused_condition, -1.0, 1.0)

        return fused_condition, color_guide


class HierarchicalProjectionCrossAttention(nn.Module):
    def __init__(self, text_dim=512, img_channels=3, scales=[(240, 320), (120, 160), (60, 80)]):
        super().__init__()
        self.device = torch.device('cuda')
        self.v_gradient_guidance = EnhancedVGradientGuidance(out_channels=2).to(
            self.device)
        self.grad_fusion = CrossAttnGradientFusion(img_channels=3, v_grad_channels=2, fused_channels=3).to(
            self.device)
        self.hs_guidance = HSColorGuidance().to(self.device)
        self.color_adaptor = nn.Sequential(
            nn.Conv2d(2, 3, 3, padding=1),
            nn.Tanh()
        ).to(self.device)


    def forward(self, x_in):
        batch_size = x_in.shape[0]
        device = x_in.device
        x_in_np = x_in.cpu().detach().permute(0, 2, 3, 1).numpy()
        processed_images = []
        for i in range(batch_size):
            enhanced_img_np = apply_enhancement_pipeline(x_in_np[i], gain_factor=1.1)
            processed_images.append(enhanced_img_np)
        corrected_img_np = np.stack(processed_images, axis=0)
        corrected_img = torch.from_numpy(corrected_img_np).permute(0, 3, 1, 2).float().to(device)

        v_grad = self.v_gradient_guidance(corrected_img)
        rgb_grad = self.grad_fusion(corrected_img, v_grad)
        color_weights = self.hs_guidance(corrected_img)
        color_correction = self.color_adaptor(color_weights)
        condition_x = x_in + 0.01 * corrected_img * rgb_grad * color_correction  
        return condition_x
