import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from model.losses import DecoupledLoss
from .base_model import BaseModel
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np
from model.improved_gradient_fusion import ImprovedGradientFusionModule

logger = logging.getLogger('base')

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


def white_balance(img, eps=1e-4, alpha=1.0, use_local=False, patch_size=64, sigma=1.0, gamma=1.0):
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    if sigma > 0:
        img_np = img.permute(0, 2, 3, 1).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        for b in range(img_np.shape[0]):
            img_np[b] = gaussian_filter(img_np[b], sigma=sigma)
        img = torch.from_numpy(img_np).permute(0, 3, 1, 2).to(img.device)

    R, G, B = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

    if use_local:
        B, _, H, W = img.shape
        corrected_img = torch.zeros_like(img)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch_R = R[:, i:i + patch_size, j:j + patch_size]
                patch_G = G[:, i:i + patch_size, j:j + patch_size]
                patch_B = B[:, i:i + patch_size, j:j + patch_size]

                R_avg = patch_R.mean(dim=[1, 2], keepdim=True)
                G_avg = patch_G.mean(dim=[1, 2], keepdim=True)
                B_avg = patch_B.mean(dim=[1, 2], keepdim=True)

                Gray_avg = (R_avg + G_avg + B_avg) / 3

                R_avg = torch.clamp(R_avg, min=eps)
                G_avg = torch.clamp(G_avg, min=eps)
                B_avg = torch.clamp(B_avg, min=eps)

                R_corrected = patch_R * (Gray_avg / R_avg)
                G_corrected = patch_G * (Gray_avg / G_avg)
                B_corrected = patch_B * (Gray_avg / B_avg)

                corrected_img[:, 0, i:i + patch_size, j:j + patch_size] = R_corrected
                corrected_img[:, 1, i:i + patch_size, j:j + patch_size] = G_corrected
                corrected_img[:, 2, i:i + patch_size, j:j + patch_size] = B_corrected
    else:
        R_avg = R.mean(dim=[1, 2], keepdim=True)
        G_avg = G.mean(dim=[1, 2], keepdim=True)
        B_avg = B.mean(dim=[1, 2], keepdim=True)

        Gray_avg = (R_avg + G_avg + B_avg) / 3

        R_avg = torch.clamp(R_avg, min=eps)
        G_avg = torch.clamp(G_avg, min=eps)
        B_avg = torch.clamp(B_avg, min=eps)

        R_corrected = R * (Gray_avg / R_avg)
        G_corrected = G * (Gray_avg / G_avg)
        B_corrected = B * (Gray_avg / B_avg)

        corrected_img = torch.stack([R_corrected, G_corrected, B_corrected], dim=1)

    corrected_img = torch.clamp(corrected_img, min=0.0, max=1.0)

    if gamma != 1.0:
        corrected_img = torch.clamp(corrected_img, min=1e-8, max=1.0)
        corrected_img = corrected_img ** (1.0 / gamma)

    corrected_img = alpha * corrected_img + (1 - alpha) * img

    corrected_img = torch.clamp(corrected_img, 0, 1)
    corrected_img = torch.nan_to_num(corrected_img, nan=0.0, posinf=1.0, neginf=0.0)

    return corrected_img


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


class BlockCrossAttn(nn.Module):
    def __init__(self, embed_dim=1, num_heads=1, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        L = query.size(0)
        outputs = []
        for i in range(0, L, self.block_size):
            q_block = query[i:i + self.block_size]
            k_block = key[i:i + self.block_size]
            v_block = value[i:i + self.block_size]
            attn_out, _ = self.attn(q_block, k_block, v_block)
            outputs.append(attn_out)
        return torch.cat(outputs, dim=0)
CrossAttnGradientFusion = ImprovedGradientFusionModule

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        self.schedule_phase = None

        self.netG = self.set_device(networks.define_G(opt))

        if self.is_train:
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_loss(self.device)
            else:
                self.netG.set_loss(self.device)

        if self.is_train:
            loss_opt = opt['train']['loss']
            self.decoupled_loss_fn = DecoupledLoss(
                lambda_recon=loss_opt['lambda_recon'],
                lambda_structure=loss_opt['lambda_structure'],
                lambda_color=loss_opt['lambda_color']
            ).to(self.device)
            optim_params = list(self.netG.parameters())
            logger.info("Optimizer will manage all parameters from netG (including sub-modules like APGM).")
            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        x0_hat_corrected, original_loss = self.netG(self.data)
        x_start = self.data['HR']
        x_in = self.data['SR']
        total_loss, loss_dict = self.decoupled_loss_fn(
            enhanced_img=x0_hat_corrected,
            clear_img=x_start,
            hazy_img=x_in
        )
        total_loss.backward()
        self.optG.step()
        self.log_dict['l_total'] = total_loss.item()
        self.log_dict['l_rec'] = loss_dict.get('loss_recon', 0).item()
        self.log_dict['l_str'] = loss_dict.get('loss_structure', 0).item()
        self.log_dict['l_col'] = loss_dict.get('loss_color', 0).item()
        self.log_dict['l_pix (monitor)'] = original_loss.sum().item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.p_sample_loop(self.data['SR'], continous)
            else:
                self.SR = self.netG.p_sample_loop(self.data['SR'], continous)
        self.netG.train()

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(f'Network G structure: {net_struc_str}, with parameters: {n:,d}')
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_gen.pth')
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_opt.pth')

        network = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)

        opt_state = {'epoch': epoch, 'iter': iter_step, 'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)
        logger.info(f'Saved model in [{gen_path}] ...')

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(f'Loading pretrained model for G [{load_path}] ...')
            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'

            network = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
            network.load_state_dict(torch.load(gen_path), strict=False)

            if self.is_train:
                opt = torch.load(opt_path)
                if 'optimizer' in opt and opt['optimizer'] is not None:
                    try:
                        self.optG.load_state_dict(opt['optimizer'])
                    except ValueError as e:
                        logger.warning(f"Could not load optimizer state: {e}. Starting with a fresh optimizer.")

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']