import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class DecoupledLoss(nn.Module):
    def __init__(self, lambda_recon=1.0, lambda_structure=1.0, lambda_color=0.5):
        super(DecoupledLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_structure = lambda_structure
        self.lambda_color = lambda_color
        self.l1_loss = nn.L1Loss()

        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.sobel_x = nn.Parameter(sobel_kernel_x.view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_kernel_y.view(1, 1, 3, 3), requires_grad=False)

    def _compute_l_gradient(self, lab_image):
        l_channel = lab_image[:, 0:1, :, :]
        grad_x = F.conv2d(l_channel, self.sobel_x, padding=1)
        grad_y = F.conv2d(l_channel, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, enhanced_img, clear_img, hazy_img):
        loss_dict = {}
        loss_recon = self.l1_loss(enhanced_img, clear_img)
        loss_dict['loss_recon'] = loss_recon * self.lambda_recon
        enhanced_0_1 = (enhanced_img.clamp(-1, 1) + 1.0) / 2.0
        clear_0_1 = (clear_img.clamp(-1, 1) + 1.0) / 2.0
        hazy_0_1 = (hazy_img.clamp(-1, 1) + 1.0) / 2.0
        lab_enhanced = kornia.color.rgb_to_lab(enhanced_0_1)
        lab_clear = kornia.color.rgb_to_lab(clear_0_1)
        lab_hazy = kornia.color.rgb_to_lab(hazy_0_1)
        grad_enhanced = self._compute_l_gradient(lab_enhanced)
        grad_clear = self._compute_l_gradient(lab_clear)
        loss_structure = self.l1_loss(grad_enhanced, grad_clear)
        loss_dict['loss_structure'] = loss_structure * self.lambda_structure
        ab_enhanced = lab_enhanced[:, 1:3, :, :]
        ab_clear = lab_clear[:, 1:3, :, :]
        ab_hazy = lab_hazy[:, 1:3, :, :]
        d_ec_color = self.l1_loss(ab_enhanced, ab_clear)
        d_eh_color = self.l1_loss(ab_enhanced, ab_hazy)
        loss_color = d_ec_color / (d_eh_color + 1e-6)
        loss_dict['loss_color'] = loss_color * self.lambda_color

        total_loss = loss_dict['loss_recon'] + loss_dict['loss_structure'] + loss_dict['loss_color']

        return total_loss, loss_dict
