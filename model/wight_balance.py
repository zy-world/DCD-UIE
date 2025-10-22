import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def enhance_image_with_paper_method_numpy(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

    V = np.maximum(np.maximum(R, G), B)
    S = np.zeros_like(V)
    min_rgb = np.minimum(np.minimum(R, G), B)
    non_zero_v_mask = V != 0

    V_safe = np.where(non_zero_v_mask, V, 1)
    S[non_zero_v_mask] = 1 - min_rgb[non_zero_v_mask] / V_safe[non_zero_v_mask]

    H = np.zeros_like(V)
    mask_s_zero = (S == 0)
    mask_v_r = (V == R) & ~mask_s_zero
    mask_v_g = (V == G) & ~mask_s_zero
    mask_v_b = (V == B) & ~mask_s_zero

    denominator = S * V
    safe_denominator = np.where(denominator != 0, denominator, 1)

    H[mask_s_zero] = 0
    H[mask_v_r] = (60 * (G[mask_v_r] - B[mask_v_r]) / safe_denominator[mask_v_r]) + 360
    H[mask_v_g] = (60 * (B[mask_v_g] - R[mask_v_g]) / safe_denominator[mask_v_g]) + 120
    H[mask_v_b] = (60 * (R[mask_v_b] - G[mask_v_b]) / safe_denominator[mask_v_b]) + 240
    H = H % 360

    S_255 = S * 255
    S_min, S_max = np.min(S_255), np.max(S_255)
    S_H = np.zeros_like(S_255)
    if S_max > S_min:
        S_H = (255 / (S_max - S_min)) * (S_255 - S_min)

    V_min, V_max = np.min(V), np.max(V)
    V_H = np.zeros_like(V)
    if V_max > V_min:
        V_H = (255 / (V_max - V_min)) * (V - V_min)

    S_norm = S_H / 255.0
    V_norm = V_H / 255.0
    C = V_norm * S_norm
    H_prime = H / 60.0
    X = C * (1 - np.abs(H_prime % 2 - 1))
    m = V_norm - C
    R_p, G_p, B_p = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)

    mask1 = (H >= 0) & (H < 60)
    mask2 = (H >= 60) & (H < 120)
    mask3 = (H >= 120) & (H < 180)
    mask4 = (H >= 180) & (H < 240)
    mask5 = (H >= 240) & (H < 300)
    mask6 = (H >= 300) & (H < 360)

    R_p[mask1], G_p[mask1], B_p[mask1] = C[mask1], X[mask1], 0
    R_p[mask2], G_p[mask2], B_p[mask2] = X[mask2], C[mask2], 0
    R_p[mask3], G_p[mask3], B_p[mask3] = 0, C[mask3], X[mask3]
    R_p[mask4], G_p[mask4], B_p[mask4] = 0, X[mask4], C[mask4]
    R_p[mask5], G_p[mask5], B_p[mask5] = X[mask5], 0, C[mask5]
    R_p[mask6], G_p[mask6], B_p[mask6] = C[mask6], 0, X[mask6]

    final_R = R_p + m
    final_G = G_p + m
    final_B = B_p + m

    final_rgb = np.stack([final_R, final_G, final_B], axis=2)
    final_rgb = np.clip(final_rgb, 0, 1)
    final_rgb_uint8 = (final_rgb * 255).astype(np.uint8)

    final_bgr = cv2.cvtColor(final_rgb_uint8, cv2.COLOR_RGB2BGR)
    return final_bgr

def process_tensor_image(img_tensor: torch.Tensor) -> torch.Tensor:
    if img_tensor is None or img_tensor.nelement() == 0:
        print("警告: process_tensor_image 收到一个空的或 None 的张量。")
        return img_tensor

    device = img_tensor.device
    batch_size = img_tensor.shape[0]
    processed_images = []

    for i in range(batch_size):
        single_img_tensor = img_tensor[i]
        img_norm_0_1 = (single_img_tensor.clamp(-1, 1) + 1) / 2.0
        img_hwc = img_norm_0_1.permute(1, 2, 0)
        img_rgb_uint8 = (img_hwc.detach().cpu().numpy() * 255).astype(np.uint8)
        img_bgr_uint8 = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
        enhanced_bgr_uint8 = enhance_image_with_paper_method_numpy(img_bgr_uint8)
        enhanced_rgb_uint8 = cv2.cvtColor(enhanced_bgr_uint8, cv2.COLOR_BGR2RGB)
        enhanced_tensor_hwc = torch.from_numpy(enhanced_rgb_uint8)
        enhanced_tensor_chw = enhanced_tensor_hwc.permute(2, 0, 1)
        enhanced_tensor_final = (enhanced_tensor_chw.float() / 255.0) * 2.0 - 1.0
        processed_images.append(enhanced_tensor_final)

    output_tensor = torch.stack(processed_images).to(device)

    return output_tensor