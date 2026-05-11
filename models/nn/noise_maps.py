import torch
import torch.nn.functional as F

# Gaussian smoothing kernel (2D)
def gaussian_kernel_2d(kernel_size=7, sigma=1.5, channels=1):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def residual_noise_map_2d(x, kernel_size=7, sigma=1.5):
    B, C, H, W = x.shape
    kernel = gaussian_kernel_2d(kernel_size, sigma, C).to(x.device)
    smoothed = F.conv2d(x, kernel, padding=kernel_size//2, groups=C)
    residual = torch.abs(x - smoothed)
    residual = residual.mean(dim=1, keepdim=True)  # → [B, 1, H, W]

    return residual

def local_variance_map_2d(x, kernel_size=7):
    B, C, H, W = x.shape
    weight = torch.ones((C,1,kernel_size,kernel_size), device=x.device) / (kernel_size**2)
    local_mean = F.conv2d(x, weight, padding=kernel_size//2, groups=C)
    local_sq_mean = F.conv2d(x**2, weight, padding=kernel_size//2, groups=C)
    local_var = local_sq_mean - local_mean**2
    local_var = torch.sqrt(torch.clamp(local_var, min=1e-8))  # [B,C,H,W]
    local_var = local_var.mean(dim=1, keepdim=True)  # → [B,1,H,W]
    return local_var

def residual_noise_map_context(context, kernel_size=7, sigma=1.5):
    """
    context: [B, L, C, H, W]
    returns: [B, L, 1, H, W]
    """
    B, L, C, H, W = context.shape
    context = context.view(B * L, C, H, W)  # merge B and L
    
    # Gaussian kernel
    kernel = gaussian_kernel_2d(kernel_size, sigma, C).to(context.device)
    
    smoothed = F.conv2d(context, kernel, padding=kernel_size//2, groups=C)
    residual = torch.abs(context - smoothed)  # [B*L, C, H, W]
    residual = residual.mean(dim=1, keepdim=True)  # → [B*L, 1, H, W]
    
    return residual.view(B, L, 1, H, W)


def local_variance_map_context(context, kernel_size=7):
    """
    context: [B, L, C, H, W]
    returns: [B, L, 1, H, W]
    """
    B, L, C, H, W = context.shape
    context = context.view(B * L, C, H, W)  # merge B and L
    
    weight = torch.ones((C,1,kernel_size,kernel_size), device=context.device) / (kernel_size**2)
    local_mean = F.conv2d(context, weight, padding=kernel_size//2, groups=C)
    local_sq_mean = F.conv2d(context**2, weight, padding=kernel_size//2, groups=C)
    local_var = local_sq_mean - local_mean**2
    local_var = torch.sqrt(torch.clamp(local_var, min=1e-8))  # [B*L, C, H, W]
    local_var = local_var.mean(dim=1, keepdim=True)  # → [B*L, 1, H, W]
    
    return local_var.view(B, L, 1, H, W)
