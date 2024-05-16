import torch
import torch.nn as nn
import numpy as np


class DiffusionModel(nn.Module):
    def __init__(self, eps_model, betas, num_timesteps, loss='l2'):
        super().__init__()
        self.eps_model = eps_model

        for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
            self.register_buffer(name, schedule)

        self.num_timesteps = num_timesteps
        
        if loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError('loss type is not supported') 

    def forward(self, target, conds):
        device = target.device
        
        timestep = torch.randint(1, self.num_timesteps + 1, (target.shape[0], ), device=device)
        eps = torch.randn_like(target, device=device)
        x_t = (
            self.sqrt_alphas_cumprod[timestep, None] * target
            + self.sqrt_one_minus_alpha_prod[timestep, None] * eps
        )

        return self.criterion(eps, self.eps_model(conds, x_t, timestep))

    def sample(self, conds): # num_samples, size, 
        device = conds.device
        sample_shape = conds.shape[:1] + (1, ) # 1=traget_dim
        x_i = torch.randn(sample_shape, device=device)
        for i in range(self.num_timesteps, 0, -1):
            z = torch.randn(sample_shape, device=device) if i > 1 else 0
            eps = self.eps_model(conds, x_i, torch.tensor(i).repeat(sample_shape[0], ).to(device))
            x_rec = self.inv_sqrt_alphas[i] * (x_i - eps * self.sqrt_one_minus_alpha_prod[i]) 
            x_i = self.gamma0[i] * x_rec + self.gamma1[i] * x_i + self.posterior_variance[i] * z if i > 1 else x_rec

        return x_i


def get_schedules(beta1, beta2, num_timesteps):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    #betas = (beta2 - beta1) * torch.arange(0, num_timesteps + 1, dtype=torch.float32) / num_timesteps + beta1
    betas = torch.linspace(beta1, beta2, num_timesteps + 1) # linear
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
    one_minus_alpha_over_prod = (1 - alphas) / sqrt_one_minus_alpha_prod
    
    posterior_variance = torch.sqrt(
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
    gamma0 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    gamma1 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    return {
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
        "one_minus_alpha_over_prod": one_minus_alpha_over_prod,
        "posterior_variance": posterior_variance, 
        "gamma0": gamma0,
        "gamma1": gamma1
    }

