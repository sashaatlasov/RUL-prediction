import torch
from torch import nn
from .eps_theta import ConditionalGuidedModel
from .diffusion import DiffusionModel
from rulpred.transformer.DAST_support import DAST

class RULpredictor(nn.Module):
    
    def __init__(self,
                 window_size,
                 betas=(1e-4, 0.02),
                 num_timesteps=100
    ):
        super().__init__()
        
        self.transformer = DAST(window_size + 2, 14, 4)
        
        self.eps_theta = ConditionalGuidedModel()
        
        self.diffusion = DiffusionModel(self.eps_theta, betas=betas, num_timesteps=num_timesteps)
    
        
    def forward(self, target, sensors):
        """
        Parameters
        ----------
        target
            (batch_size, target_dim)
        sensors
            (batch_size, window_size, input_size)  
            
        Returns
        -------
        loss
            loss for a batch
        """
        conds = self.transformer(sensors)
        loss = self.diffusion(target=target.reshape(-1, 1), conds=conds)
        
        return loss
    
    @torch.no_grad()
    def sample(self, sensors, num_traj=100):
        """
        Parameters
        ----------
        sensors
            (batch_size, window_size, input_size)  
        """
        
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=num_traj, dim=dim)
        
        conds = self.transformer(sensors)
        repeated_conds = repeat(conds, dim=0)
        
        new_samples = self.diffusion.sample(repeated_conds)


        return new_samples.reshape(
            (
                -1,
                num_traj,
                1,
                1,
            )
        )