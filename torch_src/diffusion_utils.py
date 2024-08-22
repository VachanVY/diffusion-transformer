import torch
from torch import nn, Tensor
from tqdm import trange
from typing import Optional

class DiffusionUtils:
    def __init__(self, config):
        self.num_timesteps = config.num_timesteps # (nT,)
        self.beta_range = config.var_range
        self.device = config.device
        self.get_alpha()

        self.H, self.W = config.H, config.W
        self.in_channels = config.in_channels

    def get_alpha(self):
        self.beta = torch.linspace(start=self.beta_range[0], end=self.beta_range[1], steps=self.num_timesteps, device=self.device) # (nT,)
        self.alpha = (1-self.beta) # (nT,)
        self.alpha_bar = torch.concatenate(
            (torch.tensor([1.], device=self.device), self.alpha.cumprod(axis=0)),
            axis=0
        ) # (nT,)
    
    def noisy_it(self, X:Tensor, t:Tensor): # (B, H, W, C), (B,)
        noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=self.device) # (B,)

        alpha_bar_t = self.alpha_bar[t][:, None, None, None] # (B, 1, 1, 1) <= (B,) <= (nT,)
        return {
            "noisy_images": torch.sqrt(alpha_bar_t)*X + torch.sqrt(1 - alpha_bar_t) * noise,
            "timesteps": t
        }, noise
    
    def one_step_ddpm(self, xt:Tensor, pred_noise:Tensor, t:int):
        alpha_t, alpha_bar_t = self.alpha[t, None, None, None], self.alpha_bar[t, None, None, None]
        xt_minus_1 = (
            (1/torch.sqrt(alpha_t))
            *
            (xt - (1-alpha_t)*pred_noise/torch.sqrt(1-alpha_bar_t)
            ) + torch.sqrt(self.beta[t])*torch.normal(mean=0.0, std=1.0, size=xt.shape, device=self.device)
        )
        return xt_minus_1
    
    def one_step_ddim(self, xt:Tensor, pred_noise:Tensor, t:int) -> Tensor:
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self, *,
        model:nn.Module, # x:Tensor, # (B, C = 3 or 1, H, W) t:Tensor, # (B,) y:tp.Optional[Tensor]=None, # (B,) key:tp.Optional[Tensor]=None
        labels:Optional[int]=None,
        # num_samples:int=1, # B # idk, doesnt work for more than 1, TODO: fix it
        use_ddim:bool=False, # False for now until implemented
    ):
        # assert len(labels) == num_samples if labels is not None else True
        sample_func = self.one_step_ddim if use_ddim else self.one_step_ddpm

        print(f"Generating images", "" if labels is None else "of " + str(labels))
        labels = torch.tensor(labels, device=self.device) if labels is not None else None
        x = torch.normal(mean=0.0, std=1.0, size=(1, self.in_channels, self.H, self.W), device=self.device)

        for i in trange(0, self.num_timesteps-1):
            t = torch.tensor([self.num_timesteps - i - 1]*1, device=self.device) # (B,)
            noise = model(x, t, labels)
            x = sample_func(x, noise, t)
        return x