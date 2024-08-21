import random
import time
import matplotlib.pyplot as plt; plt.style.use('dark_background') 
import dataclasses as dc
from collections import OrderedDict
from copy import deepcopy
import os

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.utils.data

from torch_src.diffusion_transformer import get_diffution_transformer, DiT
from torch_src.diffusion_utils import DiffusionUtils
from torch_src.config import CelebA_config as config
from torch_src.utils import CosineDecayWithWarmup


class CelebMaleFemale(torch.utils.data.IterableDataset):
    def __init__(
        self, *, 
        transforms:transforms.Compose, 
        split:str, 
    ):
        super().__init__()
        self.transforms = transforms
        self.split = split

    def male_or_female(self, one_hot_tensor:Tensor):
        """| (1:male, 0: female) | (68_261, 94_509) |"""
        return one_hot_tensor[20]

    def __iter__(self):
        while True:
            ds = datasets.CelebA(
                root='data', split=self.split, target_type='attr',
                download=True, transform=self.transforms
            )
            for x, y in ds:
                yield x, self.male_or_female(y)


class DataLoader:
    transforms = transforms.Compose([
        # torchvision.transforms.Resize((config.H, config.W)),
        transforms.CenterCrop((config.H, config.W)),
        transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # [0.0, 1.0] -> [-1.0, 1.0]
    ])
    
    def iter_batches(self, batch_size:int):
        ds = CelebMaleFemale(transforms=self.transforms, split="train")
        # TODO: when num_workers > 0, returns duplicates, use worker id to avoid duplicates
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, drop_last=True,
            num_workers=0, prefetch_factor=None
        )

        for x, y in dl:
            x = x.to(config.device)
            y = y.to(config.device)
            yield x, y

@torch.no_grad()
def update_ema(ema_model:DiT, model:DiT, decay:float):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1-decay)


@torch.no_grad()
def evaluate(model:nn.Module):
    model.eval()
    lbl = random.randint(0, config.num_classes-1)
    gen_image = diff_utils.generate(model=model, labels=[lbl]).detach().cpu()
    model.train()
    gen_image = gen_image.squeeze().permute(1, 2, 0).add(1.0).div(2.0) # [-1, 1] -> [0, 1]
    print("max", gen_image.max(), "min", gen_image.min())
    return gen_image, lbl


def train(model:DiT, ema:DiT, losses:list[float]):
    global start_iter
    t0 = time.time()
    losses = []
    for step in range(start_iter, config.num_steps):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (step % config.eval_freq == 0 and step > 0) or step == config.num_steps-1:
            gen_image, lbl = evaluate(model)
            plt.imshow(gen_image.numpy(), cmap="gray")
            plt.savefig(f"images/gen_{step}.png")
            plt.title("male" if lbl == 1 else "female")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            checkpoint = {
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_args": model_args,
                "step": step,
                "train_config": dc.asdict(config()),
                "losses": losses
            }
            
            print(f"Saving checkpoint to {config.ckpt_dir} ...", end=" ==> ")
            torch.save(checkpoint, os.path.join(config.ckpt_dir, "ckpt.pt"))
            print("Done.")

        optimizer.zero_grad(set_to_none=True)
        X_batch, y_batch = next(train_iterator)

        timesteps = torch.randint(
            size=(X_batch.shape[0],), low=0, high=config.num_timesteps, device=config.device
        )
        noisy_image_timestep, noise_true = diff_utils.noisy_it(X_batch, timesteps)
        
        with ctx:
            noise_pred = model(noisy_image_timestep["noisy_images"], noisy_image_timestep["timesteps"], y_batch)
            loss = F.mse_loss(noise_pred, noise_true)
        loss.backward()
        if config.clipnorm is not None:
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clipnorm, error_if_nonfinite=True)
        optimizer.step()

        update_ema(ema, model, config.ema_momentum)

        t1 = time.time()
        dt = t1-t0
        t0 = t1
        lossf = loss.cpu().item()
        if step % config.log_interval == 0:
            print(
                f"| Step: {step} || Loss: {lossf:.4f} |"
                f"| LR: {lr:e} || dt: {dt*1000:.2f}ms |", end="")
            print(f"| Norm: {norm:.4f} |" if config.clipnorm is not None else "")
        losses.append(lossf)
    return losses


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_iterator = iter(DataLoader().iter_batches(batch_size=config.batch_size))

    get_lr = CosineDecayWithWarmup(
        warmup_steps=config.warmup_steps,
        max_learning_rate=config.max_lr,
        decay_steps=config.decay_steps,
        min_learning_rate=config.min_lr
    ) if not config.no_decay else lambda _: config.max_lr

    diff_utils = DiffusionUtils(config)

    ctx = torch.autocast(
            device_type=config.device.type,
            dtype={"bfloat16": torch.bfloat16,
                "float32" : torch.float32}[config.dtype_type]
        )
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    losses, accuracies, mask_accuracies, start_iter = [], [], [], 0
    COMPILE = True # sppeeeddd impressive
    if "scratch" in config.init_from:
        model_config = config
        model_args = dc.asdict(config())
        model:DiT = get_diffution_transformer(
            config=model_config,
            compile=COMPILE,
            input_shape=(2, config.in_channels, config.H, config.W)
        )
        ema:DiT = deepcopy(model)

        best_val_loss = 1e9
        checkpoint = None
    elif "resume" in config.init_from:
        print("Resuming training using checkpoint...")
        def get_model_state(state_dict:dict[str, Tensor]):
            unwanted_prefix = "_orig_mod." # this prefix gets added when a loaded Model was compiled
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            return state_dict
                
        ckpt_path = os.path.join(config.ckpt_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=config.device)

        model_args = checkpoint["model_args"]
        model_config = config(**model_args)
        start_iter = checkpoint["step"]

        model:DiT = get_diffution_transformer(
            config=model_config,
            compile=COMPILE,
            input_shape=(2, config.in_channels, config.H, config.W)
        )
        ema:DiT = deepcopy(model)
        
        model.load_state_dict(checkpoint["model_state"])
        ema.load_state_dict(checkpoint["ema_state"])

        best_val_loss = checkpoint["best_val_loss"]
        losses, accuracies, mask_accuracies = checkpoint["losses"], checkpoint["accuracies"], checkpoint["mask_accuracies"]

    model.to(config.device, dtype=torch.float32); model.train()
    ema.to(config.device, dtype=torch.float32); ema.eval() # use ema for sampling
    ema.requires_grad_(False)
    update_ema(ema, model, 0.0) # copy the weights

    print("\nDiffusion Transformer:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "Million Parameters\n")

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device_type=config.device.type
    )

    if ("resume" in config.init_from) and ("optimizer" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None # free memory

    print("Training...")
    losses = train(model, ema, losses)
    print("Done")

    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs Steps")
    plt.savefig("images/loss_vs_steps.png")
    plt.show(block=False)
    plt.pause(interval=5)
    plt.close()

    
    for model_string, _model in {"model": model, "ema": ema}.items():
        print("Generating images using", model_string)
        model.eval()
        for blahh in range(10):
            for i in range(config.num_classes):
                gen_image = diff_utils.generate(model=_model, labels=[i]).add(1.0).div(2.0).clip(0, 1) # [-1, 1] -> [0, 1]
                plt.imshow(gen_image.detach().squeeze().permute(1, 2, 0).cpu().numpy(), cmap="gray")
                plt.title("male" if i == 1 else "female")
                plt.show(block=False)
                plt.pause(interval=2)
                plt.savefig(f"images/{model_string}_gen_{blahh}.png")
                plt.close()
        print("Succesfull")
