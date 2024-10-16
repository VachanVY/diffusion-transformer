from dataclasses import dataclass
import torch

@dataclass
class MNIST_config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"

    # Diffusion Args
    var_range:tuple[int, int] = (1e-4, 2e-2)
    num_timesteps:int = 400

    # Vit Args
    patch_size:int = 2
    H:int = 28
    W:int = 28
    in_channels:int = 1
    out_channels:int = in_channels
    N:int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # transformer Args
    d_model:int = 288
    num_heads:int = 6
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 6
    num_classes:int = 10
    dropout_rate:float = 0.0
    maxlen:int = N

    # Training Args
    cfg_dropout_rate:float = 0.0

    batch_size:int = 64
    num_steps:int = 20_000
    decay_steps:int = num_steps
    warmup_steps:int = 100
    max_lr:float = 1e-3
    min_lr:float = 0.0*max_lr
    no_decay:bool = False # for max_lr = 1e-3, if kept True, model forgets everything in middle of training
    beta1:float = 0.9
    beta2:float = 0.999
    clipnorm:float = None
    weight_decay:float = 0.0
    ema:float = 0.999
    
    patience:int = 10
    num_grad_accumalation_steps:int = 1
    checkpoint_dir:str = "checkpoints"
    return_best_train_states:bool = True
    log_interval:int = 50
    eval_freq:int = 400


@dataclass
class CelebA_config:
    """38.78 Million Parameters || 280ms per step on 4090"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_type = "bfloat16" if torch.cuda.is_available() else "float32"

    # Diffusion Args
    var_range:tuple[int, int] = (1e-4, 2e-2)
    num_timesteps:int = 1000

    # Vit Args
    patch_size:int = 4 # maybe change to 2
    H:int = 128
    W:int = 128
    in_channels:int = 3
    out_channels:int = in_channels
    N:int = H*W//patch_size**2
    assert N*patch_size**2 == H*W

    # transformer Args
    d_model:int = 512
    num_heads:int = 8
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    num_layers:int = 8
    num_classes:int = 2
    dropout_rate:float = 0.0
    maxlen:int = N

    # Training Args
    cfg_dropout_rate:float = 0.1

    batch_size:int = 112 # fits on my 4090; change accordingly
    num_steps:int = 100_000 # in paper, they trained for 400k steps | Generated images will be good
    decay_steps:int = num_steps
    warmup_steps:int = 200
    max_lr:float = 1e-4 # model forgot everything in middle of training for maxlr 3e-4 and warmup steps 700
    min_lr:float = 0.0*max_lr
    no_decay:bool = False # TODO: Got to try training with no_decay=True, like in the paper. Does loss shot up?
    beta1:float = 0.9
    beta2:float = 0.999
    clipnorm:float = None
    weight_decay:float = 0.0
    ema_momentum:float = 0.9999
    
    patience:int = 10
    num_grad_accumalation_steps:int = 1
    ckpt_dir:str = "checkpoints/celeba"
    return_best_train_states:bool = True
    log_interval:int = 1
    eval_freq:int = 400

    init_from:str = "scratch" # "resume" or "scratch"


def loss_vs_lr(config:CelebA_config|MNIST_config):
    from diffusion_transformer import get_diffution_transformer
    from diffusion_utils import DiffusionUtils

    model = get_diffution_transformer(
        config=config,
        compile=True,
        input_shape=(2, config.in_channels, config.H, config.W)
    ).cuda()
    diff_utils = DiffusionUtils(config)

    opt = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    lrs = (10**torch.linspace(-6, -2, 100)).tolist()
    lrs = [lr for lr in lrs for _ in range(2)]

    X_batch = torch.normal(
        mean=-0.88, std=1.98832, size=(4, config.in_channels, config.H, config.W)
    ).cuda().clip(-1, 1)

    ctx = torch.autocast("cuda", torch.bfloat16)

    def get_loss(lr:float):
        opt.zero_grad()
        timesteps = torch.randint(0, config.num_timesteps, (4,)).cuda()
        noisy_image_timestep, noise_true = diff_utils.noisy_it(X_batch, timesteps)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        y = torch.randint(0, config.num_classes, size=(4,))
        with ctx:
            noise_pred = model(noisy_image_timestep["noisy_images"].cuda(), noisy_image_timestep["timesteps"].cuda(), y.cuda())
            loss = torch.nn.functional.mse_loss(noise_pred, noise_true)
        loss.backward()
        opt.step()
        return loss.cpu().item()
    
    losses = [get_loss(lr) for lr in lrs]
    return losses, lrs


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    losses, lrs = loss_vs_lr(CelebA_config)
    plt.figure(figsize=(15,5))
    plt.xlabel("Log base10 Learning Rate: Do 10^(x) to get actual learning rate")
    plt.ylabel("Loss")
    plt.ylim(0.0, 1.5)
    plt.xticks([-6, -5, -4]+torch.arange(-3.5, -2.0, 0.1).tolist())
    plt.plot(torch.log10(torch.tensor(lrs)), losses)
    plt.savefig("images/loss_vs_lr.png")
    plt.show()

    # lrs             log10(lrs)
    # 3.162278e-06, -5.500000e+00
    # 3.981072e-06, -5.400000e+00
    # 5.011872e-06, -5.300000e+00
    # 6.309573e-06, -5.200000e+00
    # 7.943282e-06, -5.100000e+00
    # 1.000000e-05, -5.000000e+00
    # 1.258925e-05, -4.900000e+00
    # 1.584893e-05, -4.800000e+00
    # 1.995262e-05, -4.700000e+00
    # 2.511886e-05, -4.600000e+00
    # 3.162278e-05, -4.500000e+00
    # 3.981072e-05, -4.400000e+00
    # 5.011872e-05, -4.300000e+00
    # 6.309573e-05, -4.200000e+00
    # 7.943282e-05, -4.100000e+00
    # 1.000000e-04, -4.000000e+00 <==== Select this based on the graph
    # 1.258925e-04, -3.900000e+00
    # 1.584893e-04, -3.800000e+00
    # 1.995262e-04, -3.700000e+00
    # 2.511886e-04, -3.600000e+00

    # seeing the graph, and the above table set max_lr to 1e-4

