import os
import matplotlib.pyplot as plt

import torch

from torch_src.diffusion_transformer import get_diffution_transformer
from torch_src.diffusion_utils import DiffusionUtils
from torch_src.config import CelebA_config


diff_utils = DiffusionUtils(CelebA_config)
MODEL_PATH = "checkpoints/celeba/ckpt.pt"

if not os.path.exists(MODEL_PATH):
    from checkpoints.celeba.download_celeba_ckpt import CKPT_URL
    raise FileNotFoundError(
        F"""Model weights not found at {MODEL_PATH}
        Run `python checkpoints/celeba/download_celeba_ckpt.py` to download the weights
        Or \033[1mmanually download weights to directory {MODEL_PATH.split('ckpt')[0]}\033[0m from {CKPT_URL}
        """
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ctx = torch.autocast(
    DEVICE.type,
    torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
)

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    print("Loading model...", end=" ")
    model = get_diffution_transformer(
        config=CelebA_config,
        compile=True,
        input_shape=(2, CelebA_config.in_channels, CelebA_config.H, CelebA_config.W)
    ).to(DEVICE)
    print("model loaded")
    if args.ema:
        model.load_state_dict(ckpt["ema_state"]); model.eval()
    else:
        model.load_state_dict(ckpt["model_state"]); model.eval()
    ckpt = None

    print("Generating samples...")
    for idx, lbl in enumerate(args.labels):
        with ctx:
            gen_img = diff_utils.generate(
                model=model,
                labels=[1 if lbl=="male" else 0]
            ).detach().squeeze().add(1.0).div(2.0).permute(1, 2, 0).clip(0, 1).cpu().numpy()
            plt.imshow(gen_img)
            plt.title(lbl)
            plt.show(block=False)
            plt.pause(interval=4)
            plt.savefig(os.path.join(args.save_dir, f"sample_{args.path_idx+idx}.png"))
            plt.axis("off")
            plt.close()
    print("Done")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, nargs='+')
    parser.add_argument("--save_dir", type=str, default="images/")
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--path_idx", type=int, default=time.time())
    args = parser.parse_args()

    main(args)