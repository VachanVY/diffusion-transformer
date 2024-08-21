import requests
from tqdm import tqdm


CKPT_URL = "https://huggingface.co/vachanvy/Diffusion_Transformer/resolve/main/celeba/ckpt.pt"
def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    download_file(CKPT_URL, "checkpoints/celeba/ckpt.pt")
    print("Downloaded checkpoint to checkpoints/celeba/ckpt.pt")