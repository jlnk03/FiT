import json
import os
import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm
from preprocess.iterators import create_dataloader_imagenet_preprocessing
import argparse


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate latent space representations")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config.get("latent_folder", '../latent')
    os.makedirs(save_dir, exist_ok=True)

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    # Build dataloader
    dataloader = create_dataloader_imagenet_preprocessing(config)

    # Run inference
    records = list()
    for img, path in tqdm(dataloader):
        path = path[0]
        outdir = os.path.abspath(os.path.join(
            save_dir, os.path.basename(os.path.dirname(path))))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        dest = os.path.join(outdir, os.path.splitext(
            os.path.basename(path))[0] + ".npy")
        if os.path.isfile(dest):
            continue
        with torch.no_grad():
            enc = vae.encode(img.to(device)).latent_dist.sample() * 0.18215
        latent = enc.detach().cpu().numpy().astype(np.float16)[0]
        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
