import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from iterators import create_dataloader_imagenet_preprocessing

config = dict()

if __name__ == "__main__":
    save_dir = config.get("data_folder", '../dataset')
    os.makedirs(save_dir, exist_ok=True)

    # 2 vae
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)

    # 3. build dataloader
    dataloader = create_dataloader_imagenet_preprocessing(config)

    # 4. run inference
    records = list()
    for img, path in tqdm(dataloader):
        path = path[0]

        outdir = os.path.abspath(os.path.join(save_dir, os.path.basename(os.path.dirname(path))))

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        dest = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".npy")
        print(dest)

        if os.path.isfile(dest):
            continue

        with torch.no_grad():
            enc = vae.encode(img).latent_dist.sample() * 0.18215

        latent = enc.detach().numpy().astype(np.float16)[0]

        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
