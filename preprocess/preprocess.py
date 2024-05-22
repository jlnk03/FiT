import json
import logging
import os

import numpy as np
import torch

from tqdm import tqdm
from diffusers import AutoencoderKL

from logger import set_logger

from imagenet_dataset import create_dataloader_imagenet_preprocessing, create_dataloader_imagenet_latent

logger = logging.getLogger(__name__)

config = dict()


def get_latent_dataset():
    return create_dataloader_imagenet_latent(config)


if __name__ == "__main__":
    save_dir = config.get("data_folder", '../dataset')
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 2 vae
    logger.info("vae init")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)

    # 3. build dataloader
    dataloader = create_dataloader_imagenet_preprocessing(config)

    # 4. run inference
    records = list()
    for img, path in tqdm(dataloader.create_tuple_iterator(num_epochs=1), total=len(dataloader)):
        path = path.asnumpy().item()
        outdir = os.path.abspath(os.path.join(save_dir, os.path.basename(os.path.dirname(path))))

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        dest = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".npy")

        if os.path.isfile(dest):
            continue

        enc = vae.encode(torch.from_numpy(img.asnumpy())).latent_dist.sample() * 0.18215
        latent = enc.detach().numpy().astype(np.float16)[0]

        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
