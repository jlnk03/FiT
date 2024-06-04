import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

from iterators import create_dataloader_imagenet_preprocessing

import matplotlib.pyplot as plt

config = dict()

'''def decode_latents(latents):
    #print(type(latents))
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True, num_groups=12)
    latents = np.load(latents)
    latents = torch.from_numpy(latents).to(torch.float32)
    print(latents.shape)
    latents = 1.0 / 0.18215 * latents
    latents = latents[None, :, :, :]
    print(type(latents))
    with torch.no_grad():
        image = vae.decode(latents).sample
    save_path = os.path.join('decoded/', f"decoded_image{np.random.randint(1000)}.npy")
    print(image.shape)
    #os.makedirs(save_path, exist_ok=True)
    #image.save(save_path)
    torch.save(image, save_path)

    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
    plt.show()'''
    

if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = config.get("latent_folder", '../latent_two')
    os.makedirs(save_dir, exist_ok=True)

    # 2 vae
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).to(device)

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

        if os.path.isfile(dest):
            continue

        with torch.no_grad():
            enc = vae.encode(img.to(device)).latent_dist.sample() * 0.18215

        latent = enc.detach().numpy().astype(np.float16)[0]

        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
