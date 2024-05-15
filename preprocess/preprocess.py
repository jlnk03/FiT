#!/usr/bin/env python
"""
FiT preprocessing (for training) pipeline
"""
import argparse
import json
import logging
import os

import numpy as np

from tqdm import tqdm
from diffusers import AutoencoderKL

from model_utils import str2bool, check_cfgs_in_parser
from logger_ours import set_logger

from imagenet_dataset import create_dataloader_imagenet_preprocessing

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/preprocess/sd-vae-ft-mse-256x256.yaml",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="path to source torch checkpoint, which ends with .pt",
    )
    parser.add_argument("--outdir", default="./latent", help="Path of the output dir")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--imagenet_format", type=str2bool, default=True, help="Training with ImageNet dataset format")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 2 vae
    logger.info("vae init")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)

    # 3. build dataloader
    if args.imagenet_format:
        data_config = dict(data_folder=args.data_path, sample_size=args.image_size, patch_size=args.patch_size)

        dataloader = create_dataloader_imagenet_preprocessing(data_config)
    else:
        raise NotImplementedError("Currently only ImageNet format is supported.")

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

        latent = vae.encode(img).numpy().astype(np.float16)[0]
        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
