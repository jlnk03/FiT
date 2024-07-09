import os
import random
from typing import Dict, List, Tuple

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from pos_embed import get_2d_sincos_pos_embed, precompute_freqs_cis_2d

ALLOWED_FORMAT = {".jpeg", ".jpg", ".bmp", ".png"}


class _ResizeByMaxValue:
    def __init__(self, max_size: int = 256, vae_scale: int = 8, patch_size: int = 2) -> None:
        self.max_size = max_size
        self.scale = vae_scale * patch_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        image_area = w * h
        max_area = self.max_size * self.max_size

        if image_area > max_area:
            ratio = max_area / image_area

            new_w = w * torch.sqrt(torch.Tensor([ratio])).item()
            new_h = h * torch.sqrt(torch.Tensor([ratio])).item()
        else:
            new_w = w
            new_h = h

        round_w, round_h = (torch.round(torch.Tensor([new_w, new_h]) / self.scale) * self.scale).int().tolist()
        if round_w * round_h > max_area:
            round_w, round_h = (torch.floor(torch.Tensor([new_w, new_h]) / self.scale) * self.scale).int().tolist()

        round_w, round_h = max(round_w, self.scale), max(round_h, self.scale)

        img = img.resize((round_w, round_h), resample=Image.BICUBIC)

        return img


class ImageNetWithPathIterator(Dataset):
    def __init__(self, config) -> None:
        self.image_paths = self._inspect_images(config.get("img_folder", '../dataset'))
        self.resize = _ResizeByMaxValue(max_size=config.get("sample_size", 256),
                                        patch_size=config.get("patch_size", 2))
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def _inspect_images(self, root: str) -> List[str]:
        images_info = list()

        for dirpath, _, filenames in os.walk(root):
            for f in filenames:

                _, ext = os.path.splitext(f)

                if ext.lower() in ALLOWED_FORMAT:
                    fpath = os.path.join(dirpath, f)
                    images_info.append(fpath)

        if len(images_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        images_info = sorted(images_info)

        return images_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]

        with Image.open(path) as f:
            img = f.convert("RGB")

        img = self.resize(img)
        img = self.transform(img)

        return img, path


class ImageNetLatentIterator(Dataset):
    def __init__(self, config) -> None:
        self.latent_info = self._inspect_latent(config.get("latent_folder", '../latent'))
        self.label_mapping = self._create_label_mapping(self.latent_info)

        self.sample_size = config.get("sample_size", 256)
        self.patch_size = config.get("patch_size", 2)
        self.vae_scale = config.get("vae_scale", 8)
        self.C = config.get("C", 4)
        self.max_length = self.sample_size * self.sample_size // self.patch_size // self.patch_size // self.vae_scale // self.vae_scale

        self.embed_dim = config.get("embed_dim", 64)
        self.embed_method = config.get("embed_method", "rotate")

        self.batch_size = config.get("batch_size", 256)

    def _inspect_latent(self, root: str) -> List[Dict[str, str]]:
        latent_info = list()

        for dirpath, _, filenames in os.walk(root):
            for f in filenames:

                _, ext = os.path.splitext(f)

                if ext.lower() in ".pt":
                    fpath = os.path.join(dirpath, f)
                    latent_info.append(dict(path=fpath, label=os.path.basename(dirpath)))

        if len(latent_info) == 0:
            raise RuntimeError(f"Cannot find any image under `{root}`")

        latent_info = sorted(latent_info, key=lambda x: x["path"])

        return latent_info

    def _create_label_mapping(self, latent_info: List[Dict[str, str]]):
        labels = set([x["label"] for x in latent_info])
        labels = sorted(list(labels))
        labels = dict(zip(labels, torch.arange(len(labels), dtype=torch.int)))

        return labels

    def __len__(self):
        return len(self.latent_info)

    def _random_horiztotal_flip(self, latent: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            # perform a random horizontal flip in latent domain
            # mimic the effect of horizontal flip in image (not exactly identical)
            latent = latent[..., ::-1]

        return latent

    def _patchify(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, h, w = latent.shape
        nh, nw = h // self.patch_size, w // self.patch_size

        latent = torch.reshape(latent, (c, nh, self.patch_size, nw, self.patch_size))
        latent = torch.permute(latent, (1, 3, 2, 4, 0))  # nh, nw, patch, patch, c
        latent = torch.reshape(latent, (nh * nw, -1))  # nh * nw, patch * patch * c

        if self.embed_method == "rotate":
            pos = precompute_freqs_cis_2d(self.embed_dim, nh, nw).float()
        else:
            pos = get_2d_sincos_pos_embed(self.embed_dim, nh, nw).float()

        return latent, pos

    def subsample(self, latent, label, pos, mask, number_of_tokens):
        print(f'latent shape subsample: {latent.shape}', flush=True)
        if latent.shape[0] > number_of_tokens:
            permutation = torch.randperm(latent.shape[0])

            latent = latent[permutation][:number_of_tokens]
            pos = pos[permutation][:number_of_tokens]
            mask = mask[permutation][:number_of_tokens]
        else:
            latent = torch.nn.functional.pad(latent, (
            0, self.patch_size * self.patch_size * self.C - latent.shape[1], 0, number_of_tokens - latent.shape[0]))
            pos = torch.nn.functional.pad(pos, (0, self.embed_dim - pos.shape[1], 0, number_of_tokens - pos.shape[0]))
            mask = torch.nn.functional.pad(mask, (0, number_of_tokens - mask.shape[0]))

        return latent, torch.Tensor(label), pos, mask

    def collate(self, batch):
            number_of_tokens = random.randint(32, self.max_length)

            concatenate = [self.subsample(latent, label, pos, mask, number_of_tokens) for latent, label, pos, mask in batch]

            latent, label, pos, mask = zip(*concatenate)
            
            latent = torch.stack(latent)
            label = torch.tensor(label)  # Convert list of scalars to a tensor
            pos = torch.stack(pos)
            mask = torch.stack(mask)

            return latent, label, pos, mask

    def __getitem__(self, idx):
        x = self.latent_info[idx]

        latent = torch.load(x["path"])

        latent, pos = self._patchify(latent)
        label = self.label_mapping[x["label"]]
        mask = torch.ones(latent.shape[0], dtype=torch.bool)

        return latent, label, pos, mask


def create_dataloader_imagenet_preprocessing(
        config,
):
    dataset = ImageNetWithPathIterator(config)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    return dataloader
