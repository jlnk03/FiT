import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UniPCMultistepScheduler

from fit import FiT_models


class FiTFusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
        )
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler")
        self.FiT = FiT_models['FiT-B/2']()
        self.generator = torch.Generator(device=self.device).manual_seed(42)

    def forward(self, prompt='a photograph of an astronaut riding a horse', inference_steps=25, guidance_scale=7.5, height=256, width=256):
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        batch_size = len(prompt)
        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length",
                                      max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            generator=self.generator,
            device=self.device,
        )
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t)

            # timestep of diffusion s
            time_step = torch.tensor([t] * batch_size, device=device)

            positional_embeddings = None
            mask = None

            # predict the noise residual
            # TODO: pass correct parameters such as mask, noise, etc.
            noise_pred = self.FiT(latent_model_input, time_step, text_embeddings, positional_embeddings, mask)

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)

        return image

    def training_step(self, batch, batch_idx):
        # Map input images to latent space + normalize latents:
        x = self.vae.encode(batch).latent_dist.sample().mul_(0.18215)

        # TODO: Implement the rest of the training loop

        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
