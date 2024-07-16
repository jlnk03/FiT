import argparse

import torch
from diffusers import DDIMScheduler
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from diffusion import create_diffusion
from train import FiTModule
import math


def main(args):
    # Load the model
    model = FiTModule.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    print("Model loaded successfully")
    print(model.device)

    # Set up the diffusion process
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Load the VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(model.device)

    # Set up batch size and calculate number of batches
    batch_size = 100
    total_samples = args.num_samples
    num_batches = math.ceil(total_samples / batch_size)

    for batch in range(num_batches):
        # Calculate the number of samples for this batch
        current_batch_size = min(batch_size, total_samples - batch * batch_size)
        
        # Generate random noise
        latent_height, latent_width = args.image_height // 8, args.image_width // 8
        class_labels = torch.randint(0, args.num_classes, (current_batch_size,)).int().to(model.device)
        z = torch.randn(current_batch_size, 4, latent_height, latent_width, device=model.device)
        y = class_labels
        y_null = torch.ones_like(y) * 1000

        # Prepare the data
        y_tot = torch.cat([y, y_null], dim=0)
        x_in = torch.cat([z] * 2, dim=0)
        n, _, h, w = x_in.shape

        # Set up position embeddings and mask
        z = model._pad_latent(x_in, 2, 32, 256)
        pos, valid_t = model._create_pos_embed(h, w, 2, 256, 64, "rotate")
        mask = model._create_mask(valid_t, 256, n)
        pos = torch.Tensor(pos).to(model.device)
        mask = torch.Tensor(mask).to(model.device)

        print(f"Starting sampling for batch {batch + 1}/{num_batches}")

        # Set up model kwargs
        model_kwargs = dict(y=y_tot, pos=pos, mask=mask, cfg_scale=args.cfg_scale)

        # Run the diffusion sampling
        samples = diffusion.ddim_sample_loop(
            model.model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=model.device
        )
        samples, _ = samples.chunk(2, axis=0)
        samples = model._unpad_latent(samples, valid_t, h, w, 2)

        # Decode the latents using the VAE
        with torch.no_grad():
            images = vae.decode(samples / 0.18215).sample

        print(f'images shape: {images.shape}')

        # Save the generated images
        for i, image in enumerate(images):
            image_index = batch * batch_size + i
            print(f'generated_image_{image_index}.png')
            save_image(image, f'/storage/slurm/schnaus/gsu_2024/results/output/mfit38_2_1/generated_image_{image_index}_{class_labels[i]}.png')

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

    print("All batches processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--num_sampling_steps", type=int, default=250, help="Number of sampling steps")
    parser.add_argument("--image_height", type=int, default=256, help="Height of the generated image")
    parser.add_argument("--image_width", type=int, default=256, help="Width of the generated image")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes in the model")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"], help="Type of VAE to use")
    parser.add_argument("--cfg_scale", type=float, default=15.0, help="Classifier-free guidance scale")
    args = parser.parse_args()
    main(args)
