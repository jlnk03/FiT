import torch
import argparse
from lightning import Trainer
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler
from train import FiTModule
from torchvision.utils import save_image

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

    # Set up the noise scheduler
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Generate random noise
    latent_height, latent_width = args.image_height // 8, args.image_width // 8
    class_labels = torch.Tensor([125, 120, 260, 248, 270, 280]).int().to(model.device)
    num_samples = len(class_labels)
    z = torch.randn(num_samples, 4, latent_height, latent_width, device=model.device)
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

    print("Starting sampling")

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
        # torch.save(image, f'generated_image_{i}.pt')
        print(f'generated_image_{i}.png')
        save_image(image, f'output/generated_image_{i}.png')

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