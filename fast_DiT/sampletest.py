import torch
import argparse
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.fit import FiT_models
from samplesupport import precompute_freqs_cis_2d, apply_2d_rotary_pos, resize_call, patchify
from lightning import LightningModule

class FiTModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = FiT_models[args.model]()
        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    def predict_step(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, cfg_scale: float):
        # Setup PyTorch:
        torch.manual_seed(self.args.seed)
        torch.set_grad_enabled(False)
        device = x.device

        latent_size = x.shape[-1]
        n = x.shape[0]

        # Create sampling noise:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        x = torch.cat([x, x], 0)
        pos = torch.cat([pos, pos], 0)
        mask = torch.cat([mask, mask], 0)
        y_null = torch.full_like(y, self.args.num_classes)
        y = torch.cat([y, y_null], 0)

        # Sample images:
        def model_fn(x_t, t, y):
            x_t = apply_2d_rotary_pos(x_t, pos)
            return self.model(x_t, t, y=y, pos=pos, mask=mask, h=latent_size, w=latent_size)

        samples = self.diffusion.p_sample_loop(
            model_fn, z.shape, z, clip_denoised=False, model_kwargs=dict(y=y, cfg_scale=cfg_scale), progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = self.vae.decode(samples / 0.18215).sample

        return samples

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = FiTModule(args)
    model.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # Prepare inputs for FiT model
    latent_size = args.image_size // 8
    patch_size = 2
    embed_dim = 64
    max_tokens = 256
    n = args.num_images  # Number of images to generate

    # Create initial latent noise
    z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Patchify the latent
    latents, positions = [], []
    for i in range(n):
        latent, pos = patchify(z[i].cpu().numpy(), patch_size=patch_size, embed_dim=embed_dim)
        latents.append(torch.tensor(latent, device=device))
        positions.append(torch.tensor(pos, device=device))

    # Pad inputs to max_tokens
    latents = torch.stack([torch.nn.functional.pad(l, (0, patch_size * patch_size * 4 - l.shape[1], 0, max_tokens - l.shape[0])) for l in latents])
    positions = torch.stack([torch.nn.functional.pad(p, (0, embed_dim - p.shape[1], 0, max_tokens - p.shape[0])) for p in positions])
    
    # Prepare mask (all tokens are valid)
    mask = torch.ones(n, max_tokens, dtype=torch.bool, device=device)

    # Prepare class labels
    y = torch.tensor([args.class_label] * n, device=device)

    # Generate samples
    samples = model.predict_step(latents, None, y, positions, mask, args.cfg_scale)

    # Save and display images:
    save_image(samples, "samples.png", nrow=int(n**0.5), normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(FiT_models.keys()), default="FiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the FiT model checkpoint.")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to generate")
    args = parser.parse_args()
    main(args)