from train_lightning_diffusers_final import FiTModule
from models.fit import FiT_models
import argparse
from lightning import seed_everything, Trainer

def main(args):
    seed_everything(args.global_seed)
    
    model = FiTModule(args)
    trainer = Trainer()
    trainer.predict(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(FiT_models.keys()), default="FiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the FiT model checkpoint.")
    args = parser.parse_args()
    main(args)
