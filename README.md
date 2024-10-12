# (Masked) FiT: Flexible Vision Transformer for Diffusion Model

### FiT

#### Training

To train the model from scratch navigate to the `FiT` folder and run:
```
python train.py  --model FiT-B/2 --global-batch-size 128 --epochs 100 --feature-path /path/to/latent_train --feature-val-path /path/to/latent_val --results-dir /path/to/store/results
```

#### Inference

To sample images from a checkpoint run `sample_lightning.py`
```
python sample_lightning.py --checkpoint_path /path/to/store/results/checkpoints/checkpoint.ckpt --num_samples 50_000 --cfg_scale 1.5
```

### Masked FiT

This model incorporates masking for the input tokens to enable faster training runs. To run the code navigate to the `masked_FiT` folder

#### Training

To train the model from scratch navigate to the `FiT` folder and run:
```
python train.py  --model FiT-B/2 --global-batch-size 128 --epochs 100 --feature-path /path/to/latent_train --feature-val-path /path/to/latent_val --results-dir /path/to/store/results
```

#### Inference

To sample images from a checkpoint run `sample_lightning.py`
```
python sample_lightning.py --checkpoint_path /path/to/store/results/checkpoints/checkpoint.ckpt --num_samples 50_000 --cfg_scale 1.5
```

### Training input

To make training more efficient the inputs are preprocessed images which are fed through the Stable Diffusion Encoder (`stabilityai/sd-vae-ft-ema`)

Inside the `preprocess` folder set the paths to your dataset in `config.json` and run `preprocess.py` to encode a set of images.

```
python preprocess.py --config path/to/your/config.json
```