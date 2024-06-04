import os
from preprocess import decode_latents

latent_folder = '/Users/daniellavin/Desktop/RCI_III/SceneView/implement/GSU/latent/n01440764'
latent_files = os.listdir(latent_folder)

for i in range(3):
    latent_file = os.path.join(latent_folder, latent_files[i])
    decode_latents(latent_file)
