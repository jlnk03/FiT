import os
import numpy as np
import torch
from tqdm import tqdm
import time

def convert_npy_to_torch():
    input_folder = "input_npy_files"  # Hardcoded input folder name
    output_folder = "output_torch_files"  # Hardcoded output folder name
    
    input_folder_path = os.path.abspath(input_folder)
    output_folder_path = os.path.abspath(output_folder)
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Count total number of .npy files
    total_files = sum(len([f for f in files if f.endswith('.npy')]) for _, _, files in os.walk(input_folder_path))
    
    print(f"Found {total_files} .npy files to convert.")
    
    start_time = time.time()
    files_processed = 0
    
    with tqdm(total=total_files, unit='file') as pbar:
        for root, dirs, files in os.walk(input_folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    npy_tensor = np.load(file_path)
                    torch_tensor = torch.from_numpy(npy_tensor)
                    
                    rel_path = os.path.relpath(file_path, input_folder_path)
                    new_file_path = os.path.join(output_folder_path, rel_path)
                    new_dir = os.path.dirname(new_file_path)
                    os.makedirs(new_dir, exist_ok=True)
                    
                    torch.save(torch_tensor, new_file_path.replace('.npy', '.pt'))
                    
                    files_processed += 1
                    pbar.update(1)
                    
                    # Update progress bar description with current file and speed
                    elapsed_time = time.time() - start_time
                    speed = files_processed / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_description(f"Processing {rel_path} ({speed:.2f} files/s)")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_speed = files_processed / total_time if total_time > 0 else 0
    
    print(f"\nConversion complete. {files_processed} files processed in {total_time:.2f} seconds.")
    print(f"Average speed: {avg_speed:.2f} files/s")
    print(f"Torch files saved in {output_folder_path}")

# Usage
convert_npy_to_torch()