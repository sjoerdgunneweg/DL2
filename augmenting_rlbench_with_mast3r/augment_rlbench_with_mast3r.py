import numpy as np
np.bool = np.bool_  # Fix for deprecated `np.bool`
from tqdm import tqdm
import os
import argparse
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

os.environ["PYOPENGL_PLATFORM"] = "egl"

np.set_printoptions(threshold=sys.maxsize)

# Add required module paths
sys.path.append('../mast3r')
from mast3r.model import AsymmetricMASt3R
from utils import create_depth_maps, float_array_to_rgb_image

sys.path.append('peract_colab')
from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs

# Constants
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
IMAGE_SIZE = 128
DEPTH_SCALE = 2**24 - 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process depth maps from RLBench data')
    parser.add_argument('--data_path', type=str, default='/data/train',
                        help='Path to the data directory (default: /data/train)')
    parser.add_argument('--tasks', nargs='+', type=str, default=None,
                        help='List of specific task names to process (default: process all tasks)')
    args = parser.parse_args()

    data_path = args.data_path

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 150
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    if args.tasks:
        task_dirs = args.tasks
    else:
        task_dirs = os.listdir(data_path)

    for task_dir in task_dirs:
        print(f"Current task: {task_dir}")
        task_path = os.path.join(data_path, task_dir)
        episodes_path = os.path.join(task_path, "all_variations/episodes")
        
        if not os.path.exists(os.path.join(task_path, episodes_path)):
            print(f"Task folder is not found in the data directory: {os.path.join(task_path, episodes_path)}. Skipping...")
            continue

        episode_dirs = sorted([d for d in os.listdir(episodes_path) 
                      if os.path.isdir(os.path.join(episodes_path, d)) 
                      and d.startswith('episode')])

        for episode_idx in range(92, len(episode_dirs)):
            print(f"  Current episode: {episode_idx}")
            try:
                demo = get_stored_demo(episodes_path, episode_idx)
            except Exception as e:
                print(f"    Failed to load demo {episode_idx}: {e}")
                continue
            episode_path = os.path.join(episodes_path, f'episode{episode_idx}')

            for ts in tqdm(range(len(demo))):
                try:
                    obs_dict = extract_obs(demo._observations[ts], CAMERAS, t=ts)
                    images = [
                        np.transpose(obs_dict['front_rgb'], (1, 2, 0)),
                        np.transpose(obs_dict['left_shoulder_rgb'], (1, 2, 0)),
                        np.transpose(obs_dict['right_shoulder_rgb'], (1, 2, 0)),
                        np.transpose(obs_dict['wrist_rgb'], (1, 2, 0))
                    ]

                    # Create depth maps (np arrays single channel) using the model
                    depths = create_depth_maps(images, model, device, batch_size=1, niter=niter, schedule=schedule, lr=lr)
                
                    for depth, cam in zip(depths, CAMERAS):
                        near = demo[ts].misc[f'{cam}_camera_near']
                        far = demo[ts].misc[f'{cam}_camera_far']
                        depth = (depth - near) / (far - near)

                        depth_rgb = float_array_to_rgb_image(depth, DEPTH_SCALE)
                    
                        depth_output_path = os.path.join(episode_path, f'{cam}_depth/{ts}_mast3r.png')
                        depth_rgb.save(depth_output_path)
                except:
                    print(f"An error occured in episode {episode_idx}, ts {ts}")
