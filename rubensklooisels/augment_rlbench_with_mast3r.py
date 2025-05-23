import numpy as np
np.bool = np.bool_  # Fix for deprecated `np.bool`

import os
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
DATA_FOLDER = 'peract_colab/data'
EPISODES_FOLDER = 'colab_dataset/open_drawer/all_variations/episodes'
DEPTH_SCALE = 2**24 - 1

if __name__ == '__main__':
    data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
    TRAIN_DATA_PATH = os.path.abspath("../../data/train")  

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    task_dirs = os.listdir(TRAIN_DATA_PATH)

    for task_dir in task_dirs:
        print(f"Current task: {task_dir}")
        task_path = os.path.join(TRAIN_DATA_PATH, task_dir)
        episodes_path = os.path.join(task_path, "all_variations/episodes")
        
        if not os.path.exists(episodes_path):
            print(f"Skipping missing episode folder: {episodes_path}")
            continue

        episode_dirs = sorted(os.listdir(episodes_path))

        for episode_idx in range(len(episode_dirs)):
            print(f"  Current episode: {episode_idx}")
            try:
                demo = get_stored_demo(episodes_path, episode_idx)
            except Exception as e:
                print(f"    Failed to load demo {episode_idx}: {e}")
                continue
            episode_path = os.path.join(episodes_path, episode_dirs[episode_idx])

            for ts in range(len(demo)):
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
                    near = obs_dict[f'{cam}_camera_near']
                    far = obs_dict[f'{cam}_camera_far']
                    depth = (depth - near) / (far - near)

                    depth_rgb = float_array_to_rgb_image(depth, DEPTH_SCALE)
                    
                    depth_output_path = os.path.join(episode_path, f'{cam}_depth/{ts}_mast3r.png')
                    depth_rgb.save(depth_output_path)