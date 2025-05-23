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
from utils import create_depth_maps

sys.path.append('peract_colab')
from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs

# Constants
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
IMAGE_SIZE = 128
DATA_FOLDER = 'peract_colab/data'
EPISODES_FOLDER = 'colab_dataset/open_drawer/all_variations/episodes'
OUTPUT_FOLDER = 'train_output'

if __name__ == '__main__':
    data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
    TRAIN_DATA_PATH = os.path.abspath("../../data/train")  

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

            for ts in range(len(demo)):
                obs_dict = extract_obs(demo._observations[ts], CAMERAS, t=ts)

                images = [
                    np.transpose(obs_dict['front_rgb'], (1, 2, 0)),
                    np.transpose(obs_dict['left_shoulder_rgb'], (1, 2, 0)),
                    np.transpose(obs_dict['right_shoulder_rgb'], (1, 2, 0)),
                    np.transpose(obs_dict['wrist_rgb'], (1, 2, 0))
                ]

                # Save RGB images
                for i, image in enumerate(images):
                    output_path = os.path.join(OUTPUT_FOLDER, f'task_{task_dir}_ep_{episode_idx}_ts_{ts}_rgb{i}.png')
                    plt.imsave(output_path, image)

                # Create depth maps using the model
                depths = create_depth_maps(images, model, device, batch_size=1, niter=niter, schedule=schedule, lr=lr)

                # Save depth output
                depth_output_path = os.path.join(OUTPUT_FOLDER, f'task_{task_dir}_ep_{episode_idx}_ts_{ts}_depth0.png')
                plt.imsave(depth_output_path, np.asarray(depths[0]))

                # Save ground truth depth map
                gt_depth = np.transpose(obs_dict['front_depth'], (1, 2, 0))
                gt_depth_output_path = os.path.join(OUTPUT_FOLDER, f'task_{task_dir}_ep_{episode_idx}_ts_{ts}_gtdepth.png')
                plt.imsave(gt_depth_output_path, gt_depth)
