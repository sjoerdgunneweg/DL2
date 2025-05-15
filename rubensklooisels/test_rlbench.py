import numpy as np
np.bool = np.bool_ # bad trick to fix numpy version issue :(

import os
import sys

import matplotlib
import matplotlib.pyplot as plt

os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"


import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import sys
sys.path.append("..")

from mast3r.mast3r.model import AsymmetricMASt3R

from utils import create_depth_maps

sys.path.append('peract_colab')
from rlbench.utils import get_stored_demo
from rlbench.backend.utils import extract_obs

#constants:
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
IMAGE_SIZE =  128  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
DATA_FOLDER ='peract_colab/data'
EPISODES_FOLDER = 'colab_dataset/open_drawer/all_variations/episodes'


if __name__ == '__main__':
    
    data_path = os.path.join(DATA_FOLDER, EPISODES_FOLDER)
    TEST_DATA_PATH = os.path.abspath("../../tost/")

    

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)


    task_dirs = os.listdir(TEST_DATA_PATH)
    #Loop through all the tasks:
    for task_dir in task_dirs:
        print(f"Current task: {task_dir}")
        task_path = os.path.join(TEST_DATA_PATH, task_dir)
        episodes = os.listdir(os.path.join(task_path, "all_variations/episodes"))

        #Loop through all the episodes:
        for episode_idx in range(len(episodes)):
            print(f"Current episode: {episode_idx}")
            episode_path = os.path.join(task_path, 'all_variations/episodes')
            
            demo = get_stored_demo(episode_path, episode_idx)

            #Loop through time steps:
            for ts in range(len(demo)):
                obs_dict = extract_obs(demo._observations[ts], CAMERAS, t=ts)

                images = [obs_dict['front_rgb'], 
                        obs_dict['left_shoulder_rgb'], 
                        obs_dict['right_shoulder_rgb'], 
                        obs_dict['wrist_rgb']]
                
                depths = create_depth_maps(images, model, device, batch_size=1, niter=niter, schedule=schedule, lr=lr)
                print("lol")
                break

            break
        break
    
                