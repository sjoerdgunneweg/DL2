import os
import sys
import argparse
import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from PIL import Image


sys.path.append('peract_colab')
from peract_colab.rlbench.backend.utils import image_to_float_array

import json
from collections import defaultdict

np.bool = np.bool_  # Fix for deprecated `np.bool`

os.environ["PYOPENGL_PLATFORM"] = "egl"

np.set_printoptions(threshold=sys.maxsize)


# Metrics for depth evaluation:
def compute_depth_metrics(gt, pred):
    mask = (gt > 0)
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt + 1e-8) - np.log(pred + 1e-8)) ** 2))

    log_diff = np.log(pred + 1e-8) - np.log(gt + 1e-8)
    scale_inv_rmse = np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2)

    ssim_val = ssim(gt.reshape(gt.shape), pred.reshape(pred.shape), data_range=1.0)

    return {
        "AbsRel": abs_rel,
        "SqRel": sq_rel,
        "RMSE": rmse,
        "RMSE(log)": rmse_log,
        "ScaleInvRMSE": scale_inv_rmse,
        "δ1": a1,
        "δ2": a2,
        "δ3": a3,
        "SSIM": ssim_val
    }

def aggregate_metrics(metrics_list):
    aggregated = {}
    for key in metrics_list[0].keys():
        if key == 'valid_pixels':
            aggregated[key] = sum(m[key] for m in metrics_list)
        else:
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
            else:
                aggregated[key] = np.nan
                aggregated[f"{key}_std"] = np.nan
    
    aggregated['num_samples'] = len(metrics_list)
    return aggregated

def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {output_path}")

def normalize_depth(depth):
    depth = depth.astype(np.float32)
    return depth / depth.max()

def plot_comparison(gt, pred):
    error_map = np.abs(gt - pred)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(gt, cmap='inferno')
    axs[0].set_title("Ground Truth")
    axs[1].imshow(pred, cmap='inferno')
    axs[1].set_title("Prediction")
    axs[2].imshow(error_map, cmap='magma')
    axs[2].set_title("Absolute Error")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()



# Main function to evaluate depth maps:
def main(data_path, tasks=None):
    # Constants
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
    IMAGE_SIZE = 128
    DEPTH_SCALE = 2**24 - 1

    # output structure
    per_timestep_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    task_aggregates = []
    episode_aggregates = defaultdict(list)


    if args.tasks:
        task_dirs = args.tasks
    else:
        task_dirs = os.listdir(data_path)

    for task_dir in task_dirs:
        print(f"Current task: {task_dir}")
        task_path = os.path.join(data_path, task_dir)
        episodes_path = os.path.join(task_path, "all_variations/episodes")
        
        if not os.path.exists(episodes_path):
            print(f"Skipping missing episode folder: {episodes_path}")
            continue

        episode_dirs = sorted([d for d in os.listdir(episodes_path) 
                      if os.path.isdir(os.path.join(episodes_path, d)) 
                      and d.startswith('episode')])
        task_metrics_list = []

        for episode_idx, episode_name in enumerate(episode_dirs[:3]):
            print(f"  Current episode: {episode_name}")
            episode_path = os.path.join(episodes_path, episode_name)

            try:
                with open(os.path.join(episode_path, 'low_dim_obs.pkl'), 'rb') as f:
                    obs = pickle.load(f)
            except Exception as e:
                print(f"    Failed to load demo {episode_idx}: {e}")
                continue

            episode_metrics_list = []

            for ts in range(len(obs)):
                try:
                    ts_metrics_list = []

                    for cam in CAMERAS:
                        near = obs[ts].misc['%s_camera_near' % (cam)]   
                        far = obs[ts].misc['%s_camera_far' % (cam)]

                        gt_depth = near + image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (cam, 'depth'), f'{ts}.png')), DEPTH_SCALE) * (far - near)
                        pred_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (cam, 'depth'), f'{ts}_mast3r.png')), DEPTH_SCALE) * (far - near)

                        cam_metrics = compute_depth_metrics(gt_depth, pred_depth)
                        ts_metrics_list.append(cam_metrics)

                        # Store per-timestep metrics:
                        per_timestep_metrics[task_dir][episode_name][str(ts)][cam] = cam_metrics
                        
                    
                    episode_metrics_list.extend(ts_metrics_list)
                except Exception as e:
                    print(f"    Failed to process timestep {ts} in episode {episode_name}: {e}")
                    continue
            
            episode_metrics = aggregate_metrics(episode_metrics_list)
            episode_aggregates[task_dir].append({
                "episode": episode_name, 
                **episode_metrics})
            
            task_metrics_list.append(episode_metrics)


        task_metrics = aggregate_metrics(task_metrics_list)
        task_aggregates.append({
            "task": task_dir, 
            **task_metrics})
    
    output = {
        "per_timestep_metrics": per_timestep_metrics,
        "task_aggregates": task_aggregates,
        "episode_aggregates": episode_aggregates
    }

    with open('metrics_output.json', 'w') as f:
        json.dump(output, f, indent=4)

    print("Evaluation complete. Metrics saved to metrics_output.json")            

                        






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process depth maps from RLBench data')
    parser.add_argument('--data_path', type=str, default='/data/train',
                        help='Path to the data directory (default: /data/train)')
    parser.add_argument('--tasks', nargs='+', type=str, default=None,
                        help='List of specific task names to process (default: process all tasks)')
    args = parser.parse_args()
    main(args.data_path, args.tasks)