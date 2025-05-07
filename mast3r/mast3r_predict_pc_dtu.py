import os
import torch
import tempfile
from glob import glob
import numpy as np
import open3d as o3d
import random
import re

from mast3r.cloud_opt.triangulation import matches_to_depths
from mast3r.cloud_opt.sparse_ga import forward_mast3r, convert_dust3r_pairs_naming
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R

from dust3r.utils.image import load_images
from dust3r.utils.geometry import opencv_to_colmap_intrinsics

def save_pointcloud_as_ply(path, pts3d, colors):
    mask = np.isfinite(pts3d).all(-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d[mask].reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors[mask].reshape(-1, 3) / 255.0)
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved: {path}")

def load_dtu_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    extr_idx = lines.index("extrinsic\n")
    intr_idx = lines.index("intrinsic\n")

    extr = np.array([
        list(map(float, lines[extr_idx + i + 1].strip().split()))
        for i in range(4)
    ])[:3]
    intr = np.array([
        list(map(float, lines[intr_idx + i + 1].strip().split()))
        for i in range(3)
    ])

    return extr, intr

def run_pipeline(scan_id, image_dir, cam_dir, output_path, model, device, img_size=384):
    img_paths = sorted(glob(os.path.join(image_dir, f'scan{scan_id}_train', '*_3_r5000.png')))
    img_paths = [p for p in img_paths if int(re.findall(r'(\d+)_3_r5000', p)[0]) <= 48]
    assert len(img_paths) > 0, f"No valid images found for scan {scan_id}."

    imgs = load_images(img_paths, size=img_size, verbose=True)

    intrinsics, extrinsics = [], []
    for path in img_paths:
        idx = int(re.findall(r'(\d+)_3_r5000', path)[0])
        cam_path = os.path.join(cam_dir, f"{idx:08d}_cam.txt")
        extr, intr = load_dtu_intrinsics(cam_path)
        intrinsics.append(opencv_to_colmap_intrinsics(intr))
        extrinsics.append(extr)

    # Convert intrinsics and extrinsics to torch tensors
    intrinsics = torch.tensor(np.stack(intrinsics)).float()
    extrinsics = torch.tensor(np.stack(extrinsics)).float().to(device)

    # === Scale intrinsics to resized image size ===
    original_h, original_w = 512, 640  # DTU original resolution
    scaled_h, scaled_w = imgs[0]['img'].shape[-2:]
    scale_h = scaled_h / original_h
    scale_w = scaled_w / original_w
    intrinsics[:, 0, 0] *= scale_w  # fx
    intrinsics[:, 1, 1] *= scale_h  # fy
    intrinsics[:, 0, 2] *= scale_w  # cx
    intrinsics[:, 1, 2] *= scale_h  # cy
    intrinsics = intrinsics.to(device)

    pairs = make_pairs(imgs, scene_graph='oneref-0')
    pairs = convert_dust3r_pairs_naming(img_paths, pairs)

    # Run MASt3R forward matching
    cache_path = tempfile.mkdtemp()
    pair_paths, _ = forward_mast3r(pairs, model, cache_path=cache_path, device=device)

    H, W = imgs[0]['img'].shape[-2:]
    N = len(img_paths)

    ref_idx = 0
    ref_path = img_paths[ref_idx]
    matches = torch.full((1, N - 1, H, W, 5), float("nan"), device=device)

    for tgt_idx, tgt_path in enumerate(img_paths[1:]):
        pair_key = (ref_path, tgt_path)
        if pair_key not in pair_paths:
            continue

        _, path_corres = pair_paths[pair_key]
        if not os.path.exists(path_corres):
            continue

        _, (xy_ref, xy_tgt, confs) = torch.load(path_corres)

        # === Filter by confidence threshold ===
        conf_thresh = 0.5
        valid = confs > conf_thresh
        if valid.sum() == 0:
            continue  # skip if no valid matches

        xy_ref = xy_ref[valid]
        xy_tgt = xy_tgt[valid]
        confs = confs[valid]

        x_ref = xy_ref[:, 0].round().long().clamp(0, W - 1)
        y_ref = xy_ref[:, 1].round().long().clamp(0, H - 1)

        matches[0, tgt_idx, y_ref, x_ref, 0] = xy_ref[:, 0].float()
        matches[0, tgt_idx, y_ref, x_ref, 1] = xy_ref[:, 1].float()
        matches[0, tgt_idx, y_ref, x_ref, 2] = xy_tgt[:, 0].float()
        matches[0, tgt_idx, y_ref, x_ref, 3] = xy_tgt[:, 1].float()
        matches[0, tgt_idx, y_ref, x_ref, 4] = confs.float()

    intr_ = intrinsics.unsqueeze(0)
    extr_ = extrinsics.unsqueeze(0)
    pts3d, depths, confs = matches_to_depths(intr_, extr_, matches, batchsize=1)

    img_tensor = imgs[ref_idx]['img']
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    rgb = img_tensor.permute(1, 2, 0).cpu().numpy()

    save_pointcloud_as_ply(output_path, pts3d[0].cpu().numpy(), rgb)

def main():
    scan_id = 24
    image_dir = '../data/dtu/Rectified'
    cam_dir = '../data/dtu/Cameras/train'
    output_path = f'../predictions/scan{scan_id}.ply'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model.output_mode = 'matches'
    model = model.to(device).eval()

    run_pipeline(scan_id, image_dir, cam_dir, output_path, model, device)

if __name__ == '__main__':
    main()
