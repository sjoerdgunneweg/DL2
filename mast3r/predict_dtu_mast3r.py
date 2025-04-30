import os
import torch
import tempfile
from glob import glob
import numpy as np
import open3d as o3d
import random
import re

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R, inf


def save_pointcloud_as_ply(path, pts3d_list, confs_list, imgs_list):
    """
    Saves the reconstructed 3D points as a colored point cloud in .ply format.
    pts3d_list: list of (H, W, 3) numpy arrays (one per image)
    confs_list: list of (H, W) confidence maps
    imgs_list:  list of (H, W, 3) RGB images in [0, 255] range
    """

    all_points = []
    all_colors = []

    for pts3d, conf, img in zip(pts3d_list, confs_list, imgs_list):
        # Convert to numpy
        pts3d = np.array(pts3d.cpu()) if isinstance(pts3d, torch.Tensor) else np.array(pts3d)
        conf = np.array(conf.cpu()) if isinstance(conf, torch.Tensor) else np.array(conf)
        if isinstance(img, torch.Tensor):
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [1, 3, H, W] → [H, W, 3]

        # Reshape pts3d to match image shape
        H, W = conf.shape
        pts3d = pts3d.reshape(H, W, 3)

        # Build mask and apply
        mask = np.isfinite(pts3d).all(-1) & (conf > 1.0)
        pts = pts3d[mask]
        colors = img[mask] / 255.0  # Normalize RGB to [0,1]

        all_points.append(pts)
        all_colors.append(colors)


    if not all_points:
        print("⚠️ No valid 3D points to save.")
        return

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ Point cloud saved to: {path}")


def run_mast3r_on_scan(scan_dir, output_dir, model, device, number_of_images, image_size=384):
    print(f"\n▶️ Processing scan: {scan_dir}")

    # Load images
    all_paths = sorted(glob(os.path.join(scan_dir, '*.png')))
    image_paths = [p for p in all_paths if re.match(r'.*_[3]_r5000\.png$', os.path.basename(p))]
    if len(image_paths) < number_of_images:
        raise ValueError(f"Scan {scan_dir} has fewer than {number_of_images} images.")
    if number_of_images < 2:
        raise ValueError(f"Need at least 2 images.")
    image_paths = random.sample(image_paths, number_of_images)

    images = load_images(image_paths, size=image_size, verbose=True)
    imgs_rgb = [img['img'] for img in images]

    # Build complete image pair graph
    pairs = make_pairs(images, scene_graph="complete")

    # Set up cache dir for temp results
    cache_dir = tempfile.mkdtemp(suffix='_cache', dir=output_dir)

    # Run sparse global alignment (matching, pose estimation, triangulation)
    scene = sparse_global_alignment(
        image_paths, pairs, cache_dir,
        model,
        lr1=0.07, niter1=300,
        lr2=0.01, niter2=300,
        device=device,
        opt_depth=True,
        shared_intrinsics=False,
        matching_conf_thr=0.0
    )

    # Extract 3D points (dense)
    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)

    # Save output as PLY point cloud
    out_name = os.path.basename(os.path.normpath(scan_dir)) + ".ply"
    out_path = os.path.join(output_dir, out_name)
    save_pointcloud_as_ply(out_path, pts3d, confs, imgs_rgb)


def main():
    # Paths
    dtu_test_root = "../data/dtu/Rectified"  # ⬅️ Change this to your DTU image folder
    output_root = "../predictions"
    os.makedirs(output_root, exist_ok=True)

    # Device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build and load model
    model = eval("AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)")
    model.to(device)
    model.eval()
    ckpt = torch.load("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    # Run over selected scans (e.g., DTU test set: scans 24–49)
    test_scan_ids = [24]
    for sid in test_scan_ids:
        scan_dir = os.path.join(dtu_test_root, f"scan{sid}_train")
        if not os.path.exists(scan_dir):
            print(f"⚠️ Skipping missing scan: scan{sid}")
            continue
        run_mast3r_on_scan(scan_dir, output_root, model, device, number_of_images=49)


if __name__ == "__main__":
    main()
