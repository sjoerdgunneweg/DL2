import os
import torch
import tempfile
from glob import glob
import numpy as np
import re
import trimesh
import copy

from mast3r.cloud_opt.triangulation import batched_triangulate
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import convert_dust3r_pairs_naming, forward_mast3r, prepare_canonical_data, condense_data

from dust3r.utils.image import load_images
from dust3r.utils.geometry import opencv_to_colmap_intrinsics


def load_dtu_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    extr_idx = lines.index("extrinsic\n")
    intr_idx = lines.index("intrinsic\n")
    extr = np.array([
        list(map(float, lines[extr_idx + i + 1].strip().split())) for i in range(4)
    ])[:3]
    intr = np.array([
        list(map(float, lines[intr_idx + i + 1].strip().split())) for i in range(3)
    ])
    return extr, intr

def triangulate_sparse_matches_with_gt(imgs_slices, K_map, c2w_map, imgs, device):
    all_pts3d = []
    all_colors = []
    all_confs = []

    for s in imgs_slices:
        if len(s.pix1) == 0:
            continue

        key1 = imgs[s.img1]['instance']
        key2 = imgs[s.img2]['instance']
        K1 = K_map[key1]
        K2 = K_map[key2]
        E1 = np.linalg.inv(c2w_map[key1])[:3]
        E2 = np.linalg.inv(c2w_map[key2])[:3]
        P1 = torch.from_numpy(K1 @ E1).float().to(device).unsqueeze(0)
        P2 = torch.from_numpy(K2 @ E2).float().to(device).unsqueeze(0)
        proj_mats = torch.cat([P1, P2], dim=0).unsqueeze(0)  # (1, 2, 3, 4)

        # 2D matches
        pts2d = torch.stack([
            s.pix1.float().to(device),
            s.pix2.float().to(device)
        ]).unsqueeze(0)  # (1, 2, N, 2)

        # Triangulate
        pts3d = batched_triangulate(pts2d, proj_mats)[0]  # (N, 3)
        conf = s.confs.cpu().numpy()
        valid = torch.isfinite(pts3d).all(dim=-1).cpu().numpy()

        if valid.sum() == 0:
            continue

        pts3d = pts3d[valid].cpu().numpy()
        conf = conf[valid]

        # Sample color from first image
        pix = s.pix1[valid].cpu().numpy().astype(int)
        img_tensor = imgs[s.img1]['img']
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)  
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        colors = img_np[pix[:, 1], pix[:, 0]] 
  
        all_pts3d.append(pts3d)
        all_colors.append(colors)
        all_confs.append(conf)

    if not all_pts3d:
        return None

    pts3d = np.concatenate(all_pts3d)
    colors = np.concatenate(all_colors)
    confs = np.concatenate(all_confs)

    return pts3d, colors, confs

def run_pipeline(scan_id, image_dir, cam_dir, output_path, model, device, img_size=512):
    img_paths = sorted(glob(os.path.join(image_dir, f'scan{scan_id}_train', '*_3_r5000.png')))
    img_paths = [p for p in img_paths if int(re.findall(r'(\d+)_3_r5000', p)[0]) <= 48]
    assert len(img_paths) > 0, f"No valid images found for scan {scan_id}."
    imgs = load_images(img_paths, size=img_size, verbose=True)
    for i, img in enumerate(imgs):
        img['instance'] = img_paths[i]

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    K_map = {}
    c2w_map = {}
    H_orig, W_orig = 512, 640

    for i, path in enumerate(img_paths):
        idx = int(re.findall(r'(\d+)_3_r5000', path)[0])
        cam_path = os.path.join(cam_dir, f"{idx:08d}_cam.txt")
        extr, intr = load_dtu_intrinsics(cam_path)

        H_new, W_new = 400, 512
        scale_x = W_new / W_orig
        scale_y = H_new / H_orig
        intr[0, 0] *= scale_x
        intr[0, 2] *= scale_x
        intr[1, 1] *= scale_y
        intr[1, 2] *= scale_y

        K_map[img_paths[i]] = opencv_to_colmap_intrinsics(intr)
        c2w_map[img_paths[i]] = np.vstack([extr, [0, 0, 0, 1]])

    pairs_in = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True, sim_mat=None)
    pairs_in = convert_dust3r_pairs_naming(img_paths, pairs_in)

    os.makedirs("out", exist_ok=True)
    cache_path = tempfile.mkdtemp(suffix='_cache', dir="out") 
    pairs, cache_path = forward_mast3r(pairs_in, model,
                                    cache_path=cache_path, subsample=8,
                                    desc_conf='desc_conf', device=device)
    
    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
        prepare_canonical_data(img_paths, pairs, subsample=8, cache_path=cache_path, mode='avg-angle', device=device)

    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(img_paths, tmp_pairs, canonical_views, preds_21, dtype=torch.float32)
    
    imgs_slices = corres[2]

    pts3d, colors, confs = triangulate_sparse_matches_with_gt(imgs_slices, K_map, c2w_map, imgs, device)

    pc = trimesh.PointCloud(pts3d[confs > 2], colors=colors[confs > 2])
    pc.export(output_path)
    print(f"Point cloud saved to {output_path}")

def main():
    scan_id = 24
    image_dir = '../data/dtu/Rectified'
    cam_dir = '../data/dtu/Cameras'
    output_path = f'../predictions/scan{scan_id}.ply'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = model.to(device).eval()

    run_pipeline(scan_id, image_dir, cam_dir, output_path, model, device)

if __name__ == '__main__':
    main()
