import numpy as np
import trimesh
import os, copy, tempfile, torch
from glob import glob
import re

from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images


def load_cam_pose(image_id, cam_dir):
    cam_path = f"{cam_dir}/{image_id:08d}_cam.txt"
    with open(cam_path, 'r') as f:
        lines = f.readlines()

    extrinsic = np.array([[float(x) for x in line.strip().split()] for line in lines[1:5]])
    cam2world = np.linalg.inv(extrinsic)

    return torch.tensor(cam2world, dtype=torch.float32)

def extract_pointcloud(imgs, pts3d, mask):
    pts3d = [p if isinstance(p, np.ndarray) else p.cpu().numpy() for p in pts3d]
    imgs = [i if isinstance(i, np.ndarray) else i.cpu().numpy() for i in imgs]
    mask = [m if isinstance(m, np.ndarray) else m.cpu().numpy() for m in mask]

    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    col = np.concatenate([i[m] for i, m in zip(imgs, mask)]).reshape(-1, 3)

    valid_msk = np.isfinite(pts.sum(axis=1))
    pts = pts[valid_msk]
    col = col[valid_msk]

    pointcloud = trimesh.PointCloud(pts, colors=col)
    return pointcloud

def get_pointcloud_from_scene(scene, min_conf_thr=2, clean_depth=False, TSDF_thresh=0):
    if scene is None:
        return None

    rgbimg = scene.imgs

    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    mask = to_numpy([c > min_conf_thr for c in confs])

    return extract_pointcloud(rgbimg, pts3d, mask)

def get_pointcloud_from_images(
    cam_dir, outdir, model, device, image_size,
    filelist, optim_level, lr1, niter1, lr2, niter2,
    min_conf_thr, matching_conf_thr, clean_depth,
    scene_graph, TSDF_thresh, shared_intrinsics, silent=False, **kw
    ):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    if optim_level == 'coarse':
        niter2 = 0

    os.makedirs(outdir, exist_ok=True)
    cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir) 
    scene, root = sparse_global_alignment(
        filelist, pairs, cache_dir, model,
        lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
        opt_depth='depth' in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        **kw
    )

    pointcloud = get_pointcloud_from_scene(scene, min_conf_thr=min_conf_thr,
                                        clean_depth=clean_depth, TSDF_thresh=TSDF_thresh)

    # Extract predicted and GT camera poses
    pred_cam2w_root = scene.get_im_poses()[root]
    gt_cam2w_root = load_cam_pose(root, cam_dir).to(pred_cam2w_root.device)

    # Use another image to compute scale from camera centers
    ref = 0 if root != 0 else 1
    pred_cam2w_ref = scene.get_im_poses()[ref]
    gt_cam2w_ref = load_cam_pose(ref, cam_dir).to(pred_cam2w_root.device)

    # Compute baseline distances
    gt_dist = torch.norm(gt_cam2w_root[:3, 3] - gt_cam2w_ref[:3, 3]).item()
    pred_dist = torch.norm(pred_cam2w_root[:3, 3] - pred_cam2w_ref[:3, 3]).item()
    scale = gt_dist / pred_dist

    # Decompose rotation + translation
    R_pred = pred_cam2w_root[:3, :3]
    t_pred = pred_cam2w_root[:3, 3] * scale

    R_gt = gt_cam2w_root[:3, :3]
    t_gt = gt_cam2w_root[:3, 3]

    # Compute full similarity transform from scaled predicted to GT
    R_align = R_gt @ R_pred.T
    t_align = t_gt - R_align @ t_pred

    T = torch.eye(4)
    T[:3, :3] = R_align
    T[:3, 3] = t_align

    # Transform and scale point cloud
    pts = pointcloud.vertices * scale
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_gt = (T.cpu().numpy() @ pts_h.T).T[:, :3]
    pointcloud.vertices = pts_gt

    return pointcloud

def main():
    # dtu_test_scan_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    # dtu_test_scan_ids = [106, 110, 114, 118, 122]
    dtu_test_scan_ids = [5]
    image_dir = '../data/dtu/Cleaned'
    cam_dir = '../data/dtu/Cameras'
    output_dir = '../predictions'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = model.to(device).eval()

    for scan_id in dtu_test_scan_ids:
        print(f"\nProcessing scan {scan_id}")
        img_paths = sorted(glob(os.path.join(image_dir, f'scan{scan_id}', '*_3_r5000.png')))
        if not img_paths:
            print(f"No valid images found for scan {scan_id}. Skipping.")
            continue

        pointcloud = get_pointcloud_from_images(
            cam_dir=cam_dir,
            outdir="out",
            model=model,
            device=device,
            image_size=512,
            filelist=img_paths,
            optim_level="refine+depth",
            lr1=0.07, niter1=1000,
            lr2=0.01, niter2=1000,
            min_conf_thr=10,
            matching_conf_thr=30,
            clean_depth=True,
            scene_graph="complete",
            TSDF_thresh=0,
            shared_intrinsics=False,
            silent=False
        )

        out_path = os.path.join(output_dir, f'scan{scan_id}_new.ply')
        pointcloud.export(out_path)
        print(f"Point cloud saved to {out_path}")

if __name__ == "__main__":
    main()