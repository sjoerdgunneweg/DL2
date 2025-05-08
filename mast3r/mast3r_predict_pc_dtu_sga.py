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


def extract_pointcloud(imgs, pts3d, mask):
    # Ensure inputs are NumPy arrays
    pts3d = [p if isinstance(p, np.ndarray) else p.cpu().numpy() for p in pts3d]
    imgs = [i if isinstance(i, np.ndarray) else i.cpu().numpy() for i in imgs]
    mask = [m if isinstance(m, np.ndarray) else m.cpu().numpy() for m in mask]

    # Extract valid points and colors
    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    col = np.concatenate([i[m] for i, m in zip(imgs, mask)]).reshape(-1, 3)

    # Remove invalid points
    valid_msk = np.isfinite(pts.sum(axis=1))
    pts = pts[valid_msk]
    col = col[valid_msk]

    # Return as trimesh PointCloud or raw arrays
    pointcloud = trimesh.PointCloud(pts, colors=col)
    return pointcloud

def get_pointcloud_from_scene(scene, min_conf_thr=2, clean_depth=False, TSDF_thresh=0):
    if scene is None:
        return None

    # Get inputs
    rgbimg = scene.imgs

    # Compute 3D points
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    # Build mask based on confidence threshold
    mask = to_numpy([c > min_conf_thr for c in confs])

    # Extract and return point cloud
    return extract_pointcloud(rgbimg, pts3d, mask)

def get_pointcloud_from_images(
    outdir, model, device, image_size,
    filelist, optim_level, lr1, niter1, lr2, niter2,
    min_conf_thr, matching_conf_thr, clean_depth,
    scene_graph, TSDF_thresh, shared_intrinsics, silent=False, **kw
    ):
    # Load and preprocess images
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    # Generate image pairs
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    if optim_level == 'coarse':
        niter2 = 0

    # Run Sparse Global Alignment (MASt3R pipeline)
    os.makedirs(outdir, exist_ok=True)
    cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir) 
    scene = sparse_global_alignment(
        filelist, pairs, cache_dir, model,
        lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
        opt_depth='depth' in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        **kw
    )

    # Extract colored point cloud
    pointcloud = get_pointcloud_from_scene(scene, min_conf_thr=min_conf_thr,
                                           clean_depth=clean_depth, TSDF_thresh=TSDF_thresh)
    
    return pointcloud

def main():
    scan_id = 24
    image_dir = '../data/dtu/Rectified'

    img_paths = sorted(glob(os.path.join(image_dir, f'scan{scan_id}_train', '*_3_r5000.png')))
    img_paths = [p for p in img_paths if int(re.findall(r'(\d+)_3_r5000', p)[0]) <= 48]
    assert len(img_paths) > 0, f"No valid images found for scan {scan_id}."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = model.to(device).eval()    

    pointcloud = get_pointcloud_from_images(
        outdir="out",
        model=model,
        device=device,
        image_size=512,
        filelist=img_paths,
        optim_level="refine+depth",
        lr1=1e-3, niter1=0,
        lr2=5e-4, niter2=0,
        min_conf_thr=2,
        matching_conf_thr=0.2,
        clean_depth=True,
        scene_graph="oneref-0",
        TSDF_thresh=0,
        shared_intrinsics=False,
        silent=False
    )

    out_path = f'../predictions/scan{scan_id}_new_sga.ply'
    pointcloud.export(out_path)
    print(f"Point cloud saved to {out_path}")

if __name__ == "__main__":
    main()




