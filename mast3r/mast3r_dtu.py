import numpy as np
import trimesh
import os, copy, torch
from glob import glob
import re
from collections import defaultdict
import torchvision.transforms as tvf

from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import convert_dust3r_pairs_naming
from mast3r.cloud_opt.triangulation import matches_to_depths
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import select_pairs_of_crops
from visloc_demo import crops_inference, crop

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.cloud_opt.base_opt import clean_pointcloud
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
from dust3r.inference import inference
from dust3r_visloc.datasets.utils import get_HW_resolution


def torgb(x): return (x[0].permute(1, 2, 0).numpy() * .5 + .5).clip(min=0., max=1.)

def load_gt_camera(image_path, cam_dir, device='cuda', orig_size=(1200, 1600), new_size=(1200, 1600)):
    scale_y = new_size[0] / orig_size[0]
    scale_x = new_size[1] / orig_size[1]

    # Extract image ID
    image_id = int(re.search(r'(\d+)_3_r5000', image_path).group(1)) - 1
    cam_path = f"{cam_dir}/{image_id:08d}_cam.txt"

    with open(cam_path, 'r') as f:
        lines = f.readlines()

    ext = np.array([[float(x) for x in line.strip().split()] for line in lines[1:5]])
    K = np.array([[float(x) for x in line.strip().split()] for line in lines[7:10]])

    # Rescale intrinsics
    K[0, 0] *= scale_x  # fx
    K[1, 1] *= scale_y  # fy
    K[0, 2] *= scale_x  # cx
    K[1, 2] *= scale_y  # cy

    # Convert to torch tensors
    K_tensor = torch.tensor(K, dtype=torch.float32, device=device)         # [3, 3]
    ext_tensor = torch.tensor(ext[:3], dtype=torch.float32, device=device) # [3, 4]

    return K_tensor, ext_tensor

def resize_image_to_max(max_image_size, rgb, K):
    if rgb.ndim == 4:
        rgb = rgb.squeeze(0)
    H, W = rgb.shape[1:]

    if max_image_size and max(W, H) > max_image_size:
        islandscape = (W >= H)
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))

        resize_op = tvf.Resize(size=[HMax, WMax])
        rgb_tensor = resize_op(rgb)

        to_orig_max = np.array([[W / WMax, 0, 0],
                                [0, H / HMax, 0],
                                [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0],
                                  [0, HMax / H, 0],
                                  [0, 0, 1]])

        K_np = K.cpu().numpy() if isinstance(K, torch.Tensor) else K
        new_K = opencv_to_colmap_intrinsics(K_np)
        new_K[0, :] *= WMax / W
        new_K[1, :] *= HMax / H
        new_K = colmap_to_opencv_intrinsics(new_K)

        if isinstance(K, torch.Tensor):
            new_K = torch.tensor(new_K, dtype=torch.float32, device=K.device)
    else:
        rgb_tensor = rgb
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        new_K = K

    return rgb_tensor.unsqueeze(0), new_K, to_orig_max, to_resize_max, (HMax, WMax)

def extract_pointcloud(imgs, pts3d, mask):
    # Convert to numpy arrays if needed
    pts3d = [p if isinstance(p, np.ndarray) else p.cpu().numpy() for p in pts3d]
    imgs = [i if isinstance(i, np.ndarray) else i.cpu().numpy() for i in imgs]
    mask = [m if isinstance(m, np.ndarray) else m.cpu().numpy() for m in mask]

    # Fix: Reshape pts3d to 2D before applying mask
    pts_list = []
    col_list = []
    
    for p, i, m in zip(pts3d, imgs, mask):
        # Reshape 3D points from [H, W, 3] to [H*W, 3]
        p_reshaped = p.reshape(-1, 3)
        # Reshape images from [H, W, 3] to [H*W, 3] 
        i_reshaped = i.reshape(-1, 3)
        # Flatten mask from [H, W] to [H*W]
        m_flat = m.ravel()
        
        # Apply mask
        pts_list.append(p_reshaped[m_flat])
        col_list.append(i_reshaped[m_flat])
    
    # Concatenate all valid points and colors
    pts = np.concatenate(pts_list, axis=0)
    col = np.concatenate(col_list, axis=0)

    # Remove invalid points (NaN or inf)
    valid_msk = np.isfinite(pts.sum(axis=1))
    pts = pts[valid_msk]
    col = col[valid_msk]

    pointcloud = trimesh.PointCloud(pts, colors=col)

    return pointcloud

def get_coarse_matches(query_view, map_view, model, device, pixel_tol, fast_nn_params, edge_border=3):
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], []

    if pixel_tol == 0:
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params)
        HM, WM = map_view['rgb_rescaled'].shape[1:]
        HQ, WQ = query_view['rgb_rescaled'].shape[1:]
        # ignore small border around the edge
        valid_matches_map = (matches_im_map[:, 0] >= edge_border) & (matches_im_map[:, 0] < WM - edge_border) & (
            matches_im_map[:, 1] >= edge_border) & (matches_im_map[:, 1] < HM - edge_border)
        valid_matches_query = (matches_im_query[:, 0] >= edge_border) & (matches_im_query[:, 0] < WQ - edge_border) & (
            matches_im_query[:, 1] >= edge_border) & (matches_im_query[:, 1] < HQ - edge_border)
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        matches_confs = []
    else:
        H1, W1, DIM1 = PM.shape
        H2, W2, DIM2 = PQ.shape
        assert DIM1 == DIM2
        yM, xM = np.mgrid[0:H1, 0:W1].reshape(2, -1)
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
        )

    # from cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # rescale coordinates
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    # from colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5

    return matches_im_query, matches_im_map, matches_confs

def get_fine_matches(query_views, map_views, model, device, pixel_tol, fast_nn_params):
    assert pixel_tol > 0

    # Run the network on the cropped image regions and extract descriptors + confidence
    output = crops_inference([query_views, map_views], model, device, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()        # Descriptors for query image crops
    descs2 = pred2['desc'].clone()        # Descriptors for map image crops
    confs1 = pred1['desc_conf'].clone()   # Confidence for query descriptors
    confs2 = pred2['desc_conf'].clone()   # Confidence for map descriptors

    # Store outputs across all image pairs
    matches_im_map, matches_im_query, matches_confs = [], [], []

    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        H1, W1, DIM1 = pp2.shape
        H2, W2, DIM2 = pp1.shape
        assert DIM1 == DIM2

        y1, x1 = np.mgrid[0:H1, 0:W1].reshape(2, -1)

        # Run reciprocal NN match with fast filtering
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(
            pp2, pp1, subsample_or_initxy1=(x1, y1), pixel_tol=pixel_tol, **fast_nn_params
        )

        # Compute confidence for each match as the minimum of the two confidences
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]
        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )

        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi].cpu().numpy(), matches_im_map_ppi.copy(), norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi].cpu().numpy(), matches_im_query_ppi.copy(), norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        matches_confs.append(matches_confs_ppi)

    if len(matches_im_map) == 0:
        return [], [], []

    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    matches_confs = np.concatenate(matches_confs, axis=0)
    
    return matches_im_query, matches_im_map, matches_confs

def get_matches(query_view, map_view, coarse_to_fine, model, device, pixel_tol):
    maxdim = max(model.patch_embed.img_size)

    H, W = query_view['true_shape'][0]
    map_rgb_tensor = map_view['img'].squeeze(0).permute(1, 2, 0)
    query_rgb_tensor = query_view['img'].squeeze(0).permute(1, 2, 0)
    map_K = map_view['intrinsics'].cpu().numpy()
    query_K = query_view['intrinsics'].cpu().numpy()

    # Resize images and get intrinsic transforms
    query_view['rgb_rescaled'], _, query_view['to_orig'], query_to_resize_max, _ = resize_image_to_max(
        maxdim, query_view['img'], query_view['intrinsics'])
    map_view['rgb_rescaled'], _, map_view['to_orig'], map_to_resize_max, _ = resize_image_to_max(
        maxdim, map_view['img'], map_view['intrinsics'])
    
    assert torch.allclose(query_rgb_tensor, query_view['img'].squeeze(0).permute(1, 2, 0)), "query_view['img'] was modified"
    assert np.allclose(query_K, query_view['intrinsics'].cpu().numpy()), "query_view['intrinsics'] was modified"
    
    fast_nn_params = dict(device=device, dist='dot', block_size=2**13)

    if coarse_to_fine and (maxdim < max(W, H)):
        # Coarse matches on downscaled images
        coarse_matches_im0, coarse_matches_im1, _ = get_coarse_matches(
            query_view, map_view, model, device, 0, fast_nn_params)

        coarse_matches_im0 = geotrf(query_to_resize_max, coarse_matches_im0, norm=True)
        coarse_matches_im1 = geotrf(map_to_resize_max, coarse_matches_im1, norm=True)

        # Prepare image crops around coarse matches
        crops1, crops2 = [], []
        to_orig1, to_orig2 = [], []
        resolution = get_HW_resolution(H, W, maxdim=maxdim, patchsize=model.patch_embed.patch_size)

        for crop_q, crop_b, _ in select_pairs_of_crops(map_rgb_tensor,
                                                       query_rgb_tensor,
                                                       coarse_matches_im1,
                                                       coarse_matches_im0,
                                                       maxdim=maxdim,
                                                       overlap=0.5,
                                                       forced_resolution=[resolution, resolution]):
            
            # Crop map and query image regions, return uncropping transforms
            c1, _, _, trf1 = crop(map_rgb_tensor, None, None, crop_q, map_K)
            c2, _, _, trf2 = crop(query_rgb_tensor, None, None, crop_b, query_K)
            crops1.append(c1)
            crops2.append(c2)
            to_orig1.append(trf1)
            to_orig2.append(trf2)

        if len(crops1) == 0 or len(crops2) == 0:
            matches_im_query, matches_im_map, matches_conf = [], [], []
        else:
            crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
            if len(crops1.shape) == 3:  # single crop fallback
                crops1, crops2 = crops1[None], crops2[None]
            to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)

            # Create input dicts for fine matching
            map_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                    instance=['1' for _ in range(crops1.shape[0])],
                                    to_orig=to_orig1)
            query_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                    instance=['2' for _ in range(crops2.shape[0])],
                                    to_orig=to_orig2)

            # Run fine matching on high-res cropped regions
            matches_im_query, matches_im_map, matches_conf = get_fine_matches(
                query_crop_view, map_crop_view,
                model, device,
                pixel_tol, fast_nn_params
            )
    else:
        # Use only coarse matching (no fine refinement)
        matches_im_query, matches_im_map, matches_conf = get_coarse_matches(
            query_view, map_view, model, device, pixel_tol, fast_nn_params)

    return matches_im_query, matches_im_map, matches_conf

@torch.no_grad()
def get_pointcloud_from_images(
    cam_dir, model, device, image_size,
    filelist, min_conf_thr, clean_depth,
    scene_graph, symmetrize, coarse_to_fine, pixel_tol, silent=False):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=symmetrize, sim_mat=None)

    # Convert pair naming convention from dust3r to mast3r
    pairs = convert_dust3r_pairs_naming(filelist, pairs)
    
    ref_to_pairs = defaultdict(list)

    for pair in pairs:
        ref_path = pair[0]['instance']
        ref_to_pairs[ref_path].append(pair)
    
    matches_list = []
    intrinsics_list = []
    extrinsics_list = []

    for ref_path, pair_list in ref_to_pairs.items():
        intrinsics = []
        extrinsics = []

        # Add reference camera first
        K_ref, ext_ref = load_gt_camera(ref_path, cam_dir)
        intrinsics.append(K_ref)
        extrinsics.append(ext_ref)

        H, W = pair_list[0][0]['true_shape'][0]
        matches = torch.zeros((len(pair_list), H, W, 5), dtype=torch.float32, device=device)

        for i, (ref, other) in enumerate(pair_list):
            K, ext = load_gt_camera(other['instance'], cam_dir)
            intrinsics.append(K)
            extrinsics.append(ext)
            other['intrinsics'] = K
            ref['intrinsics'] = K_ref

            matches_im_query, matches_im_map, matches_conf = get_matches(ref, other, coarse_to_fine, model, device, pixel_tol)

            for idx in range(len(matches_im_query)):
                x_ref, y_ref = matches_im_query[idx]
                x_other, y_other = matches_im_map[idx]
                confidence = float(matches_conf[idx])

                if 0 <= x_ref < W and 0 <= y_ref < H:
                    matches[i, y_ref, x_ref, 0] = x_ref
                    matches[i, y_ref, x_ref, 1] = y_ref
                    matches[i, y_ref, x_ref, 2] = x_other
                    matches[i, y_ref, x_ref, 3] = y_other
                    matches[i, y_ref, x_ref, 4] = confidence

        matches_list.append(matches.unsqueeze(0))                    # [1, Npairs, H, W, 5]
        intrinsics_list.append(torch.stack(intrinsics).unsqueeze(0))       # [1, Ncams, 3, 3]
        extrinsics_list.append(torch.stack(extrinsics).unsqueeze(0))       # [1, Ncams, 3, 4]

    matches_tensor = torch.cat(matches_list, dim=0)            # [B, Npairs, H, W, 5]
    intrinsics_tensor = torch.cat(intrinsics_list, dim=0)  # [B, Ncams, 3, 3]
    extrinsics_tensor = torch.cat(extrinsics_list, dim=0)  # [B, Ncams, 3, 4]

    pts3d, depthmaps, confs = matches_to_depths(
        intrinsics=intrinsics_tensor,
        extrinsics=extrinsics_tensor,
        matches=matches_tensor,
        min_num_valids_ratio=0.3,
    )

    if clean_depth:
        confs = clean_pointcloud(confs, intrinsics_tensor[:, 0], extrinsics_tensor[:, 0], depthmaps, pts3d)

    mask = (torch.stack(confs) > min_conf_thr)
    rgbimg = [torgb(pair_list[0][0]['img']) for pair_list in ref_to_pairs.values()]

    return extract_pointcloud(rgbimg, pts3d, mask)

def main():
    # dtu_test_scan_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    dtu_test_scan_ids = [24]
    image_dir = '../data/dtu/Cleaned'
    cam_dir = '../data/dtu/Cameras'
    output_dir = '../predictions'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = model.to(device).eval()

    for scan_id in dtu_test_scan_ids:
        print(f"\nProcessing scan {scan_id}")
        img_paths = glob(os.path.join(image_dir, f'scan{scan_id}', '*_3_r5000.png'))
        img_paths = sorted(
            img_paths,
            key=lambda p: int(re.search(r'(\d+)_3_r5000', os.path.basename(p)).group(1))
        )

        if not img_paths:
            print(f"No valid images found for scan {scan_id}. Skipping.")
            continue

        if len(img_paths) != 49:
            print(f"Different amount of images for this scan: {len(img_paths)} in stead of 49.")

        pointcloud = get_pointcloud_from_images(
            cam_dir=cam_dir,
            model=model,
            device=device,
            image_size=1600,
            filelist=img_paths,
            min_conf_thr=1.5,
            clean_depth=True,
            scene_graph="oneref-0",
            symmetrize=False,
            coarse_to_fine=True,
            pixel_tol=5
        )

        out_path = os.path.join(output_dir, f'scan{scan_id}_fine.ply')
        pointcloud.export(out_path)
        print(f"Point cloud saved to {out_path}")

if __name__ == "__main__":
    main()