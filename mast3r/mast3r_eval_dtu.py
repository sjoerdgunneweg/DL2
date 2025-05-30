# -----------------------------------------------------------------------------
# Adapted from: https://github.com/jzhangbs/DTUeval-python/blob/master/eval.py
# Description: Modified to evaluate a list of DTU scans instead of a single scan
# -----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse
import os

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def evaluate_single_scan(scan_id, args):
    if args.mode == 'mesh':
        data_mesh = o3d.io.read_triangle_mesh(args.data.format(scan_id))
        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = args.downsample_density * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)
        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)
        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)
    else:
        data_pcd_o3d = o3d.io.read_point_cloud(args.data.format(scan_id))
        data_pcd = np.asarray(data_pcd_o3d.points)

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=args.downsample_density, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=args.downsample_density, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{scan_id}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)
    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) == 3
    data_in = data_down[inbound]
    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{scan_id:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    nn_engine.fit(stl)
    dist_d2s, _ = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    mean_d2s = dist_d2s[dist_d2s < args.max_dist].mean()

    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{scan_id}.mat')['P']
    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, _ = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < args.max_dist].mean()

    over_all = (mean_d2s + mean_s2d) / 2

    print(f"Scan {scan_id:03}: Accuracy = {mean_d2s:.3f}, Completeness = {mean_s2d:.3f}, Chamfer distance = {over_all:.3f}")
    return mean_d2s, mean_s2d, over_all

if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path with {} placeholder for scan ID")
    parser.add_argument('--scan_list', type=int, nargs='+', default=[1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118])
    parser.add_argument('--mode', type=str, default='pcd', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()

    all_results = []
    for scan_id in args.scan_list:
        try:
            result = evaluate_single_scan(scan_id, args)
            all_results.append((scan_id, *result))
        except Exception as e:
            print(f"Error in scan {scan_id:03}: {e}")

    print("\n=== Summary ===")
    if all_results:
        all_d2s = np.mean([r[1] for r in all_results])
        all_s2d = np.mean([r[2] for r in all_results])
        all_avg = np.mean([r[3] for r in all_results])
        
        print(f"\nAverage accuracy: {all_d2s:.3f}")
        print(f"Average completeness: {all_s2d:.3f}")
        print(f"Average chamfer distance: {all_avg:.3f}")
