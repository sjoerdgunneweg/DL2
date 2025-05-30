import numpy as np
import trimesh
import argparse
import os

def load_pointcloud(file_path):
    # Load point cloud from .ply file using trimesh
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    pcd = trimesh.load(file_path, process=False)
    if not isinstance(pcd, trimesh.points.PointCloud):
        raise TypeError(f"Expected a PointCloud, got {type(pcd)}")
    return pcd

def denoise_pointcloud(pcd, quantile=0.99):
    # Remove outlier points far from the median centroid
    centroid = np.median(pcd.vertices, axis=0)
    dist_to_centroid = np.linalg.norm(pcd.vertices - centroid, axis=-1)
    dist_thr = np.quantile(dist_to_centroid, quantile)
    valid = dist_to_centroid < dist_thr
    return trimesh.points.PointCloud(pcd.vertices[valid], color=pcd.colors[valid])

def main(file_path, denoise=False, point_size=2):
    pcd = load_pointcloud(file_path)

    if denoise:
        pcd = denoise_pointcloud(pcd)

    scene = trimesh.Scene()
    scene.add_geometry(pcd)
    scene.show(line_settings={'point_size': point_size})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .ply point cloud file")
    parser.add_argument("--denoise", action="store_true", help="Enable outlier filtering")
    parser.add_argument("--point_size", type=int, default=2, help="Point size in the viewer")
    args = parser.parse_args()

    main(args.file, denoise=args.denoise, point_size=args.point_size)
