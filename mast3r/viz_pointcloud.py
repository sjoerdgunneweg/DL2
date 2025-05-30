import numpy as np
import open3d as o3d
import argparse
import os

def load_pointcloud(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError("Loaded point cloud is empty.")
    if not pcd.has_colors():
        print("Warning: point cloud does not contain colors.")
    return pcd

def denoise_pointcloud(pcd, quantile=0.99):
    # Remove outliers based on distance to median
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    centroid = np.median(points, axis=0)
    dist_to_centroid = np.linalg.norm(points - centroid, axis=-1)
    dist_thr = np.quantile(dist_to_centroid, quantile)
    valid = dist_to_centroid < dist_thr

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[valid])
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    return filtered_pcd

def main(file_path, denoise=False, point_size=2):
    pcd = load_pointcloud(file_path)

    if denoise:
        pcd = denoise_pointcloud(pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Colored Point Cloud", width=800, height=600)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the .ply point cloud file")
    parser.add_argument("--denoise", action="store_true", default=True, help="Enable outlier filtering")
    parser.add_argument("--point_size", type=int, default=2, help="Point size in the viewer")
    args = parser.parse_args()

    main(args.file, denoise=args.denoise, point_size=args.point_size)
