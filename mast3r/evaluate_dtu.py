import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.datasets import get_data_loader


def parse_args():
    parser = argparse.ArgumentParser('MASt3R DTU Evaluation')
    parser.add_argument('--model', type=str,
                        default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        help="model constructor string")
    parser.add_argument('--checkpoint', required=True, type=str,
                        help="path to saved model checkpoint (.pth)")
    parser.add_argument('--dataset', type=str, default='DTU',
                        help="name of the DTU dataset split to evaluate")
    parser.add_argument('--batch_size', default=1, type=int,
                        help="batch size (number of scenes per batch)")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="number of data loading workers")
    parser.add_argument('--device', default='cuda', choices=['cuda','cpu'],
                        help="compute device")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="where to save evaluation logs")
    return parser.parse_args()


def compute_mvs_metrics(pred_points, gt_points):
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    accuracy = float(dist_pred_to_gt.mean())
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points)
    completeness = float(dist_gt_to_pred.mean())
    chamfer = 0.5 * (accuracy + completeness)
    return accuracy, completeness, chamfer


def evaluate_dtu(loader, model, device):
    metrics_list = []
    for data in loader:
        images = data['images'].to(device)
        extrinsics = data['extrinsics'].numpy()
        gt_points = data['gt_points'].numpy()

        with torch.no_grad():
            outputs = model(images)
        # Expect pointmaps: [N, H, W, 3]
        pointmaps = outputs['pointmaps'].cpu().numpy()

        pts = []
        for i in range(pointmaps.shape[0]):
            pm = pointmaps[i].reshape(-1, 3)
            pts_cam = pm
            R = extrinsics[i][:3, :3]
            t = extrinsics[i][:3, 3]
            pts_world = (R @ pts_cam.T + t[:, None]).T
            pts.append(pts_world)
        pred_points = np.concatenate(pts, axis=0)

        acc, comp, cham = compute_mvs_metrics(pred_points, gt_points)
        metrics_list.append((acc, comp, cham))

    avg = np.mean(metrics_list, axis=0)
    return {'accuracy': float(avg[0]), 'completeness': float(avg[1]), 'chamfer': float(avg[2])}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    # Build and load model
    model = eval(args.model)
    model.to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    # Build DTU loader
    loader = get_data_loader(f"{args.dataset}",
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_mem=True,
                             shuffle=False,
                             drop_last=False)
    print(f"Loaded DTU dataset '{args.dataset}' with {len(loader)} scenes.")

    # Evaluate
    print("Starting DTU evaluation...")
    results = evaluate_dtu(loader, model, device)

    print("\nDTU Evaluation Results:")
    print(f"Accuracy   : {results['accuracy']:.4f}")
    print(f"Completeness: {results['completeness']:.4f}")
    print(f"Chamfer    : {results['chamfer']:.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = Path(args.output_dir) / 'dtu_eval.json'
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()