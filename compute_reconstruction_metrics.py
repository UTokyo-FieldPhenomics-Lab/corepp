import open3d as o3d
import os 

from tqdm import tqdm
from metrics_3d import chamfer_distance, precision_recall


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="shape completion main file, assume a pretrained deepsdf model")
    arg_parser.add_argument(
        "--ground-truth-dir",
        "-gt",
        dest="gt_dir",
        required=True,
    )
    arg_parser.add_argument(
        "--prediction-dir",
        "-pt",
        dest="pt_dir",
        required=True,
    )

    args = arg_parser.parse_args()

    cd = chamfer_distance.ChamferDistance()
    pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

    pt_dir = args.pt_dir
    gt_dir = os.path.join(args.gt_dir, '{}/laser/fruit.ply')

    for fname in tqdm(os.listdir(pt_dir)):
        pt_mesh = o3d.io.read_triangle_mesh(os.path.join(pt_dir,fname))
        gt_pcd = o3d.io.read_point_cloud(gt_dir.format(fname[:-4]))
        cd.update(gt_pcd,pt_mesh)
        pr.update(gt_pcd,pt_mesh)

    cd.compute()
    pr.compute_at_threshold(0.005)
