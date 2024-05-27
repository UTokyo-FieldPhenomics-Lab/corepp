import os
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def get_concave_hull_3d(pcd, alpha_st=0.005, alpha_step=0.005):
    alpha_value = alpha_st
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
    while not alpha_mesh.is_watertight():
        alpha_value += alpha_step
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)

    return alpha_mesh, alpha_mesh.get_volume(), alpha_value


if __name__ == "__main__":
    folder_path = "/home/pieter/shape_completion/deepsdf/experiments/potato_32/Reconstructions/130/Meshes/complete/test"
    gt_dir = "/home/pieter/shape_completion/data/potato"
    files = [f for f in os.listdir(folder_path) if f.endswith(".ply")]

    df = pd.read_csv("/mnt/data/PieterBlok/Potato/Data/ground_truth_measurements/ground_truth.csv")
    columns = ['potato_id', 'split', 'sfm_volume_ml', 'deepsdf_volume_ml', 'diff_ml', 'abs_diff_ml', 'rel_error_%']
    save_df = pd.DataFrame(columns=columns)

    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    cmap = plt.get_cmap('RdYlGn_r')

    for file_name in tqdm(files):
        file_path = os.path.join(folder_path, file_name)
        instance_name = os.path.splitext(file_name)[0]
        gt_path = os.path.join(gt_dir, instance_name) + "/laser/fruit.ply"

        gt = o3d.io.read_point_cloud(gt_path)
        gt_volume = df.loc[df['label'] == instance_name, 'volume_metashape'].values[0]

        deepsdf = o3d.io.read_point_cloud(file_path)
        hull, _ = deepsdf.compute_convex_hull()

        hull.remove_degenerate_triangles()
        hull.remove_duplicated_triangles()
        hull.remove_duplicated_vertices()
        hull.remove_non_manifold_edges()
        hull.remove_unreferenced_vertices()

        if hull.is_watertight():
            deepsdf_volume = hull.get_volume() * 1e6
            deepsdf_mesh = hull
        else:
            deepsdf_mesh, deepsdf_volume, alpha_value = get_concave_hull_3d(deepsdf)
            deepsdf_volume = deepsdf_volume * 1e6

        diff = gt_volume - deepsdf_volume
        abs_diff = abs(diff)
        rel_error = (abs_diff / gt_volume) * 100

        mesh_error = o3d.geometry.TriangleMesh()
        mesh_error.vertices = deepsdf_mesh.vertices
        mesh_error.triangles = deepsdf_mesh.triangles

        pcd = o3d.geometry.PointCloud()
        pcd.points = deepsdf_mesh.vertices
        dist_pt_2_gt = np.asarray(pcd.compute_point_cloud_distance(gt))
        dist_pt_2_gt -= dist_pt_2_gt.min()
        dist_pt_2_gt /= 0.01 ## dist_pt_2_gt /= dist_pt_2_gt.max()
        color = cmap(dist_pt_2_gt)[:,:-1]
        mesh_error.vertex_colors = o3d.utility.Vector3dVector(color)

        # window_name = f"{file_name}, GT: {gt_volume:.0f} ml, DSF: {deepsdf_volume:.0f} ml"
        # o3d.visualization.draw_geometries([mesh_error, gt, cs], window_name=window_name)
        # o3d.visualization.draw_geometries([mesh_error.translate([.1,0,0]), gt, cs], window_name=window_name)
        
        cur_data = {
                    'potato_id': instance_name,
                    'split': 'test',
                    'sfm_volume_ml': gt_volume,
                    'deepsdf_volume_ml': round(deepsdf_volume, 1),
                    'diff_ml': round(diff, 1),
                    'abs_diff_ml': round(abs_diff, 1),
                    'rel_error_%': round(rel_error, 1)
                    }
        save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
        save_df.to_csv("/mnt/data/PieterBlok/Potato/Results/deepsdf_results_test.csv", mode='w+', index=False)

    filtered = save_df[save_df['deepsdf_volume_ml'] != 0]
    rmse_volume = mean_squared_error(filtered['sfm_volume_ml'].values, filtered['deepsdf_volume_ml'].values, squared=False)

    cur_data = {
                'potato_id': "",
                'split': "",
                'sfm_volume_ml': "",
                'deepsdf_volume_ml': "",
                'diff_ml': "",
                'abs_diff_ml': "rmse",
                'rel_error_%': round(rmse_volume, 1)
                }
    save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
    save_df.to_csv("/mnt/data/PieterBlok/Potato/Results/deepsdf_results_test.csv", mode='w+', index=False)