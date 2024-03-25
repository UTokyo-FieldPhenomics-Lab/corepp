import os
import trimesh

import numpy as np
import open3d as o3d

def mesh2pcd(mesh_path, points_num):
    # read and sample mesh by trimesh
    mesh_tri = trimesh.load(mesh_path, force='mesh')
    samples, face_idx, colors = trimesh.sample.sample_surface(mesh_tri, points_num, sample_color=True)

    o3d_rgb = colors[:,0:3] / 255

    # convert trimesh to open3d objects
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(samples)
    final_pcd.colors = o3d.utility.Vector3dVector(o3d_rgb)

    return final_pcd


if __name__ == "__main__":

    POINTS_NUM = 30000

    mesh_folder = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/2_SfM/1_mesh'
    pcd_folder = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/2_SfM/2_pcd'
    mesh_id_list = [i for i in os.listdir(mesh_folder)]

    already_exist = [i.split('_')[0] for i in os.listdir(pcd_folder) if '.ply' in i]

    for potato_idx in mesh_id_list:

        print(f'Convert [{potato_idx}]')

        if potato_idx in already_exist:
            continue

        sampled = mesh2pcd(os.path.join(mesh_folder, potato_idx, potato_idx + '.obj'), POINTS_NUM)

        pcd_path = os.path.join( pcd_folder, f"{potato_idx}_{POINTS_NUM}.ply")

        o3d.io.write_point_cloud(
            pcd_path,
            sampled
        )

        print(f" ---> Save to {os.path.abspath(pcd_path)}")