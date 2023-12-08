import os 
import open3d as o3d


ids = [
    '3R6-4', '4R5-1', '5R1-7', 'R7-7', 'R13-8'
]

for id in ids:
    fname = "/media/federico/2.0_TB_HDD/ipb_x_qut/PieterData/3d_scans_for_federico/files_for_federico/{}.obj".format(id)
    mesh = o3d.io.read_triangle_mesh(fname, enable_post_processing=True, print_progress=False)

    ## get point cloud from mesh
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = mesh.vertices
    mesh_pcd.colors = mesh.vertex_colors
    mesh_pcd.normals = mesh.vertex_normals

    ## sample points from mesh
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    # o3d.visualization.draw_geometries([pcd], window_name="point cloud from mesh, uniformly sampled")

    pcd_name = '/home/federico/Desktop/{}.ply'.format(id)
    o3d.io.write_point_cloud(pcd_name,pcd)
