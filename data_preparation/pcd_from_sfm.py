import os 
import argparse
import open3d as o3d


def main(src, dst):
    src_files = os.listdir(src)
    for name in src_files:
        if 'obj' not in name: continue
        id = name.replace('.obj','')
        fname = os.path.join(src,name)
        mesh = o3d.io.read_triangle_mesh(fname, enable_post_processing=True, print_progress=False)

        ## get point cloud from mesh
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = mesh.vertices
        mesh_pcd.colors = mesh.vertex_colors
        mesh_pcd.normals = mesh.vertex_normals

        ## sample points from mesh
        pcd = mesh.sample_points_uniformly(number_of_points=20000)
        # o3d.visualization.draw_geometries([pcd], window_name="point cloud from mesh, uniformly sampled")

        os.makedirs(dst, exist_ok=True)
        os.makedirs(os.path.join(dst,id,'laser'), exist_ok=True)
        os.makedirs(os.path.join(dst,id,'realsense'), exist_ok=True)
        os.makedirs(os.path.join(dst,id,'tf'), exist_ok=True)
        pcd_name = os.path.join(dst,id,'laser/fruit.ply')
        o3d.io.write_point_cloud(pcd_name,pcd)

if __name__ == "__main__":
  #%% Parse arguments
  parser = argparse.ArgumentParser(description="Prepare deep sdf training with measurement arm data")
  parser.add_argument("src", help="data source where the point meshes are stored")
  parser.add_argument("dst", help="destination folder")   
  args = parser.parse_args()

  main(args.src, args.dst)
