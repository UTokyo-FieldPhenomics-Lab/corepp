#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare data for deepsdf training using the measurement arm data.

Created on Wed Jun  2 11:39:41 2021

@author: laebe
"""
import os
import numpy as np
import open3d as o3d
import argparse

def np2o3d(xyz, colors=None, normals=None):
    """
    convert numpy point cloud to open3d

    Args: 
      xyz: numpy array representing a cloud

    Returns;
      pcd: ope3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz)[:, :3])

    if normals: pcd.normals = o3d.utility.Vector3dVector(np.asarray(xyz)[:, 3:6])
    if colors: pcd.colors = o3d.utility.Vector3dVector(np.asarray(xyz)[:, 6:9])

    return pcd


def generate_tsdf_samples(swl_points: np.ndarray, no_samples_per_point: int,
                          tsdf_positive: float, tsdf_negative) -> np.ndarray:
  """ 
  Generate samples of coordinates near the surface together with their sdf values.
  
  Args:
    swl_points: A no_points x 6 array of points together with the viewpoints
                (0:3 -> points, 3:6 corresponding viewpoint)
    no_samples_per_point: each point is sampled no_samples_per_point times.
    tsdf_positive: maximum positive (outside) distance from surface
    tsdf_negative: maximum negative (inside) distance from surface 
                   (give a positive float here)
                
  Returns: 
    a tuple (pos, neg) of two numpy arrays, one for positive 
    sdf values (outside), one for negative (inside)
    Each array is a no_samplesx4 float32 array which includes 
    the point coordinates (colums 0,1,2) and sdf value (colum 3)
  """
  no_points = swl_points.shape[0]
  
  offset_vectors = swl_points[:,3:6] - swl_points[:,0:3]
  offset_vectors = offset_vectors / np.expand_dims(np.linalg.norm(offset_vectors, axis=1), axis=-1)
  offset_vectors = np.repeat(offset_vectors, no_samples_per_point, axis=0)

  tsdf_positive_offset = tsdf_positive * np.random.rand(no_points*no_samples_per_point, 1)
  tsdf_positive_offset_env = 0.1 * np.random.rand(no_points*no_samples_per_point, 1)
  tsdf_negative_offset = (-1.0) * tsdf_negative * np.random.rand(no_points*no_samples_per_point, 1)
  
  sample_points = np.repeat(swl_points[:,0:3], no_samples_per_point, axis=0)

  pos = sample_points + tsdf_positive_offset * offset_vectors
  env = sample_points + tsdf_positive_offset_env * offset_vectors
  neg = sample_points + tsdf_negative_offset * offset_vectors
  
  pos = np.concatenate ((pos, tsdf_positive_offset), axis=1)
  env = np.concatenate ((env, tsdf_positive_offset_env), axis=1)
  pos = np.concatenate((pos,env), axis=0)
  neg = np.concatenate ((neg, tsdf_negative_offset), axis=1)
    
  return (pos,neg)

def show_swl_points(swl_points: np.ndarray, window_name='Points') -> None:
  pcdswl = o3d.geometry.PointCloud()
  pcdswl.points = o3d.utility.Vector3dVector(swl_points[:,0:3])
  pcdswl.paint_uniform_color(np.array([0,1,0]))
  
  pcdviewpoints = o3d.geometry.PointCloud()
  pcdviewpoints.points = o3d.utility.Vector3dVector(swl_points[:,3:6])
  pcdviewpoints.paint_uniform_color(np.array([1,0,0]))
  
  coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
  
  o3d.visualization.draw_geometries([pcdswl, pcdviewpoints, coordinate_system], 
                                    point_show_normal=False, window_name=window_name)

def show_pos_neg(pos: np.ndarray, neg: np.ndarray, swl_points: np.ndarray = None) -> None:
  pcdpos = o3d.geometry.PointCloud()
  pcdpos.points = o3d.utility.Vector3dVector(pos[:,0:3])
  pcdpos.paint_uniform_color(np.array([0,1,0]))

  pcdneg = o3d.geometry.PointCloud()
  pcdneg.points = o3d.utility.Vector3dVector(neg[:,0:3])
  pcdneg.paint_uniform_color(np.array([1,0,0]))
  
  if swl_points is not None:
    pcdswl = o3d.geometry.PointCloud()
    pcdswl.points = o3d.utility.Vector3dVector(swl_points[:,0:3])
    pcdswl.paint_uniform_color(np.array([0,0,1]))

    o3d.visualization.draw_geometries([pcdpos, pcdneg, pcdswl], window_name="green: positive sdf, red: negative sdf, blue: original fruit")           
    o3d.visualization.draw_geometries([pcdneg, pcdswl], window_name="red: negative sdf, blue: original fruit")           
    o3d.visualization.draw_geometries([pcdpos, pcdswl], window_name="green: positive sdf, blue: original fruit")               
  else:
    o3d.visualization.draw_geometries([pcdpos, pcdneg], window_name="green: positive sdf, red: negative sdf")

  
if __name__ == "__main__":
  #%% Parse arguments
  parser = argparse.ArgumentParser(description="Prepare deep sdf training with measurement arm data")
  parser.add_argument("--src", default='./data/exp-name', type=str, 
                      help="name of the folder containing the point clouds processed by 'pcd_from_sfm.py' (either .swl files from the measurement arm or .ply pointcloud files)")
  parser.add_argument("--no_of_samples", default=100000, type=int,
                      help="Number of positive/negative samples, default: 100.000")
                      
  parser.add_argument("--tsdf_positive", default=0.04, type=float,
                      help="Maximal positive (outside) sdf value, default:0.04")
  parser.add_argument("--tsdf_negative", default=0.01, type=float,
                      help="Maximal negative (inside) sdf value, default:0.01")
  # parser.add_argument("--bounding_box", nargs=6, type=float,
  #                     help="Bounding box around the object: minx miny minz  maxx maxy maxz. If not given, no cut is done.")
  parser.add_argument("--bounding_box", action="store_true",
                      help="Bounding box around the object: minx miny minz  maxx maxy maxz. If not given, no cut is done.")
  parser.add_argument("--remove_noise", action="store_true",
                      help="Remove noise. (Not done per default)")
  parser.add_argument("--show_sdf_points", action="store_true",
                      help="show 3D view of the 3D points of the generated samples")
  parser.add_argument("--show_input_points", action="store_true",
                      help="show 3D view of complete 3D points and viewpoints loaded from the swl/ply file.")
  parser.add_argument("--show_truncated_points", action="store_true",
                      help="show 3D view of the 3D points and viewpoints after appling the bound box cutout.")

  args = parser.parse_args()
  
    
  #%% Load
  print('Load %s ...' % args.src, flush=True)
  for fname in os.listdir(args.src):
    
    path_to_file = os.path.join(args.src, fname, 'laser/fruit.ply')
    # filename, file_extension = os.path.splitext(path_to_file)

    swl_points = o3d.io.read_point_cloud(path_to_file)
    points = np.asarray(swl_points.points)
    viewpoints = points + np.asarray(swl_points.normals)
    swl_points = np.concatenate([points, viewpoints], axis=1)

      
    #%% Show
    if args.show_input_points:
      # import ipdb; ipdb.set_trace()
      show_swl_points(swl_points, 'All loaded points (green: points, red: viewpoints')
    
    #%% Cut out 
    if args.bounding_box:
      print('Cutout object ...', flush=True)
      # bbox_min=args.bounding_box[0:3]
      # bbox_max=args.bounding_box[3:6]
      
      bbox_min=np.array([-1, -1.0, 0.34])
      bbox_max=np.array([ 1, 1, 1.0])
      swl_points = swl_points[(swl_points[:,0]>=bbox_min[0]) & (swl_points[:,0]<=bbox_max[0]) & 
                              (swl_points[:,1]>=bbox_min[1]) & (swl_points[:,1]<=bbox_max[1]) & 
                              (swl_points[:,2]>=bbox_min[2]) & (swl_points[:,2]<=bbox_max[2]), : ]
      
      if args.show_truncated_points:
        show_swl_points(swl_points, 'Points after applying bounding box')
 
    #%% Removing noise
    if args.remove_noise:
      print('Remove noise ...', flush=True)
      swl_pcd = np2o3d(swl_points, normals=True)
      swl_pcd = swl_pcd.voxel_down_sample(voxel_size=0.001)
      o3d.visualization.draw_geometries([swl_pcd])
      cl, ind = swl_pcd.remove_statistical_outlier(nb_neighbors=50,
                                                          std_ratio=.5)
      o3d.visualization.draw_geometries([cl])
    
      swl_points = np.asarray(cl.points)
      swl_viewpoint = np.asarray(cl.normals)
    
      swl_points = np.hstack((swl_points, swl_viewpoint))
      # np.savetxt("/home/federico/Datasets/swl_points_test_leaf.txt", swl_points)

    #%% Generate sdf samples
    print('Generate sdf samples ...', flush=True)
    no_samples_per_point = int(np.ceil(args.no_of_samples/swl_points.shape[0]))
    (pos,neg)  = generate_tsdf_samples(swl_points, no_samples_per_point=no_samples_per_point,
                            tsdf_positive=args.tsdf_positive, tsdf_negative=args.tsdf_negative)
    # Truncate to number of samples
    pos = pos[np.random.choice(pos.shape[0], args.no_of_samples, replace=False), :]
    neg = neg[np.random.choice(neg.shape[0], args.no_of_samples, replace=False), :]
    
    #%% Save
    output_filename = os.path.join(args.src, fname, 'laser/samples.npz')
    print('Save to %s ...' % output_filename, flush=True)
    np.savez(os.path.join(output_filename), pos=pos, neg=neg)

    #%% Saving pcd for inspection
    # pcd_fname = output_filename.replace('.npz','.ply')
    # pcd = np2o3d(swl_points)
    # pcd.translate(-pcd.get_center())
    # pcd.estimate_normals()
    # pcd.orient_normals_towards_camera_location([0,0,0])
    # pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    # o3d.io.write_point_cloud(pcd_fname, pcd)
    
    #%% Show result
    if args.show_sdf_points:
      show_pos_neg(pos, neg, swl_points)


