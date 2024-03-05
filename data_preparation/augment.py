#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import pathlib
import open3d as o3d
import numpy as np
import copy
import argparse
import json

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from prepare_deepsdf_training_data import generate_tsdf_samples, show_pos_neg


SAVE = True

if __name__ == "__main__":
  #%% Parse arguments
  parser = argparse.ArgumentParser(description="Data augmentation for DeepSDF pretraining")
  parser.add_argument("--json_config_filename", help="json filename with the parameters for augmentation")
  parser.add_argument("--src", default=".",
                    help="Path to dataset root, where listdir will find p1,p2,p3,etc. Default: .")
  parser.add_argument("--dst", default="/tmp",
                    help="Path to store the augmented dataset")
  parser.add_argument("--show_augmented_pointcloud", action="store_true",
                    help="Show the augmented pointcloude")
  parser.add_argument("--show_augmented_sdfsamples", action="store_true",
                    help="Show the generated sdf samples")

  args = parser.parse_args()
  
  # Load parameters from json
  with open(args.json_config_filename) as json_file:
    config = json.load(json_file)
  
  # Fields which should be there in the config:
  #min_scalefactor = 0.5
  #max_scalefactor = 2.0
  #max_rotation_angle_degree = 30.0
  #max_shear = 0.5
  #no_of_augmentations = 10
  
  # For visu
  coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(
          size=0.1, origin=[0, 0, 0])

  for idx in os.listdir(args.src):
    path = os.path.join(args.src,idx)
    if not os.path.isdir(path): continue

    path = os.path.join(path,'laser/fruit.ply')
    if not os.path.isfile(path): 
      print('%s does not exist, continue.' % path)
      continue
    
    print("Reading pointcloud ", path)
    pcd = o3d.io.read_point_cloud(path)

    for jdx in range(config['no_of_augmentations']):

      # 1. Scale
      I = np.eye(4)
      scale = np.random.uniform(config['min_scalefactor'],
                                config['max_scalefactor'], size=(1,4)) 
      print("Using scales %f, %f, %f" % (scale[0,0], scale[0,1], scale[0,2]), flush=True)      
      # don't scale the homogeneous coord !
      scale[0,-1] = 1
      T = scale*I

      # 2. Rotation around Z
      angle = np.random.uniform(-config['max_rotation_angle_degree']* np.pi/180.0,
                                +config['max_rotation_angle_degree']* np.pi/180.0)
      print("Using rotation of %d degrees" % (angle * 180/np.pi))
      R = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray([[0,0, angle]]).T)
      T_R = np.eye(4)
      T_R[0:3,0:3] = R
      T = T_R @ T
      
      # 3. Shear in x direction
      shear = np.random.uniform(-config['max_shear'], +config['max_shear'], size=(2,)) 
      print("Using shear of %f, %f" % (shear[0], shear[1]), flush=True)
      T_shear = np.eye(4)
      T_shear[0,1] = shear[0]
      T_shear[0,2] = shear[1]
      T = T_shear @ T

      tmp = copy.deepcopy(pcd)
      tmp.transform(T)

      # forcing normals to point away from origin
      tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
      tmp.orient_normals_towards_camera_location()
      tmp.normals = o3d.utility.Vector3dVector(-np.array(tmp.normals))

      # from data_preparation
      tsdf_positive = 0.02
      tsdf_negative = 0.01
      swl_points = np.hstack((np.array(tmp.points),np.array(tmp.normals)))

      no_of_samples = 100000
      no_samples_per_point = int(np.ceil(no_of_samples/swl_points.shape[0]))

      (pos, neg) = generate_tsdf_samples(swl_points, no_samples_per_point=no_samples_per_point,
                      tsdf_positive=tsdf_positive, tsdf_negative=tsdf_negative)

      pos = pos[np.random.choice(pos.shape[0], no_of_samples, replace=False), :]
      neg = neg[np.random.choice(neg.shape[0], no_of_samples, replace=False), :]
      
      #%% Save
      if SAVE:
        output_dir = os.path.join(args.dst, str(idx)+'_%02d' % jdx, "laser")
        print('Save pointcloud and samples to %s ...' % output_dir, flush=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # The pointcloud
        o3d.io.write_point_cloud(os.path.join(output_dir,'fruit.ply'), tmp, compressed=True)
        # The samples
        np.savez(os.path.join(output_dir,'samples.npz'), pos=pos, neg=neg)
        
      #%% Show
      if args.show_augmented_pointcloud:
        pcdvisu = copy.deepcopy(pcd)
        pcdvisu.translate(np.asarray([[0.2,0,0]]).T)
        o3d.visualization.draw_geometries([pcdvisu, tmp, coordinate_system])
      if args.show_augmented_sdfsamples: 
        show_pos_neg(pos, neg, swl_points)
