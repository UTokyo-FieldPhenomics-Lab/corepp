#!/usr/bin/env python3

from re import I
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToTensor, Compose, Resize

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
import deepsdf.deep_sdf.o3d_utils as o3d_utils


from sdfrenderer.grid import Grid3D
from dataloaders.transforms import Pad
from dataloaders.cameralaser_w_masks import MaskedCameraLaserData

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled
import networks.utils as net_utils

import open3d as o3d
import numpy as np

import time
import json

from utils import sdf2mesh, tensor_dict_2_float_dict
# from metrics_3d import chamfer_distance, precision_recall

# cd = chamfer_distance.ChamferDistance()
# pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)


def from_pred_sdf_to_mesh(pred_sdf, grid_points, voxel_size, t=0):
    keep_idx = pred_sdf<t
    keep_points = grid_points[keep_idx.squeeze()]
    pcd_grid = o3d.geometry.PointCloud()
    pcd_grid.points = o3d.utility.Vector3dVector(keep_points.detach().cpu())
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_grid, depth=3)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_grid, voxel_size=voxel_size)
    hull, _ = pcd_grid.compute_convex_hull()
    hull.remove_degenerate_triangles()
    hull.remove_duplicated_triangles()
    hull.remove_duplicated_vertices()
    hull.remove_non_manifold_edges()
    hull.remove_unreferenced_vertices()
    return hull, viewpoint_grid

def main_function(decoder, pretrain, cfg, latent_size):
    volumes = []
    print("\n WARNING I'M NOT SAVING THE PREDICTIONS AT THE MOMENT \n")

    exec_time = []

    with open(cfg) as json_file:
        param = json.load(json_file)

    device = 'cuda'

    # creating variables for 3d grid for diff SDF renderer
    threshold = param['threshold']
    grid_density = 40  # param['grid_density']
    precision = torch.float32

    # define encoder
    if param['encoder'] == 'big':
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)

    ckpt = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    # import ipdb;ipdb.set_trace()
    encoder.load_state_dict(torch.load(ckpt)['encoder_state_dict'])
    decoder.load_state_dict(torch.load(ckpt)['decoder_state_dict'])

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    # transformations
    tfs = [Pad(size=param["image_size"])]
    tf = Compose(tfs)

    cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                        tf=tf, pretrain=pretrain,
                                        pad_size=param["image_size"],
                                        supervised_3d=True,
                                        sdf_loss=param["3D_loss"],
                                        grid_density=param["grid_density"],
                                        split='test',
                                        overfit=False,
                                        species=param["species"]
                                        )    
    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():

        for n_iter, item in enumerate(tqdm(iter(dataset))):
            volume,marching_cubes_volume,vgrid_volume = 0,0,0
            box = tensor_dict_2_float_dict(item['bbox'])
            voxel_size = (box['xmax'] - box['xmin'])/grid_density

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())

            # unpacking inputs
            rgbd = torch.cat((item['rgb'], item['depth']), 1).to(device)

            start = time.time()
            latent = encoder(rgbd)

            
            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            pred_sdf = decoder(deepsdf_input)
            mesh, vgrid = from_pred_sdf_to_mesh(pred_sdf, grid_3d.points, t=0, voxel_size=voxel_size)
            if mesh.is_watertight():
                volume = mesh.get_volume()
                vgrid_volume = len(vgrid.get_voxels()) * voxel_size**3
            else:
                print(item['frame_id'])
            # o3d.visualization.draw_geometries([hull, gt, cs], mesh_show_wireframe=True)
            inference_time = time.time() - start
            # for the deployment we can delete from here to the end of the file
            if n_iter > 0:
                exec_time.append(inference_time)
            # continue
            # print(n_iter, item['fruit_id'], inference_time)

            start = time.time()
            pred_mesh = sdf2mesh(pred_sdf, voxel_size, grid_density)
            if pred_mesh.is_watertight():
                marching_cubes_volume = pred_mesh.get_volume()
            volumes.append([volume,marching_cubes_volume,vgrid_volume])
            # pred_mesh.translate(np.full((3, 1), -(box['xmax'] - box['xmin'])/2))
            # pred_mesh = pred_mesh.filter_smooth_simple(number_of_iterations=2)
            # o3d.visualization.draw_geometries([pred_mesh.translate([.1,0,0]), gt, cs], mesh_show_wireframe=True)
            # o3d.visualization.draw_geometries([pred_mesh, gt, cs], mesh_show_wireframe=True)
            # cd.update(gt,pred_mesh)
            # pr.update(gt,pred_mesh)


        # print('inference time: {}'.format(mean(exec_time)))
        # cd.compute()
        # pr.compute_at_threshold(0.005)
        np.savetxt('./volumes.txt',volumes)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="shape completion main file, assume a pretrained deepsdf model")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--cfg",
        "-c",
        dest="cfg",
        required=True,
        help="Config file for the outer network.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    path = args.experiment_directory + '/ModelParameters/latest.pth'
    model_state = net_utils.load_without_parallel(torch.load(path))
    decoder.load_state_dict(model_state)
    decoder = net_utils.set_require_grad(decoder, False)

    main_function(decoder=decoder,
                  pretrain=args.experiment_directory,
                  cfg=args.cfg,
                  latent_size=latent_size)
