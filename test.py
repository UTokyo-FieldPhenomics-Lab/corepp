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

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, PointCloudEncoder, PointCloudEncoderLarge, FoldNetEncoder
import networks.utils as net_utils

import open3d as o3d
import open3d.core as o3c
import torch.utils.dlpack
import numpy as np

import time
import json

from utils import sdf2mesh, tensor_dict_2_float_dict
# from metrics_3d import chamfer_distance, precision_recall

# cd = chamfer_distance.ChamferDistance()
# pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)


def from_pred_sdf_to_mesh(pred_sdf, grid_points, t=0):
    keep_idx = torch.lt(pred_sdf, t)
    keep_points = grid_points[torch.squeeze(keep_idx)]

    o3d_t = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keep_points))
    pcd_gpu = o3d.t.geometry.PointCloud(o3d_t)
    _, ind = pcd_gpu.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
    pcd_gpu_filt = pcd_gpu.select_by_mask(ind)
    down_pcd_gpu = pcd_gpu_filt.voxel_down_sample(voxel_size=0.005)

    hull_gpu = down_pcd_gpu.compute_convex_hull()
    hull = hull_gpu.to_legacy()
    hull.remove_degenerate_triangles()
    hull.remove_duplicated_triangles()
    hull.remove_duplicated_vertices()
    hull.remove_non_manifold_edges()
    hull.remove_unreferenced_vertices()

    return hull

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
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'small_pool':
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'point_cloud':
        encoder = PointCloudEncoder(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'point_cloud_large':
        encoder = PointCloudEncoderLarge(in_channels=3, out_channels=latent_size).to(device)
    elif param['encoder'] == 'foldnet':
        encoder = FoldNetEncoder(in_channels=3, out_channels=latent_size).to(device)
    else:
        encoder = Encoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)

    ckpt = os.path.join(param['checkpoint_dir'], param['checkpoint_file'])
    # import ipdb;ipdb.set_trace()
    encoder.load_state_dict(torch.load(ckpt)['encoder_state_dict'])
    decoder.load_state_dict(torch.load(ckpt)['decoder_state_dict'])

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    # transformations
    tfs = [Pad(size=param["input_size"])]
    tf = Compose(tfs)

    cl_dataset = MaskedCameraLaserData(data_source=param["data_dir"],
                                        tf=tf, pretrain=pretrain,
                                        pad_size=param["input_size"],
                                        detection_input=param["detection_input"],
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
            volume,marching_cubes_volume = 0,0
            box = tensor_dict_2_float_dict(item['bbox'])
            voxel_size = (box['xmax'] - box['xmin'])/grid_density

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())

            # unpacking inputs
            if param['encoder'] != 'point_cloud' and param['encoder'] != 'point_cloud_large' and param['encoder'] != 'foldnet':
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
            else: 
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) ## be aware: the current partial pcd is not registered to the target pcd!

            start = time.time()
            latent = encoder(encoder_input)

            
            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            pred_sdf = decoder(deepsdf_input)
            mesh = from_pred_sdf_to_mesh(pred_sdf, grid_3d.points, t=0.005)
            if mesh.is_watertight():
                volume = mesh.get_volume()
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
            volumes.append([volume,marching_cubes_volume])
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
    arg_parser.add_argument(
        "--checkpoint_decoder",
        dest="checkpoint",
        default="3500",
        help="The checkpoint weights to use. This should be a number indicated an epoch",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    specs = ws.load_experiment_specifications(args.experiment_directory)
    latent_size = specs["CodeLength"]
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    path = os.path.join(args.experiment_directory, 'ModelParameters', args.checkpoint) + '.pth'
    model_state = net_utils.load_without_parallel(torch.load(path))
    decoder.load_state_dict(model_state)
    decoder = net_utils.set_require_grad(decoder, False)

    pretrain_path = os.path.join(args.experiment_directory, 'Reconstructions', args.checkpoint, 'Codes', 'complete')

    main_function(decoder=decoder,
                  pretrain=pretrain_path,
                  cfg=args.cfg,
                  latent_size=latent_size)
