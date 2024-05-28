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

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, DoubleEncoder
import networks.utils as net_utils

import open3d as o3d
import numpy as np

import time
import json

from utils import sdf2mesh, tensor_dict_2_float_dict
from metrics_3d import chamfer_distance, precision_recall

import pandas as pd
from sklearn.metrics import mean_squared_error

cd = chamfer_distance.ChamferDistance()
pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

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
    torch.manual_seed(133)
    np.random.seed(133)
    
    df = pd.read_csv("/mnt/data/PieterBlok/Potato/Data/ground_truth_measurements/ground_truth.csv")
    columns = ['potato_id',
                'frame_id',
                'vertical_pos',
                'weight_g',
                'gt_volume_ml',
                'sfm_volume_ml',
                'mesh_volume_ml',
                'chamfer_distance',
                'precision',
                'recall',
                'f1']
    save_df = pd.DataFrame(columns=columns)

    exec_time = []

    with open(cfg) as json_file:
        param = json.load(json_file)

    device = 'cuda'

    # creating variables for 3d grid for diff SDF renderer
    threshold = param['threshold']
    grid_density = param['grid_density']
    precision = torch.float32

    # define encoder
    if param['encoder'] == 'big':
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'small_pool':
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["image_size"]).to(device)
    elif param['encoder'] == 'double':
        encoder = DoubleEncoder(out_channels=latent_size, size=param["image_size"]).to(device)
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
            volume, chamfer_distance, prec, rec, f1 = 0, 0, 0, 0, 0
            box = tensor_dict_2_float_dict(item['bbox'])
            voxel_size = (box['xmax'] - box['xmin'])/grid_density

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())

            start = time.time()

            # unpacking inputs
            rgbd = torch.cat((item['rgb'], item['depth']), 1).to(device)
            latent = encoder(rgbd)
            
            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            pred_sdf = decoder(deepsdf_input)
            mesh, vgrid = from_pred_sdf_to_mesh(pred_sdf, grid_3d.points, t=0, voxel_size=voxel_size)
            if mesh.is_watertight():
                volume = mesh.get_volume()
                # vgrid_volume = len(vgrid.get_voxels()) * voxel_size**3
            else:
                pass

            inference_time = time.time() - start

            if n_iter > 0:
                exec_time.append(inference_time)

            cd.reset()
            cd.update(gt, mesh)
            chamfer_distance = cd.compute()

            pr.reset()
            pr.update(gt, mesh)
            prec, rec, f1, _ = pr.compute_at_threshold(0.005)

            cur_data = {
                'potato_id': item['fruit_id'][0],
                'frame_id': item['frame_id'][0],
                'vertical_pos': int(item['frame_id'][0].split("_")[-1]),
                'weight_g': df.loc[df['label'] == item['fruit_id'][0], 'weight_g_inctack'].values[0],
                'gt_volume_ml': df.loc[df['label'] == item['fruit_id'][0], 'volume_ml'].values[0],
                'sfm_volume_ml': df.loc[df['label'] == item['fruit_id'][0], 'volume_metashape'].values[0],
                'mesh_volume_ml': round(volume * 1e6, 1),
                'chamfer_distance': round(chamfer_distance, 6),
                'precision': round(prec, 1),
                'recall': round(rec, 1),
                'f1': round(f1, 1)
                }
            save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
            save_df.to_csv("shape_completion_results.csv", mode='w+', index=False)
            
        print(f"Average time for 3D shape completion, including postprocessing: {mean(exec_time)*1e3:.1f} ms")

        try:
            analysis_values = [[0, 100], [100, 150], [150, 200], [200, 250], 
                                [250, 300], [300, 350], [350, 400], [400, 450], 
                                [450, 500], [500, 550], [550, 600], [600, 650], 
                                [650, 720]]
            
            for start, end in analysis_values:
                subset_df = save_df[(save_df['vertical_pos'] >= start) & (save_df['vertical_pos'] <= end)]
                filtered_subset_df = subset_df[subset_df['mesh_volume_ml'] != 0]
                subset_rmse_volume = mean_squared_error(filtered_subset_df['sfm_volume_ml'].values, filtered_subset_df['mesh_volume_ml'].values, squared=False)
                print(f"RMSE volume: {round(subset_rmse_volume, 1)} between {start}-{end} pixels")

            filtered_mesh = save_df[save_df['mesh_volume_ml'] != 0]
            rmse_volume = mean_squared_error(filtered_mesh['sfm_volume_ml'].values, filtered_mesh['mesh_volume_ml'].values, squared=False)
            avg_cd = sum(save_df['chamfer_distance'].values) / len(save_df['chamfer_distance'].values)
            avg_p = sum(save_df['precision'].values) / len(save_df['precision'].values)
            avg_r = sum(save_df['recall'].values) / len(save_df['recall'].values)
            avg_f1 = sum(save_df['f1'].values) / len(save_df['f1'].values)

            cur_data = {
                    'potato_id': "",
                    'frame_id': "",
                    'weight_g': "",
                    'gt_volume_ml': "",
                    'sfm_volume_ml': "",
                    'mesh_volume_ml': round(rmse_volume, 1),
                    'chamfer_distance': round(avg_cd, 6),
                    'precision': round(avg_p, 1),
                    'recall': round(avg_r, 1),
                    'f1': round(avg_f1, 1)
                    }
            
            save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
            save_df.to_csv("shape_completion_results.csv", mode='w+', index=False)
        except:
            pass


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
        default="500",
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