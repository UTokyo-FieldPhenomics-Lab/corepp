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

from networks.models import Encoder, EncoderBig, ERFNetEncoder, EncoderBigPooled, EncoderPooled, DoubleEncoder, PointCloudEncoder, PointCloudEncoderLarge, FoldNetEncoder
import networks.utils as net_utils

import open3d as o3d
import numpy as np

import time
import json

from utils import sdf2mesh_cuda, tensor_dict_2_float_dict
from metrics_3d import chamfer_distance, precision_recall

cd = chamfer_distance.ChamferDistance()
pr = precision_recall.PrecisionRecall(0.001, 0.01, 10)

torch.autograd.set_detect_anomaly(True)
import pandas as pd
from sklearn.metrics import mean_squared_error


def main_function(decoder, pretrain, cfg, latent_size):
    torch.manual_seed(133)
    np.random.seed(133)
    
    df = pd.read_csv("./data/3DPotatoTwinDemo/ground_truth.csv")
    columns = ['potato_id',
                'frame_id',
                'vertical_pos',
                'cultivar',
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
        encoder = EncoderBig(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'small_pool':
        encoder = EncoderPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'erfnet':
        encoder = ERFNetEncoder(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'pool':
        encoder = EncoderBigPooled(in_channels=4, out_channels=latent_size, size=param["input_size"]).to(device)
    elif param['encoder'] == 'double':
        encoder = DoubleEncoder(out_channels=latent_size, size=param["input_size"]).to(device)
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
                                        tf=tf, 
                                        color_tf = None,
                                        pretrain=pretrain,
                                        pad_size=param["input_size"],
                                        detection_input=param["detection_input"],
                                        normalize_depth=param["normalize_depth"],
                                        depth_min=param["depth_min"],
                                        depth_max=param["depth_max"],
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

            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
            gt = o3d.geometry.PointCloud()
            gt.points = o3d.utility.Vector3dVector(item['target_pcd'][0].numpy())

            start = time.time()

            # unpacking inputs
            if param['encoder'] != 'point_cloud' and param['encoder'] != 'point_cloud_large' and param['encoder'] != 'foldnet':
                encoder_input = torch.cat((item['rgb'], item['depth']), 1).to(device)
            else: 
                encoder_input = item['partial_pcd'].permute(0, 2, 1).to(device) ## be aware: the current partial pcd is not registered to the target pcd!

            latent = encoder(encoder_input)

            # save the latent vector for further inspection
            latent_save = latent.detach().to('cpu').squeeze()
            save_path = os.path.join(os.path.dirname(pretrain), "encoder")
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            torch.save(latent_save, os.path.join(save_path, item['frame_id'][0] + ".pth"))

            grid_3d = Grid3D(grid_density, device, precision, bbox=box)
            deepsdf_input = torch.cat([latent.expand(grid_3d.points.size(0), -1),
                                        grid_3d.points], dim=1).to(latent.device, latent.dtype)
            pred_sdf = decoder(deepsdf_input)

            try:
                mesh = sdf2mesh_cuda(pred_sdf, grid_3d.points, t=0.0)
                if mesh.is_watertight():
                    volume = mesh.get_volume()
                else:
                    print(item['frame_id'])
                    pass
            except:
                pass

            inference_time = time.time() - start

            if n_iter > 0:
                exec_time.append(inference_time)

            cd.reset()
            cd.update(gt, mesh)
            chamfer_distance = cd.compute(print_output=False)

            pr.reset()
            pr.update(gt, mesh)
            prec, rec, f1, _ = pr.compute_at_threshold(0.005, print_output=False)

            cur_data = {
                'potato_id': item['fruit_id'][0],
                'frame_id': item['frame_id'][0],
                'vertical_pos': int(item['frame_id'][0].split("_")[-1]),
                'cultivar': df.loc[df['label'] == item['fruit_id'][0], 'cultivar'].values[0],
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
        print("Results saved in: " + os.getcwd() + "/shape_completion_results.csv")

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