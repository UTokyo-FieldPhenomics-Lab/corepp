#!/usr/bin/env python3

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToTensor, Compose, Resize

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from skimage import measure

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws
import deepsdf.deep_sdf.o3d_utils as o3d_utils

from sdfrenderer.grid import Grid3D

from dataloaders.cameralaser_w_masks import MaskedCameraLaserData
from dataloaders.transforms import Pad

from networks.models import Encoder, EncoderBig
import networks.utils as net_utils

import open3d as o3d
import numpy as np

import time
import json

DEBUG = True

torch.autograd.set_detect_anomaly(True)

def tensor_dict_2_float_dict(tensor_dict):
    for k, v in tensor_dict.items():
        tensor_dict[k] = float(v)
    return tensor_dict

def visualize_sdf(sdf_data):
    xyz = sdf_data[:, :-1]
    val = sdf_data[:, -1]

    xyz_min = xyz[val<0] + np.array([0,0,0])
    xyz_max = xyz[val>0]

    val_min = val[val<0]
    val_max = val[val>0]

    val_min += np.min(val_min)
    val_min /= np.max(val_min)

    val_max -= np.min(val_max)
    val_max /= np.max(val_max)

    colors_min = np.zeros(xyz_min.shape)
    colors_min[:, 0] =  val_min

    colors_max = np.zeros(xyz_max.shape)
    colors_max[:, 2] =  val_max

    pcd_min = o3d.geometry.PointCloud()
    pcd_min.points = o3d.utility.Vector3dVector(xyz_min)
    pcd_min.colors = o3d.utility.Vector3dVector(colors_min)

    pcd_max = o3d.geometry.PointCloud()
    pcd_max.points = o3d.utility.Vector3dVector(xyz_max)
    pcd_max.colors = o3d.utility.Vector3dVector(colors_max)

    o3d.visualization.draw_geometries([pcd_min, pcd_max])

def viz_pcd(sdf):
    pcd_min = o3d.geometry.PointCloud()
    pcd_min.points = o3d.utility.Vector3dVector(sdf.cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd_min])

def save_input(item, out, e, i):

    c = 250
    crop_dim = item['dimension']             

    rgb = item['rgb'].squeeze().permute(1,2,0) / 255
    d = item['depth'].squeeze()
    mask = item['mask'].squeeze()
    renderer = out.permute(1,2,0).squeeze()

    fig, axs = plt.subplots(2,2)
    axs[0][0].imshow(rgb)
    axs[0][1].imshow(d)
    axs[1][0].imshow(mask)
    axs[1][1].imshow(renderer.detach().cpu())

    [axi.set_axis_off() for axi in axs.ravel()]

    fig.tight_layout()
    fig.savefig('cache/input_{}_{}'.format(e,i))
    plt.close(fig)

def criterion_latent(latent, epoch, reg_lambda=0.1):
    l2_size_loss = torch.sum(torch.norm(latent, dim=1))
    reg_loss = reg_lambda * min(1, (epoch+1) / 100) * l2_size_loss
    return reg_loss

def generate_point_grid(grid_density=10):
    """
    Initial 3D point grid generation

    Args:
        grid_density (int): grid point density

    Returns: 3D point grid

    """

    # Set up the grid
    grid_density_complex = grid_density * 1j
    X, Y, Z = np.mgrid[-.2:.2:grid_density_complex, -.2:.2:grid_density_complex, -.2:.2:grid_density_complex]
    grid_np = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

    
    # Make an offset for every second z grid plane
    grid_np[1::2, :2] += ((X.max() - X.min()) / grid_density / 2)
    grid= torch.from_numpy(grid_np.astype(np.float32))

    return grid

# def sdf2mesh(decoder, latent, mesh_filename):
def sdf2mesh(pred_sdf, voxel_size):
    # d = 10
    # xyz = generate_point_grid(d).cuda()
    # latent = latent.expand(d*d*d, -1).cuda()

    # inputs = torch.cat([latent, xyz], 1).cuda()
    # pred_sdf = decoder(inputs).detach().cpu().reshape(d,d,d).numpy()
    # print(xyz.shape, pred_sdf.shape)
    verts, faces, normals, values = measure.marching_cubes(pred_sdf.reshape((40,40,40)).detach().cpu().numpy(), level=0.0, spacing=[voxel_size] * 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def main_function(decoder, latent_size):

    # some variables that will be later added to a config
    device = 'cuda'
    latent_trained = '/home/pieter/shape_completion/deepsdf/experiments/potato/LatentCodes/latest.pth'
    latents = torch.load(latent_trained)['latent_codes']['weight']


    torch.set_printoptions(linewidth=500, sci_mode=False)
    # for l in latents:
    #     print(torch.linalg.norm(l))

    # import ipdb;ipdb.set_trace()
    i = torch.randint(0,len(latents), (1,))
    j = torch.randint(0,len(latents), (1,))
    # import ipdb;ipdb.set_trace()

    l1 = latents[i]
    l2 = latents[j]

    interpolated  = np.linspace(l1, l2, num=7)


    box = {'xmin': -0.07712149275362304, 'xmax': 0.07712149275362304,
           'ymin': -0.07712149275362304, 'ymax': 0.07712149275362304,
           'zmin': -0.07712149275362304, 'zmax': 0.07712149275362304}

    # creating variables for 3d grid for diff SDF renderer
    grid_density = 40
    precision = torch.float32

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    geometries = []
    for idx, l in enumerate(interpolated):

        l = torch.from_numpy(l).to(device)
        start = time.time()

        grid_3d = Grid3D(grid_density, device, precision, bbox=box) 
        deepsdf_input = torch.cat([l.expand(grid_3d.points.size(0), -1),
                                    grid_3d.points],dim=1).to(l.device, l.dtype)

        pred_sdf = decoder(deepsdf_input)
        # print('min-max: ', torch.min(pred_sdf), torch.max(pred_sdf))

        inference_time = time.time() - start
        # print("inference_time", inference_time)
        
        start = time.time()
        voxel_size = (box['xmax'] - box['xmin'])/grid_density
        pred_mesh = sdf2mesh(pred_sdf, voxel_size)
        pred_mesh = pred_mesh.filter_smooth_simple(number_of_iterations=5)
        voxel_size = max(pred_mesh.get_max_bound() - pred_mesh.get_min_bound()) / 16
        pred_mesh = pred_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
        # import ipdb;ipdb.set_trace()
        lset =  o3d.geometry.LineSet().create_from_triangle_mesh(pred_mesh)
        marching_time = time.time() - start
        # print("marching_time", marching_time)

        t = np.array([0.2,0,0]) * idx
        geometries.append(lset.translate(t)) # translate 

    o3d.visualization.draw_geometries(geometries)

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
        "--checkpoint_decoder",
        "-c",
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

    main_function(decoder, latent_size)
