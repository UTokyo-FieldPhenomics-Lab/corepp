#!/usr/bin/env python3

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import glob
import ast

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
from utils import sdf2mesh_cuda

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

def main_function(experiment_directory, split_filename, decoder, latent_names, transformation, color, geometries):
    device = 'cuda'
    search_pattern = os.path.join(experiment_directory, '**', 'Reconstructions', '**', 'Codes', '**', 'encoder', '**', '*.pth')
    pth_files = glob.glob(search_pattern, recursive=True)

    with open(split_filename, "r") as f:
        split = json.load(f)

    split_list = []
    for _, value in split.items():
        if isinstance(value, dict):
            nested_value = next(iter(value.values()))  # Get the first (and only) value in the inner dictionary
            if isinstance(nested_value, list):
                split_list = nested_value
                break

    latents = []
    tuber_names = []
    frame_names = []
    for pth_file in pth_files:
        tuber_name = os.path.splitext(os.path.basename(pth_file))[0].split("_")[0]
        if tuber_name in split_list:
            latent = torch.load(pth_file)
            latent_detach = latent.detach().to('cpu').squeeze()
            latents.append(latent_detach)
            tuber_names.append(tuber_name)
            frame_names.append(os.path.splitext(os.path.basename(pth_file))[0])
    latents = torch.vstack(latents)

    idx1 = frame_names.index(latent_names[0])
    idx2 = frame_names.index(latent_names[1])

    l1 = latents[idx1]
    l2 = latents[idx2]

    torch.set_printoptions(linewidth=500, sci_mode=False)

    interpolated  = np.linspace(l1, l2, num=7)


    box = {'xmin': -0.07712149275362304, 'xmax': 0.07712149275362304,
           'ymin': -0.07712149275362304, 'ymax': 0.07712149275362304,
           'zmin': -0.07712149275362304, 'zmax': 0.07712149275362304}

    # creating variables for 3d grid for diff SDF renderer
    grid_density = 20
    precision = torch.float32

    ##############################
    #  TESTING LOOP STARTS HERE  #
    ##############################

    decoder.to(device)

    print("")
    print(experiment_directory)
    
    for idx, l in enumerate(interpolated):

        l = torch.from_numpy(l).to(device)
        start = time.time()

        grid_3d = Grid3D(grid_density, device, precision, bbox=box) 
        deepsdf_input = torch.cat([l.expand(grid_3d.points.size(0), -1),
                                    grid_3d.points],dim=1).to(l.device, l.dtype)

        pred_sdf = decoder(deepsdf_input)
        
        start = time.time()
        voxel_size = (box['xmax'] - box['xmin'])/grid_density
        
        try:
            pred_mesh_or = sdf2mesh_cuda(pred_sdf, grid_3d.points, t=0.0)
            pred_mesh = pred_mesh_or.filter_smooth_simple(number_of_iterations=5)
            voxel_size = max(pred_mesh.get_max_bound() - pred_mesh.get_min_bound()) / 16
            pred_mesh = pred_mesh.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average)

            t1 = np.array([0.1, 0, 0]) * idx
            t2 = np.array(transformation)

            pred_mesh_t1 = pred_mesh_or.translate(t1)
            pred_mesh_t2 = pred_mesh_t1.translate(t2)
            pred_mesh_t2.paint_uniform_color(color)
            
            print(f"Volume (ml): {round(pred_mesh_or.get_volume()*1e6, 1)}")
            geometries.append(pred_mesh_t2)
        except:
            pass

    return geometries 

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="shape completion main file, assume a pretrained deepsdf model")
    arg_parser.add_argument(
        "--experiments",
        "-e",
        dest="experiment_directory",
        nargs='+',
        type=str,
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--checkpoints_decoder",
        "-c",
        dest="checkpoint",
        default="3500",
        nargs='+',
        type=str,
        help="The checkpoint weights to use. This should be a number indicated an epoch",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--latent_names",
        "-ln",
        dest="latent_names",
        nargs='+',
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--transformations",
        "-t",
        dest="transformations",
        nargs='+',
        type=str,
        required=True
    )
    arg_parser.add_argument(
        "--colors",
        "-cl",
        dest="colors",
        nargs='+',
        type=str,
        required=True
    )


    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    # loading deepsdf model
    geometries = []
    for experiment, checkpoint, transformation, color in zip(args.experiment_directory, args.checkpoint, args.transformations, args.colors):
        specs = ws.load_experiment_specifications(experiment)
        latent_size = specs["CodeLength"]
        arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
        decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

        path = os.path.join(experiment, 'ModelParameters', checkpoint) + '.pth' 
        model_state = net_utils.load_without_parallel(torch.load(path))
        decoder.load_state_dict(model_state)
        decoder = net_utils.set_require_grad(decoder, False)

        geometries = main_function(experiment, args.split_filename, decoder, args.latent_names, ast.literal_eval(transformation), ast.literal_eval(color), geometries)

    o3d.visualization.draw_geometries(geometries, mesh_show_wireframe=True)