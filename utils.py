import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import json
import torch.utils.dlpack
from skimage import measure
import torch

def save_model(encoder, decoder, epoch, optim, loss, name):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
        }, name)

def tensor_dict_2_float_dict(tensor_dict):
    for k, v in tensor_dict.items():
        tensor_dict[k] = float(v[0])
    return tensor_dict

def sdf2mesh(pred_sdf, voxel_size, grid_size):
    verts, faces, _, _ = measure.marching_cubes(pred_sdf.reshape((grid_size,grid_size,grid_size)).detach().cpu().numpy(),
                                                 level=0.0, spacing=[voxel_size] * 3)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

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
    return pcd_min+pcd_max

def viz_pcd(sdf):
    pcd_min = o3d.geometry.PointCloud()
    pcd_min.points = o3d.utility.Vector3dVector(sdf.cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd_min])

def save_input(item, out, e, i):

    rgb = item['rgb'][0].squeeze().permute(1,2,0)
    d = item['depth'][0].squeeze()
    mask = item['mask'][0].squeeze()
    renderer = out[0]#.permute(1,2,0).squeeze()


    fig, axs = plt.subplots(2,2)
    axs[0][0].imshow(rgb)
    axs[0][1].imshow(d)
    axs[1][0].imshow(mask)
    axs[1][1].imshow(renderer.detach().cpu())

    [axi.set_axis_off() for axi in axs.ravel()]

    fig.tight_layout()
    fig.savefig('cache/input_{}_{}'.format(e,i))
    plt.close(fig)

def sample_freespace(pcd, n_samples=20000):
    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)

    xyz_free = []
    mu_free = []
    for _ in range(n_samples):

        mu_freespace = np.random.uniform(0.001, 0.01)
        idx = np.random.randint(len(xyz))

        p = xyz[idx]
        pn = n_xyz[idx]

        sample = p + mu_freespace*pn

        xyz_free.append(sample)
        mu_free.append(mu_freespace)

    return np.asarray(xyz_free), np.asarray(mu_free)

def generate_deepsdf_target(pcd, mu=0.001, align_with=np.array([0.0, 1.0, 0.0])):
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction(align_with)

    # o3d.visualization.draw_geometries([pcd])

    xyz = np.asarray(pcd.points)
    n_xyz = np.asarray(pcd.normals)
    
    xyz_pos = xyz + mu*n_xyz
    xyz_neg = xyz - mu*n_xyz
    sdf_val_pos = np.repeat(mu, len(xyz))
    sdf_val_neg = np.repeat(-mu, len(xyz))

    xyz_free, sdf_val_free = sample_freespace(pcd)

    # merging positive samples from depth and from freespace
    xyz_pos = np.vstack((xyz_pos, xyz_free))
    sdf_val_pos = np.concatenate((sdf_val_pos, sdf_val_free))

    # packing deepsdf input
    sdf_pos = np.empty((len(xyz_pos), 4))
    sdf_pos[:, :3] = xyz_pos
    sdf_pos[:, 3] = sdf_val_pos 

    sdf_neg = np.empty((len(xyz_neg), 4))
    sdf_neg[:, :3] = xyz_neg
    sdf_neg[:, 3] = sdf_val_neg

    # visualize_sdf(np.vstack((sdf_pos, sdf_neg))) 

    return sdf_pos, sdf_neg


def sdf2mesh_cuda(pred_sdf, grid_points, t=0):
    voxel_size = 0.0
    keep_idx = torch.lt(pred_sdf, t)
    keep_points = grid_points[torch.squeeze(keep_idx)]

    o3d_t = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keep_points))
    pcd_gpu = o3d.t.geometry.PointCloud(o3d_t)

    hull_gpu = pcd_gpu.compute_convex_hull()
    hull = hull_gpu.to_legacy()

    mesh = hull.subdivide_loop(number_of_iterations=1)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    while not mesh.is_watertight():
        voxel_size += 0.01
        down_pcd_gpu = pcd_gpu.voxel_down_sample(voxel_size=voxel_size)
        hull_gpu = down_pcd_gpu.compute_convex_hull()
        hull = hull_gpu.to_legacy()
        
        mesh = hull.subdivide_loop(number_of_iterations=1)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    return mesh


def visualize_errors(pred_mesh, gt_pcd, max_error):
    cmap = plt.get_cmap('RdYlGn_r')

    mesh_error = o3d.geometry.TriangleMesh()
    mesh_error.vertices = pred_mesh.vertices
    mesh_error.triangles = pred_mesh.triangles

    pcd = o3d.geometry.PointCloud()
    pcd.points = pred_mesh.vertices
    dist_pt_2_gt = np.asarray(pcd.compute_point_cloud_distance(gt_pcd))
    dist_pt_2_gt -= dist_pt_2_gt.min()
    dist_pt_2_gt /= max_error ## dist_pt_2_gt /= dist_pt_2_gt.max()
    color = cmap(dist_pt_2_gt)[:,:-1]
    mesh_error.vertex_colors = o3d.utility.Vector3dVector(color)

    return mesh_error


def mesh_and_volume(pcd):
    mesh, _ = pcd.compute_convex_hull()

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if mesh.is_watertight():
        volume = mesh.get_volume() * 1e6
    else:
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.005)
        while not alpha_mesh.is_watertight():
            alpha_value += 0.005
            alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_value)
        volume = alpha_mesh.get_volume() * 1e6  
        mesh = alpha_mesh

    return mesh, volume


def read_matrix_json(json_path):
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
            return data
    else:
        raise FileNotFoundError(f"Could not locate the given json file [{json_path}]")
    

## thanks to: https://github.com/isl-org/Open3D/issues/2
def text_3d(text, pos, direction=None, degree=0.0, font='Ubuntu-R.ttf', font_size=16, density=10):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    left, top, right, bottom = font_obj.getbbox(text)
    text_width = (right - left)
    text_height = (bottom - top) + 20
    font_dim = (text_width, text_height)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)

    return pcd