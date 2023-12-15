import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
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
