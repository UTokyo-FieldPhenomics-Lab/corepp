import open3d as o3d
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import copy
import getpass

import os
from diskcache import FanoutCache


cache = FanoutCache(
    directory=os.path.join("/tmp", "fanoutcache_" + getpass.getuser() + "/"),
    shards=64,
    timeout=1,
    size_limit=3e11,
)


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

    o3d.visualization.draw_geometries([pcd_max, pcd_min])

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

    # free = o3d.geometry.PointCloud()
    # free.points = o3d.utility.Vector3dVector(np.asarray(xyz_free))
    # free = free.paint_uniform_color([0,0.5,1])

    # frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([pcd, free, frame], point_show_normal=False, window_name='Free space')

    return np.asarray(xyz_free), np.asarray(mu_free)

def generate_pcd_from_virtual_depth(filename):
    # param for virtual depth
    height = 480
    width = 640

    # create open3d visualizer
    mesh = o3d.geometry.TriangleMesh()
    vis = o3d.visualization.Visualizer()
    vis.add_geometry(mesh)
    vis.create_window(height=height, width=width)

    pcd = o3d.io.read_point_cloud(filename.replace('samples.npz', 'fruit.ply'))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
    mesh.paint_uniform_color(np.array([0,0,1]))

    rot = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()

    mesh_t = copy.deepcopy(mesh)
    mesh_t.rotate(rot)

    vis.clear_geometries()
    vis.add_geometry(mesh_t)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.rotate(0.0, 0.0)

    depth = vis.capture_depth_float_buffer()

    param = ctr.convert_to_pinhole_camera_parameters()

    pcd = o3d.geometry.PointCloud()
    pcd = pcd.create_from_depth_image(depth, param.intrinsic, param.extrinsic)
    pcd.rotate(rot.T, center=mesh.get_center())
    return pcd

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

def read_depth_as_pcd(filename, pose=True):

    frame_id = int(filename.split('/')[-1][:-4])

    # getting depth
    depth = np.load(filename, allow_pickle=True)
    depth = o3d.geometry.Image(depth)

    # getting rgb
    image_path = filename.replace('depth', 'color') # os.path.join('/',*filename.split('/')[:-3])#, 'color/', filename.replace('npy', 'png'))
    color_file = image_path.replace('npy', 'png')
    mask_file = color_file.replace('color','masks')

    color = o3d.io.read_image(color_file)
    mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE) // 255
    rgb_np = np.asarray(color)
    rgb_np = np.copy(rgb_np[:,:,0:3])
    rgb_np[np.where(mask==0)] = 0
    color = o3d.geometry.Image(rgb_np)

    # getting pose
    invpose = np.eye(4)
    if pose:
        posesfilename = os.path.join('/',*filename.split('/')[:-3], 'tf/tf_allposes.npz')
        poses = np.load(posesfilename)
        invpose = np.linalg.inv(poses['arr_0'][frame_id-1])

    # getting bbox
    bbfilename = os.path.join('/',*filename.split('/')[:-3], 'tf/bounding_box.npz')
    bb_coordinates = np.load(bbfilename)['arr_0'] #* 1000

    bb = o3d.geometry.AxisAlignedBoundingBox()
    bb = bb.create_from_points(o3d.utility.Vector3dVector(bb_coordinates[:, :3]))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    # getting intrinsic
    intrinsicfilename = os.path.join('/',*filename.split('/')[:-2], 'intrinsic.json')
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsicfilename)
  
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale = 1000,
        depth_trunc=1.0,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, invpose,
                                                          project_valid_depth_only=True)
    pcd_colors = np.asarray(pcd.colors)
    valid_mask = pcd_colors.sum(axis=1)
    pcd = pcd.select_by_index(np.where(valid_mask!=0)[0])
    pcd.translate(-pcd.get_center())
    pcd = pcd.crop(bb)
    return pcd

if __name__ == "__main__":
    filename = './data/cameralaser/peppers/p17/realsense/depth/00050.npy'
    read_depth_as_pcd(filename)