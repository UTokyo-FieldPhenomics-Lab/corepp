#!/usr/bin/env python3

from base64 import decode
import copy
import json
import re
import numpy as np
import os
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
import cv2 as cv
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision.transforms.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

from transforms import Pad
# from dataloaders.transforms import Pad

mpl.rcParams['image.cmap'] = 'gray'

NAME = '0'


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

def show_cv(img, fruit_id):
    cv.imshow(fruit_id, img) 
    cv.waitKey(0)
    cv.destroyAllWindows()

class MaskedCameraLaserData(torch.utils.data.Dataset):
    def __init__(self, data_source, pad_size, pretrain, grid_density, 
                tf=None, supervised_3d=False, split=None, sdf_loss=False, sdf_trunc=0.015, overfit=False, decoder=False):
	    
        self.overfit = overfit
        self.data_source = data_source
        self.update_decoder = decoder
        
        if 'Strawberry' in data_source:
            self.species = 'Strawberry'
        else:
            self.species = 'SweetPepper'

        with open(os.path.join(self.data_source, 'split.json')) as json_file:
            self.split_ids = json.load(json_file)[split]

        # latents dics has to be defined before filtering not useable fruits
        # self.latents_dict, trained_latents = self.get_latents_dict(pretrain) 
        self.split_ids = self.check_is_useable()

        # loading output of realsense registration
        registration_outputs = self.get_registration_outputs()
        
        self.Ks = registration_outputs['K']
        self.poses = registration_outputs['pose']
        self.bboxs = registration_outputs['bbox']
        self.global_bbox = self.find_global_bbox()

        self.files = self.get_instance_filenames()

        self.sdf_loss = sdf_loss
        self.sdf_trunc = sdf_trunc

        self.grid_density = grid_density
        self.supervised_3d = supervised_3d
        self.tf = tf

        self.pad_size = pad_size
        # self.target_mean = torch.mean(trained_latents, dim=0)
        # self.target_cov = torch.cov(trained_latents.T)

    def get_latents_dict(self, path):
        "create dictionary of pairs fruit_id:latent given pretrained model"

        species = 'SweetPepper' if self.update_decoder else self.species
 
        spect_path = os.path.join(path, 'specs.json')
        spec_file = open(spect_path,'r')
        split_file = open(json.load(spec_file)['TrainSplit'],'r')
        split = json.load(split_file)['.'][species]

        latent = torch.load(os.path.join(path, 'LatentCodes/latest.pth'))['latent_codes']['weight']
        latent_dictionary = dict(zip(split, latent))

        return latent_dictionary, latent

    @staticmethod
    def preprocess_images(rgb, depth, mask):
        """ 
        Crop images and depths around the fruit
        
        Args:
            rgb: full image as loaded from opencv
            depth: full depth as generated from realsense camera
            mask: semantic mask
                        
        Returns: 
            rgb: rgb cropped around the fruit
            depth: depth cropped around the fruit
            mask: mask cropped around the fruit
        """

        # cropping
        offset = 20
        indices = np.where(mask)
        min_i = indices[0].min() - offset
        min_j = indices[1].min() - offset
        max_i = indices[0].max() + offset
        max_j = indices[1].max() + offset

        # fucking python with negative indexes
        if min_i < 0: min_i = 0
        if min_j < 0: min_j = 0

        rgb = rgb[min_i:max_i, min_j:max_j, :]
        depth = depth[min_i:max_i, min_j:max_j]
        mask = mask[min_i:max_i, min_j:max_j]

        w = max_i - min_i
        h = max_j - min_j

        return rgb, depth, mask, (min_i, min_j), (w,h), np.ones(mask.shape)

    @staticmethod
    def load_K(path):
        """ 
        Load intrinsic params
        
        Args:
            path: path to json
                        
        Returns: 
            k: intrinsic matrix
        """
        f = open(path,'r')
        data = json.load(f)['intrinsic_matrix']
        k = np.reshape(data, (3, 3), order='F') 
        return k

    @staticmethod
    def bbox2dict(bb):
        """
        Convert bounding box from array with shapes 2x3 to dictionary

        Args:
            bb: bounding box extreme points

        Returns:
            box: bounding box as dictionary
        """

        x_min, y_min, z_min = bb[0] 
        x_max, y_max, z_max = bb[1]

        box = {}
        box['xmin'] = x_min
        box['xmax'] = x_max
        box['ymin'] = y_min
        box['ymax'] = y_max
        box['zmin'] = z_min
        box['zmax'] = z_max
        return box

    def update_intrinsic(self, crop_origin, shape_origin, k):
        pad_x = self.pad_size - shape_origin[0]
        pad_y = self.pad_size - shape_origin[1]
        
        intrinsic = copy.deepcopy(k)

        intrinsic[0,2] -= crop_origin[1] - pad_y/2 + 0.5
        intrinsic[1,2] -= crop_origin[0] - pad_x/2 + 0.5
        return intrinsic
    
    def find_global_bbox(self):
        dmax = 0
        for fruit_id in self.bboxs:
            dx = self.bboxs[fruit_id]['xmax'] - self.bboxs[fruit_id]['xmin']
            dy = self.bboxs[fruit_id]['ymax'] - self.bboxs[fruit_id]['ymin']
            dz = self.bboxs[fruit_id]['zmax'] - self.bboxs[fruit_id]['zmin']
            local_dmax = max(dx,dy,dz)
            if local_dmax > dmax:
                dmax = local_dmax
        global_bbox = {'xmin': -dmax/2, 'xmax': dmax/2, 
                       'ymin': -dmax/2, 'ymax': dmax/2, 
                       'zmin': -dmax/2, 'zmax': dmax/2}    
        return global_bbox
        
    def get_instance_filenames(self):
        """ 
        Load image names
        
        Returns: 
            files: list of file names ( of rgb frames) to be used for training/testing
        """

        files = []
        for count_id, id in enumerate(self.split_ids):

            ####################### 
            if self.overfit:
                if count_id == 1: break
            #######################  

            max_frame_id = len(self.poses[id])
            if self.species == 'Strawberry': 
                lst_images = os.listdir(self.data_source+id+'/realsense/masks_cntr/')
            else:
                lst_images = os.listdir(self.data_source+id+'/realsense/masks/')
            lst_images.sort()
            for count_img, fname in enumerate(lst_images):


                frame_id = int(fname[:-4]) - 1

                if frame_id == max_frame_id: break
                
                ####################### 
                # if frame_id > 600: break
                if count_img % 10 != 0: continue  # keeping one frame each sec, max 20 images per fruit
                ####################### 


                files.append(self.data_source+id+'/realsense/color/'+fname)

            # print(frame_id)

        files.sort() 
        return files

    def get_registration_outputs(self):
        """ 
        Load registration parameters
        
        Args:
      
        Returns: 
            data: dictionary containing intrinsic, boudning box 
                  and poses of each frame of each fruit
        """       
        Ks = {}
        poses = {}
        bboxs = {}

        for fruit in self.split_ids:
            path = os.path.join(self.data_source, fruit)
            k_path = os.path.join(path, 'realsense/intrinsic.json')
            pose_path = os.path.join(path, 'tf/tf_allposes.npz')
            bbox_path = os.path.join(path, 'tf/bounding_box.npz')

            k = self.load_K(k_path)
            bbox = np.load(bbox_path)['arr_0']
            pose = np.load(pose_path)['arr_0']

            Ks[fruit] = k
            bboxs[fruit] = self.bbox2dict(bbox)
            poses[fruit] = pose
        
        return {'K': Ks, 'bbox': bboxs, 'pose':poses}

    def compute_target_sdf(self, rgb, depth, pose, k):

        bbox = self.global_bbox

        grid_density_complex = self.grid_density * 1j
        X, Y, Z = np.mgrid[bbox['xmin']:bbox['xmax']:grid_density_complex, bbox['ymin']:bbox['ymax']:grid_density_complex, bbox['zmin']:bbox['zmax']:grid_density_complex]
        grid = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

        bb = o3d.geometry.AxisAlignedBoundingBox()
        bb = bb.create_from_points(o3d.utility.Vector3dVector(grid))

        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid)
        grid_pcd.paint_uniform_color(np.array((.5,.5,.5)))

        invpose = np.linalg.inv(pose)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                  o3d.geometry.Image(depth),
                                                                  depth_scale = 1000,
                                                                  depth_trunc=1.0,
                                                                  convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(height=rgb.shape[0],
                                 width=rgb.shape[1],
                                 fx=k[0,0],
                                 fy=k[1,1],
                                 cx=k[0,2],
                                 cy=k[1,2],
                                 )

        volume_len = self.global_bbox['xmax'] - self.global_bbox['xmin']
        volume = o3d.pipelines.integration.UniformTSDFVolume(length=volume_len,
                                                             resolution=self.grid_density,
                                                             sdf_trunc= self.sdf_trunc,
                                                             origin = np.full((3,1),-volume_len/2),
                                                             color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

        volume.integrate(rgbd, intrinsic, invpose)
        tsdf = volume.extract_volume_tsdf()

        val = np.asarray(tsdf)[:,0]
        weight = np.asarray(tsdf)[:,1]

        val = val.reshape((self.grid_density**3,1))
        weight = weight.reshape((self.grid_density**3,1))
        return val, weight, grid

    def check_is_useable(self):
        useable_fruits = []
        for fruit in self.split_ids:
            # print('checking {} usability...'.format(fruit))
            with open(os.path.join(self.data_source, fruit, 'dataset.json')) as json_file:
                specs = json.load(json_file)
                if specs['is_useable']:
                    useable_fruits.append(fruit)
        return useable_fruits

    def __len__(self):
        # print(len(self.files))
        return len(self.files)

    def __getitem__(self, idx):

        # ugly stuff to get consistent ids
        fruit_id = self.files[idx].split('/')[-4]
        frame_id = int(self.files[idx].split('/')[-1][:-4]) - 1 # realsense idxs start with 1 
        
        target_pcd_fname = os.path.join(self.data_source, fruit_id+'/laser/fruit.ply')
        target_pcd = o3d.io.read_point_cloud(target_pcd_fname)

        # print(idx, fruit_id, frame_id)

        image_path = self.files[idx]
        depth_path = image_path.replace('color', 'depth')
        depth_path = depth_path.replace('png', 'npy')
        if self.species == 'Strawberry':
            mask_path = image_path.replace('color', 'masks_cntr')
        else:
            mask_path = image_path.replace('color', 'masks')

        rgb = cv.imread(image_path)
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) // 255
        
        # getting intrinsic and camera pose
        k = self.Ks[fruit_id]
        pose = self.poses[fruit_id][frame_id]
        bbox = self.global_bbox

        if self.sdf_loss:
            target_sdf, target_sdf_weights, _ = self.compute_target_sdf(rgb, depth, pose, k)

        # cropping image to region of interest
        rgb, depth, mask, crop_origin, crop_dim, padding_mask = self.preprocess_images(rgb, depth, mask)
        
        # fig, axs = plt.subplots(1,3)
        # axs[0].imshow(rgb)
        # axs[1].imshow(depth)
        # axs[2].imshow(mask*255)

        # [axi.set_axis_off() for axi in axs.ravel()]
        # plt.show()

        intrinsic = self.update_intrinsic(crop_origin, depth.shape, k)

        item = {
            'dimension': crop_dim,
            'fruit_id': fruit_id,
            'frame_id': frame_id,
            'pose': pose,
            'K': intrinsic,
            'bbox': bbox,
        }

        if self.sdf_loss:
            item['target_sdf'] = torch.from_numpy(target_sdf)
            item['target_sdf_weights'] = torch.from_numpy(target_sdf_weights)

        if self.tf:
            rgb = torch.from_numpy(np.array(self.tf(rgb))).permute(2,0,1)
            depth = torch.from_numpy(np.array(self.tf(depth))).unsqueeze(dim=0) 
            mask = torch.from_numpy(np.array(self.tf(mask))).unsqueeze(dim=0)
            padding_mask = torch.from_numpy(np.array(self.tf(padding_mask))).unsqueeze(dim=0)

        item['rgb'] = rgb.float()/255 
        item['depth'] = depth.float()/1000 # suppose depth is in mm
        item['mask'] = mask
        item['padding_mask'] = padding_mask
        item['target_pcd'] = torch.Tensor(np.asarray(target_pcd.points))

        if self.supervised_3d:
            
            trained_latent = self.latents_dict[fruit_id]
            item['latent'] = trained_latent

        return item


if __name__ == '__main__':

    # tfs = [Pad(size=128), ToTensor()]
    tfs = [Pad(size=500)]
    tf = transforms.Compose(tfs)
    cl_dataset = MaskedCameraLaserData(data_source="/ipb14/export/igg_fruit/processed/SweetPepper/", 
                                tf=tf, supervised_3d=False,
                                pretrain='../deepsdf/experiments/sweetpeppers_latent12/', 
                                pad_size=250,
                                sdf_loss=True,
                                grid_density=50,
                                split='train')

    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False)

    c = 0
    for item in iter(dataset):
        import ipdb;ipdb.set_trace()
        ids = item['fruit_id']
        c+=1
        print(c*32)
