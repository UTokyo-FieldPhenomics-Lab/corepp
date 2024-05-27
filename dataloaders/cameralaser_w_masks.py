#!/usr/bin/env python3

import copy
import json
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

# from transforms import Pad
from dataloaders.transforms import Pad

mpl.rcParams['image.cmap'] = 'gray'

NAME = '0'

def imshow(img):
    # plt.imshow(img[:, :, [2, 1, 0]])
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    

## load intrinsics file
def load_intrinsics(intrinsics_file):
    with open(intrinsics_file) as json_file:
        data = json.load(json_file)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(data['width'], data['height'], data['intrinsic_matrix'][0], data['intrinsic_matrix'][4], data['intrinsic_matrix'][6], data['intrinsic_matrix'][7])
    return intrinsics, data['depth_scale']


## Histogram filter for the depth image
def histogram_filtering(dimg, mask, max_depth_range=50, max_depth_contribution=0.05):
    mask = mask.astype(np.uint8)
    mask_bool = mask.astype(bool)
    
    z = np.expand_dims(dimg, axis=2)
    z_mask = z[mask_bool]
    z_mask_filtered = z_mask[z_mask != 0]

    if z_mask_filtered.size > 1: 
        z_mask_filtered_range = np.max(z_mask_filtered)-np.min(z_mask_filtered)

        if (z_mask_filtered_range > max_depth_range):
            hist, bin_edges = np.histogram(z_mask_filtered, density=False) 
            hist_peak = np.argmax(hist)
            lb = bin_edges[hist_peak]
            ub = bin_edges[hist_peak+1]

            bc = np.bincount(np.absolute(z_mask_filtered.astype(np.int64)))
            peak_id = np.argmax(bc)

            if peak_id > int(lb) and peak_id < int(ub):
                peak_id = peak_id
            else:
                bc_clip = bc[int(lb):int(ub)]
                peak_id = int(lb) + np.argmax(bc_clip)

            pixel_counts = np.zeros((10), dtype=np.int64)

            for j in range(10):
                lower_bound = peak_id-(max_depth_range - (j * 10))
                upper_bound = lower_bound + max_depth_range
                z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
                pixel_counts[j] = z_final.size

            pix_id = np.argmax(pixel_counts)
            lower_bound = peak_id-(max_depth_range - (pix_id * 10))
            upper_bound = lower_bound + max_depth_range
            z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
            
        else:
            z_final = z_mask_filtered

        hist_f, bin_edges_f = np.histogram(z_final, density=False)
        norm1 = hist_f / np.sum(hist_f)

        sel1 = bin_edges_f[np.where(norm1 >= max_depth_contribution)]
        sel2 = bin_edges_f[np.where(norm1 >= max_depth_contribution)[0]+1]
        edges = np.concatenate((sel1,sel2), axis=0)
        final_bins = np.unique(edges)
 
        z_min = np.min(final_bins)
        z_max = np.max(final_bins)
    else:
        z_min = np.min(z_mask_filtered)
        z_max = np.max(z_mask_filtered)
    
    return z_min, z_max


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
    def __init__(self, data_source, pad_size, detection_input, normalize_depth, depth_min, depth_max, pretrain, grid_density, 
                tf=None, color_tf=None, supervised_3d=False, split=None, sdf_loss=False, sdf_trunc=0.015, overfit=False, species=None):
	    
        self.overfit = overfit
        self.data_source = data_source
        self.species = species

        with open(os.path.join(self.data_source, 'split.json')) as json_file:
            self.split_ids = json.load(json_file)[split]

        # latents dics has to be defined before filtering not useable fruits
        self.latents_dict = self.get_latents_dict(pretrain) 
        self.split_ids = self.check_is_useable()

        # loading output of realsense registration
        registration_outputs = self.get_intrinsics()
        self.Ks = registration_outputs['K']
        self.box = self.find_global_bbox()
        self.files = self.get_instance_filenames()

        self.grid_density = grid_density
        self.supervised_3d = supervised_3d
        self.tf = tf
        self.color_tf = color_tf

        self.pad_size = pad_size
        self.detection_input = detection_input
        self.normalize_depth = normalize_depth
        self.depth_min = depth_min  # in mm
        self.depth_max = depth_max  # in mm

    def get_latents_dict(self, path):
        "create dictionary of pairs fruit_id:latent given pretrained model"
        latent_dictionary = {}
        for fname in os.listdir(path):
            latent = torch.load(os.path.join(path,fname))
            key = fname[:-4]
            latent_dictionary[key] = latent
        return latent_dictionary

    @staticmethod
    def preprocess_images(rgb, depth, mask, intrinsic_file, detection_input="box"):
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

        # create partial point cloud
        intrinsics, depth_scale = load_intrinsics(intrinsic_file)
        img_mask = np.multiply(rgb, np.expand_dims(mask, axis=2))
        dimg_mask = np.multiply(depth, mask)
        z_min, z_max = histogram_filtering(depth, mask, 50, 0.05)
        dimg_mask[dimg_mask < z_min] = 0
        dimg_mask[dimg_mask > z_max] = 0

        rgb_mask = o3d.geometry.Image((img_mask).astype(np.uint8))
        depth_mask = o3d.geometry.Image(dimg_mask)
        rgbd_mask = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_mask, depth_mask, depth_scale=depth_scale, depth_trunc=0.4, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_mask, intrinsics)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_filtered = pcd.select_by_index(ind)
        pcd_filtered.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # cropping
        offset = 20
        indices = np.where(mask)
        min_i = indices[0].min() - offset
        min_j = indices[1].min() - offset
        max_i = indices[0].max() + offset
        max_j = indices[1].max() + offset

        if min_i < 0: min_i = 0
        if min_j < 0: min_j = 0

        rgb = rgb[min_i:max_i, min_j:max_j, :]
        depth = depth[min_i:max_i, min_j:max_j]
        mask = mask[min_i:max_i, min_j:max_j]

        ## mask the RGB and depth image and apply histogram filtering on the depth image
        if detection_input == "mask":
            depth[depth < z_min] = 0
            depth[depth > z_max] = 0
            depth = depth * mask
            mask = np.clip(depth, 0.0, 1.0)
            rgb = rgb * np.expand_dims(mask.astype(bool), 2)
        
        w = max_i - min_i
        h = max_j - min_j

        return rgb, depth, mask, (min_i, min_j), (w,h), np.ones(mask.shape), pcd_filtered

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
        for fruit_id in self.split_ids:
            target_pcd_fname = os.path.join(self.data_source, fruit_id+'/laser/fruit.ply')
            target_pcd = o3d.io.read_point_cloud(target_pcd_fname)
            box = target_pcd.get_axis_aligned_bounding_box()
            dx, dy, dz = np.abs(box.get_max_bound() - box.get_min_bound())
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

            lst_images = os.listdir(self.data_source+id+'/realsense/masks/')
            lst_images.sort()
            for count_img, fname in enumerate(lst_images):
                files.append(self.data_source+id+'/realsense/color/'+fname)

        files.sort() 
        return files

    def get_intrinsics(self):
        """ 
        Load registration parameters
        
        Args:
      
        Returns: 
            data: dictionary containing intrinsic, boudning box 
                  and poses of each frame of each fruit
        """       
        Ks = {}
        for fruit in self.split_ids:
            path = os.path.join(self.data_source, fruit)
            k_path = os.path.join(path, 'realsense/intrinsic.json')
            k = self.load_K(k_path)
            Ks[fruit] = k
        
        return {'K': Ks}

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
        return len(self.files)

    def __getitem__(self, idx):

        # ugly stuff to get consistent ids
        fruit_id = self.files[idx].split('/')[-4]
        frame_id = self.files[idx].split('/')[-1][:-4]
        
        target_pcd_fname = os.path.join(self.data_source, fruit_id+'/laser/fruit.ply')
        target_pcd = o3d.io.read_point_cloud(target_pcd_fname)

        image_path = self.files[idx]
        depth_path = image_path.replace('color', 'depth')
        depth_path = depth_path.replace('png', 'npy')
        mask_path = image_path.replace('color', 'masks')

        realsense_dir = os.path.dirname(os.path.dirname(self.files[idx]))
        intrinsic_file = os.path.join(realsense_dir, "intrinsic.json")

        rgb = cv.imread(image_path)
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) // 255

        # getting intrinsic and camera pose
        k = self.Ks[fruit_id]
        rgb, depth, mask, crop_origin, crop_dim, padding_mask, partial_pcd = self.preprocess_images(rgb, depth, mask, intrinsic_file, self.detection_input)

        ## downsample the partial cloud
        down_pcd = partial_pcd.random_down_sample((self.pad_size+100)/len(partial_pcd.points))
        random_list = np.random.choice(len(down_pcd.points), size=self.pad_size, replace=False)
        down_pcd = down_pcd.select_by_index(random_list)

        intrinsic = self.update_intrinsic(crop_origin, depth.shape, k)

        item = {
            'dimension': crop_dim,
            'fruit_id': fruit_id,
            'frame_id': frame_id,
            # 'pose': pose,
            'K': intrinsic,
            'bbox': self.box,
        }

        # if self.sdf_loss:
        #     item['target_sdf'] = torch.from_numpy(target_sdf)
        #     item['target_sdf_weights'] = torch.from_numpy(target_sdf_weights)

        if self.normalize_depth:
            depth_original = copy.deepcopy(depth)
            depth = (depth_original - self.depth_min) / (self.depth_max - self.depth_min)
            depth[depth_original == 0] = 0
        else:
            depth = depth / self.depth_max

        if self.color_tf:
            rgb = torch.from_numpy(np.array(self.color_tf(rgb))).permute(2,0,1)
        else:
            if self.tf:
                rgb = torch.from_numpy(np.array(self.tf(rgb))).permute(2,0,1)

        if self.tf:
            depth = torch.from_numpy(np.array(self.tf(depth))).unsqueeze(dim=0) 
            mask = torch.from_numpy(np.array(self.tf(mask))).unsqueeze(dim=0)
            padding_mask = torch.from_numpy(np.array(self.tf(padding_mask))).unsqueeze(dim=0)

        item['rgb'] = rgb.float()/255 
        item['depth'] = depth.float()
        item['mask'] = mask
        item['padding_mask'] = padding_mask
        item['target_pcd'] = torch.Tensor(np.asarray(target_pcd.points))
        item['partial_pcd'] = torch.Tensor(np.asarray(down_pcd.points))

        if self.supervised_3d:
            trained_latent = self.latents_dict[fruit_id]
            item['latent'] = trained_latent.squeeze()

        return item


if __name__ == '__main__':

    # tfs = [Pad(size=128), ToTensor()]
    tfs = [Pad(size=500)]
    tf = transforms.Compose(tfs)
    cl_dataset = MaskedCameraLaserData(data_source="/home/federico/Datasets/shape_completion/potato/Potato/", 
                                tf=tf, supervised_3d=True,
                                pretrain='/home/federico/Datasets/shape_completion/potato/deepsdf_potato_weights/', 
                                pad_size=250,
                                sdf_loss=False,
                                grid_density=50,
                                split='train',
                                species='Potato')

    dataset = DataLoader(cl_dataset, batch_size=1, shuffle=False)

    c = 0
    for item in iter(dataset):
        # import ipdb;ipdb.set_trace()
        ids = item['fruit_id']
        c+=1
        print(c*32)
