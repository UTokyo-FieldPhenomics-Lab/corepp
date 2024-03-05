# -*- coding: utf-8 -*-
# @Author: Pieter Blok
# @Date:   2024-03-05 06:55:15
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2024-03-05 09:04:19
import argparse
import os 
import cv2
import numpy as np
from tqdm import tqdm

def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder

def main(src, dst):
    for potato_id in tqdm(os.listdir(src)):
        id_folder_path = os.path.join(src,potato_id)
        fnames = [f for f in os.listdir(id_folder_path) if 'rgb' in f and 'png' in f]
        for name in fnames:
            img_path = os.path.join(id_folder_path, name)
            dep_path = img_path.replace('_rgb_','_depth_')

            rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = rgba[:,:,:-1]
            mask = rgba[:,:,-1]

            dep = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED) # TODO check if some normalization needs to be done

            dst_img_dir = os.path.join(dst, potato_id, 'realsense/color/')
            dst_dep_dir = os.path.join(dst, potato_id, 'realsense/depth/')
            dst_ann_dir = os.path.join(dst, potato_id, 'realsense/masks/')

            check_direxcist(dst_img_dir)
            check_direxcist(dst_dep_dir)
            check_direxcist(dst_ann_dir)

            dst_name = os.path.splitext(name.replace('_rgb',''))[0]

            dst_img = os.path.join(dst_img_dir, dst_name+'.png')
            dst_dep = os.path.join(dst_dep_dir, dst_name+'.npy')
            dst_ann = os.path.join(dst_ann_dir, dst_name+'.png')

            with open(dst_dep, 'wb') as f:
                np.save(f, dep)
            cv2.imwrite(dst_ann, mask)
            cv2.imwrite(dst_img, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training the encoder")
    parser.add_argument('--src', type=str, default='./data/3DPotatoTwin/1_rgbd/1_image/', help="data source where the point meshes are stored")
    parser.add_argument('--dst', type=str, default='./data/potato/', help="destination folder")    
    args = parser.parse_args()

    main(args.src, args.dst)