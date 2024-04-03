import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import argparse

def file_finder(data_dir, str_search, exclude_files):
    file_selection = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if str_search in file_path:
                potato_id = file_path.split(data_dir)[-1].split("/")[0]
                if potato_id not in exclude_files:
                    file_selection.append(file_path)
    return file_selection


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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", required=True)
    arg_parser.add_argument("--offset", required=True)
    arg_parser.add_argument("--split_file", required=True)
    args = arg_parser.parse_args()

    with open(args.split_file) as json_file:
        test_files = json.load(json_file)['test']

    largest_box = [0, 0]

    mask_files = file_finder(args.data_dir, "realsense/masks", test_files)

    for mask_file in tqdm(mask_files):
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) // 255
        indices = np.where(mask)
        min_i = indices[0].min()
        min_j = indices[1].min()
        max_i = indices[0].max()
        max_j = indices[1].max()

        if (max_i - min_i) > largest_box[0]:
            largest_box[0] = max_i - min_i
        if (max_j - min_j) > largest_box[1]:
            largest_box[1] = max_j - min_j

    print(f"Set 'input_size' in config file at least to: {np.max(largest_box) + 2 * int(args.offset)}")

    depth_files = file_finder(args.data_dir, "realsense/depth", test_files)

    lowest_zmin = np.inf
    highest_zmax = 0.0

    for depth_file in tqdm(depth_files):
        potato_id = depth_file.split(args.data_dir)[-1].split("/")[0]
        mask_file = depth_file.replace("depth", "masks")
        mask_file = mask_file.replace("npy", "png")
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) // 255
        depth = np.load(depth_file).astype(np.float32)
        dimg_mask = np.multiply(depth, mask)
        z_min, z_max = histogram_filtering(depth, mask, 50, 0.05)

        if z_min < lowest_zmin:
            lowest_zmin = z_min
        if z_max > highest_zmax:
            highest_zmax = z_max

    print(f"Lowest Z-min is: {lowest_zmin} mm")
    print(f"Highest Z-max is: {highest_zmax} mm")