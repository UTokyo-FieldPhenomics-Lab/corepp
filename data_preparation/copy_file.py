import argparse
import os
import shutil

def main(src, dst, subdir):
    subdirectories = next(os.walk(dst))[1]

    # Copy the file to each immediate subdirectory
    for subdirectory_name in subdirectories:
        if subdir != '':
            subdirectory_path = os.path.join(dst, subdirectory_name, subdir)
        else:
            subdirectory_path = os.path.join(dst, subdirectory_name)
        copy_path = os.path.join(subdirectory_path, os.path.basename(src))
        shutil.copyfile(src, copy_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training the encoder")
    parser.add_argument('--src', type=str, default='./3DPotatoTwin/1_rgbd/1_image/', help="data source where the point meshes are stored")
    parser.add_argument('--dst', type=str, default='./data/potato/', help="destination folder")
    parser.add_argument('--subdir', type=str, default='tf', help="the subdirectory where it needs to be copied to, see the structure in ./data/potato_example")    
    args = parser.parse_args()

    main(args.src, args.dst, args.subdir)