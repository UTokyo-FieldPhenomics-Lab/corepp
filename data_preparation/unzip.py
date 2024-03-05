# -*- coding: utf-8 -*-
# @Author: Pieter Blok
# @Date:   2024-03-05 08:07:15
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2024-03-05 08:45:27

import argparse
import os
import zipfile


def unzip_to_folder(zip_files, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            print(f"Extracted '{zip_file}' to '{destination_folder}'")
        except zipfile.BadZipFile:
            print(f"Error: '{zip_file}' is not a valid zip file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip multiple zip files into a single folder.")
    parser.add_argument("--src", help="The directory containing the zip files.")
    parser.add_argument("--dst", help="The destination folder for the extracted files.")
    args = parser.parse_args()

    zip_files = [os.path.join(args.src, filename) for filename in os.listdir(args.src) if filename.endswith(".zip") and "3DPotatoTwin" in filename]

    unzip_to_folder(sorted(zip_files), args.dst)

    print("All files extracted successfully!")