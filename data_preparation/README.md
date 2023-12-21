# Data Preparation

1. Convert all the SfM files into simple point cloud files `python3 pcd_from_sfm.py <src-dir> <dst-dir>`
2. Create augmentations `python3 augment.py <path-to>/augment.json <src-dir> <dst-dir>`
3. Generate npz files from point clouds `pyhon3 prepare_deepsdf_training_data.py <src-dir> <dst-dir>`
4. Create train and test split files into `<path-to-shape_completion>/deepsdf/experiments/splits/`. See the provided example.
5. Create an experiment folder inside `<path-to-shape_completion>/deepsdf/experiments/<exp-name>`. See the provided example.
