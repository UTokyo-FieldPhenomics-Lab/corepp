# CoRe++: High-Throughput 3D Shape Completion of RGB-D Images

![CoRe++](data/git_promo.gif)

### Installation

[INSTALL.md](INSTALL.md)
<br/><br/>

### Dataset
Our dataset will be made public as soon as our manuscript is published.
<br/><br/>

### Instructions

1. Download our [demo dataset](https://github.com/UTokyo-FieldPhenomics-Lab/corepp/releases/tag/demo_dataset).
2. Place the zip file in the data folder and unzip the files
3. Prepare the dataset for training DeepSDF
    ```python
    python data_preparation/pcd_from_sfm.py --src ./data/3DPotatoTwinDemo/2_sfm/1_mesh --dst ./data/potato
    python data_preparation/augment.py --json_config_filename ./data_preparation/augment.json --src ./data/potato --dst ./data/potato_augmented
    python data_preparation/prepare_deepsdf_training_data.py --src ./data/potato
    python data_preparation/prepare_deepsdf_training_data.py --src ./data/potato_augmented
    ```
4. Change the file paths in **deepsdf/experiments/potato/specs.json** such that they correspond to your file paths
5. Train DeepSDF
    ```python
    python train_deep_sdf.py --experiment ./deepsdf/experiments/potato
    ```
6. Reconstruct the 3D shapes with DeepSDF
    ```console
    bash run_scripts_reconstruct.sh
    ```
7. Compute the reconstructing metrics and determine the best weights file
    ```console
    bash run_scripts_metrics.sh
    ```
8. For the best weights run the following 3 commands. In this example the best weights are at checkpoint 500.
    ```python
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_train_without_augmentations.json
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_val.json
    python reconstruct_deep_sdf.py --experiment ./deepsdf/experiments/potato --data ./data --checkpoint 500 --split ./deepsdf/experiments/splits/potato_test.json
    ```
9. Prepare the dataset for training the encoder
    ```python
    python data_preparation/organize_data.py --src ./data/3DPotatoTwinDemo/1_rgbd/1_image --dst ./data/potato
    python data_preparation/copy_file.py --src ./data/potato_example/dataset.json --dst ./data/potato --subdir ""
    python data_preparation/copy_file.py --src ./data/potato_example/tf/tf.npz --dst ./data/potato --subdir "tf"
    python data_preparation/copy_file.py --src ./data/potato_example/tf/bounding_box.npz --dst ./data/potato --subdir "tf"
    python data_preparation/copy_file.py --src ./data/potato_example/realsense/intrinsic.json --dst ./data/potato --subdir "realsense"
    ```
10. Change the file paths in **configs/super3d.json** such that they correspond to your file paths
11. Train the encoder
    ```python
    python train.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato/ --checkpoint_decoder 500
    ```
12. Test the encoder
    ```python
    python test.py --cfg ./configs/super3d.json --experiment ./deepsdf/experiments/potato/ --checkpoint_decoder 500
    ```
<br/>

### Acknowledgements
CoRe++ is the updated version of Federico Magistri's original CoRe implementation: <br/>
https://github.com/magistri/shape_completion<br/>
https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/magistri2022ral-iros.pdf
