# DeepSDF modified from the original repository

This is a simplified version of DeepSDF where I removed all the preprocessing steps, those are outdated and not relevant for us at the moment.
The original repository is [here](https://github.com/facebookresearch/DeepSDF)

Below you can read the original instructions

# DeepSDF

This is an implementation of the CVPR '19 paper "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" by Park et al. See the paper [here][6]. 


## Citing DeepSDF

If you use DeepSDF in your research, please cite the
[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html):
```
@InProceedings{Park_2019_CVPR,
author = {Park, Jeong Joon and Florence, Peter and Straub, Julian and Newcombe, Richard and Lovegrove, Steven},
title = {DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## File Organization

The various Python scripts assume a shared organizational structure such that the output from one script can easily be used as input to another. This is true for both preprocessed data as well as experiments which make use of the datasets.

##### Data Layout

The DeepSDF code allows for pre-processing of meshes from multiple datasets and stores them in a unified data source. It also allows for separation of meshes according to class at the dataset level. The structure is as follows:

```
<data_source_name>/
    .datasources.json
    SdfSamples/
        <dataset_name>/
            <class_name>/
                <instance_name>.npz
```

Subsets of the unified data source can be reference using split files, which are stored in a simple JSON format. For examples, see `experiments/splits/`. 

##### Experiment Layout

Each DeepSDF experiment is organized in an "experiment directory", which collects all of the data relevant to a particular experiment. The structure is as follows:

```
<experiment_name>/
    specs.json
    Logs.pth
    LatentCodes/
        <Epoch>.pth
    ModelParameters/
        <Epoch>.pth
    OptimizerParameters/
        <Epoch>.pth
    Reconstructions/
        <Epoch>/
            Codes/
                <MeshId>.pth
            Meshes/
                <MeshId>.pth
    Evaluations/
        Chamfer/
            <Epoch>.json
        EarthMoversDistance/
            <Epoch>.json
```

The only file that is required to begin an experiment is 'specs.json', which sets the parameters, network architecture, and data to be used for the experiment.

## How to Use DeepSDF


### Training a Model

Once data has been preprocessed, models can be trained using:

```
python train_deep_sdf.py -e <experiment_directory>
```

Parameters of training are stored in a "specification file" in the experiment directory, which (1) avoids proliferation of command line arguments and (2) allows for easy reproducibility. This specification file includes a reference to the data directory and a split file specifying which subset of the data to use for training.

##### Visualizing Progress

All intermediate results from training are stored in the experiment directory. To visualize the progress of a model during training, run:

```
python plot_log.py -e <experiment_directory>
```

By default, this will plot the loss but other values can be shown using the `--type` flag.

##### Continuing from a Saved Optimization State

If training is interrupted, pass the `--continue` flag along with a epoch index to `train_deep_sdf.py` to continue from the saved state at that epoch. Note that the saved state needs to be present --- to check which checkpoints are available for a given experiment, check the `ModelParameters', 'OptimizerParameters', and 'LatentCodes' directories (all three are needed).

### Reconstructing Meshes

To use a trained model to reconstruct explicit mesh representations of shapes from the test set, run:

```
python reconstruct_deep_sdf.py -e <experiment_directory>
```

This will use the latest model parameters to reconstruct all the meshes in the split. To specify a particular checkpoint to use for reconstruction, use the ```--checkpoint``` flag followed by the epoch number. Generally, test SDF sampling strategy and regularization could affect the quality of the test reconstructions. For example, sampling aggressively near the surface could provide accurate surface details but might leave under-sampled space unconstrained, and using high L2 regularization coefficient could result in perceptually better but quantitatively worse test reconstructions.

## License

DeepSDF is relased under the MIT License. See the [LICENSE file][5] for more details.

[5]: https://github.com/magistri/shape_completion/blob/master/deepsdf/LICENSE
[6]: http://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html
