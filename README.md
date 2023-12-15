# Shape Completion

### How to Cite

If you use this code in your research, please cite our
[paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/magistri2022ral-iros.pdf):
```
@article{magistri2022ral-iros,
author = {Federico Magistri and Elias Marks and Sumanth Nagulavancha and Ignacio Vizzo and Thomas L{\"a}be and Jens Behley and Michael Halstead and Chris McCool and Cyrill Stachniss},
title = {Contrastive 3D Shape Completion and Reconstruction for Agricultural Robots using RGB-D Frames},
journal = ral,
url = {https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/magistri2022ral-iros.pdf},
year = {2022},
volume={7},
number={4},
pages={10120-10127},
videourl = {https://www.youtube.com/watch?v=2ErUf9q7YOI},
}
```

### Installation

Look at [INSTALL.md](INSTALL.md).

### Training the Network 

At the moment we can run a simple training by issuing:
* `python main.py -e ./deepsdf/experiments/<name> -c ./configs/<cfg>.json`
NB: you need to pretrain a DeepSDF network first

### Testing the Network 

At the moment we can run a simple testing by issuing:
* `python test.py -e ./deepsdf/experiments/<name> -c ./configs/<cfg>.json` 

### Pre-Train DeepSDF

To pretrain a DeepSDF model:
* `python train_deep_sdf.py -e ./deepsdf/experiments/<name>`

To test DeepSDF results:
* `python reconstruct_deep_sdf.py -e ./deepsdf/experiments/<name> -d ./data/<name> -s ./deepsdf/experiments/splits/<name>.json`

the framework expect a specific structure, read the instruction in the `deepsdf` folder. 
