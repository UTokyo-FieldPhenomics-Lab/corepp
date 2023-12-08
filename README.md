# Shape Completion

Used to train DeepSDF

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
