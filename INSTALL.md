## Installation
Tested on Ubuntu 22.04 with a NVIDIA GeForce RTX 4090 Laptop GPU <br/>
Software: Pytorch 2.1.0, torchvision 0.16.0, Python 3.10, CUDA 12.1 <br/> <br/>

**1) Download and install Anaconda:**
- download anaconda: https://www.anaconda.com/download (python 3.x version)
- install anaconda (using the terminal, cd to the directory where the file has been downloaded): bash Anaconda3-[distribution].sh <br/> <br/>

**2) Make a virtual environment (called corepp) using the terminal:**
- conda create --name corepp python=3.10 pip
- conda activate corepp <br/> <br/>

**3) Download the code repository:**
- git clone https://github.com/UTokyo-FieldPhenomics-Lab/corepp.git 
- cd corepp <br/> <br/>

**4) Install the required software libraries (in the corepp virtual environment, using the terminal):**
- pip install -U torch==2.1.0 torchvision==0.16.0 -f https://download.pytorch.org/whl/cu121/torch_stable.html
- pip install open3d==0.17.0
- pip install scikit-image==0.22.0
- pip install plyfile==1.0.2
- pip install Pillow==9.5.0 
- pip install trimesh==4.0.5 
- pip install diskcache==5.6.3
- pip install tensorboard==2.15.1
- pip install numba==0.58.1 
- pip install opencv-python==4.8.1.78 <br/> <br/>

**5) Check if Pytorch links with CUDA (in the corepp virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 12.1)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**Optional**: alter ~/.bashrc file to prevent libGL error when doing open3d visualization, refer to [link](https://github.com/conda-forge/ctng-compilers-feedstock/issues/95)
- cd ..
- sudo gedit ~/.bashrc
- add this line at the end of the bashrc file: **export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6**
- save and close the bashrc file
- source ~/.bashrc <br/> <br/>
