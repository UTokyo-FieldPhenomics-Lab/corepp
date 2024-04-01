import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
import operator

import networks.blocks as blocks

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBig(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBig, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBigPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBigPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class ERFNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        self.initial_block = blocks.DownsamplerBlock(in_channels,16)

        self.layers = [self.initial_block]

        self.layers.append(blocks.DownsamplerBlock(16,64))

        # for x in range(0, 1):    #5 times
        #    self.layers.append(blocks.non_bottleneck_1d(64, 0.03, 1)) 

        # self.layers.append(blocks.DownsamplerBlock(64,128))

        for x in range(0, 1):    #2 times
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 2))
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 4))
            # self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 8))
            # self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 16))

        self.layers.append(nn.Flatten())
        
        encoder = nn.Sequential(*self.layers)
        self.num_features = functools.reduce(operator.mul, list(encoder(torch.rand(1, *input_dim)).shape))

        self.layers.append(nn.Linear(self.num_features, self.latent_size, bias=True))
        self.encoder = nn.Sequential(*self.layers)
    
    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()


## thanks to: https://github.com/cihanongun/Point-Cloud-Autoencoder
class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudEncoder, self).__init__()
        
        self.latent_size = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, self.latent_size)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        return x


## thanks to: https://github.com/HAN-oQo/Pointnetautoencoder_pytorch/blob/main/model/model.py
class PointCloudEncoderLarge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudEncoderLarge, self).__init__()

        self.latent_size = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512) # make global feature dimension smaller
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_size) #latent space

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2, keepdim= True)[0]
        x = x.view(-1, 1024) # 1*1024

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    def forward(self, x):
        x = self.encoder(x)
        return x
    

## thanks to: https://github.com/antao97/UnsupervisedPointCloudReconstruction/blob/master/model.py
class FoldNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FoldNetEncoder, self).__init__()
        self.latent_size = out_channels
        self.k = 16
        self.feat_dims = 512
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, self.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(self.feat_dims, self.latent_size, 1),
        )

    def knn(self, x, k):
        batch_size = x.size(0)
        num_points = x.size(2)

        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

        if idx.get_device() == -1:
            idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
        else:
            idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        return idx


    def local_cov(self, pts, idx):
        batch_size = pts.size(0)
        num_points = pts.size(2)
        pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
    
        _, num_dims, _ = pts.size()

        x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
        x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
        x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

        x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
        # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
        x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

        x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)
        return x


    def local_maxpool(self, x, idx):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
        x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
        x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
        x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)
        return x

    def graph_layer(self, x, idx):           
        x = self.local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = self.local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        idx = self.knn(pts, k=self.k)
        x = self.local_cov(pts, idx)            # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, latent_size, 1)
        x = x.view(-1, self.latent_size)        # (batch_size, latent_size, 1) -> (batch_size, latent_size)
        return x                                # (batch_size, latent_size)