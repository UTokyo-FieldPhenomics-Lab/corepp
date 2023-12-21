import torch
from torch.autograd import Variable
import numpy as np

# Store grads for normals
grads = {}


# Hook definition
def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


class Grid3D:
    def __init__(self, density=30, device='cpu', precision=torch.float32, min_val=-1, max_val=1, points=None, bbox=None, offset=False):

        self.min_val = min_val
        self.max_val = max_val
        self.offset = offset

        if bbox:
            self.points=Variable(self.generate_point_from_bbox(density, bbox).to(device, precision), requires_grad=True)
        else:
            if isinstance(points, torch.Tensor): self.points = Variable(points.to(device, precision), requires_grad=True)
            else: self.points = Variable(self.generate_point_grid(density).to(device, precision), requires_grad=True)
        
        
        self.points.register_hook(save_grad('grid_points'))

    def generate_point_grid(self, grid_density):
        """
        Initial 3D point grid generation

        Args:
            grid_density (int): grid point density

        Returns: 3D point grid

        """
        # Set up the grid
        grid_density_complex = grid_density * 1j
        X, Y, Z = np.mgrid[self.min_val:self.max_val:grid_density_complex, self.min_val:self.max_val:grid_density_complex, self.min_val:self.max_val:grid_density_complex]
        grid_np = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

        # Make an offset for every second z grid plane
        if self.offset:
            grid_np[1::2, :2] += ((X.max() - X.min()) / grid_density / 2)
        grid = torch.from_numpy(grid_np.astype(np.float32))

        return grid

    def generate_point_from_bbox(self, grid_density, bbox):
        """
        Initial 3D point grid generation

        Args:
            grid_density (int): grid point density

        Returns: 3D point grid

        """
        # Set up the grid
        grid_density_complex = grid_density * 1j
        X, Y, Z = np.mgrid[bbox['xmin']:bbox['xmax']:grid_density_complex, bbox['ymin']:bbox['ymax']:grid_density_complex, bbox['zmin']:bbox['zmax']:grid_density_complex]
        grid_np = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=-1).reshape((-1, 3))

        # Make an offset for every second z grid plane
        grid_np[1::2, :2] += ((X.max() - X.min()) / grid_density / 2)
        grid = torch.from_numpy(grid_np.astype(np.float32))

        return grid

    def get_surface_points(self, pred_sdf_grid, threshold=0.03):
        """
        Zero isosurface projection

        Args:
            pred_sdf_grid (N,1): output of DeepSDF
            threshold (float): band of points to be projected onto the surface

        Returns: projected points (N,3), NOCS (N,3), normals (N,3)

        """
        # Get Jacobian / normals: backprop to vertices to get a jacobian
        pred_sdf_grid.sum().backward(retain_graph=True)
        normals = grads['grid_points'][:, :]

        normals_norm = torch.norm(normals, p=2, dim=1).detach()
        normals = normals / ((normals_norm.unsqueeze(-1).expand_as(normals)) + 0.000001)
        # print(torch.isnan(normals).any())

        # Project points onto the surface
        points = self.points - (pred_sdf_grid * normals)

        # Keep only SDF points close to the surface
        surface_mask = pred_sdf_grid.abs() < threshold
        points_masked = points.masked_select(surface_mask).view(-1, 3)
        normals_masked = normals.masked_select(surface_mask).view(-1, 3)
        points_masked_normed = (points_masked + 1) / 2

        return points_masked.to(pred_sdf_grid.to(pred_sdf_grid.dtype)
                                ), points_masked_normed.to(pred_sdf_grid.to(pred_sdf_grid.dtype)
                                                           ), normals_masked.to(pred_sdf_grid.dtype)
