import torch
import matplotlib.pyplot as plt

path = '/home/federico/Projects/magistri2022icra/shape_completion/deepsdf/experiments/laserdata/LatentCodes/latest.pth'

latents = torch.load(path)
latents = latents['latent_codes']['weight'].numpy()

print(latents.shape)

plt.scatter(latents[:, 0], latents[:, 1])
plt.show()