import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from Networks import Generator

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

generator = torch.load("generator.pt")

fixed_noise = torch.randn(64, 100, 1, 1,device='cuda')
images = []

img = generator(fixed_noise).detach().cpu()
images.append(vutils.make_grid(img, padding=2, normalize=True))
    

# Plot the fake images from the last epoch
plt.figure(figsize=(10,5))
plt.axis("off")
plt.title("Fake Terrain")
plt.imshow(np.transpose(images[-1],(1,2,0)))
plt.show()