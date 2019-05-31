from __future__ import print_function
#%matplotlib inline
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
import torchsummary
import sys
import psutil
import gc
# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Root directory for dataset
dataroot = "split/DSMWRAPPER"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 8

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 512

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_distribution_sampler(mu,sigma):
    return lambda n: torch.Tensor(np.random.normal(mu,sigma,(1,n)))

def get_generator_input_sampler():
    return lambda m,n: torch.rand(m,n)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator,self).__init__()
        
        # ngf is Number Generator Features
        # nz is Noise input vector shape
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf*4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d( ngf*4, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//4),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf//4, ngf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//8),
            nn.ReLU(True),
            
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(ngf//8, nc, 4, 2, 1, bias=False),
            #nn.Tanh()
            # state size. nc x 256 x 256
        )
        
        
    def forward(self,x):
        return self.main(x)        
netG = Generator(1).to(device)

netG.apply(weights_init)

torchsummary.summary(netG,input_size=((100,1,1)))

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator,self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
                # state size nc x 64 x 64
                nn.Conv2d(nc,ndf,6,3,1,bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                # state size ndf x 32 x 32
                nn.Conv2d(ndf, ndf, 4, 2, 1,bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2,inplace=True),
                
                # state size ndf*2 x 16 x 16
                nn.Conv2d(ndf, ndf, 4, 2, 1,bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2,inplace=True),
                
                # state size ndf*4 x 8 x 8
                nn.Conv2d(ndf,ndf//2,4,2,1,bias=False),
                nn.BatchNorm2d(ndf//2),    
                nn.LeakyReLU(0.2,inplace=True),
                
                # state size ndf*4 x 8 x 8
                nn.Conv2d(ndf//2,ndf//2,4,2,1,bias=False),
                nn.BatchNorm2d(ndf//2),    
                nn.LeakyReLU(0.2,inplace=True),
               
                # state size ndf*4 x 8 x 8
                nn.Conv2d(ndf//2,ndf//4,4,2,1,bias=False),
                nn.BatchNorm2d(ndf//4),    
                nn.LeakyReLU(0.2,inplace=True),
                
                # state size ndf*4 x 8 x 8
                nn.Conv2d(ndf//4,ndf//8,4,2,2,bias=False),
                nn.BatchNorm2d(ndf//8),    
                nn.LeakyReLU(0.2,inplace=True),
                              
                # state size nc x 4 x 4
                nn.Conv2d(ndf//8,1,3,1,0,bias=False),
                nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.main(x)
        
netD = Discriminator(ngpu).to(device)

netD.apply(weights_init)

torchsummary.summary(netD,input_size=(1,512,512))


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create a batch of latent vectors that we will use to visuallize
# as we train the generator
fixed_noise = torch.randn(8, nz, 1, 1, device=device)

# Set labelling convention
real_label = 0
fake_label = 1

# Set up the two optimizers for G and D, we use ADAM since it's in the paper
optimizerD = optim.Adam(netD.parameters(),lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr, betas=(beta1, 0.999))

fixed_noise.shape
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(30,30))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))


fake = netG(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()