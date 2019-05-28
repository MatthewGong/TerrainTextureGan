from __future__ import print_function


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

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "split/DSMWRAPPER"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

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


netG = Generator(1).to(device)

netG.apply(weights_init)

torchsummary.summary(netG,input_size=(100,1,1))

netD = Discriminator(ngpu).to(device)

netD.apply(weights_init)

torchsummary.summary(netD,input_size=(1,64,64))


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create a batch of latent vectors that we will use to visuallize
# as we train the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Set labelling convention
real_label = 1
fake_label = 0

# Set up the two optimizers for G and D, we use ADAM since it's in the paper
optimizerD = optim.Adam(netD.parameters(),lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr, betas=(beta1, 0.999))

# Training Loop

# lists to keep track of losses and progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop")

#for epoch in range(25):
    
for epoch in range(num_epochs):
    
    # for each batch in the dataloader
    for i, data in enumerate(dataloader,0):
        
        ####
        # Train the discriminator: Maximize log(D(x) + log(1- D(G(z)))
        ####
        
        ## Train with all real batch
        netD.zero_grad()
        
        # format the batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0) #checks to see uniform batch size
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass for real batch through the Discriminator
        output = netD(real_cpu).view(-1)
        
        # Calculate loss
        errD_real = criterion(output, label)
        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        
        D_x = output.mean().item()
        
        ## Train with all fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        
        # Generate Fake images using the Generator
        fake = netG(noise)
        label.fill_(fake_label)
        
        # Classify fake images
        output = netD(fake.detach()).view(-1)
        
        # Calculate the Discriminators loss on the fake batch
        errD_fake = criterion(output,label)
        
        # Calculate the gradients of the backward pass for the Generator
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # Add the gradients from the real and fake batches
        errD = errD_real + errD_fake
        optimizerD.step()
        
        ####
        # training the generator: Maximize log(D(G(z)))
        ###
        
        netG.zero_grad()
        label.fill_(real_label) # fake labels are real for the generator cost
        
        # since we just updated D, we perform another forward pass of all fake through D
        output = netD(fake).view(-1)
        
        # Calculate generator loss basd on the output
        errG = criterion(output,label)
        
        # Calculate gradients
        errG.backward()
        D_G_z2 = output.mean().item()
        
        # update G
        optimizerG.step()
        
        # Output stats during training
        if i%50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D; %.4f\tLoss_g:%.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                 % (epoch, num_epochs, i, len(dataloader),
                   errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(30,30))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


torch.save(netD,"discriminator.pt")
torch.save(netG,"generator.pt")