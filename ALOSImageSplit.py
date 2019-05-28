import os
import numpy as np

from scipy.misc import imsave
from PIL import Image
from matplotlib import pyplot as plt


def SplitImage(img,name,strideX=256,strideY=256,width=512,height=512):
    X,Y = img.shape

    for i in range(X//strideX - 1):
        xStart = i*strideX
        xEnd = xStart+width       
        for j in range(Y//strideY - 1):
            yStart = j*strideY
            yEnd = yStart + height 
            
            imsave(name+"x{}y{}.jpg".format(i,j),img[xStart:xEnd,yStart:yEnd])

tifs = []
for root, dirs, files in os.walk(os.path.join("..","TopographicalData"), topdown=False):
   
    for name in files:
        if os.path.join(root, name).endswith("STK.tif"):
            tifs.append(os.path.join(root, name))

for tif in tifs:
    print("splitting " + tif)
    im = Image.open(tif)
    
    image_array = np.array(im,dtype="float") 
    image_array -= np.amin(image_array)
    image_array += .0001 #added in epsilon in case the max is zero 
    image_array /= np.amax(image_array)

    SplitImage(image_array,os.path.join("..","split","STK",tif.split(os.sep)[-1][:-4]))


im = Image.open("TopographicalData\\N035W120_N040W115\\N035W116_AVE_STK.tif")

print(im.size)

image_array = np.array(im,dtype="float") 
print(np.amin(image_array),np.amax(image_array))

plt.imshow(image_array,cmap="plasma")

