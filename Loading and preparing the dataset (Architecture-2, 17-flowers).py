# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 13:32:01 2018

@author: Simran Tinani
"""

import tarfile
import collections
import os
import cv2
import numpy as np

# Extract the image files from the downloaded tar file
# Here, we use the larger dataset with 102 flower types. 
# The same code works for the 17-label dataset, by merely changing the file/variable names.

k=tarfile.open('17flowers.tgz') 
k.extractall()
k.close()

# The images are now extracted into the working directory in .jpg form. 
# We write the image file names into a list and then into a text file named "flowers102" for future use.

files=list()
DirectoryIndex = collections.namedtuple('DirectoryIndex', ['root', 'dirs', 'files'])

for file_name in DirectoryIndex(*next(os.walk('.\\working_folder'))).files:
# insert the path of the working directory in the quotes
    if file_name.endswith(".jpg"):
        files.append(file_name)
        
# Create a blank text file named flowers102 in the working directory
textf  = open('flowers17.txt','w')
for file in files:
    textf.write(file + "\n")
textf.close()

# Now we extract the digital forms (pixel representations) of all the images.

line_file=0
flowers=np.zeros((len(files),28,28,3)) # the tensor which will hold the digital forms of the images

l=0

for line in open("flowers17.txt"): # loop over all entries of the text file
    # print(l+1)
    line=line.strip('\n')
    line_file=line
    # print line_4
    img = cv2.imread(line_file,cv2.IMREAD_COLOR)
    img1 = cv2.resize(img, (28,28)) # resize the image to have 28 rows and 28 columns
    img2 = np.asarray(img1, dtype='float32') # convert to numpy array
    flowers[l]=img2 # populate the tensor with pixel values of each image
    l=l+1 # l ranges from 0 to 8188


print(flowers.shape)
