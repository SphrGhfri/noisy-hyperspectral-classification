# Copyright 2014-2017 Bert Carremans
# Author: Bert Carremans <bertcarremans.be>
#
# License: BSD 3 clause


import os
import random
from shutil import copyfile,rmtree

       
source = ['/noiseless',
          '/gaussian/noise5',
          '/gaussian/noise10',
          '/gaussian/noise15',
          '/gaussian/noise20',
          '/gaussian/noise25',
          '/saltPepper/noise5',
          '/saltPepper/noise10',
          '/saltPepper/noise15',
          '/saltPepper/noise20',
          '/saltPepper/noise25',
          '/stripe/noise5',
          '/stripe/noise10',
          '/stripe/noise15',
          '/stripe/noise20',
          '/stripe/noise25',
          ]       
for i in range(16):
    if not os.path.exists('Pavia_Split_data/train'+source[i]):
        os.makedirs('Pavia_Split_data/train'+source[i])
    if not os.path.exists('Pavia_Split_data/validation'+source[i]):
        os.makedirs('Pavia_Split_data/validation'+source[i])

        
for i in range(16):

    subdirs = [subdir for subdir in os.listdir('Pavia_pix2img'+source[i]) if os.path.isdir(os.path.join('Pavia_pix2img'+source[i], subdir))]
    
    for subdir in subdirs:
        subdir_fullpath = os.path.join('Pavia_pix2img'+source[i], subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break
    
        train_subdir = os.path.join('Pavia_Split_data/train'+source[i], subdir)
        validation_subdir = os.path.join('Pavia_Split_data/validation'+source[i], subdir)
    
        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)
    
        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)
    
        train_counter = 0
        validation_counter = 0
    
        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                fileparts = filename.split('.')
    
                if random.uniform(0, 1) <= 0.7:
                    copyfile(os.path.join(subdir_fullpath, filename), os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename), os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                    validation_counter += 1
                    
        print('Copied ' + str(train_counter) + ' images to data/train'+source[i]+'/'+ subdir)
        print('Copied ' + str(validation_counter) + ' images to data/validation'+source[i]+'/'+ subdir)
        

rmtree('Pavia_Split_data/train/gaussian')
rmtree('Pavia_Split_data/train/saltPepper')
rmtree('Pavia_Split_data/train/stripe')