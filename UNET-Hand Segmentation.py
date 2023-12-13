# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:01:46 2020

@author: Meng-Fen Tsai
"""

import os
import numpy as np
import skimage
from skimage.transform import resize
import segmentation_models as sm
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import re

def atoi(text):
    return int(text) if text.isdigit() else text
    

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

target_list='{path to the folder with all tasks to segment}/pipeline/';
ModelPath='{path to the model}/model-UNET-epoch100-stroke.h5';
im_resize=[320,320]; #image size

#Load the model
model = load_model(ModelPath, custom_objects={'binary_crossentropy_plus_jaccard_loss':sm.losses.bce_jaccard_loss,'iou_score':sm.metrics.iou_score});#import the loss function from segmentation model
print(model.summary()) 

#load the images to test
folder=next(os.walk(target_list))[1];
folder.sort(key=natural_keys);

for idx_folder in range(0,len(folder)):
    i=folder[idx_folder];
    num_image=0;
    X_test=[];mask_test=[];frameName_test=[];
    mask_savePath=target_list+i+'/Mask/';
    if not os.path.exists(mask_savePath):
        os.makedirs(mask_savePath)
    ids=next(os.walk(target_list+'/' + i +'/RGB405p2/'))[2];
    for frames in ids:
        if frames[-4:]=='.jpg':
            # Load images
            img = load_img(target_list+i+'/RGB405p2/'+frames);
            x_img = img_to_array(img); ## Convert the image pixels to a numpy array
            x_img = resize(x_img, (im_resize[0],im_resize[1], 3), mode='constant', preserve_range=True); # resize images for predictions
            #save predicted mask 
            x_test=np.array([x_img])/255;
            preds_test = model.predict(x_test, verbose=0);
            maskImg=preds_test.reshape(preds_test[0].shape[0:2]);
            A=skimage.transform.resize(maskImg,(405,720)); #resize to original image size
            # set a threshold to increase clarity of the mask image
            A = (A > 0.5);
            matplotlib.image.imsave(mask_savePath + frames[:-4] + '_masked.jpg', A,cmap='gray'); # make it binary (black and white)
    print('============= Finished: '+i+' ===============')
