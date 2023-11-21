# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:04:51 2020

@author: TsaiM
"""
##### must run under darkflow! C:\Users\TsaiM\Anaconda3\envs\tensorflow-Mengfen\darkflow #######

import cv2
import os
os.chdir('C:/Users/TsaiM/Anaconda3/envs/tensorflow-Mengfen/darkflow/')

from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
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

def yolo_to_txt(image_name,img,visualize_dets,RGBLocation,Output_Location,BBX_location):    
    bbx_savename=BBX_location+image_name[:-4]+'.txt';
    pred_result= tfnet.return_predict(img)
    
    image=cv2.imread(RGBLocation+'/'+image_name)
    if visualize_dets=='Yes' or 'yes' or 'Y' or 'y':
        predictImg_savename=Output_Location+'/dets/'+image_name[:-4]+'_detected.jpg';
        cv2.imwrite(predictImg_savename, image)
    output_file = open(bbx_savename, 'w') 
    output_file.write(str('Filename: '+image_name)+'\n')
    for i in range(0,len(pred_result)):# num of hands   
        xmax=pred_result[i]["bottomright"]["x"];
        ymax=pred_result[i]["bottomright"]["y"];
        xmin=pred_result[i]["topleft"]["x"];
        ymin=pred_result[i]["topleft"]["y"];
        output_file.write(pred_result[i]["label"]+': '+str(int(round(pred_result[i]["confidence"]*100)))+'% ['+str(xmin)+', '+str(ymin)+', '+str(xmax)+', '+str(ymax)+']\n');
        if visualize_dets=='Yes' or 'yes' or 'Y' or 'y':
            cv2.rectangle(image, (xmin,ymin),(xmax,ymax),(0,255,0),3)
            cv2.putText(image,pred_result[i]["label"]+': '+str(int(round(pred_result[i]["confidence"]*100)))+'%', (xmax,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0))
            cv2.imwrite(predictImg_savename, image)        
    output_file.close()
    
##################################################################################################
# load the pretrained YOLOv2 for hand detection
options = {"model": "C:/Users/TsaiM/Anaconda3/envs/tensorflow-Mengfen/darkflow/cfg/yolov2_hand.cfg", 
           "load": "C:/Users/TsaiM/Anaconda3/envs/tensorflow-Mengfen/darkflow/bin/yolov2_hand.weights", 
           "threshold": 0.15, 
           "gpu": 1.0}

tfnet = TFNet(options)

Target_Folder='//idapt_shares/NET/Stroke Hand Use/DO NOT BACK UP/Hand Labeling_Home/pipeline/';
subfolders=next(os.walk(Target_Folder))[1];
subfolders.sort(key=natural_keys)

for i in range(0,len(subfolders)):##########################
    if not os.path.exists(Target_Folder+'/'+subfolders[i]+'/bboxCoords/'):
        Output_Location = Target_Folder+'/'+subfolders[i]+'/DetectOutput/';        
        BBX_location=Target_Folder+'/'+subfolders[i]+'/bboxCoords/';        
        RGBLocation = Target_Folder+'/'+subfolders[i]+'/RGB405p2/';
        visualize_dets='Yes';
        
        if not os.path.exists(Target_Folder+'/'+subfolders[i]+'/bboxCoords/'):
            os.makedirs(Target_Folder+'/'+subfolders[i]+'/bboxCoords/')
        
        if visualize_dets == 'Yes':
            if not os.path.exists(Output_Location+'/dets/'):
                os.makedirs(Output_Location+'/dets/')
        
        img_list=next(os.walk(RGBLocation))[2];
        for j in range(0,len(img_list)):
            img_name=img_list[j];
            if img_name[-4:]=='.jpg':
                img=cv2.imread(RGBLocation+'/'+img_list[j]);
            #    plt.imshow(img)
                yolo_to_txt(img_name,img,visualize_dets,RGBLocation,Output_Location,BBX_location)
            
        print('Complete! Detections saved to: '+Output_Location)
