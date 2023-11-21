# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:03:41 2020

Feature extraction with UNET hand segmentation

@author: Meng-Fen Tsai
"""
import numpy as np
#import matplotlib.pyplot as plt
import os
import re
import cv2
import scipy.io as sio

from skimage.feature import hog
from skimage import color

def atoi(text):
    return int(text) if text.isdigit() else text
    
def optFlow(prev, now):
    flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return mag, ang

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def hogFeature(frame):
    image = color.rgb2gray(frame)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, transform_sqrt=True)
    
    return fd, hog_image

def CalFea(framename,RGBfolder,Maskfolder,BBXfolder):
    #load RGB and mask
    Frame_RGB=cv2.imread(RGBfolder+framename);
    Frame_HSV=cv2.cvtColor(Frame_RGB, cv2.COLOR_BGR2HSV)
    #change color for optical flow
    Frame_OPT=cv2.cvtColor(Frame_RGB, cv2.COLOR_BGR2GRAY)
    
    #plt.imshow(Frame_RGB)
    Frame_mask=cv2.imread(Maskfolder+framename[:-4]+'_masked.jpg',0);# binary
    
    # make all black BG
    BlackBG = np.ones((img_height, img_width), np.uint8)
     
    #load bbx
    BBXfilename = framename[:-4]
    Frame_bbx_R=[];Frame_bbx_L=[];    
    # Check if the txt file exists
    if os.path.exists(BBXfolder + BBXfilename + '.txt'):
        # and read it
        with open(BBXfolder + BBXfilename + '.txt', 'r') as b:
            listBox = [line.strip() for line in b]
        if len(listBox)>1:
            for j in range(0,len(listBox)):
                if listBox[j-1][0]=='R':
                    Frame_bbx_R=listBox[j-1].split('[')[1].strip(']').split(', ');
                    for l in range(0,4):
                        if int(Frame_bbx_R[l])<0:
                            Frame_bbx_R[l]='0';
                    Frame_RGB_crop_R=Frame_RGB[int(Frame_bbx_R[1]):int(Frame_bbx_R[3]), int(Frame_bbx_R[0]):int(Frame_bbx_R[2])]
                    #plt.imshow(Frame_RGB_crop_R)
                    Frame_mask_crop_R=Frame_mask[int(Frame_bbx_R[1]):int(Frame_bbx_R[3]), int(Frame_bbx_R[0]):int(Frame_bbx_R[2])]
                    
                    # make bbx_R in white in the BG
                    BlackBG[int(Frame_bbx_R[1]):int(Frame_bbx_R[3]), int(Frame_bbx_R[0]):int(Frame_bbx_R[2])] = 0
                    #plt.imshow(BlackBG)
                    
                    #### Locations #####
                    hand_loc_R_crop=np.where(Frame_mask_crop_R>=200)#crop coord
                    #hand region area
                    area_hand_R=np.shape(hand_loc_R_crop)[1]
                    nonhand_loc_R_crop=np.where(Frame_mask_crop_R<200)#crop coord
                    # for whole img coord
                    hand_loc_R=np.array(np.where(Frame_mask_crop_R>=200));
                    hand_loc_R=hand_loc_R.T;
                    hand_loc_R=hand_loc_R+[int(Frame_bbx_R[1]),int(Frame_bbx_R[0])];
                    hand_loc_R=hand_loc_R.T;
                    hand_loc_R=tuple(hand_loc_R)
                            
                    nonhand_loc_R=np.array(np.where(Frame_mask_crop_R<200));
                    nonhand_loc_R=nonhand_loc_R.T;
                    nonhand_loc_R=nonhand_loc_R+[int(Frame_bbx_R[1]),int(Frame_bbx_R[0])];
                    nonhand_loc_R=nonhand_loc_R.T;
                    nonhand_loc_R=tuple(nonhand_loc_R)
                            
                    ######## HOG ########
                    resize_RGB_crop_R=cv2.resize(Frame_RGB_crop_R,(int(0.1*img_width),int(0.1*img_height)),interpolation = cv2.INTER_CUBIC)
                    fullHOGfea_R, hogImg_R=hogFeature(resize_RGB_crop_R)
                    resize_RGB_crop_R=[];
                    reshape_fullHOGfea_R=np.reshape(fullHOGfea_R,((len(fullHOGfea_R)//9),9))
                    meanHOGfea_R=np.mean(reshape_fullHOGfea_R,axis=0)
                    stdHOGfea_R=np.std(reshape_fullHOGfea_R,axis=0)
                    
                    ####### Colour ######
                    Frame_bbx_crop_HSV_R=cv2.cvtColor(Frame_RGB_crop_R, cv2.COLOR_BGR2HSV)
                    Frame_hand_HSV_R=Frame_bbx_crop_HSV_R[hand_loc_R_crop] 
                    Frame_nonhand_HSV_R=Frame_bbx_crop_HSV_R[nonhand_loc_R_crop]
                    
                    #histogram of R hand colour
                    y_BG, x_BG = np.shape(Frame_hand_HSV_R)
                    blank_hand_R = np.zeros((y_BG, 1, x_BG), np.uint8)
                    blank_hand_R[:,0,0] = Frame_hand_HSV_R[:,0]
                    blank_hand_R[:,0,1] = Frame_hand_HSV_R[:,1]
                    blank_hand_R[:,0,2] = Frame_hand_HSV_R[:,2]
                    #histogram of color-hand R
                    histHandR = cv2.calcHist([blank_hand_R], [0, 1, 2], None, [30, 32, 32], [0, 179, 0, 255, 0, 255])

                    #histogram of R nonhand colour
                    y_BG, x_BG = np.shape(Frame_nonhand_HSV_R)
                    blank_nonhand_R = np.zeros((y_BG, 1, x_BG), np.uint8)
                    blank_nonhand_R[:,0,0] = Frame_nonhand_HSV_R[:,0]
                    blank_nonhand_R[:,0,1] = Frame_nonhand_HSV_R[:,1]
                    blank_nonhand_R[:,0,2] = Frame_nonhand_HSV_R[:,2]
                    #histogram of color-hand R
                    histNonhandR = cv2.calcHist([blank_nonhand_R], [0, 1, 2], None, [30, 32, 32], [0, 179, 0, 255, 0, 255])
            
                    # change colour space for optical flow
                    Frame_OPT_crop_R= cv2.cvtColor(Frame_RGB_crop_R, cv2.COLOR_BGR2GRAY)
                        
                elif listBox[j-1][0]=='L':
                    Frame_bbx_L=listBox[j-1].split('[')[1].strip(']').split(', ');
                    for l in range(0,4):
                        if int(Frame_bbx_L[l])<0:
                            Frame_bbx_L[l]='0';
                    Frame_RGB_crop_L=Frame_RGB[int(Frame_bbx_L[1]):int(Frame_bbx_L[3]), int(Frame_bbx_L[0]):int(Frame_bbx_L[2])]
                    #plt.imshow(Frame_RGB_crop_L)
                    Frame_mask_crop_L=Frame_mask[int(Frame_bbx_L[1]):int(Frame_bbx_L[3]), int(Frame_bbx_L[0]):int(Frame_bbx_L[2])]
                    
                    # make bbx_L in white in the BG
                    BlackBG[int(Frame_bbx_L[1]):int(Frame_bbx_L[3]), int(Frame_bbx_L[0]):int(Frame_bbx_L[2])] = 0
                    #plt.imshow(BlackBG)
                    
                    #### Locations #####
                    hand_loc_L_crop=np.where(Frame_mask_crop_L>=200)#crop coord
                    area_hand_L=np.shape(hand_loc_L_crop)[1]
                    
                    nonhand_loc_L_crop=np.where(Frame_mask_crop_L<200)#crop coord
                    # for whole img coord
                    hand_loc_L=np.array(np.where(Frame_mask_crop_L>=200));
                    hand_loc_L=hand_loc_L.T;
                    hand_loc_L=hand_loc_L+[int(Frame_bbx_L[1]),int(Frame_bbx_L[0])];
                    hand_loc_L=hand_loc_L.T;
                    hand_loc_L=tuple(hand_loc_L)
                            
                    nonhand_loc_L=np.array(np.where(Frame_mask_crop_L<200));
                    nonhand_loc_L=nonhand_loc_L.T;
                    nonhand_loc_L=nonhand_loc_L+[int(Frame_bbx_L[1]),int(Frame_bbx_L[0])];
                    nonhand_loc_L=nonhand_loc_L.T;
                    nonhand_loc_L=tuple(nonhand_loc_L)
                    ######## HOG ########
                    resize_RGB_crop_L=cv2.resize(Frame_RGB_crop_L,(int(0.1*img_width),int(0.1*img_height)),interpolation = cv2.INTER_CUBIC)
                    fullHOGfea_L, hogImg_L=hogFeature(resize_RGB_crop_L)
                    resize_RGB_crop_L=[];
                    reshape_fullHOGfea_L=np.reshape(fullHOGfea_L,((len(fullHOGfea_L)//9),9))
                    meanHOGfea_L=np.mean(reshape_fullHOGfea_L,axis=0)
                    stdHOGfea_L=np.std(reshape_fullHOGfea_L,axis=0)
                    
                    ####### Colour ######
                    Frame_bbx_crop_HSV_L=cv2.cvtColor(Frame_RGB_crop_L, cv2.COLOR_BGR2HSV)
                    Frame_hand_HSV_L=Frame_bbx_crop_HSV_L[hand_loc_L_crop]
                    Frame_nonhand_HSV_L=Frame_bbx_crop_HSV_L[nonhand_loc_L_crop]
                    
                    #histogram of L hand colour
                    y_BG, x_BG = np.shape(Frame_hand_HSV_L)
                    blank_hand_L = np.zeros((y_BG, 1, x_BG), np.uint8)
                    blank_hand_L[:,0,0] = Frame_hand_HSV_L[:,0]
                    blank_hand_L[:,0,1] = Frame_hand_HSV_L[:,1]
                    blank_hand_L[:,0,2] = Frame_hand_HSV_L[:,2]
                    #histogram of color-hand L
                    histHandL = cv2.calcHist([blank_hand_L], [0, 1, 2], None, [30, 32, 32], [0, 179, 0, 255, 0, 255])
                    
                    #histogram of L nonhand colour
                    y_BG, x_BG = np.shape(Frame_nonhand_HSV_L)
                    blank_nonhand_L = np.zeros((y_BG, 1, x_BG), np.uint8)
                    blank_nonhand_L[:,0,0] = Frame_nonhand_HSV_L[:,0]
                    blank_nonhand_L[:,0,1] = Frame_nonhand_HSV_L[:,1]
                    blank_nonhand_L[:,0,2] = Frame_nonhand_HSV_L[:,2]
                    #histogram of color-nonhand L
                    histNonhandL = cv2.calcHist([blank_nonhand_L], [0, 1, 2], None, [30, 32, 32], [0, 179, 0, 255, 0, 255])
        
                    # change colour space
                    Frame_OPT_crop_L= cv2.cvtColor(Frame_RGB_crop_L, cv2.COLOR_BGR2GRAY)
                       
                    
        BG_loc=np.where(BlackBG==1)
        ####### Colour ######
        Frame_BG_HSV=Frame_HSV[BG_loc]
        y_BG, x_BG = np.shape(Frame_BG_HSV)
        blank_BG = np.zeros((y_BG, 1, x_BG), np.uint8)
        blank_BG[:,0,0] = Frame_BG_HSV[:,0]
        blank_BG[:,0,1] = Frame_BG_HSV[:,1]
        blank_BG[:,0,2] = Frame_BG_HSV[:,2]
        #histogram of color-BG
        histBG = cv2.calcHist([blank_BG], [0, 1, 2], None, [30, 32, 32], [0, 179, 0, 255, 0, 255])
        ################ save data ##################  
        Frame_OPT_full=BBXfilename,Frame_OPT
        if (Frame_bbx_R==[]) and (Frame_bbx_L!=[]):# predict > 2 L bbx (no R bbx)
            #only save L hand            
            Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX=[[BBXfilename,[Frame_OPT_crop_L,hand_loc_L_crop,nonhand_loc_L_crop],[fullHOGfea_L,meanHOGfea_L,stdHOGfea_L],[histHandL,histNonhandL,histBG],area_hand_L,[hand_loc_L,nonhand_loc_L,BG_loc],Frame_bbx_L,'L']];
        elif (Frame_bbx_R!=[]) and (Frame_bbx_L==[]):# predict > 2 R bbx (no L bbx)
            # only save R hand
            Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX=[[BBXfilename,[Frame_OPT_crop_R,hand_loc_R_crop,nonhand_loc_R_crop],[fullHOGfea_R,meanHOGfea_R,stdHOGfea_R],[histHandR,histNonhandR,histBG],area_hand_R,[hand_loc_R,nonhand_loc_R,BG_loc],Frame_bbx_R,'R']]
        elif (Frame_bbx_R==[]) and (Frame_bbx_L==[]):#has >= 2 bbx predicted            
            Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX=[]                
        else: 
            #filename,location: hand, nonhand, HOG: full, mean, std, HSV(histogram): hand, nonhand, BG, Colour:hand,nonhand,BG, HandSize, Global loc:hand, nonhand,BG
            A=BBXfilename,[Frame_OPT_crop_R,hand_loc_R_crop,nonhand_loc_R_crop],[fullHOGfea_R,meanHOGfea_R,stdHOGfea_R],[histHandR,histNonhandR,histBG],area_hand_R,[hand_loc_R,nonhand_loc_R,BG_loc],Frame_bbx_R,'R'
            B=BBXfilename,[Frame_OPT_crop_L,hand_loc_L_crop,nonhand_loc_L_crop],[fullHOGfea_L,meanHOGfea_L,stdHOGfea_L],[histHandL,histNonhandL,histBG],area_hand_L,[hand_loc_L,nonhand_loc_L,BG_loc],Frame_bbx_L,'L'
            Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX=A,B
    else:#no bbx file
        Frame_OPT_full=BBXfilename,Frame_OPT;
        Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX=[];    
     
    return Frame_OPT_full,Frame_OPT_crop_HOG_full_mean_std_HSV_hand_nonhand_bg_HandSize_GlobLoc_nonhand_BG_BBX
#############################################################
    
folder_Path='//idapt_shares/NET/Stroke Hand Use/DO NOT BACK UP/Hand Labeling_Home/pipeline/';
folder_list=os.listdir(folder_Path);
folder_list.sort(key=natural_keys);
N=10;

for k in range(0,len(folder_list)):
    print(folder_list[k])
    RGBfolder=folder_Path+folder_list[k]+'/RGB405p2/';
    Maskfolder=folder_Path+folder_list[k]+'/Mask/';
    BBXfolder=folder_Path+folder_list[k]+'/Shan_bbx/';
    
    if os.path.exists(BBXfolder):
        saveFeatPath=folder_Path+folder_list[k]+'/Feature_N10_ShanBBX/';
        filename_fea=[];
        img_width=720;
        img_height=405;
        if not os.path.exists(saveFeatPath):
            try:
                os.makedirs(saveFeatPath)        
                #read in RGB folder
                framename_list=[]
                for root, dirs, files in os.walk(RGBfolder): 
                    for i in range(0,len(files)):
                        if files[i][-4:] == '.jpg':  
                            framename_list.append(files[i])
                            # sort the list
                    framename_list.sort(key=natural_keys)
                    
                for i in range(1,len(framename_list)):
                    OPTfull_prev=[];OPTfull_now=[];
                    OPTcrop_HOG_HSV_HandSize_now=[];OPTcrop_HOG_HSV_HandSize_now=[];
                    preFeature=[];
                    
                    OPTfull_prev,OPTcrop_HOG_HSV_HandSize_prev=CalFea(framename_list[i-1],RGBfolder,Maskfolder,BBXfolder) 
                    OPTfull_now,OPTcrop_HOG_HSV_HandSize_now=CalFea(framename_list[i],RGBfolder,Maskfolder,BBXfolder)
                    
                    # Full optical flow: previoud frame, current frame
                    OPT_Mag_full, OPT_Ang_full=optFlow(OPTfull_prev[1],OPTfull_now[1])
                    if (len(OPTcrop_HOG_HSV_HandSize_prev)>0) & (len(OPTcrop_HOG_HSV_HandSize_now)>0): #both frames had hands
                        for j in range(0,len(OPTcrop_HOG_HSV_HandSize_now)):
                            for l in range(0,len(OPTcrop_HOG_HSV_HandSize_prev)):
                                if OPTcrop_HOG_HSV_HandSize_now[j][-1]==OPTcrop_HOG_HSV_HandSize_prev[l][-1]:#match hands
                                    if OPTcrop_HOG_HSV_HandSize_now[j][-1]=='R':hand=0
                                    elif OPTcrop_HOG_HSV_HandSize_now[j][-1]=='L':hand=1
                                    # back ground global location/coordiniates
                                    BBX=OPTcrop_HOG_HSV_HandSize_now[j][6]
                                    BG_loc=OPTcrop_HOG_HSV_HandSize_now[j][5][2];        
                                    GLOB_hand_loc=OPTcrop_HOG_HSV_HandSize_now[j][5][0];
                                    GLOB_nohand_loc=OPTcrop_HOG_HSV_HandSize_now[j][5][1];
                                    Crop_hand_loc=OPTcrop_HOG_HSV_HandSize_now[j][1][1];            
                                    Crop_nonhand_loc=OPTcrop_HOG_HSV_HandSize_now[j][1][2];
                                    ################## HOG ###################
                                    hogFull=OPTcrop_HOG_HSV_HandSize_now[j][2][0];    
                                    hogMean=OPTcrop_HOG_HSV_HandSize_now[j][2][1];
                                    hogStd=OPTcrop_HOG_HSV_HandSize_now[j][2][2];
                                    
                                    ################## Colour (HSV) ###################
                                    HSV_hand=OPTcrop_HOG_HSV_HandSize_now[j][3][0];
                                    HSV_nonhand=OPTcrop_HOG_HSV_HandSize_now[j][3][1];
                                    HSV_BG=OPTcrop_HOG_HSV_HandSize_now[j][3][2];
                                    # hand - BG
                                    HSV_hand_BG=cv2.compareHist(HSV_hand,HSV_BG,cv2.HISTCMP_BHATTACHARYYA)
                                    # nonhand - BG
                                    HSV_nonhand_BG=cv2.compareHist(HSV_nonhand,HSV_BG,cv2.HISTCMP_BHATTACHARYYA)
                                    # hand-nonhand
                                    HSV_hand_nonhand=cv2.compareHist(HSV_hand,HSV_nonhand,cv2.HISTCMP_BHATTACHARYYA)
                                    
                                    ################# HandSize #################
                                    HandSize_pre=OPTcrop_HOG_HSV_HandSize_prev[l][4]; 
                                    HandSize_now=OPTcrop_HOG_HSV_HandSize_now[j][4]; 
                                    HandSize_change=HandSize_now-HandSize_pre;             
                                    
                                    ############## Optical Flow ################
                                    OPTcrop_prev=OPTfull_prev[1][int(BBX[1]):int(BBX[3]),int(BBX[0]):int(BBX[2])] # current BBX in the prvious frame
                                    OPTcrop_now=OPTcrop_HOG_HSV_HandSize_now[j][1][0]
                                    #### Global coord #####
                                    # BG optical flow histogram:
                                    hisMag_BG, binMag_BG=np.histogram(OPT_Mag_full[BG_loc],15,[0,15],density=True)# used density=True rather than normed=True? Did the same and it's still accurate in non-uniform bins
                                    hisAng_BG, binAng_BG=np.histogram(OPT_Ang_full[BG_loc],15,[0,15],density=True)
                                    # non-hand optical flow histogam 
                                    hisMag_GLOB_hand, binMag_GLOB_hand=np.histogram(OPT_Mag_full[GLOB_hand_loc],15,[0,15],density=True)
                                    hisAng_GLOB_hand, binAng_GLOB_hand=np.histogram(OPT_Ang_full[GLOB_hand_loc],15,[0,15],density=True)
                                    # non-hand optical flow histogam (as old feature)
                                    hisMag_GLOB_nohand, binMag_GLOB_nohand=np.histogram(OPT_Mag_full[GLOB_nohand_loc],15,[0,15],density=True)
                                    hisAng_GLOB_nohand, binAng_GLOB_nohand=np.histogram(OPT_Ang_full[GLOB_nohand_loc],15,[0,15],density=True)
                                    #### Cropped coord #####
                                    OPT_Mag_crop, OPT_Ang_crop=optFlow(OPTcrop_prev,OPTcrop_now)
                                    # hand optical flow
                                    hisMag_hand, binMag_hand=np.histogram(OPT_Mag_crop[Crop_hand_loc],15,[0,15],density=True)
                                    hisAng_hand, binAng_hand=np.histogram(OPT_Ang_crop[Crop_hand_loc],15,[0,15],density=True)
                                    hisMag_nonhand, binMag_nonhand=np.histogram(OPT_Mag_crop[Crop_nonhand_loc],15,[0,15],density=True)
                                    hisAng_nonhand, binAng_nonhand=np.histogram(OPT_Ang_crop[Crop_nonhand_loc],15,[0,15],density=True)                                    
                                    
                                    # BG - hand (global coord)
                                    hisMag_BG_GLOB_hand=np.abs(hisMag_BG-hisMag_GLOB_hand)
                                    hisAng_BG_GLOB_hand=np.abs(hisAng_BG-hisAng_GLOB_hand)
                                    ## BG-hand ##doesn't make sense, different coord
                                    hisMag_BG_hand=np.abs(hisMag_BG-hisMag_hand);
                                    hisAng_BG_hand=np.abs(hisAng_BG-hisAng_hand);     
            #                        # BG - nonhand (global coord)
                                    hisMag_BG_GLOB_nohand=np.abs(hisMag_BG-hisMag_GLOB_nohand)
                                    hisAng_BG_GLOB_nohand=np.abs(hisAng_BG-hisAng_GLOB_nohand)
                                    # BG- nonhand (cropped coord) ##doesn't make sense, different coord
                                    hisMag_BG_nonhand=np.abs(hisMag_BG-hisMag_nonhand)
                                    hisAng_BG_nonhand=np.abs(hisAng_BG-hisAng_nonhand)
                                    # hand -nonhand (cropped coord)
                                    hisMag_hand_nonhand=np.abs(hisMag_hand-hisMag_nonhand)
                                    hisAng_hand_nonhand=np.abs(hisAng_hand-hisAng_nonhand)
                                    # New_Feature_N10
                                    fea= np.concatenate((hisAng_BG_GLOB_hand,hisMag_BG_GLOB_hand,hisAng_BG_GLOB_nohand,hisMag_BG_GLOB_nohand,hisAng_hand_nonhand,hisMag_hand_nonhand,hisAng_BG_hand,hisMag_BG_hand,hisAng_BG_nonhand,hisMag_BG_nonhand,hogFull,hogMean,hogStd,[HSV_hand_BG,HSV_nonhand_BG,HSV_hand_nonhand,HandSize_now,HandSize_change,hand]), axis=0) # 0 for right hand
                                    preFeature.append(fea)
                        filename_fea.append([framename_list[i][:-4],preFeature])
            
                
                for h in range(0,(len(filename_fea)-N)):        
                    FinalFeature=[]
                    if len(filename_fea[h][1])>0:#has hand
                        for m in range(0,len(filename_fea[h][1])):
                            handSize_diff=[]
                            handSize=filename_fea[h][1][m][-3];
                            handSize_change=filename_fea[h][1][m][-2];
                            hand=filename_fea[h][1][m][-1];
                            for n in range(0,N):
                                if len(filename_fea[h+n][1])==0:#no hand
                                    handSize_diff.append(np.inf)
                                elif len(filename_fea[h+n][1])==1:
                                    if hand==filename_fea[h+n][1][0][-1]:#match hand
                                        handSize_diff.append(filename_fea[h+n][1][0][-2])
                                    else:handSize_diff.append(np.inf)
                                elif len(filename_fea[h+n][1])==2:
                                    for o in range(0,2):
                                        if hand==filename_fea[h+n][1][o][-1]:#match hand
                                            handSize_diff.append(filename_fea[h+n][1][o][-2])
                            #print(handSize_diff/handSize)
                            fea_wHandSizePrec=np.concatenate((filename_fea[h][1][m][0:-3],handSize_diff/handSize,[hand]),axis=0);
                            FinalFeature.append(fea_wHandSizePrec)
                    s1 = saveFeatPath + filename_fea[h][0] + '.mat' 
                    sio.savemat(s1, mdict={filename_fea[h][0]:FinalFeature})  
                print('====== Finished: ',folder_list[k],' ========' )
            except OSError:
                pass
                
            
            