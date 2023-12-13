# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:01:08 2021

@author: tsaiM
"""

#clear all
clear = lambda: os.system('clear')

import pickle as pk
import numpy as np
from sklearn.model_selection import LeaveOneOut
import scipy.io
import os
import re
import pandas as pd
import math
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
import scipy.io as sio
def atoi(text):
    return int(text) if text.isdigit() else text
    

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def Orgniza_Data(GT_labeling_Folder,Pipeline_Folder,folder_name,target_list,PCA_location,len_fea,hand_threshold):
    sub_vset_framestart_frameend=[];
    for i in range(0,len(target_list)):
        sub_vset_framestart_frameend.append([[target_list[i][2],target_list[i][5]],target_list[i][3],target_list[i][4]]);
    # load GT labeling (.xlsx)
    subj_list=[]
    for root, dirs, files in os.walk(GT_labeling_Folder): 
        GT_filename=files
        # sort the list
        GT_filename.sort(key=natural_keys)
        for filename in files:
            subj_list.append(filename[-9])
    
    GT_label_all_R=[]
    GT_label_all_L=[]
    for f in GT_filename: # go through annotation files one by one
        GT_label = pd.read_excel(GT_labeling_Folder+f)
        GT_label_all_R.append([f,GT_label.iloc[: , label_col[0]]])
        GT_label_all_L.append([f,GT_label.iloc[: , label_col[1]]])
        
    GT_R_vector=[]
    GT_L_vector=[]
    for i in GT_label_all_R: #tasks
        if i[0][15:22]=='HomeLab':
            GT_subj='Dd'+i[0][26:28]
            GT_Vset=i[0][29:31]
        else:#Home
            GT_subj='Ed'+i[0][23:25]
            GT_Vset=i[0][26:28]
        #GT number    
        for index in i[1].iteritems(): # index:(index,value)
            for j in range(0,len(sub_vset_framestart_frameend)):
                #match subj, vset, frame
                if ([int(GT_subj[2:4]),int(GT_Vset)]==sub_vset_framestart_frameend[j][0]) and (index[0]+2>=sub_vset_framestart_frameend[j][1]) and (index[0]+2<=sub_vset_framestart_frameend[j][2]):
                    if math.isnan(index[1]) == False:            
                        if (int(index[1])==0 or int(index[1])==1) == True:
                            GT_R_vector.append([GT_subj,GT_Vset,index[0]+2,int(index[1])])#index[0]+2 is the actual frame
                        elif int(index[1])==2:
                            GT_R_vector.append([GT_subj,GT_Vset,index[0]+2,0]) 
                
    for i in GT_label_all_L: #tasks
        if i[0][15:22]=='HomeLab':
            GT_subj='Dd'+i[0][26:28]
            GT_Vset=i[0][29:31]
        else:#Home
            GT_subj='Ed'+i[0][23:25]
            GT_Vset=i[0][26:28]
        #GT number    
        for index in i[1].iteritems(): # index:(index,value)
            #match subj, vset, frame
            for j in range(0,len(sub_vset_framestart_frameend)):
                if ([int(GT_subj[2:4]),int(GT_Vset)]==sub_vset_framestart_frameend[j][0]) and (index[0]+2>=sub_vset_framestart_frameend[j][1]) and (index[0]+2<=sub_vset_framestart_frameend[j][2]):
                    if math.isnan(index[1]) == False:                
                        if (int(index[1])==0 or int(index[1])==1) == True:
                            GT_L_vector.append([GT_subj,GT_Vset,index[0]+2,int(index[1])])#index[0]+2 is the actual frame
                        elif int(index[1])==2:
                            GT_L_vector.append([GT_subj,GT_Vset,index[0]+2,0]) 
    
    if len(PCA_location)>0:
        pca_model=pk.load(open(PCA_location,'rb'))
        case_PCA='with PCA';
    else:case_PCA='no PCA';
    
    FeatureVector=[];
    key_filename=[];
    bbx_frameList=[];
    subfolders=next(os.walk(Pipeline_Folder))[1];
    subfolders.sort(key=natural_keys);
    
    for k in range(0, len(subfolders)):
        for root, dirs, files in os.walk(Pipeline_Folder+'/'+subfolders[k]+'/'+folder_name+'/'):  
              FeatureName_list=files
              FeatureName_list.sort(key=natural_keys)             
        FeatureVectorStack=[];
        
        for FeatureFileName in FeatureName_list:
            mat=scipy.io.loadmat(Pipeline_Folder+'/'+subfolders[k]+'/'+folder_name+'/'+FeatureFileName)
            FeatureVectorStack.append(mat)             
        
        for i in range(0,len(FeatureVectorStack)):
            for key in FeatureVectorStack[i].keys():
                if (key.find('Dd')!=-1) or (key.find('Ed')!=-1):# -1 means didn't find
                    key_filename.append(key);
                    # sub,video set, frame number
                    subj=key[:4];
                    Vset=key[6:8];
                    frame=int(key[10:16]);
                    B=FeatureVectorStack[i].get(key);# feature vector in each frame
                    
                    with open(Pipeline_Folder+'/'+subfolders[k]+'/Shan_bbx/'+key+'.txt', 'r') as b:
                        listBox = [line.strip() for line in b]
                        heading_len=0;loc=[]
                        for j in range(0,len(listBox)):
                            if (listBox[j][0]!='R') & (listBox[j][0]!='L'):# others' hand
                                heading_len=heading_len+1;
                            else:
                                loc.append(j)
                        if np.size(B)>0: # is not []
                            if (len(loc)>1) or (len(loc)==len(B)):#has 2 hands
                                if case_PCA=='no PCA':
                                    fea=np.concatenate([B[:,range(30,60)],B[:,range(90,120)]],axis=1);               
                                    FeatureVector.append([subj,Vset,frame,fea])
                                    bbx_frameList.append([subj,Vset,frame])
                                elif case_PCA=='with PCA':
                                    pca_HOG=pca_model.transform(B[:,range(150,276)])      
                                    # New_Feature_N10
                                    fea=np.concatenate([B[:,range(60,90)],B[:,range(120,150)],pca_HOG,B[:,range(277,279)]],axis=1);
                                    FeatureVector.append([subj,Vset,frame,fea])
                                    bbx_frameList.append([subj,Vset,frame])
                            elif len(loc)==0: continue # no hand
                            elif (len(loc)==1) and (len(B)>1):
                                if case_PCA=='no PCA':
                                    if listBox[int(loc[:])][0]=='R':
                                        fea=np.concatenate([B[0,range(30,60)],B[0,range(90,120)]],axis=1);
                                    else:fea=np.concatenate([B[1,range(30,60)],B[1,range(90,120)]],axis=1);    
                                    FeatureVector.append([subj,Vset,frame,fea])
                                    bbx_frameList.append([subj,Vset,frame])
                                elif case_PCA=='with PCA':
                                    if listBox[int(loc[:])][0]=='R':
                                        pca_HOG=pca_model.transform(B[0,range(150,276)].reshape(1,-1))      
                                        # New_Feature_N10
                                        fea=np.concatenate([B[0,range(60,90)].reshape(1,30),B[0,range(120,150)].reshape(1,30),pca_HOG,B[0,range(277,279)].reshape(1,2)],axis=1);
                                    else:
                                        pca_HOG=pca_model.transform(B[1,range(150,276)].reshape(1,-1)) #L     
                                        # New_Feature_N10
                                        fea=np.concatenate([B[1,range(60,90)].reshape(1,30),B[1,range(120,150)].reshape(1,30),pca_HOG,B[1,range(277,279)].reshape(1,2)],axis=1);
                                    FeatureVector.append([subj,Vset,frame,fea])
                                    bbx_frameList.append([subj,Vset,frame])
    
    # organizing by frame
    Match_GT_Fea_R=[]
    Predictions_noBBX_R=[]
    for i in GT_R_vector:         
        for j in FeatureVector:            
            if (j[0:3] == i[0:3]) == True:
                Match_GT_Fea_R.append([[i[0][2:4],i[1],i[2],i[3]],j[3][0]]) # subj,vset,frame,GT, feature_R
        if i[0:3] not in bbx_frameList:#no bbx 
            Predictions_noBBX_R.append([[i[0][2:4],i[1],i[2],i[3]],0])
            
    Match_GT_Fea_L=[]
    Predictions_noBBX_L=[]
    for i in GT_L_vector:
        for j in FeatureVector:
            if (j[0:3] == i[0:3]) == True:
                if len(j[3]) > 1: # more than 1 hand
                    Match_GT_Fea_L.append([[i[0][2:4],i[1],i[2],i[3]],j[3][1]]) # subj,vset,frame,GT,  feature_L
                else:# no left hand   
                    Predictions_noBBX_L.append([[i[0][2:4],i[1],i[2],i[3]],0])
                
        if i[0:3] not in bbx_frameList:#no bbx or no R hand in the frame
            Predictions_noBBX_L.append([[i[0][2:4],i[1],i[2],i[3]],0])# subj,vset,frame,GT, prediction 
    
    # matching by subj
    subj=[]
    Match_subj_GT_Fea_L=[]
    Fea_subj_L=[];
    GT_subj_L=[];
    for i in Match_GT_Fea_L:
        if (i[0][0] in subj)==False:#find new subj
            subj.append(i[0][0])
            Match_subj_GT_Fea_L.append([i[0][0],[i[0][-1],i[1]]]) # subj, [GT, feature]
            Fea_subj_L.append([i[0][0],i[1]])# [subj, feature]       
            GT_subj_L.append([i[0][0],i[0][-1]])# [subj, GT]
        else:#same subj
            for j in Match_subj_GT_Fea_L:
                if j[0] == i[0][0]:# subj id
                    C=Match_subj_GT_Fea_L.index(j)
                    if len(i)>1: # hand exists
                        Match_subj_GT_Fea_L[C].append([i[0][-1],i[1]])
                        Fea_subj_L[C].append(i[1])
                        GT_subj_L[C].append(i[0][-1])  
    
    
    subj=[]
    Match_subj_GT_Fea_R=[]
    Fea_subj_R=[];
    GT_subj_R=[];
    for i in Match_GT_Fea_R:
        if (i[0][0]in subj)==False:#find new subj
            subj.append(i[0][0])
            Match_subj_GT_Fea_R.append([i[0][0],[i[0][-1],i[1]]]) # subj, [GT, feature]
            Fea_subj_R.append([i[0][0],i[1]])# [subj, feature]       
            GT_subj_R.append([i[0][0],i[0][-1]])# [subj, GT]
        else:#same subj
            for j in Match_subj_GT_Fea_R:
                if j[0] == i[0][0]:# subj id
                    C=Match_subj_GT_Fea_R.index(j)
                    Match_subj_GT_Fea_R[C].append([i[0][-1],i[1]])
                    Fea_subj_R[C].append(i[1])
                    GT_subj_R[C].append(i[0][-1])    
                    
    return Fea_subj_R,Fea_subj_L,GT_subj_R,GT_subj_L,subj,Predictions_noBBX_R,Predictions_noBBX_L      
         
def Pool_hands(Fea_subj_R,Fea_subj_L,GT_subj_R,GT_subj_L,Participant_info):
    Fea_subj=copy.deepcopy(Fea_subj_R);
    num_R_frames_BySubj=[];
    num_L_frames_BySubj=[];
    for i in Fea_subj:
        num_R_frames_BySubj.append([i[0],len(i)-1]) 
        for j in Fea_subj_L:        
            if i[0] == j[0]:            
                C=Fea_subj.index(i)
                Fea_subj[C].extend(j[1:])
                num_L_frames_BySubj.append([i[0],len(j)-1])
                
    
    subj_num=Participant_info.iloc[:,0];
    Aff_Hand_info=Participant_info.iloc[:,3] #sub,affted hand (R=1;L=0)
    Aff_Hand_list=Aff_Hand_info.values.tolist() # make it a list
    aff_frames_num_BySubj=[];unaff_frames_num_BySubj=[];
    for i in num_R_frames_BySubj:    
        if Aff_Hand_list[int(np.where(int(i[0])==subj_num)[0])]==1: # R hand affected; i[0] is subj number
            aff_frames_num_BySubj.append([i[0],int(0),i[1]])# R [subj ID, start frame , end frame in the vector (not frame number)]
            for j in num_L_frames_BySubj:
                if i[0]==j[0]:
                    unaff_frames_num_BySubj.append([i[0],i[1],i[1]+j[1]]) #L [subj ID, start frame , end frame in the vector (not frame number)]
        else:
            unaff_frames_num_BySubj.append([i[0],int(0),i[1]]) #R
            for k in num_L_frames_BySubj:
                if i[0]==k[0]:
                    aff_frames_num_BySubj.append([i[0],i[1],i[1]+k[1]]) #L
    
    #
    GT_subj=copy.deepcopy(GT_subj_R);
    for i in GT_subj:
        for j in GT_subj_L:
            if i[0] == j[0]:
                C=GT_subj.index(i)
                GT_subj[C].extend(j[1:]) 
    
    X=[];#feature
    Y=[];#GT/labels
    for i in Fea_subj:
        num_frames=len(i[1:])
        X.append(np.reshape(np.concatenate(i[1:],axis=0),(num_frames,len(len_fea))))
        
    for i in GT_subj:
        Y.append(i[1:])
        
    # LOOCV    
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    print('Split', loo.get_n_splits(X),'(total number of subj) times')
    
    BySUBJ_Fea_TrainIndex_TestIndex=[]
    BySUBJ_GT_TrainIndex_TestIndex=[]
    for train_index, test_index in loo.split(X): 
        X_train=[]; Y_train=[]; X_test=[]; Y_test=[]; X_train_subj=[]; Y_train_subj=[]; # clear all the sets each time
        num_train_subj=len(train_index)
        for i in range(num_train_subj):   
            X_train.append(X[int(train_index[i])])
            Y_train.append(Y[int(train_index[i])]) 
            X_train_subj.append(Fea_subj[int(train_index[i])][0])
            Y_train_subj.append(GT_subj[int(train_index[i])][0])
    
        X_test = X[int(test_index)]
        Y_test = Y[int(test_index)]    
        BySUBJ_Fea_TrainIndex_TestIndex.append([X_train_subj,np.concatenate(X_train,axis=0), Fea_subj[int(test_index)][0], X_test]) # X,Y remain same sequence of subj ID with Fea_subj, GT_subj
        BySUBJ_GT_TrainIndex_TestIndex.append([Y_train_subj,list(np.concatenate(Y_train,axis=0)), GT_subj[int(test_index)][0], Y_test])

    return BySUBJ_Fea_TrainIndex_TestIndex, BySUBJ_GT_TrainIndex_TestIndex, aff_frames_num_BySubj, unaff_frames_num_BySubj
  
########## Home ############
GT_labeling_folder_Home='{path to annotation folder}/Documents for Home/Labeled xlsx/';
Pipeline_folder_Home='{path to the folder with all tasks}/pipeline/';
fea_foldername_Home='Features';
target_label_Home = pd.read_excel('{path to the list of all tasks}/Documents for Home/TargetLabeling_Home.xlsx');
target_info_Home=target_label_Home.iloc[2:,0:6] #task,b/o,sub,start,end, vset # all tasks
target_list_Home=target_info_Home.values.tolist();

Participant_info=pd.read_excel('{path to participant info}/participants_info.xlsx',sheet_name='demographic');
subj_num=Participant_info.iloc[:,0];
Aff_Hand_info=Participant_info.iloc[:,3]; #sub,affted hand (R=1;L=0)
Aff_Hand_list=Aff_Hand_info.values.tolist();
PCA_location='{path to a pretrained PCA model}/PCA_Model_Stroke_sub1-9_Interaction.joblib'; # if no PCA: location=[]
label_col=[0,1];saveFilename='Interaction_';
feaName='OPT_HOGpca_Colour';len_fea=range(0,122);
hand_threshold=0;# 0: take every bbx

rf_cl = RandomForestClassifier(n_estimators=150) # n_estimators=150 trees.
saveRoot='{path to store results}/Interaction/LOSOCV/Home/';
save_model_filename=saveRoot+'Model_Home_Stroke_Sub1-26_Interaction.joblib'

#### Home #####
Fea_subj_R_home,Fea_subj_L_home,GT_subj_R_home,GT_subj_L_home,subj,Predictions_noBBX_R_home,Predictions_noBBX_L_home =Orgniza_Data(GT_labeling_folder_Home,Pipeline_folder_Home,fea_foldername_Home,target_list_Home,PCA_location,len_fea,hand_threshold)
BySUBJ_Fea_TrainIndex_TestIndex_Home, BySUBJ_GT_TrainIndex_TestIndex_Home, aff_frames_num_BySubj_Home, unaff_frames_num_BySubj_Home = Pool_hands(Fea_subj_R_home,Fea_subj_L_home,GT_subj_R_home,GT_subj_L_home,Participant_info);

num_interaction=0;num_non_interaction=0;
for i in range(0,len(GT_subj_R_home)):
    num_interaction=num_interaction+len(np.where(np.array(GT_subj_R_home[i][1:])==1)[0]);
    num_non_interaction=num_non_interaction+len(np.where(np.array(GT_subj_R_home[i][1:])==0)[0]);
    num_interaction=num_interaction+len(np.where(np.array(GT_subj_L_home[i][1:])==1)[0]);
    num_non_interaction=num_non_interaction+len(np.where(np.array(GT_subj_L_home[i][1:])==0)[0]);

print("\nHome: Total Number of Interaction Instance: "+str(num_interaction+num_non_interaction))
print("Home: Number of Interaction Instance: "+str(num_interaction/(num_interaction+num_non_interaction)*100)+" %")
print("Home: Number of Non-Interaction Instance: "+str(num_non_interaction/(num_interaction+num_non_interaction)*100)+" %")

################ Leave 1 Subj Out Cross Validation ##############################

#BySUBJ still stored the original data, the data fed in random forest is replacing nan and inf with 0

###################
BySUBJ_Fea_TrainIndex_TestIndex_post=[]
BySUBJ_overall_GT_predict=[];BySUBJ_aff_GT_predict=[];BySUBJ_unaff_GT_predict=[];
BySUBJ_overall_precision_recall_F1=[];BySUBJ_aff_precision_recall_F1=[];BySUBJ_unaff_precision_recall_F1=[];
BySUBJ_overall_ACC= [];BySUBJ_overall_matthews=[];BySUBJ_aff_matthews=[];BySUBJ_unaff_matthews=[];
ACC=[];ACC_aff=[];ACC_Uaff=[];BySUBJ_aff_ACC=[];BySUBJ_unaff_ACC=[];
F1=[];F1_aff=[];F1_Uaff=[];Matthews=[];Matthews_aff=[];Matthews_Uaff=[];
ACC_bbx=[];F1_bbx=[];SUBJ_overall_GT_predict_bbx=[];SUBJ_overall_ACC_bbx=[];SUBJ_overall_precision_recall_F1_bbx=[];
ACC_Uaff_bbx=[];F1_Uaff_bbx=[];SUBJ_unaff_GT_predict_bbx=[];SUBJ_unaff_ACC_bbx=[];SUBJ_unaff_precision_recall_F1_bbx=[];
ACC_aff_bbx=[];F1_aff_bbx=[];SUBJ_aff_GT_predict_bbx=[];SUBJ_aff_ACC_bbx=[];SUBJ_aff_precision_recall_F1_bbx=[];
prec=[];prec_aff=[];prec_Uaff=[];
recall=[];recall_aff=[];recall_Uaff=[];All_overall_GT=[];All_aff_GT=[];All_unaff_GT=[];
All_overall_predict=[];All_aff_predict=[];All_unaff_predict=[];
num_frames_testing_Home=[];num_frames_training_Home=[]
percent_1_testing_Home=[];percent_1_training_Home=[];percent_0_testing_Home=[];percent_0_training_Home=[];

for i in range(0, len(subj)):
    X_train_post_Home = BySUBJ_Fea_TrainIndex_TestIndex_Home[i][1];
    X_train_post_Home[np.where(np.isnan(X_train_post_Home))]= 0;
    X_train_post_Home[np.where(np.isinf(X_train_post_Home))] = 0;
    
    Y_train_Home=BySUBJ_GT_TrainIndex_TestIndex_Home[i][1];
    X_train_post=X_train_post_Home;
    Y_train=Y_train_Home;
    
    X_test_post=BySUBJ_Fea_TrainIndex_TestIndex_Home[i][3];
    X_test_post[np.where(np.isnan(X_test_post))] = 0;
    X_test_post[np.where(np.isinf(X_test_post))] = 0;

    Y_test=BySUBJ_GT_TrainIndex_TestIndex_Home[i][3];
    # append features
    num_frames_training_Home.append(len(Y_train_Home));
    num_frames_testing_Home.append(len(Y_test));
    percent_1_training_Home.append(len(np.where(np.array(Y_train_Home)==1)[0])/len(Y_train_Home));
    percent_0_training_Home.append(len(np.where(np.array(Y_train_Home)==0)[0])/len(Y_train_Home));
    percent_1_testing_Home.append(len(np.where(np.array(Y_test)==1)[0])/len(Y_test));
    percent_0_testing_Home.append(len(np.where(np.array(Y_test)==0)[0])/len(Y_test));
    
    #shape = [n_samples, n_features]
    model = rf_cl.fit(X_train_post,Y_train);
    predictions= rf_cl.predict(X_test_post);
    
    test_subj=BySUBJ_Fea_TrainIndex_TestIndex_Home[i][2];
    
    ACC_bbx.append(accuracy_score(Y_test,predictions));
    F1_bbx.append(f1_score(Y_test,predictions));
    
    SUBJ_overall_GT_predict_bbx.append([test_subj,Y_test,predictions]);
    SUBJ_overall_ACC_bbx.append([test_subj,accuracy_score(Y_test,predictions)]);
    SUBJ_overall_precision_recall_F1_bbx.append([test_subj,precision_score(Y_test,predictions),recall_score(Y_test,predictions),f1_score(Y_test,predictions)]);
    
    ##### include the hand wasn't showed (no bbx) #########
    Y_test_withnobbx=copy.deepcopy(Y_test);
    predictions_withnobbx=copy.deepcopy(list(predictions));    
   
    for l in Predictions_noBBX_R_home: # include R hand 
        if l[0][0]==test_subj:# match subj
            Y_test_withnobbx.append(l[0][3]); # append the GT 
            predictions_withnobbx.append(l[1]); # append predictions 
            
    for m in Predictions_noBBX_L_home:# include L hand
        if m[0][0]==test_subj:# match subj
            Y_test_withnobbx.append(m[0][3]); # append the GT 
            predictions_withnobbx.append(m[1]); # append predictions
            
    ##### include the hand wasn't showde (no bbx): classify as non-interaction #########
    Matthews.append(matthews_corrcoef(Y_test_withnobbx,predictions_withnobbx));
    ACC.append(accuracy_score(Y_test_withnobbx,predictions_withnobbx));
    F1.append(f1_score(Y_test_withnobbx,predictions_withnobbx));
    prec.append(precision_score(Y_test_withnobbx,predictions_withnobbx));
    recall.append(recall_score(Y_test_withnobbx,predictions_withnobbx));
    
    All_overall_GT=All_overall_GT+Y_test_withnobbx;
    All_overall_predict=All_overall_predict+predictions_withnobbx;
    BySUBJ_overall_matthews.append([test_subj,matthews_corrcoef(Y_test_withnobbx,predictions_withnobbx)])
    BySUBJ_overall_GT_predict.append([test_subj,Y_test_withnobbx,predictions_withnobbx]);
    BySUBJ_overall_ACC.append([test_subj,accuracy_score(Y_test_withnobbx,predictions_withnobbx)]);
    BySUBJ_overall_precision_recall_F1.append([test_subj,precision_score(Y_test_withnobbx,predictions_withnobbx),recall_score(Y_test_withnobbx,predictions_withnobbx),f1_score(Y_test_withnobbx,predictions_withnobbx)]);
    
    # same training set, just calculate F1 & ACC with two hands seperately
    for j in unaff_frames_num_BySubj_Home:
        if test_subj==j[0]:
            unaff_frame_start_num=j[1];
            unaff_frame_end_num=j[2];
            ##### include the hand wasn't showde (no bbx): classify as non-interaction #########
            Y_test_unaff=copy.deepcopy(Y_test[unaff_frame_start_num:unaff_frame_end_num]);
            prediction_unaff=copy.deepcopy(list(predictions[unaff_frame_start_num:unaff_frame_end_num]));
            
            ACC_Uaff_bbx.append(accuracy_score(Y_test_unaff,prediction_unaff));
            F1_Uaff_bbx.append(f1_score(Y_test_unaff,prediction_unaff)) ;
            
            SUBJ_unaff_GT_predict_bbx.append([test_subj,Y_test[unaff_frame_start_num:unaff_frame_end_num],list(predictions[unaff_frame_start_num:unaff_frame_end_num])]);
            SUBJ_unaff_ACC_bbx.append([test_subj,accuracy_score(Y_test_unaff,prediction_unaff)]);
            SUBJ_unaff_precision_recall_F1_bbx.append([test_subj,precision_score(Y_test_unaff,prediction_unaff),recall_score(Y_test_unaff,prediction_unaff),f1_score(Y_test_unaff,prediction_unaff)]);
            
            if Aff_Hand_list[int(np.where(int(test_subj)==subj_num)[0])]==1: # match subj number, right hand affected
                for n1 in Predictions_noBBX_L_home: # subj 
                    if n1[0][0]==test_subj:
                        Y_test_unaff.append(n1[0][3]) 
                        prediction_unaff.append(n1[1]) 
            elif Aff_Hand_list[int(np.where(int(test_subj)==subj_num)[0])]==0: # match subj number, left hand affected
                for n2 in Predictions_noBBX_R_home: # subj 
                    if n2[0][0]==test_subj:
                        Y_test_unaff.append(n2[0][3])
                        prediction_unaff.append(n2[1]) 
                
            ##### include the hand wasn't showde (no bbx): classify as non-interaction #########
            
            Matthews_Uaff.append(matthews_corrcoef(Y_test_unaff,prediction_unaff));
            ACC_Uaff.append(accuracy_score(Y_test_unaff,prediction_unaff));
            F1_Uaff.append(f1_score(Y_test_unaff,prediction_unaff));
            prec_Uaff.append(precision_score(Y_test_unaff,prediction_unaff));
            recall_Uaff.append(recall_score(Y_test_unaff,prediction_unaff));
            
            All_unaff_GT=All_unaff_GT+Y_test_unaff;
            All_unaff_predict=All_unaff_predict+prediction_unaff;
            BySUBJ_unaff_matthews.append([test_subj,matthews_corrcoef(Y_test_unaff,prediction_unaff)]);
            BySUBJ_unaff_GT_predict.append([test_subj,Y_test_unaff,prediction_unaff]);
            BySUBJ_unaff_ACC.append([test_subj,accuracy_score(Y_test_unaff,prediction_unaff)]);
            BySUBJ_unaff_precision_recall_F1.append([test_subj,precision_score(Y_test_unaff,prediction_unaff),recall_score(Y_test_unaff,prediction_unaff),f1_score(Y_test_unaff,prediction_unaff)]);

    for k in aff_frames_num_BySubj_Home:
        if test_subj==k[0]:
            aff_frame_start_num=k[1];
            aff_frame_end_num=k[2];
            
            ##### include the hand wasn't showde (no bbx): classify as non-interaction #########
            Y_test_aff=copy.deepcopy(Y_test[aff_frame_start_num:aff_frame_end_num]);
            prediction_aff=copy.deepcopy(list(predictions[aff_frame_start_num:aff_frame_end_num]));
            
            ACC_aff_bbx.append(accuracy_score(Y_test_aff,prediction_aff));
            F1_aff_bbx.append(f1_score(Y_test_aff,prediction_aff));
            
            SUBJ_aff_GT_predict_bbx.append([test_subj,Y_test[aff_frame_start_num:aff_frame_end_num],list(predictions[aff_frame_start_num:aff_frame_end_num])]);
            SUBJ_aff_ACC_bbx.append([test_subj,accuracy_score(Y_test_aff,prediction_aff)]);
            SUBJ_aff_precision_recall_F1_bbx.append([test_subj,precision_score(Y_test_aff,prediction_aff),recall_score(Y_test_aff,prediction_aff),f1_score(Y_test_aff,prediction_aff)]);
      
            if Aff_Hand_list[int(np.where(int(test_subj)==subj_num)[0])]==1: # match subj number, right hand affected
                for o1 in Predictions_noBBX_R_home: # subj 
                    if o1[0][0]==test_subj:
                        Y_test_aff.append(o1[0][3]) 
                        prediction_aff.append(o1[1])
            elif Aff_Hand_list[int(np.where(int(test_subj)==subj_num)[0])]==0: # match subj number, left hand affected
                for o2 in Predictions_noBBX_L_home: # subj  
                    if o2[0][0]==test_subj:
                        Y_test_aff.append(o2[0][3]) 
                        prediction_aff.append(o2[1]) 
                
            ##### include the hand wasn't showde (no bbx): classify as non-interaction #########
            Matthews_aff.append(matthews_corrcoef(Y_test_aff,prediction_aff));         
            ACC_aff.append(accuracy_score(Y_test_aff,prediction_aff));
            F1_aff.append(f1_score(Y_test_aff,prediction_aff));
            prec_aff.append(precision_score(Y_test_aff,prediction_aff));
            recall_aff.append(recall_score(Y_test_aff,prediction_aff));
            
            All_aff_GT=All_aff_GT+Y_test_aff;
            All_aff_predict=All_aff_predict+prediction_aff;
            BySUBJ_aff_matthews.append([test_subj,matthews_corrcoef(Y_test_aff,prediction_aff)]);
            BySUBJ_aff_GT_predict.append([test_subj,Y_test_aff,prediction_aff]);
            BySUBJ_aff_ACC.append([test_subj,accuracy_score(Y_test_aff,prediction_aff)]);
            BySUBJ_aff_precision_recall_F1.append([test_subj,precision_score(Y_test_aff,prediction_aff),recall_score(Y_test_aff,prediction_aff),f1_score(Y_test_aff,prediction_aff)]);
          

print('\n=========== Dataset Info ===========')
print('Average number of training set: ', str(round(np.mean(num_frames_training_Home),2)),' +-',  str(round(np.std(num_frames_training_Home),2)), ' instances')
print(' Manipulation : ', str(round(np.mean(percent_1_training_Home)*100,2)),' % +-',  str(round(np.std(percent_1_training_Home)*100,2)), '%')
print(' non-Manipulation : ', str(round(np.mean(percent_0_training_Home)*100,2)),' % +-',  str(round(np.std(percent_0_training_Home)*100,2)), '%')

print('Average number of testing set: ', str(round(np.mean(num_frames_testing_Home),2)),' +-',  str(round(np.std(num_frames_testing_Home),2)), ' instances')
print(' Manipulation : ', str(round(np.mean(percent_1_testing_Home)*100,2)),' % +-',  str(round(np.std(percent_1_testing_Home)*100,2)), '%')
print(' non-Manipulation : ', str(round(np.mean(percent_0_testing_Home)*100,2)),' % +-',  str(round(np.std(percent_0_testing_Home)*100,2)), '%')

print('\n======== Macro Results ========')
print('\n======== Affected Hand ========')
print('Average MCC (By Subj): ', round(np.mean(Matthews_aff),2), ' +- ', round(np.std(Matthews_aff),2))
print('Average F1 Score (By Subj): ', round(np.mean(F1_aff),2), ' +- ', round(np.std(F1_aff),2))
print('Average Precision (By Subj): ', round(np.mean(prec_aff),2), ' +- ', round(np.std(prec_aff),2))
print('Average Recall (By Subj): ', round(np.mean(recall_aff),2), ' +- ', round(np.std(recall_aff),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(ACC_aff),2), ' +- ', round(np.std(ACC_aff),2))

print('\n======= Unaffected Hand =======')
print('Average MCC (By Subj): ', round(np.mean(Matthews_Uaff),2), ' +- ', round(np.std(Matthews_Uaff),2))
print('Average F1 Score (By Subj): ', round(np.mean(F1_Uaff),2), ' +- ', round(np.std(F1_Uaff),2))   
print('Average Precision (By Subj): ', round(np.mean(prec_Uaff),2), ' +- ', round(np.std(prec_Uaff),2))
print('Average Recall (By Subj): ', round(np.mean(recall_Uaff),2), ' +- ', round(np.std(recall_Uaff),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(ACC_Uaff),2), ' +- ', round(np.std(ACC_Uaff),2)) 

print('\n=========== Overall ===========')
print('Average MCC (By Subj): ', round(np.mean(Matthews),2), ' +- ', round(np.std(Matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(F1),2), ' +- ', round(np.std(F1),2))
print('Average Precision (By Subj): ', round(np.mean(prec),2), ' +- ', round(np.std(prec),2))
print('Average Recall (By Subj): ', round(np.mean(recall),2), ' +- ', round(np.std(recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(ACC),2), ' +- ', round(np.std(ACC),2),'\n')


print('\n======== Micro Average ========')
print('\n======== Affected Hand ========')
print('Average MCC: ', matthews_corrcoef(All_aff_GT,All_aff_predict))
print('Average F1 Score: ', f1_score(All_aff_GT,All_aff_predict))
print('Average Precision: ', precision_score(All_aff_GT,All_aff_predict))
print('Average Recall: ', recall_score(All_aff_GT,All_aff_predict))
print('Average Accuracy Score: ' , accuracy_score(All_aff_GT,All_aff_predict),'\n')

print('\n======= Unaffected Hand =======')
print('Average MCC: ', matthews_corrcoef(All_unaff_GT,All_unaff_predict))
print('Average F1 Score: ', f1_score(All_unaff_GT,All_unaff_predict))
print('Average Precision: ', precision_score(All_unaff_GT,All_unaff_predict))
print('Average Recall: ', recall_score(All_unaff_GT,All_unaff_predict))
print('Average Accuracy Score: ' , accuracy_score(All_unaff_GT,All_unaff_predict),'\n')

print('\n=========== Overall ===========')
print('Average MCC: ', matthews_corrcoef(All_overall_GT,All_overall_predict))
print('Average F1 Score: ', f1_score(All_overall_GT,All_overall_predict))
print('Average Precision: ', precision_score(All_overall_GT,All_overall_predict))
print('Average Recall: ', recall_score(All_overall_GT,All_overall_predict))
print('Average Accuracy Score: ' , accuracy_score(All_overall_GT,All_overall_predict),'\n')
###### save reuslt #########                     
s3 = saveRoot + 'LOSOCV_Home_'+saveFilename+feaName+'_overall_results.mat' 
s4 = saveRoot + 'LOSOCV_Home_'+saveFilename+feaName+'_aff_results.mat' 
s5 = saveRoot + 'LOSOCV_Home_'+saveFilename+feaName+'_unaff_results.mat' 
                    
sio.savemat(s3, mdict={'LOSOCV_overall_GT_prediction':BySUBJ_overall_GT_predict,'LOSOCV_overall_GT_prediction_bbx':SUBJ_overall_GT_predict_bbx,'LOSOCV_overall_ACC':BySUBJ_overall_ACC,'LOSOCV_overall_precision_recall_F1':BySUBJ_overall_precision_recall_F1,'LOSOCV_overall_matthews':BySUBJ_overall_matthews})
sio.savemat(s4, mdict={'LOSOCV_aff_GT_prediction':BySUBJ_aff_GT_predict,'LOSOCV_aff_GT_prediction_bbx':SUBJ_aff_GT_predict_bbx,'LOSOCV_aff_ACC':BySUBJ_aff_ACC,'LOSOCV_aff_precision_recall_F1':BySUBJ_aff_precision_recall_F1,'LOSOCV_aff_matthews':BySUBJ_aff_matthews})
sio.savemat(s5, mdict={'LOSOCV_unaff_GT_prediction':BySUBJ_unaff_GT_predict,'LOSOCV_unaff_GT_prediction_bbx':SUBJ_unaff_GT_predict_bbx,'LOSOCV_unaff_ACC':BySUBJ_unaff_ACC,'LOSOCV_unaff_precision_recall_F1':BySUBJ_unaff_precision_recall_F1,'LOSOCV_unaff_matthews':BySUBJ_unaff_matthews})
