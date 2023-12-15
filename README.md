This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  
In the publication, three machine learning models, including a random forest classifier, SlowFast network, and Hand Object Detector, were trained to identify hand-object interaction in daily living for stroke survivors.  

# Hand-Object Interaction and Hand Role Classification Using Random Forest Classifier  
## 1. Create three folders under the pipeline folder
`pipeline/{task name}/RGB405p2` folder: store raw images.  
`pipeline/{task name}/Mask` folder: stores hand segmentation images.  
`pipeline/{task name}/Shan_bbx` folder: stores the text files of detected bounding boxes of hands generated from [the GitHub](https://github.com/mft2023/hand-object-detector).  

## 2. Segment hand regions
Download UNET weights, _model-UNET-epoch100-stroke.h5_, for segmentation [here](https://drive.google.com/drive/folders/149ZD2eIGfj0Z4Crf4vAhN4Vu5URR70i4?usp=drive_link) and change the _ModelPath_ in the [UNET-Hand Segmentation.py](https://github.com/mft2023/Random-Forest-Classifier/blob/main/UNET-Hand%20Segmentation.py).
```
python UNET_Hand_Segmentation.py
```
## 3. Extract features  
Check the pipeline folder path is correct in [FeatureExtraction.py](https://github.com/mft2023/main/blob/Rondom-Forest-Classifier/FeatureExtraction.py), and run it to generate features in `Features` for both interactions and hand roles.
```
python FeatureExtraction.py
```
## 4. Run Leave-One-Subject-Out-Cross-Validation
Download the PCA model, _PCA_Model_Stroke_sub1-9_Interaction.joblib_, [here](https://drive.google.com/drive/folders/149ZD2eIGfj0Z4Crf4vAhN4Vu5URR70i4?usp=drive_link).  
For hand-object interaction detection, check the folder paths in [line 298-316](https://github.com/mft2023/Random-Forest-Classifier/blob/ec97f33b6e85b64076c96f30b744f1ad7df7df60/LOSOCV_Interaction.py#L298C1-L298C1) are correct and run LOSOCV_Interaction.py.
```
python LOSOCV_Interaction.py
```
For hand role classification, check each folder path in [line 312-337](https://github.com/mft2023/Random-Forest-Classifier/blob/ec97f33b6e85b64076c96f30b744f1ad7df7df60/LOSOCV_HandRole_weighted.py#L312C1-L312C1) is correct and run LOSOCV_HandRole_weighted.py.
```
python LOSOCV_HandRole_weighted.py
```

# Cite
If you find this repository useful in your research, please consider citing:
```
@article{
    Author = {Meng-Fen Tsai,Rosalie H. Wang, and Zariffa, José},
    Title = {Recognizing hand use and hand role at home after stroke from egocentric video},
    Journal = {PLOS Digital Health 2.10: e0000361},
    Year = {2023}
}
```
