This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).  
In the publication, three machine learning models, including a random forest classifier, SlowFast network, and Hand Object Detector, were trained to identify hand-object interaction in daily living for stroke survivors.  

# Hand-Object Interaction and Hand Role Classification Using Random Forest Classifier  
## 1. Create three folders under the `pipeline` folder
`RGB405p2` folder: store raw images.  
`Mask` folder: stores hand segmentation images.  
`Shan_bbx` folder: stores the text files of detected bounding boxes of hands from [Hand Object Detector](https://github.com/ddshan/hand_object_detector).  

## 2. Segment hand regions
Download UNET weights for segmentation [here](https://drive.google.com/drive/folders/149ZD2eIGfj0Z4Crf4vAhN4Vu5URR70i4?usp=sharing) and change the _ModelPath_ in the [UNET_Hand Segmentation.py](UNET-Hand Segmentation.py).
## 2. Extract features  
After the files and folders are ready, run [FeatureExtraction.py](https://github.com/mft2023/main/blob/Rondom-Forest-Classifier/FeatureExtraction.py) to generate features in `Features`.

