This repository stores open-source codes for the publication: [Recognizing hand use and hand role at home after stroke from egocentric video](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000361).

In the publication, three machine learning models, including a random forest classifier, SlowFast network, and Hand Object Detector, were trained to identify hand-object interaction in daily living for stroke survivors.
The inputs for the three models were the images of cropped hand regions of the detected hand bounding boxes using [Hand Object Detector](https://github.com/ddshan/hand_object_detector). 

#1. Random Forest Classifier

Three folders were created to store raw images (_RGB405p2_), hand segmentation images (_Mask_), and text files of detected bounding boxes of hands (_Shan_bbx_).
The generated features will be saved in a folder named: _Feature_N10_ShanBBX_.
The weights for hand segmentation using UNET are available based on request.
