Detecting Objects in a Dense Clutter 
=========================================
# 1. A report (as a "readme")
Report has been included as a pdf file in the repository here: 
# 2. Current version of programs with instructions on how to run them: 
## Mask-RCNN Usage: 
Download the project
  
     git clone https://github.com/Ying2019F/CVII_SemesterProject.git

Navigate to the cluster directory 

    cd Mask_RCNN/samples/cluster/
    
For Predicting on a new image: 

Sample running on test image: 

<p align="center"> <img src="splash_20200508T205239.png" width="400" height="350"/> </p>

Sample running on validation image: 

<p align="center"> <img src="splash_20200508T211952.png" width="400" height="300"/> </p>

## YOLO Usage: 

## SSD Usage: 
#install Caffe

details can be found on https://github.com/weiliu89/caffe

#Train SSD

Follow the Pascal VOC dataset format to prepare the input dataset.

#Generate LMDB file

1. Run create_list.sh to generate test_name_size.txt, test.txt, and trainval.txt in data/amazon/
2. Modify labelmap_amazon.prototxt 
3. Run create_data.sh to create LMDB database and make a soft link in examples/amazon/

#Training and evaluation

1. Run ssd_pascal_orig.py to train the model
2. Run score_ssd_pascal.py to evaluate the model

#Visualization

To run ssd_amazon_detect.ipynb to do the detection on a single tote image





# 3. Consent to forward to Amazon Robotics
# 4. For Groups: 
  * Mask-RCNN: Sophia Abraham 
  * YOLO: Bhakti Sharma 
  * SSD: Ying Qiu 
