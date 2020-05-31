Detecting Objects in a Dense Clutter 
=========================================
### NOTE: This is a course project for Computer Vision II at University of Notre Dame for 2020 Spring semester
  - Class code: CSE 40536/60536
  - the syllabus: http://adamczajka.pl/files/Computer-Vision-2-syllabus.pdf 

# 1. A report (as a "readme")
Report has been included as a pdf file in the repository here

Google drive for images and weights: https://drive.google.com/drive/folders/1O4ad1TXfSJZaml829rID7T84JjUvs4IM?usp=sharing
# 2. Current version of programs with instructions on how to run them: 
## Mask-RCNN Usage: 
Download the project
  
     git clone https://github.com/Ying2019F/CVII_SemesterProject.git

Navigate to the cluster directory 

    cd Mask_RCNN/samples/cluster/
    
For Predicting on a new image: 
Use 'Python3.7' and download requirements from requirements.txt

Weights and images included in google drive :) 

Easiest way would be to use the jupyter notebook in cluster! 

Example usage:

    python3.7 samples/cluster/cluster.py splash --weights='mask_rcnn_object_0099.h5' --image='LAYOUT2_PL2_WEBCAM_TOP_Photo_on_1-27-20_at_3.06_PM.jpg'

Sample running on test image: 

<p align="center"> <img src="figures/splash_20200508T205239.png" width="400" height="350"/> </p>

Sample running on validation image: 

<p align="center"> <img src="figures/96422398_276259940079522_8946705619629375488_n.png" width="400" height="300"/> </p>


## YOLO Usage: 
Navigate to the cluster directory 

    cd Yolov3

**Requirements**: 
Use Python 3.7, requirements.txt file includes all the packages required.

Weights and data are in google drive.
Data folder has data for all 3 cases - 
 - training on mobile images, testing on webcam images (Sensor1vsSensor2)
 - training on layout 1 images, testing on layout 2 images (Layout1vsLayout2)
 - training on top angle images, testing on side angle images (TopvsSide)

 unzip these folders and move the train.txt file, val.txt file, images folder to the data folder.

**Training**:

    python3 train.py --cfg cfg/yolov3.cfg --data data/yolo.data --weights weights/best_mobile_500.pt

training requires the training images in path /data/images/.. and train.txt and val.txt in path /data/..

**Testing**:

    python3 detect.py --weights weights/best_mobile_500.pt --source data/test_webcam_images

**Validation Testing**:

    python3 detect.py --weights weights/best_mobile_500.pt --source data/validation

Sample running on test image: 

<p align="center"> <img src="figures/test4.JPG" width="400" height="350"/> </p>

Sample running on validation image: 

<p align="center"> <img src="figures/Validation.jpg" width="400" height="300"/> </p>


## SSD Usage: 

### A webcam tote image was detected

<p align="center"> <img src="figures/Webcam_tote_detected.jpg" width="600" height="480"/> </p>

### An Iphone tote image was detected

<p align="center"> <img src="figures/Iphone_tote_detected.jpg" width="1000" height="800"/> </p>

### Requirements
For training
  - python/2.7.15 
  - cuda/10.0 
  - cudnn/7.4 
  - opencv/3.4 
  - intel/19.0 
  - tcl/8.6.8 
  - gcc/8.3.0 
  - hdf5/1.8.20-gcc 
  - mkl

For detection
  - caffe version: 1.0.0-rc3
  - Input datasets and trained models (weights) are available on this google drive link: https://drive.google.com/drive/folders/1O4ad1TXfSJZaml829rID7T84JjUvs4IM?usp=sharing

More details can be found here (https://github.com/Ying2019F/CVII_SemesterProject/tree/master/SSD/environment.yaml)

### For the sample datasets

1. Install Caffe (version: 1.0.0-rc3)

   follow the installation steps on this github (https://github.com/weiliu89/caffe/tree/ssd)
      - git clone https://github.com/weiliu89/caffe.git
      - cd caffe
      - git checkout ssd
      - cp Makefile.config.example Makefile.config (Modify Makefile.config according to your Caffe installation. My         Makefile.config is available in SSD folder)
      - make clean
      - make -j8
      - make py ( Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.)
      - make test -j8
      - make runtest -j8 (Optional)

2. Download input datasets and trained models (weights)from the google drive

3. Download labelmap_amazon.prototxt and deploy.prototxt from here ( https://github.com/Ying2019F/CVII_SemesterProject/tree/master/SSD)

4. Make corresponding modifications on the path in ssd_amazon_detect.ipynb

    - labelmap_file = 'data/amazon/labelmap_amazon.prototxt'

    - model_def = 'models/VGGNet/amazon/SSD_300x300_orig/deploy.prototxt'

    - model_weights = 'models/VGGNet/amazon/SSD_300x300_orig/VGG_amazon_SSD_300x300_orig_iter_8000.caffemodel'

    - image = caffe.io.load_image('examples/images/IMG_5165.jpg')


### For customized datasets
#### Install Caffe

Dependencies for installing Caffe are listed here https://caffe.berkeleyvision.org/installation.html#prerequisites

Details can be found on https://github.com/weiliu89/caffe

#### Dataset preparation

1. Follow the Pascal VOC dataset format to prepare the customized dataset (instructions can be found here https://medium.com/deepquestai/object-detection-training-preparing-your-custom-dataset-6248679f0d1d).

2. Generate LMDB file

    - Create trainval.txt (including the image index for training and validation) and test.txt (including the image index for testing)
    - Run create_list.sh to generate test_name_size.txt, test.txt, and trainval.txt in data/amazon/
    - Modify labelmap_amazon.prototxt 
    - Run create_data.sh to create LMDB database and make a soft link in examples/amazon/

### Training and evaluation

1. python ssd_pascal_orig.py to train the model
2. python score_ssd_pascal.py to evaluate the model

Note: For this project, different training schemes were implemented. There are two trained models in the google drive, including webcam model and Iphone model. The webcam model was trained on the images collected by webcam (Logitech, webcam c615) and Iphone model was trained on the images collected by Iphone. More training models and results are available in the project final report.

### Visualization
1. install caffe
2. download input dataset and pretrained models
3. To run ssd_amazon_detect.ipynb to do the detection on a single tote image
### Evaluation of SSD on detecting objects in a dense clutter

- Pros:
     - Achieved high accuracy when the detection scenario (object size, position, and layout) is similar to the training scenario
     - Low requirement on training datasize, decent results can be obtained with even several training images
     - Robust to large occlusions

- Cons: 
     - Complicated to install Caffe due to lots of dependencies
     - Low accuracy on new detection scenario or unseen scenario
     - Bad detections when the target object has similar color or pattern with the unrelated object
