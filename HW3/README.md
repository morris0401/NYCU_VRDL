# NYCU Computer Vision 2025 Spring HW3
**StudentID** : **111550177** \
**Name** : **Ting-Lin Wu (吳定霖)**

## Introduction
This assignment focuses on the task of instance segmentation in colored medical images. The dataset contains colored medical images with instances of 4 different types of cells (class1, class2, class3, and class4). The dataset consists of 209 images for training and validation and 101 images for testing. The raw images and masks are provided in .tif format, requiring data processing for model training and evaluation.
The primary goal of this assignment is to train instance segmentation models to predict the segmentation masks for each instance of the target cells. Predicted masks need to be converted into a specific submission format.
For this task, we are required to develop a model based on Mask R-CNN [1]. The model should typically comprise key components: (1) The backbone responsible for extracting feature maps from the input images. (2) The Region Proposal Network (RPN) [3] to generate regions of interest that potentially contain objects. (3) The heads, which extend the object detection heads of Faster R-CNN [2] to predict the bounding box, class, and importantly, a segmentation mask for each proposed region. Pretrained weights, specifically from ImageNet [3], are permitted. A key constraint for this assignment is that the total number of trainable parameters in the model must be less than 200 million.


## How to install
How to install dependences
```bash
# clone this repo
git clone https://github.com/morris0401/NYCU_VRDL.git
cd NYCU_VRDL/HW3

# create environment
conda create -n VRDL_hw3 python=3.11
conda activate VRDL_hw3
pip install -r requirements.txt
```

## How to install dataset
```bash
mkdir models
mkdir dataset
cd dataset
gdown --id 1B0qWNzQZQmfQP7x7o4FDdgb9GvPDoFzI
tar -xvzf hw3-data-release.tar.gz
```

## How to run
How to execute the code
```
# Training
python ./code/train.py

# Testing
python ./code/test.py
```

## Model Weight Download
[https://drive.google.com/file/d/1DB58w778_kE1TYBj6i2B4P2D7QBxWOZB/view?usp=sharing](https://drive.google.com/file/d/1DB58w778_kE1TYBj6i2B4P2D7QBxWOZB/view?usp=sharing)

## Performance snapshot
A shapshot of the leaderboard
![image](assets/leaderboard.png)

## Reference 
[1] K. He, G. Gkioxari, P. Dollár, and R. B. Girshick, ‘Mask R-CNN’, CoRR, vol. abs/1703.06870, 2017.

[2] S. Ren, K. He, R. B. Girshick, and J. Sun, ‘Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks’, CoRR, vol. abs/1506.01497, 2015.

[3] O. Russakovsky et al., ‘ImageNet Large Scale Visual Recognition Challenge’, CoRR, vol. abs/1409.0575, 2014.
