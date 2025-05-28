# NYCU Computer Vision 2025 Spring HW4
**StudentID** : **111550177** \
**Name** : **Ting-Lin Wu (吳定霖)**

## Introduction
This assignment is an image restoration competition, focusing on restoring images degraded by two types of atmospheric noise: rain and snow. The dataset includes 1600 degraded images per type (rain and snow) and their corresponding clean images for training and validation, along with 100 degraded test images (50 per type, named 0.png to 99.png) with unspecified degradation types. The target is to produce clean images corresponding to each degraded test image, evaluated using Peak Signal-to-Noise Ratio (PSNR) [2] on a private leaderboard.
The task requires training a single model to handle both rain and snow degradations. The model is restricted to using PromptIR [1] as the base architecture, a vision-based model designed for blind image restoration. PromptIR consists of three main components: (1) a backbone to extract multi-scale features from degraded images, (2) a prompt module with tunable parameters to encode degradation-specific information, and (3) a restoration head to reconstruct clean images. Unlike traditional methods like AirNet [4], which rely on contrastive learning [5] and additional encoders, PromptIR leverages prompts to efficiently adapt to multiple degradation types, as required by the assignment. There are no model size limitations, but external data and pretrained weights are prohibited to ensure fair training from scratch.


## How to install
How to install dependences
```bash
# clone this repo
git clone https://github.com/morris0401/NYCU_VRDL.git
cd NYCU_VRDL/HW4/PromptIR_code/

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
[1] V. Potlapalli, S. W. Zamir, S. H. Khan, and F. Shahbaz Khan, ‘Promptir: Prompting for all-in-one image restoration’, Advances in Neural Information Processing Systems, vol. 36, pp. 71275–71293, 2023.
[2] F. A. Fardo, V. H. Conforto, F. C. de Oliveira, and P. S. Rodrigues, ‘A formal evaluation of PSNR as quality measurement parameter for image segmentation algorithms’, arXiv preprint arXiv:1605. 07116, 2016.
[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, ‘Image quality assessment: from error visibility to structural similarity’, IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[4] E. Chee and Z. Wu, ‘Airnet: Self-supervised affine registration for 3d medical images using neural networks’, arXiv preprint arXiv:1810. 02583, 2018.
[5] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, ‘A simple framework for contrastive learning of visual representations’, in International conference on machine learning, 2020, pp. 1597–1607.


## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 3 degradation types (noise, rain, and haze).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. The pretrained model can be downloaded [here](https://drive.google.com/file/d/1j-b5Od70pGF7oaCqKAfUzmf-N-xEAjYl/view?usp=sharingg), alternatively, it is also available under the releases tab. To perform the evalaution use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehaazing and 3 for all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 3
```

## Demo
To obtain visual results from the model ```demo.py``` can be used. After placing the saved model file in ```ckpt``` directory, run:
```
python demo.py --test_path {path_to_degraded_images} --output_path {save_images_here}
```
Example usage to run inference on a directory of images:
```
python demo.py --test_path './test/demo/' --output_path './output/demo/'
```
Example usage to run inference on an image directly:
```
python demo.py --test_path './test/demo/image.png' --output_path './output/demo/'
```
To use tiling option while running ```demo.py``` set ```--tile``` option to ```True```. The Tile size and Tile overlap parameters can be adjusted using ```--tile_size``` and ```--tile_overlap``` options respectively.




## Results
Performance results of the PromptIR framework trained under the all-in-one setting

<summary><strong>Table</strong> </summary>

<img src = "prompt-ir-results.png"> 

<summary><strong>Visual Results</strong></summary>

The visual results of the PromptIR model evaluated under the all-in-one setting can be downloaded [here](https://drive.google.com/drive/folders/1Sm-mCL-i4OKZN7lKuCUrlMP1msYx3F6t?usp=sharing)



## Citation
If you use our work, please consider citing:

    @inproceedings{potlapalli2023promptir,
      title={PromptIR: Prompting for All-in-One Image Restoration},
      author={Potlapalli, Vaishnav and Zamir, Syed Waqas and Khan, Salman and Khan, Fahad},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023}
    }


## Contact
Should you have any questions, please contact pvaishnav2718@gmail.com


**Acknowledgment:** This code is based on the [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet) and [Restormer](https://github.com/swz30/Restormer) repositories. 

