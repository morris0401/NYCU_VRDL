# NYCU Computer Vision 2025 Spring HW4
**StudentID** : **111550177** \
**Name** : **Ting-Lin Wu (吳定霖)**

## Introduction
This assignment is an image restoration competition, focusing on restoring images degraded by two types of atmospheric noise: rain and snow. The dataset includes 1600 degraded images per type (rain and snow) and their corresponding clean images for training and validation, along with 100 degraded test images (50 per type, named 0.png to 99.png) with unspecified degradation types. The target is to produce clean images corresponding to each degraded test image, evaluated using Peak Signal-to-Noise Ratio (PSNR) [2] on a private leaderboard.
The task requires training a single model to handle both rain and snow degradations. The model is restricted to using PromptIR [1] as the base architecture, a vision-based model designed for blind image restoration. PromptIR consists of three main components: (1) a backbone to extract multi-scale features from degraded images, (2) a prompt module with tunable parameters to encode degradation-specific information, and (3) a restoration head to reconstruct clean images. Unlike traditional methods like AirNet [4], which rely on contrastive learning [5] and additional encoders, PromptIR leverages prompts to efficiently adapt to multiple degradation types, as required by the assignment. There are no model size limitations, but external data and pretrained weights are prohibited to ensure fair training from scratch.

This github repository is modified from [https://github.com/va1shn9v/PromptIR/tree/main](https://github.com/va1shn9v/PromptIR/tree/main)

## How to install
How to install dependences
```bash
# clone this repo
git clone https://github.com/morris0401/NYCU_VRDL.git
cd NYCU_VRDL/HW4/PromptIR_code/

# create environment
conda create -n promptir python=3.9
conda activate promptir
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```

## How to install dataset
```bash
cd data/Train
gdown --id 1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w
unzip hw4_realse_dataset.zip
```

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise,Derain or Dehaze.
After placing the training data the directory structure would be as follows:
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
```
You should place the rainy dataset into Derain folder, and snowy dataset into Dehaze dataset. I use dehaze to implment desnow.

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. The test directory after setup:
```
├───dehaze
│   ├───input
│   └───target
├───denoise
│   ├───bsd68
│   └───urban100
└───derain
    └───Rain100L
        ├───input
        └───target
```
You should place the rainy dataset into derain folder, and snowy dataset into dehaze dataset. I use dehaze to implment desnow.

## How to run

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py --de_type derain dehaze
```
This command train derain and desnow.

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory.
```
python test.py --mode 3 --ckpt_name {model ckpt name}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehaazing and 3 for all-in-one setting. ```ckpt_name``` is use to specify the model weight .ckpt file under './ckpt/' folder.

## Inference
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

Example usage to run inference with test-time augmentation on a directory of images:
```
python demo_tta.py --test_path './test/demo/' --output_path './output/demo/'
```

## Model Weight Download
[https://drive.google.com/file/d/1-XqU5Z5Wgs2aIF5MskEflFBICuu2eUlz/view?usp=sharing](https://drive.google.com/file/d/1-XqU5Z5Wgs2aIF5MskEflFBICuu2eUlz/view?usp=sharing)
You can place the model weight in the './ckpt/' folder

## Performance snapshot
A shapshot of the leaderboard
![image](assets/leaderboard.png)

## Reference 
[1] V. Potlapalli, S. W. Zamir, S. H. Khan, and F. Shahbaz Khan, ‘Promptir: Prompting for all-in-one image restoration’, Advances in Neural Information Processing Systems, vol. 36, pp. 71275–71293, 2023.
[2] F. A. Fardo, V. H. Conforto, F. C. de Oliveira, and P. S. Rodrigues, ‘A formal evaluation of PSNR as quality measurement parameter for image segmentation algorithms’, arXiv preprint arXiv:1605. 07116, 2016.
[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, ‘Image quality assessment: from error visibility to structural similarity’, IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.
[4] E. Chee and Z. Wu, ‘Airnet: Self-supervised affine registration for 3d medical images using neural networks’, arXiv preprint arXiv:1810. 02583, 2018.
[5] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, ‘A simple framework for contrastive learning of visual representations’, in International conference on machine learning, 2020, pp. 1597–1607.


