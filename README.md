# DeepFCD
Detect focal cortical dysplasias using a generative modelling approach as presented on OHBM 2019 in Rome (see our homepage for e-Poster, OHBM oral session and more information: https://www.translationalneuroimaging.de/ohbm-gan-talk ).

Repository is WIP, nothing interesting to see here *yet*. If you need a state-of-the-art program to automatically detect the FCDs in your datset, please check out MAP18 (https://www.translationalneuroimaging.de/map18).

## Software used
Python 3.7.4  
Tensorflow 2.1.0 (& 2.2.0-dev for saving and loading models using a CPU only)  
Nibabel 2.5.1  
Numpy 1.17.2  
PIL 6.2.0  

### For 3D-GAN and 3D-CNN training:
CUDA release 10.2

GAN inspired by pix2pix (Isola, Zhu et al., 2016)  
(github repo: https://github.com/phillipi/pix2pix)  

## GAN example output
#### real FLAIR - real T1 - synthetic FLAIR - synthetic T1
<img src="./assets/example_outputs/T1_FLAIR_SYNTH-T1_SYNTH-FLAIR_00.gif" width=1000 align="center">
