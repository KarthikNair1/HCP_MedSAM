# MedSAM Fine-tuned for Neuroanatomy Segmentation 

<p align="center">
<img src="https://github.com/KarthikNair1/HCP_MedSAM/assets/43316987/e4d392b6-6664-4a1f-94cb-60ce6df4a596" width=25% height=25%>
</p>

## Overview
The goal of this project is to adapt the foundation model MedSAM (https://github.com/bowang-lab/MedSAM) for segmentation of neuroanatomical structures and compare the performance against a Unet baseline. We use 1,113 T1w MPRAGE MRI's from the Human Connectome Project (HCP) Young Adult cohort, along with the matching FreeSurfer segmentations, for finetuning the model.

## Installation
1.  <code>git clone https://github.com/KarthikNair1/HCP_MedSAM</code>
2.  Create a conda virtual environment using <code>conda env create --name medsam_hcp --file=environments.yml</code>
3. Enter the <code>modified_medsam_repo</code> folder and run <code>pip install -e .</code> to install the MedSAM library
   
## Results



## Preprocessing

<code>medsam_preprocess_img.py</code>: Passes all MRI slices through the encoder component of MedSAM and saves embeddings as .npy files. These .npy files are used as input for downstream MedSAM finetuning. Note that this means that the encoder is not trained during the training phase, only the decoder.

## Training

## Inference

## Evaluation

## Visualization

## Other Experiments
