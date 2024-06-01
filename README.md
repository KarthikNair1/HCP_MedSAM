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

<code>generate_path_df_and_training_split.py</code>: Generates dataframe pointing to all paths for embeddings and segmentation npy's. This dataframe is used for all downstream machine learning. Additionally, a train-val-test split is generated and saved. This split should be generated only once to ensure reproducibility.

## Training

<code>train_multi_gpus_modified_multiclass.py</code>: Code to train MedSAM model using HCP data. Supports many different training options, including single-task or multi-task and on multiple GPU's or nodes. Example scripts are located in the scripts folder in this directory.

<code>train_unet.py</code>: Code to train Unet model using HCP data, designed for single-task segmentation. Single GPU/node.

<code>medsam_finetuning_organ_dataset.ipynb</code> and <code>medsam_finetuning_hcp_ya.ipynb</code> are useful notebooks for looking at the step-by-step process to training a MedSAM model. The former is finetuning on CT scans, and the latter is specifically for finetuning on the HCP dataset.

## Inference
<code>perform_inference_using_model_multinode.py</code>: Predicts segmentations for all MRI slices in the input path_df (by default, will perform for all train, val, and test MRI's).

<code>medsam_hcp_predict_basics.ipynb</code>: A tutorial notebook that walks through the steps in segmenting a CT scan and a brain MRI using MedSAM without any HCP finetuning. Useful for understanding the steps in prediction.

<code>medsam_hcp_segment_using_pretraining_only.ipynb</code>: A notebook that explores using out-of-the-box MedSAM without any HCP-finetuning as a segmentation tool on HCP MRI's. Some examples are shown in the notebook.

## Evaluation
<code>full_eval_val_test.py</code>: Runs a MedSAM model (single-task or multi-task) over either the validation or test set and saves dice scores. Supports multi-node. This file will require some changes to filepaths within the code to run properly.

<code>analyze_eval_results_val.ipynb</code>: Contains graphs and other analysis of the validation results obtained from running <code>full_eval_val_test.py</code>.

<code>analyze_eval_results_TEST.ipynb</code>: Analogous, but for the test results.

## Visualization
<code>


## Other Experiments
