import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import pickle
import random
from PIL import Image
from glob import glob
import re
#from adjustText import adjust_text
import seaborn as sns
import statannot
import argparse
#import torch
import wandb
import sys
sys.path.append('./modified_medsam_repo')
from MedSAM_HCP.dataset import MRIDataset_Imgs, load_datasets
from MedSAM_HCP.MedSAM import MedSAM
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *
from segmentation_models_pytorch.encoders import get_preprocessing_fn

path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv'
df_hcp = pd.read_csv('/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv')
df_desired = pd.read_csv('/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
label_converter = LabelConverter(df_hcp, df_desired)
num_classes=102

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')


train, val, test = load_datasets(
            path,
            '/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
            label_id = None, bbox_shift=0, 
                sample_n_slices = None, label_converter=label_converter, NUM_CLASSES=num_classes+1, 
                as_one_hot=True, pool_labels=False, preprocess_fn = preprocess_input, dataset_type = MRIDataset_Imgs)

train_loader = DataLoader(train, batch_size=64, shuffle=False, num_workers=4)

slice_sums = torch.zeros(102)
pixel_sums = torch.zeros(102)
total_num_slices = 0
for step, (image, gt2D) in enumerate(tqdm(train_loader)):
    image = image.to('cuda')
    gt2D = gt2D.to('cuda') # B x C x H x W
    boolgt2d = gt2D > 0
    has_class = boolgt2d.view((boolgt2d.shape[0], boolgt2d.shape[1], -1)).any(dim=2, keepdim=False) # should now be B x C
    has_class = has_class.sum(dim=0) # now C
    slice_sums += has_class.cpu().detach()


    #label_idxs = torch.argmax(gt2D, dim=1) # now B x H x W
    pixel_sums += gt2D.sum(dim=(0,2,3)).cpu().detach()

    total_num_slices += gt2D.shape[0]

df = pd.DataFrame({'label': list(range(1,103)), 'total_slices': slice_sums.tolist(), 'total_pixels': pixel_sums.tolist()})
df['avg_pixels_per_slice_when_present'] = df['total_pixels'] / df['total_slices']
df['avg_pixels_per_all_slices'] = df['total_pixels'] / total_num_slices
df['fraction_slices_present'] = df['total_slices'] / total_num_slices
df.to_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/training_set_class_statistics_10-15-24/class_statistics.csv', header=False)
