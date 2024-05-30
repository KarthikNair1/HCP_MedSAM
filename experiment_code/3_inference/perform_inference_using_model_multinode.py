import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
import nibabel as nib
import pickle
import time

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--node_rank", type=int)
args = parser.parse_args()

df_hcp = pd.read_csv('/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv')
df_desired = pd.read_csv('/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
NUM_CLASSES = len(df_desired)
label_converter = LabelConverter(df_hcp, df_desired)

model = build_sam_vit_b_multiclass(num_classes=NUM_CLASSES, checkpoint='/gpfs/home/kn2347/results/models_8-9-23/scratch_loss_reweighted_lr1e-4_ce_only_longer3_model_20230804-133537/model_best.pth')
medsam_model = MedSAM(image_encoder=model.image_encoder, 
                        mask_decoder=model.mask_decoder,
                        prompt_encoder=model.prompt_encoder
                        ).cuda()

num_nodes = 1
this_node_rank = args.node_rank
fracs = np.linspace(0, 1, num_nodes+1)

# load dataframe of slice paths
path_df = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv')

# load train val test ids
dicto = pickle.load(open('/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle', 'rb'))
train_ids = dicto['train']
val_ids = dicto['val']
test_ids = dicto['test']
total_ids = train_ids + val_ids + test_ids

cutpoints = [round(x * len(total_ids)) for x in fracs]
assert cutpoints[-1] == len(total_ids)
total_ids = total_ids[cutpoints[this_node_rank] : cutpoints[this_node_rank+1]]

total_df = path_df[path_df['id'].isin(total_ids)].reset_index(drop=True)

total_dataset = MRIDataset(total_df, None, 0, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=True)


def proc_arr(arr):
    # arr has shape (B, classes, *)
    ax_starts = np.argmax(arr, axis=2) # shape (B, classes)
    ax_ends = arr.shape[2] - 1 - np.argmax(arr[:,:,::-1], axis=2) # shape (B, classes)

    maxs = np.max(arr, axis = 2) # shape (B, classes)
    ax_starts = np.where(maxs == 1, ax_starts, np.nan)
    ax_ends = np.where(maxs == 1, ax_ends, np.nan)

    return ax_starts, ax_ends


def get_bounding_box(seg_arr):
    # seg_tens has shape (B, C,256,256)
    # return shape (4, classes) - rmin, rmax, cmin, cmax
    
    cols = np.any(seg_arr, axis=2) # (B, classes, W)
    rows = np.any(seg_arr, axis=3) # (B, classes, H) of True/False, now find min row and max row with True
    
    rmin, rmax = proc_arr(rows)
    cmin, cmax = proc_arr(cols)

    return np.stack([cmin, cmax, rmin, rmax], axis = 2) # (B, C, 4)

dataloader = DataLoader(
        total_dataset,
        batch_size = 32,
        shuffle = False,
        num_workers = 0,
        pin_memory = True
)

rt_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/saved_round1_segmentations_bbox'

for step, (image_embedding, gt2D, boxes, slice_names) in enumerate(tqdm(dataloader)):
    image_embedding, gt2D, boxes = image_embedding.cuda(), gt2D.cuda(), boxes.cuda()
    out = medsam_inference(medsam_model, image_embedding, boxes, 256, 256, True, True).astype(bool)
    # out has shape (B, C, H, W)

    bboxes = get_bounding_box(out) # (B, C, 4)
    id_names = [x.split('_')[0] for x in slice_names]
    
    for i in range(len(slice_names)):
        id_name = slice_names[i].split('_')[0]
        if not os.path.exists(os.path.join(rt_path, id_name)):
            os.makedirs(os.path.join(rt_path, id_name))
        suffix = slice_names[i].split('slice')[-1] # should be something like "0.npy"

        full_path_this = os.path.join(rt_path, id_name, suffix)
        np.save(full_path_this, bboxes[i])
        
        

