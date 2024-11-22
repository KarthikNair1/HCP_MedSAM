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

path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv'
df_hcp = pd.read_csv('/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv')
df_desired = pd.read_csv('/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
label_converter = LabelConverter(df_hcp, df_desired)
num_classes=102

train, val, test = load_datasets(
            path,
            '/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
            label_id = None, bbox_shift=0, 
                sample_n_slices = None, label_converter=label_converter, NUM_CLASSES=num_classes+1, 
                as_one_hot=True, pool_labels=False, preprocess_fn = None, dataset_type = MRIDataset)

train_dl = DataLoader(train, batch_size=128, shuffle=False, num_workers=1)
val_dl = DataLoader(val, batch_size=128, shuffle=False, num_workers=1)
test_dl = DataLoader(test, batch_size=128, shuffle=False, num_workers=1)

def operate(dataloader):
    list_names = []
    list_arr = []
    for step, (image, gt2D, boxes, img_slicename) in enumerate(tqdm(dataloader)):
        #gt2D = gt2D.cuda()
        list_names.extend(list(img_slicename))
        class_sums = gt2D.sum(dim=(2,3)) # BCHW -> BC
        class_sums = class_sums.detach().cpu().numpy()
        list_arr.append(class_sums)

    return np.vstack(list_arr), list_names

train_arr, train_names = operate(train_dl)
val_arr, val_names = operate(val_dl)
test_arr, test_names = operate(test_dl)

total_arr = np.vstack([train_arr, val_arr, test_arr])
total_names = train_names + val_names + test_names

reto = pd.DataFrame(total_arr, columns = [f'label{x}' for x in range(0,103)])
reto = reto.drop('label0', axis=1)

reto['id'] = [x.split('_')[0] for x in total_names]
reto['slice'] = [int(x.split('_slice')[1].split('.npy')[0]) for x in total_names]
order = ['id', 'slice']
for i in range(1,103):
    order.append(f'label{i}')

reto = reto.loc[:, order]

reto.to_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/class_counts/class_counts_per_slice.csv', index=False)
