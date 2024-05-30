import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from skimage import io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks import one_hot
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from skimage import io, transform
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import nibabel as nib
import pickle
from torch.utils.data import RandomSampler
import random
import scipy
import torch.nn.functional as F
import img2pdf
from torchmetrics import F1Score
from PIL import Image

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, medsam_inference
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--node_idx', type=int)
args = parser.parse_args()

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


df_hcp = pd.read_csv('/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv')
df_desired = pd.read_csv('/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
NUM_CLASSES = len(df_desired)
label_converter = LabelConverter(df_hcp, df_desired)

path_df_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv'
train_test_splits_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle'
train_dataset, val_dataset, test_dataset = load_datasets(path_df_path, train_test_splits_path, label_id = None, bbox_shift=0, sample_n_slices = None, label_converter=label_converter, NUM_CLASSES=NUM_CLASSES)


def proc_arr(arr):
    # arr has shape (classes, *)
    ax_starts = np.argmax(arr, axis=1) # shape (classes)
    ax_ends = arr.shape[1] - 1 - np.argmax(arr[:,::-1], axis=1) # shape (classes)

    maxs = np.max(arr, axis = 1) # shape (classes)
    ax_starts = np.where(maxs == 1, ax_starts, np.nan)
    ax_ends = np.where(maxs == 1, ax_ends, np.nan)

    return ax_starts, ax_ends


def get_bounding_box(seg_tens):
    # seg_tens has shape (256,256,256)
    # return shape (4, classes) - rmin, rmax, cmin, cmax
    
    cols = np.any(seg_tens, axis=1) # (classes, W)
    rows = np.any(seg_tens, axis=2) # (classes, H) of True/False, now find min row and max row with True
    
    rmin, rmax = proc_arr(rows)
    cmin, cmax = proc_arr(cols)
    
    return np.array((rmin, rmax, cmin, cmax))

def conv_format(seg_tens):

    # return shape (num_classes, 4) - x, y, width, height
    if type(seg_tens) == torch.Tensor:
        seg_tens = seg_tens.cpu().detach().numpy()
    res = get_bounding_box(seg_tens) # 4 x num_classes - rmin, rmax, cmin, cmax
    x_center = (res[2, :]+res[3,:])/2.0
    y_center = (res[0, :]+res[1,:])/2.0
    width = res[3, :] - res[2, :]
    height = res[1, :] - res[0, :]

    TOTAL_WIDTH = seg_tens.shape[2]
    TOTAL_HEIGHT= seg_tens.shape[1]

    return np.array((x_center/TOTAL_WIDTH, y_center/TOTAL_HEIGHT, width/TOTAL_WIDTH, height/TOTAL_HEIGHT)).T

def write_for_fraction(dataset, tag = 'train', start_frac = 0, end_frac = 1):

    # write 
    path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/dir_structure_for_yolov7/{tag}/labels/'
    img_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/dir_structure_for_yolov7/{tag}/images/'
    for i in range(int(round(start_frac * len(dataset))), int(round(end_frac * len(dataset)))):
        _, seg_tens, _, img_slice_name = dataset[i] # seg_tens is 256x256x256, img_tens is 256x256
        bbox = conv_format(seg_tens) # (256, 4) -> rmin, rmax, ymin, ymax
        #print(img_slice_name) # e.g. 100206_slice0.npy
        img_slc = img_slice_name.split('.npy')[0]
        this_path  = os.path.join(path, img_slc + '.txt')
        with open(this_path, 'w') as f:
        
            for class_num in range(bbox.shape[0]):
                if not np.isnan(bbox[class_num, 0]):
                    this_r = bbox[class_num, :].astype(float)

                    output_line = f'{class_num} {this_r[0]} {this_r[1]} {this_r[2]} {this_r[3]}'
                    f.write(output_line + '\n')
            f.close()

        # save the png image as well
        img_write_path = os.path.join(img_path, img_slc + '.png')
        img_itself = np.repeat(dataset.load_image(i)[:,:,None], 3, axis=-1) #(256,256,3)
        img_pil = Image.fromarray(img_itself)
        img_pil.save(img_write_path)
    


n_nodes = 5
spacers = np.linspace(0, 1, num=n_nodes+1) 

if args.node_idx >= n_nodes:
    print('over limit')
    exit

st = spacers[args.node_idx]
en = spacers[args.node_idx+1]

#write_for_fraction(train_dataset, tag = 'train', start_frac = st, end_frac = en)
write_for_fraction(val_dataset, tag = 'val', start_frac = st, end_frac = en)
write_for_fraction(test_dataset, tag = 'test', start_frac = st, end_frac = en)

    
