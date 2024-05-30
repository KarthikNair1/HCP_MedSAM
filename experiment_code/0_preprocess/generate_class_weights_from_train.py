# This code writes weights for each class, such that classes with low overall number of pixels have higher weight
# and thus can be prioritized more during training a multi-class model

# Steps:

# 1. Calculate mean number of pixels per slice for each class across the entire dataset (so 102 total values)
# 2. Call this list "mean_pixels" with 102 elements
# 3. Weight for i'th class = median{mean_pixels} / mean_pixels[i]


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
import multiprocessing as mp

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.utils_hcp import *

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)


# load stuff
path_df_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv'
train_test_splits_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle'

path_df = pd.read_csv(path_df_path)
dicto = pickle.load(open(train_test_splits_path, 'rb'))
train_ids = dicto['train']
train_df = path_df[path_df['id'].isin(train_ids)].reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True) # randomly permute rows
print(train_df.columns)

path_list = train_df['segmentation_slice_path'].tolist()

# for a single segment of our list of MRI slices, this function returns counts for each region
def process_segment_of_list(start, end):

    global_counts = dict()
    num_imgs = 0

    for i in range(start, end):
        if i % 1000 == 0:
            print(f'on {i}')
        seg_arr = np.load(path_list[i])
        num_imgs+=1
        unique, counts = np.unique(seg_arr, return_counts=True)
        for idx in range(len(unique)):

            if unique[idx] not in global_counts.keys():
                global_counts[unique[idx]] = 0
            
            global_counts[unique[idx]] += counts[idx]
    
    return global_counts

# multiprocessing to speed up counting of pixels across whole dataset
num_workers = 47
output = mp.Queue()
pool = mp.Pool(processes=num_workers)
result_dicts = []

# for each worker, pass it a segment of the overall dataset to count on
for i in range(num_workers):
    start = len(path_list) // num_workers * i
    end = len(path_list) // num_workers * (i+1)
    if i == num_workers - 1: # last iter
        end = len(path_list)
    
    result = pool.apply_async(process_segment_of_list, args = (start,end)).get()
    result_dicts.append(result)

# combine results for each worker
global_counts = result_dicts[0]
print(type(global_counts))
for i in range(1, num_workers):
    for k in global_counts.keys():
        global_counts[k] += result_dicts[i][k]

# calculate mean pixels per class
means = dict()
num_imgs = len(path_list)
listo = []
for k, v in global_counts.items():
    means[k] = v / num_imgs
    listo.append(means[k])

# calculate median
median_freq = np.median(listo)

# calculate weights
class_weights = dict()
for k, v in means.items():
    class_weights[k] = median_freq / (v+1e-6)

# save
save_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/class_weights_256.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(class_weights, file)

print('Done')





    