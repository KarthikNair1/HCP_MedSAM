import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
from glob import glob
import argparse
from PIL import Image
import pandas as pd
import re

def get_slice_to_path_dict(list_paths):
    dicto = {}
    for p in list_paths:
        last_part = p.split('/')[-1].split('.')[0] # get filename before .filetype
        # figure out how to regex out the suffix int - this will be the slice number
        slice_num = int(re.search(r"(\d+)$", last_part).group())
        dicto[slice_num] = p
    
    return dicto
def dict_get_or_none(dicto, key):
    if key in dicto:
        return dicto[key]
    else:
        return None

parser = argparse.ArgumentParser()

# accepts start_idx and end_idx for processing whole dataset across multiple nodes
parser.add_argument("--model_type", type=str, choices = ['singletask_unprompted', 'singletask_unet'])
parser.add_argument("--mri_id", type=str)
parser.add_argument("--img_encoding_dir_pattern", type=str, default = None)
parser.add_argument("--seg_dir_pattern", type=str)
parser.add_argument("--img_dir_pattern", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

# initialize the dictionary to track dataframe values
collect_dict = {
    'id':[],
    'slice':[],
    'image_embedding_slice_path':[],
    'segmentation_slice_path':[],
    'image_path':[],
    'bbox_0':[],
    'bbox_1':[],
    'bbox_2':[],
    'bbox_3':[]
}
if args.model_type == 'singletask_unet':
    collect_dict = {
        'id':[],
        'slice':[],
        'image_path':[],
        'segmentation_slice_path':[]
    }
elif args.model_type == 'singletask_unprompted':
    collect_dict = {
        'id':[],
        'slice':[],
        'image_embedding_slice_path':[],
        'segmentation_slice_path':[],
        'image_path':[],
        'bbox_0':[],
        'bbox_1':[],
        'bbox_2':[],
        'bbox_3':[]
    }

seg_paths = glob(args.seg_dir_pattern)
img_paths = glob(args.img_dir_pattern)

if args.model_type == 'singletask_unprompted':
    img_encoding_paths = glob(args.img_encoding_dir_pattern)
    img_encoding_dict = get_slice_to_path_dict(img_encoding_paths)
seg_dict = get_slice_to_path_dict(seg_paths)
img_dict = get_slice_to_path_dict(img_paths)

for sli in img_dict.keys():
    collect_dict['id'].append(args.mri_id)
    collect_dict['slice'].append(sli)        
    collect_dict['segmentation_slice_path'].append(dict_get_or_none(seg_dict, sli))
    collect_dict['image_path'].append(dict_get_or_none(img_dict, sli))

    if args.model_type == 'singletask_unprompted':
        collect_dict['image_embedding_slice_path'].append(dict_get_or_none(img_encoding_dict, sli))
        collect_dict['bbox_0'].append(0)
        collect_dict['bbox_1'].append(0)
        collect_dict['bbox_2'].append(256)
        collect_dict['bbox_3'].append(256)

df = pd.DataFrame(collect_dict)
df = df.sort_values(by='slice').reset_index(drop=True)
if args.model_type == 'singletask_unet':
    df = df.rename({'image_path': 'img_slice_path'}, axis=1)

out_path = os.path.join(args.output_dir, f'path_df_{args.model_type}.csv')
df.to_csv(out_path, index=False)