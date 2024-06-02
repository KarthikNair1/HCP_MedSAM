import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import pickle
from torch.utils.data import RandomSampler
import random
import scipy
import torch.nn.functional as F
from PIL import Image
from glob import glob
import sys
sys.path.append('./modified_medsam_repo')
from MedSAM_HCP.utils_hcp import *

def read_format_table(path, read_gt = False):
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    if read_gt:
        cols = ['class', 'x_center', 'y_center', 'width', 'height']
    else:
        cols = ['class', 'x_center', 'y_center', 'width', 'height', 'confidence']
    df = pd.read_csv(path, delimiter=' ', header=None, names=cols)
    df = df.sort_values('class').reset_index(drop=True)
    df['x_center'] = (df['x_center'] * IMG_WIDTH)
    df['y_center'] = (df['y_center'] * IMG_HEIGHT)
    df['width'] = (df['width'] * IMG_WIDTH)
    df['height'] = (df['height'] * IMG_HEIGHT)

    return df
def yolov7_format_to_bbox_format(x_center, y_center, width, height):
    x0 = x_center - width / 2.0
    y0 = y_center - height / 2.0

    x1 = x_center + width / 2.0
    y1 = y_center + height / 2.0

    return x0, y0, x1, y1

def extract_box_np_from_df(df, label):
    row = df.loc[df['class'] == label]
    if len(row) == 0:
        return np.full((4,), np.nan)
    if len(row)>1:
        print(row)
    box = np.array(yolov7_format_to_bbox_format(row['x_center'].item(), 
                                      row['y_center'].item(),
                                      row['width'].item(), 
                                      row['height'].item()))
    return box




def run_tag(collect_lists, tag='train', conf_thresh = 0.225):
    # collect_lists should be list of NUM_CLASSES lists, each containing (6,) np arrays representing id, slice, and bounding box for each sample for that class
    
    for _, file in enumerate(tqdm(glob(f'/gpfs/data/luilab/karthik/pediatric_seg_proj/yolov7_results/train_hcp_bbox/{tag}_run/labels/*.txt'))):
        basename = os.path.basename(file)
        id_num = int(basename.split('_')[0])
        slice_num = int(basename.split('_slice')[-1].split('.txt')[0])
        dfo = read_format_table(file, read_gt=False)
        dfo = dfo[dfo['confidence'] >= conf_thresh]
        dfo = dfo.sort_values('confidence').drop_duplicates('class', keep='last')    
        dfo['bbox_0'] = (dfo['x_center'] - dfo['width']/2.0).round().astype(int)
        dfo['bbox_1'] = (dfo['y_center'] - dfo['height']/2.0).round().astype(int)
        dfo['bbox_2'] = (dfo['x_center'] + dfo['width']/2.0).round().astype(int)
        dfo['bbox_3'] = (dfo['y_center'] + dfo['height']/2.0).round().astype(int)


        for i, r in dfo.iterrows():
            if r['class']==0:
                continue
            ans = np.array([id_num, slice_num, r['bbox_0'], r['bbox_1'], r['bbox_2'], r['bbox_3']]).astype(int)
            collect_lists[int(r["class"])].append(ans)
    
    return collect_lists



NUM_CLASSES=103
conf_thresh = 0.10
collect_lists = []
for i in range(NUM_CLASSES):
    collect_lists.append([])

collect_lists = run_tag(collect_lists, 'train', conf_thresh)
collect_lists = run_tag(collect_lists, 'val', conf_thresh)
collect_lists = run_tag(collect_lists, 'test', conf_thresh)

ori_df = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df.csv')
for class_num in range(1, NUM_CLASSES):
    this_class_df = pd.DataFrame(collect_lists[class_num], columns = ['id', 'slice', 'bbox_0', 'bbox_1', 'bbox_2', 'bbox_3'])
    merged = ori_df.merge(this_class_df, how = 'right', on = ['id', 'slice'])

    save_path = os.path.join('/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/yolov7/conf_thresh_010', f'path_df_label{class_num}_only_with_bbox_yolov7.csv')
    merged.to_csv(save_path)
