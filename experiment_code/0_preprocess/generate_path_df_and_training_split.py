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
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import SamPredictor, sam_model_registry, build_sam_vit_b_multiclass
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from skimage import io, transform
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import nibabel as nib
import pickle
from torch.utils.data import RandomSampler
from typing import Callable
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, medsam_inference
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *

parser = argparse.ArgumentParser()

# accepts start_idx and end_idx for processing whole dataset across multiple nodes
parser.add_argument("--image_encoding_dir", type = str, help = "Directory with all image encodings, e.g. /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/pretrained_image_encoded_slices")
parser.add_argument("--seg_dir", type = str, help = "Directory with all image segmentations, e.g. /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/segmentation_slices")
parser.add_argument("--mri_dir", type = str, help = "Directory containing all MRI .mgz files. Required dir structure is mri_dir/id_number/mri/T1.mgz. e.g. /gpfs/data/cbi/hcp/hcp_seg/data_orig")
parser.add_argument("--dest_save_dir", type = str, help = "Destination directory to save dataframes containing pathnames for all MRI slices and train-test split")

args = parser.parse_args()

dest_dir = args.dest_save_dir


# =========================================================
# Generate path_df
# =========================================================

# requires medsam_preprocess_img.py to be run, resulting in encoded slices folder and segmentation slices folder
# place the names of these folders below
data_path = args.image_encoding_dir
data_labels_path = args.seg_dir
image_path = args.mri_dir # path to the original MRI files (*.mgz)
folder_paths = sorted(glob(os.path.join(data_path, '*')))

# construct dataframe with all data and slices
# columns:

# ID    Slice #      Path_To_Slice_Npy     Path_To_Slice_Segmentation_Npy     Path_To_Image

# also add columns for the area of each segmentation in the slice

collector_dict = {'id': [], 
                'slice': [], 
                'image_embedding_slice_path': [],
                'segmentation_slice_path': [],
                'image_path': []
}

NUM_CLASSES = 256
region_areas = []
for i, elem in enumerate(tqdm(folder_paths)):
    id = elem.split('/')[-1]
    seg_path = os.path.join(data_labels_path, id)
    for slice_name in os.listdir(elem):
        slice_id = slice_name.split('.')[0]
        
        data_path_this_slice = os.path.join(elem, slice_name)
        seg_path_this_slice = os.path.join(seg_path, f'seg_{slice_name}')

        path_to_overall_image = os.path.join(image_path, id, 'mri', 'T1.mgz')

        collector_dict['id'].append(int(id))
        collector_dict['slice'].append(int(slice_id))
        collector_dict['image_embedding_slice_path'].append(data_path_this_slice)
        collector_dict['segmentation_slice_path'].append(seg_path_this_slice)
        collector_dict['image_path'].append(path_to_overall_image)

df = pd.DataFrame.from_dict(collector_dict)
df = df.sort_values(by = ['id', 'slice'])
df = df.reset_index(drop=True)

df.to_csv(os.path.join(dest_dir, 'path_df.csv'), index=False)

# make version of the dataframe with constant bbox (0,0,256,256) for MedSAM input
df_constant_bbox = df
df_constant_bbox['bbox_0'] = 0
df_constant_bbox['bbox_1'] = 0
df_constant_bbox['bbox_2'] = 256
df_constant_bbox['bbox_3'] = 256

df_constant_bbox.to_csv(os.path.join(dest_dir, 'path_df_constant_bbox.csv'), index=False)

# make version of the dataframe w/ constant bbox and discard 90% of blank images (defined as slice_num <=30 or >=225)
df = df_constant_bbox
df_trim = df[(30 < df['slice']) & (df['slice'] < 225)]
df_edges = df[(df['slice'] <= 30) | (225 <= df['slice'])]
df_edges_subsampled = df_edges.sample(frac = 0.10, replace=False, random_state=182)

df_total = pd.concat([df_trim, df_edges_subsampled], axis=0).reset_index()
df_total.to_csv(os.path.join(dest_dir, 'path_df_constant_bbox_remove_most_blank.csv'), index=False)

# =========================================================
# Generate train-val-test split in ratio 80%:10%:10%
# =========================================================
ids = [int(x.split('/')[-1]) for x in folder_paths]

# train, val, test split on the id's
size_val = round(0.1 * len(ids))
size_test = round(0.1 * len(ids))

trainval_ids, test_ids = train_test_split(ids, test_size=size_test, random_state = 2023, shuffle=True)
train_ids, val_ids = train_test_split(trainval_ids, test_size=size_val, random_state = 2023, shuffle=True)

# save the id's to a place permanently

dicto = {'train':train_ids, 'val':val_ids, 'test':test_ids}
with open(os.path.join(dest_dir, 'train_val_test_split.pickle'), 'wb') as file:
    pickle.dump(dicto,file)