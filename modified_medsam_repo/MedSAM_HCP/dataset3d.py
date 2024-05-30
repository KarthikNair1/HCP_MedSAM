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
from segment_anything import sam_model_registry, build_sam_vit_b_multiclass
import torch.nn.functional as F
import random
from datetime import datetime
import pandas as pd
import nibabel as nib
import pickle

class MRIDataset3D(Dataset): 
    def __init__(self, data_frame, label_id=None, bbox_shift=0, label_converter=None, neighborhood_dim = 5):
        self.data_frame = data_frame
        self.bbox_shift = bbox_shift
        self.label_id = label_id
        self.label_converter = label_converter
        self.neighborhood_dim = neighborhood_dim

        if self.label_converter is None:
            print('Initializing with no label converter, are you sure the labels are correct?')

        #if self.pool_labels and self.label_converter is not None: actually, will already come compressed
        #    self.data_frame['label_number'] = self.label_converter.hcp_to_compressed(self.data_frame['label_number'])
        #print(f'number of images: {data_frame.shape[0]}')
    
    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        # load image embeddings in neighborhood as npy
        this_slice_num = self.data_frame.loc[index,'slice']
        img_embed_path_to_folder = os.path.dirname(self.data_frame.loc[index,'image_embedding_slice_path'])
        list_embeddings = []
        for slice in range(this_slice_num - (self.neighborhood_dim-1)//2, this_slice_num + (self.neighborhood_dim-1)//2 + 1):
            if slice < 0 or slice >= 256:
                img_embed_npy = np.zeros((256,64,64))
            else:
                read_path = os.path.join(img_embed_path_to_folder, f'{slice}.npy')
                if slice == this_slice_num:
                    main_path = read_path
                img_embed_npy = np.load(read_path) # (256, 64, 64)
            list_embeddings.append(img_embed_npy)

        stacked_embeddings = np.stack(list_embeddings, axis=-1) # now (256, 64, 64, self.neighborhood_dim)

        img_slice_name = '_slice'.join(main_path.split('/')[-2:])

        # load segmentation mask as npy
        seg_path = self.data_frame.loc[index,'segmentation_slice_path']
        seg_npy = np.load(seg_path) # (256, 256)

        if self.label_converter is not None:
            #print(f'pre label max: {seg_npy.max()}')
            seg_npy = self.label_converter.hcp_to_compressed(seg_npy)
            #print(f'post label max: {seg_npy.max()}')

        # use all classes
        # currently seg_npy is (H,W)
        seg_tens = torch.LongTensor(seg_npy[None, :, :]) # B, H, W
        seg_tens = torch.nn.functional.one_hot(seg_tens, num_classes=103) # B, H, W, C
        
        seg_tens = torch.permute(seg_tens, (0, 3, 1, 2)) # B, C, H, W
        seg_tens = seg_tens[0] # exclude batch dimension -> C, H, W


        bboxes = np.load(self.data_frame.loc[index,'box_path']) # need to add this column to df
        if bboxes.shape[0] == 102:
            # need to insert row at the beginning to represent "unknown" entries
            bboxes = np.concatenate([np.full((1, 4), np.nan), bboxes])
        # must be np array in shape (103, 4) with entries x_min, y_min, x_max, y_max

        
        return torch.tensor(stacked_embeddings).float(), seg_tens, torch.tensor(bboxes).float(), img_slice_name
        # (256, 64, 64, 5) , (103, 256, 256), (103, 4), (1)
    
    def load_image(self, index):
        img_path = self.data_frame.loc[index, 'image_path']
        slice_num = self.data_frame.loc[index, 'slice']
        img = nib.load(img_path).get_fdata()[:,slice_num,:].astype(np.uint8)
        return img # returns as (256, 256)
    

# code to load train, val, test datasets
def load_datasets_3d(path_df_path, train_test_splits_path, bbox_shift=0, label_converter=None, neighborhood_dim = 5):
    # load dataframe of slice paths
    path_df = pd.read_csv(path_df_path)

    # load train val test ids
    dicto = pickle.load(open(train_test_splits_path, 'rb'))
    train_ids = dicto['train']
    val_ids = dicto['val']
    test_ids = dicto['test']

    train_df = path_df[path_df['id'].isin(train_ids)].reset_index(drop=True)
    val_df = path_df[path_df['id'].isin(val_ids)].reset_index(drop=True)
    test_df = path_df[path_df['id'].isin(test_ids)].reset_index(drop=True)

    train_dataset = MRIDataset3D(train_df, bbox_shift, label_converter = label_converter, neighborhood_dim=neighborhood_dim)
    val_dataset = MRIDataset3D(val_df, 0, label_converter = label_converter, neighborhood_dim=neighborhood_dim)
    test_dataset = MRIDataset3D(test_df, 0, label_converter = label_converter, neighborhood_dim=neighborhood_dim)

    return train_dataset, val_dataset, test_dataset