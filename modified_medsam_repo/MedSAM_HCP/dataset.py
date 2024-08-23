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
from PIL import Image


class MRIDataset(Dataset): 
    def __init__(self, data_frame, label_id=None, bbox_shift=0, label_converter=None, NUM_CLASSES = 256, as_one_hot = True, pool_labels = False, preprocess_fn = None):
        self.data_frame = data_frame
        self.bbox_shift = bbox_shift
        self.label_id = label_id
        self.label_converter = label_converter
        self.NUM_CLASSES = NUM_CLASSES
        self.as_one_hot = as_one_hot
        self.pool_labels = pool_labels
        self.preprocess_fn = preprocess_fn

        if self.label_converter is None:
            print('Initializing with no label converter, are you sure the labels are correct?')

        #if self.pool_labels and self.label_converter is not None: actually, will already come compressed
        #    self.data_frame['label_number'] = self.label_converter.hcp_to_compressed(self.data_frame['label_number'])
        #print(f'number of images: {data_frame.shape[0]}')
    
    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        # load image embedding as npy
        img_embed_path = self.data_frame.loc[index,'image_embedding_slice_path']
        img_embed_npy = np.load(img_embed_path) # (256, 64, 64)

        if self.preprocess_fn is not None:
            img_embed_npy = self.preprocess_fn(img_embed_npy)

        img_slice_name = '_slice'.join(img_embed_path.split('/')[-2:])

        # load segmentation mask as npy
        seg_path = self.data_frame.loc[index,'segmentation_slice_path']
        seg_npy = np.load(seg_path) # (256, 256)

        if self.label_converter is not None:
            #print(f'pre label max: {seg_npy.max()}')
            seg_npy = self.label_converter.hcp_to_compressed(seg_npy)
            #print(f'post label max: {seg_npy.max()}')
        if self.label_id is None: # use all classes
            # currently seg_npy is (H,W)
            seg_tens = torch.LongTensor(seg_npy[None, :, :]) # B, H, W
            seg_tens = torch.nn.functional.one_hot(seg_tens, num_classes=self.NUM_CLASSES) # B, H, W, C
            
            seg_tens = torch.permute(seg_tens, (0, 3, 1, 2)) # B, C, H, W
            seg_tens = seg_tens[0] # exclude batch dimension -> C, H, W
  
        else: # use 1 class only
            if self.pool_labels:
                label_number = self.data_frame.loc[index,'label_number']
            else:
                label_number = self.label_id

            seg_npy = (seg_npy==label_number).astype(np.uint8)
            seg_tens = torch.tensor(seg_npy[None, :, :]).long()

        # load bounding box coordinates from data frame
        x_min, x_max = self.data_frame.loc[index, 'bbox_0'], self.data_frame.loc[index, 'bbox_2']
        y_min, y_max = self.data_frame.loc[index, 'bbox_1'], self.data_frame.loc[index, 'bbox_3']
        
        if not np.any(np.isnan([x_min, x_max, y_min, y_max])): # if no nans
            # add perturbation to bounding box coordinates
            H, W = seg_npy.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])

        return torch.tensor(img_embed_npy).float(), seg_tens, torch.tensor(bboxes).float(), img_slice_name
    
    def load_image(self, index):
        img_path = self.data_frame.loc[index, 'image_path']
        imgo_obj = Image.open(img_path)
        img = np.array(imgo_obj) # 256,256,3
        img = img[:, :, 0] # 256,256
        print(img.min())
        print(img.max())
        #slice_num = self.data_frame.loc[index, 'slice']
        #img = nib.load(img_path).get_fdata()[:,slice_num,:].astype(np.uint8)
        return img # returns as (256, 256)
class MRIDatasetForPooled(MRIDataset):
    def __init__(self, data_frame, label_id=None, bbox_shift=0, label_converter=None, NUM_CLASSES = 256, as_one_hot = True, pool_labels = False):
        super().__init__(data_frame, label_id, bbox_shift, label_converter, NUM_CLASSES, as_one_hot, pool_labels)
    def __len__(self):
        return super().__len__()
    def __getitem__(self, index):
        return super().__getitem__(index) + (self.data_frame.loc[index, 'label_number'],)
    def load_image(self, index):
        return super().load_image(index)

# Dataset class initially written for training UNet
# Allows for loading images for training rather than np arrays. Designed for single or multi-class
# No bounding box or slice name
class MRIDataset_Imgs(MRIDataset): 
    def __init__(self, data_frame, label_id=None, bbox_shift=0, label_converter=None, NUM_CLASSES = 256, as_one_hot = True, pool_labels = False, preprocess_fn=None):
        super().__init__(data_frame, label_id, bbox_shift, label_converter, NUM_CLASSES, as_one_hot, pool_labels, preprocess_fn)
    def __len__(self):
        return super().__len__()
    def __getitem__(self, index):
        # load image as npy (256x256x3)
        img_path = self.data_frame.loc[index,'img_slice_path']
        img = Image.open(img_path)

        img_npy = np.array(img)

        if self.preprocess_fn is not None:
            img_npy = self.preprocess_fn(img_npy)
            
        img_npy = np.transpose(img_npy, axes = (2, 0, 1))
        
        # load segmentation mask as npy
        seg_path = self.data_frame.loc[index,'segmentation_slice_path']
        seg_npy = np.load(seg_path) # (256, 256)

        if self.label_converter is not None:
            seg_npy = self.label_converter.hcp_to_compressed(seg_npy)
        
        if self.label_id is not None:
           seg_npy = (seg_npy == self.label_id)

        # currently seg_npy is (H,W)
        seg_tens = torch.LongTensor(seg_npy[None, :, :]) # B, H, W
        seg_tens = torch.nn.functional.one_hot(seg_tens, num_classes=self.NUM_CLASSES) # B, H, W, C
        
        seg_tens = torch.permute(seg_tens, (0, 3, 1, 2)) # B, C, H, W
        seg_tens = seg_tens[0] # exclude batch dimension -> C, H, W
        # ignore 0th channel
        seg_tens = seg_tens[1:,:,:]
        
        return torch.tensor(img_npy).float(), seg_tens

    def get_slice_name(self, index):
        img_path = self.data_frame.loc[index,'img_slice_path']
        ido = self.data_frame.loc[index,'id']
        sliceo = self.data_frame.loc[index,'slice']
        img_slice_name = f'{ido}_{sliceo}'
        return img_slice_name

# code to load train, val, test datasets
def load_datasets(path_df_path, train_test_splits_path, label_id, bbox_shift=0, 
                sample_n_slices = None, label_converter=None, NUM_CLASSES=256, 
                as_one_hot=True, pool_labels=False, preprocess_fn = None,
                dataset_type = MRIDataset):
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

    if sample_n_slices is not None:
        train_df = train_df.sample(n=sample_n_slices, replace=False, random_state=2023).reset_index(drop=True)
        val_df = val_df.sample(n=sample_n_slices, replace=False, random_state=2023).reset_index(drop=True)
        test_df = test_df.sample(n=sample_n_slices, replace=False, random_state=2023).reset_index(drop=True)
    train_dataset = dataset_type(train_df, label_id, bbox_shift, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=as_one_hot, pool_labels=pool_labels, preprocess_fn = preprocess_fn)
    val_dataset = dataset_type(val_df, label_id, 0, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=as_one_hot, pool_labels=pool_labels, preprocess_fn = preprocess_fn)
    test_dataset = dataset_type(test_df, label_id, 0, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=as_one_hot, pool_labels=pool_labels, preprocess_fn = preprocess_fn)

    return train_dataset, val_dataset, test_dataset



class LabelConverter():
    def __init__(self, hcp_mapping: pd.DataFrame, desired_mapping: pd.DataFrame):
        # hcp_mapping and desired_mapping are dataframes with the structure:
        '''
        region_name     region_index
        None            0
        region_1_name   1
        region_2_name   2
        '''

        # note: the region "None" has to be present in both dataframes, and with region_index=0
        # also, each region has to incremental region index
        # also, desired_mapping's region_names must be a subset of those in hcp_mapping

        assert len(hcp_mapping.columns) == len(desired_mapping.columns) == 2
        assert list(hcp_mapping.columns) == list(desired_mapping.columns) == ['region_name', 'region_index']
        assert hcp_mapping[hcp_mapping['region_name'] == 'Unknown']['region_index'].squeeze() == 0
        assert desired_mapping[desired_mapping['region_name'] == 'Unknown']['region_index'].squeeze() == 0
        assert desired_mapping['region_name'].isin(hcp_mapping['region_name']).all()

        merged = pd.merge(desired_mapping, hcp_mapping, on='region_name', how = 'left', suffixes = ['_desired', '_hcp'])
        self.lookup_table = merged # relevant columns are region_name, region_index_desired, region_index_hcp

        # everything starts out mapping to class index 0 (Unknown) or name label "None"
        self.hcp_to_comp_d = dict.fromkeys(hcp_mapping['region_index'].values, 0)
        self.hcp_to_name_d = dict.fromkeys(hcp_mapping['region_index'].values, 'None')
        self.comp_to_hcp_d = dict.fromkeys(merged['region_index_hcp'].values, 0)
        self.comp_to_name_d = dict.fromkeys(merged['region_index_hcp'].values, 'None')
        #print(merged)
        # then go through and fill the regions that were desired and update the dictionary for these
        for index, row in merged.iterrows():
            region_name, region_index_desired, region_index_hcp = row
            self.hcp_to_name_d[region_index_hcp] = region_name
            self.hcp_to_comp_d[region_index_hcp] = region_index_desired
            self.comp_to_name_d[region_index_desired] = region_name
            self.comp_to_hcp_d[region_index_desired] = region_index_hcp

        # the dictionaries are now ready to be mapped across the index arrays
        return

    def hcp_to_compressed(self, index_array: np.array) -> np.array:
        return np.vectorize(self.hcp_to_comp_d.get)(index_array)
    def compressed_to_hcp(self, index_array: np.array) -> np.array:
        return np.vectorize(self.comp_to_hcp_d.get)(index_array)
    def hcp_to_name(self, index_array: np.array) -> np.array:
        return np.vectorize(self.hcp_to_name_d.get)(index_array)
    def compressed_to_name(self, index_array: np.array) -> np.array:
        return np.vectorize(self.comp_to_name_d.get)(index_array)
        