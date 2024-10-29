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
import wandb
import re
from adjustText import adjust_text
import seaborn as sns
import nibabel as nib
import scipy
import statannot
import argparse
import sys
#print(glob('../../*'))
sys.path.append('./modified_medsam_repo')
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.dataset import MRIDataset, MRIDatasetForPooled, MRIDataset_Imgs, load_datasets, LabelConverter
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def load_model(model_type, model_path, num_classes):
    result = torch.load(model_path)
    try:
        if 'model' in result.keys():
            splits = model_path.split('/')
            new_path = os.path.join('/'.join(splits[:-1]), f'{splits[-1].split(".pth")[0]}_sam_readable.pth')
            print(f'model path converted to sam readable format and saved to {new_path}')

            result = result['model']

            # now remove the "module." prefix
            result_dict = {}
            for k,v in result.items():
                key_splits = k.split('.')
                assert key_splits[0] == 'module'
                new_k = '.'.join(key_splits[1:])
                result_dict[new_k] = v

            torch.save(result_dict, new_path)
            model_path = new_path

    except (AttributeError):
        # already in the correct format
        print('model path in readable format already')

    if model_type == 'multitask_unprompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path).to('cuda')
    elif model_type == 'pooltask_yolov7_prompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path).to('cuda')
    elif model_type == 'singletask_unet':
        model = torch.load(model_path)
    else:
        # singletask model
        model = build_sam_vit_b_multiclass(3, checkpoint=model_path).to('cuda')

    model.eval()
    return model

def run_model_over_dataset_for_segmentations(model, dataset, model_type):
    batch_sz = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size = batch_sz,
        shuffle = False,
        num_workers = 0,
        pin_memory = True
    )
    num_classes = 1
    listo = []
    for step, tup in enumerate(tqdm(dataloader)):
        if isinstance(dataset, MRIDatasetForPooled):
            image_embedding, gt2D, boxes, slice_names, label_nums = tup
        elif isinstance(dataset, MRIDataset_Imgs):
            image_embedding, gt2D = tup # "image_embedding" here is really just the tensor of the raw image since unet does not do pre-embedding
        else:
            image_embedding, gt2D, boxes, slice_names = tup
        image_embedding, gt2D = image_embedding.cuda(), gt2D.cuda()
        if model_type == 'singletask_unet':
            pred = model(image_embedding).cuda()
            pred = (pred > 0.5).to(torch.uint8)
        else:
            boxes = boxes.cuda()
            pred = torch.as_tensor(
                medsam_inference(model, image_embedding, boxes, 256, 256, as_one_hot=True,
                model_trained_on_multi_label=(model_type=='multitask_unprompted'), num_classes = num_classes),
                dtype=torch.uint8
            ).cuda()
        
        if model_type == 'multitask_unprompted':
            assert len(pred.shape) == 4 and pred.shape[1] == 103 # (B,C,H,W)
            assert len(gt2D.shape) == 4 and gt2D.shape[1] == 103
        else:
            assert len(pred.shape) == 4 and pred.shape[1] == 1 # (B, C, H, W)
            assert len(gt2D.shape) == 4 and gt2D.shape[1] == 1
        
        pred = pred.cpu().detach().numpy()
        listo.append(pred)
    
    total_combine = np.concatenate(listo, axis=0) # (B, C, H, W)
    return total_combine
        

# steps:

# 1. Load a dataframe for the MRI we are interested in segmenting
#   - We should be specifying an MRI ID as input
#   - Then, given the model type, we can pick the appropriate dataframe
#   - We should also input in the model path
#   - As well as the model type
#   - Then, we make a dataset and dataloader from the MRI
#   - We run the model on the dataloader
#   - Then we collate the outputs into a 256x256x256 array
#   - And then save it somewhere

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices = ['singletask_unprompted', 'multitask_unprompted',
                                                          'singletask_medsam_prompted', 'singletask_yolov7_prompted',
                                                          'singletask_yolov7_longer_prompted', 'pooltask_yolov7_prompted',
                                                          'singletask_unet'])
parser.add_argument("--mri_id", type=str, default=None)

#parser.add_argument("--explicit_model_path", type=str, default=None)
parser.add_argument("--explicit_dataframe_path", type=str, default=None)
#parser.add_argument("--label", type=int) # only relevant for singletask models?
#parser.add_argument("--tag", type=str, choices = ['val', 'test'])
#parser.add_argument('-train_test_splits', type=str,
#                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
#                    help='path to pickle file containing a dictionary with train, val, and test IDs')
parser.add_argument('--batch_size', type=int, default = 16)
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
parser.add_argument('--output_dir', type=str, default = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/eval_results_test_10-13-23')
args = parser.parse_args()

# load mapping dataframes and label converter
df_hcp = pd.read_csv(args.df_starting_mapping_path)
df_desired = pd.read_csv(args.df_desired_path)
label_converter = LabelConverter(df_hcp, df_desired)

# load dataframe
df = pd.read_csv(args.explicit_dataframe_path, dtype={'id': str})
df = df[df['id'] == args.mri_id].reset_index(drop=True)

print(f'After filtering for ID: {df.shape[0]} slices')

# iterate on labels if single-task
total_combine = None
if args.model_type == 'singletask_unet':

    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    segs_for_each_label = []
    for label in tqdm(range(1, 103)):
        # prep the model and dataset for this
        model_path = glob(f'/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_all_labels_9-9-24/training/{label}/*-best_model.pth')[0]
        model = load_model(args.model_type, model_path, num_classes=1)
        dataset = MRIDataset_Imgs(df, label_id = label, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=2, as_one_hot=True, pool_labels=False, preprocess_fn=preprocess_input)
        res = run_model_over_dataset_for_segmentations(model, dataset, args.model_type)
        segs_for_each_label.append(res)
    total_combine = np.concatenate(segs_for_each_label, axis = 1)
elif args.model_type == 'singletask_unprompted':
    segs_for_each_label = []
    for label in tqdm(range(1, 103)):
        # prep the model and dataset for this
        model_path = glob(f'/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_retrain_9-12-24/training/{label}/0.001/*/medsam_model_best_sam_readable.pth')[0]
        model = load_model(args.model_type, model_path, num_classes=1)
        dataset = MRIDataset(df, label_id = label, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=2, as_one_hot=True, pool_labels=False)
        res = run_model_over_dataset_for_segmentations(model, dataset, args.model_type)
        segs_for_each_label.append(res)
    total_combine = np.concatenate(segs_for_each_label, axis = 1)
elif args.model_type == 'singletask_yolov7_prompted':
    raise NotImplementedError
elif args.model_type == 'pooltask_yolov7_prompted':
    raise NotImplementedError
else:
    raise NotImplementedError

# total_combine now has size (B,102,H,W) where B should be the number of rows in args.df_starting_mapping_path
# add back the "background" class to make it (B, 103, H, W)
bg_prediction = 1 - np.max(total_combine, axis = 1, keepdims = True) # will be 1 if no other class is predicted, and 0 if another class is predicted. Size is now (B, 1, H, W)
total_combine = np.concatenate((bg_prediction, total_combine), axis = 1)

# now, we should convert this to (B,H,W) of label indices
total_combine_label_idxs = np.argmax(total_combine, axis = 1) # now (B,H,W)

save_path = os.path.join(args.output_dir, str(args.mri_id))
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.save(os.path.join(save_path, 'singletask_seg_all.npy'), total_combine_label_idxs)