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

def load_data_from_label_and_index(model_type, label, label_index, explicit_dataframe_path, explicit_yolo_bbox_dataframe_path, mri_id) -> pd.DataFrame:
    if model_type in ['singletask_unet', 'singletask_unprompted']:
        df = pd.read_csv(explicit_dataframe_path, dtype={'id': str})
        df = df[df['id'] == mri_id].reset_index(drop=True)
    elif model_type in ['singletask_yolov7_prompted', 'singletask_oracle']:

        # load dataframe containing bboxes
        if model_type == 'singletask_yolov7_prompted':
            this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/yolov10/100/{label}/path_df_only_with_bbox_yolov10.csv'
        elif model_type == 'singletask_oracle':
            if args.foundation_model_type == 'SAM':
                this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/{label}/0.1/isolated_path_df_SAM_bboxes_from_ground_truth.csv'
            elif args.foundation_model_type == 'MedSAM':
                this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/{label}/0.1/isolated_path_df_bboxes_from_ground_truth.csv'
                
            print(f'Using data path: {this_path}')
        if explicit_yolo_bbox_dataframe_path is not None:
            this_path = explicit_yolo_bbox_dataframe_path
        df_boxes = pd.read_csv(this_path, dtype={'id': str})
        df_boxes = df_boxes[df_boxes['id'] == mri_id].reset_index(drop=True) # subset for just this MRI

        # load dataframe containing all samples
        df_all_samples = pd.read_csv(args.explicit_dataframe_path, dtype={'id': str})
        df_all_samples = df_all_samples[df_all_samples['id'] == mri_id].reset_index(drop=True) # subset for just this MRI
        df_all_samples = df_all_samples.drop(columns = ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3'])

        # merge sample df with bbox df
        df = df_all_samples.merge(df_boxes, how='left', on=['id','slice','image_embedding_slice_path', 'segmentation_slice_path', 'image_path'])

    return df


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
        
def get_path_using_label_and_index(paths, label, label_index):
    # given a list of paths, return the element that contains the label at the specific label_index when splitting the path by '/'
    if len(paths) == 1 and label_index == 999:
        # label_index set to 999 represents no label in the filename, i.e. a single model taking care of all labels
        # useful for segmentation using un-finetuned model
        # thus, just return the only path
        return paths[0]

    print("Path pattern to search: {}")
    for path in paths:
        splits = path.split('/')
        if splits[label_index] == str(label):
            return path
    
    raise ValueError(f'Could not find a path with label {label} at index {label_index}')

# helper functions for argparse
def none_or_str(value):
    if value == 'None':
        return None
    return str(value)
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices = ['singletask_unprompted', 'multitask_unprompted',
                                                          'singletask_medsam_prompted', 'singletask_yolov7_prompted',
                                                          'singletask_yolov7_longer_prompted', 'pooltask_yolov7_prompted',
                                                          'singletask_unet', 'singletask_oracle'])
parser.add_argument("--mri_id", type=str, default=None)

parser.add_argument("--explicit_model_path", type=str, default=None)
parser.add_argument("--explicit_model_path_label_index", type=int, default=None)
parser.add_argument("--foundation_model_type", type=str, choices = ['SAM', 'MedSAM'], default='MedSAM')
parser.add_argument("--explicit_dataframe_path", type=str, default=None)
parser.add_argument("--explicit_yolo_bbox_dataframe_path", type=none_or_str, default=None, help = 'Used only when the model_type involves a yolo model. This supplies the bboxes for samples where yolo detected the class. For samples where yolo did not detect the class, the bbox values are merged with all samples and result in NAs at these locations.')
parser.add_argument("--single_label", type=none_or_int, default=None, help = 'Label to use if only one label should be outputted in the segmentation')
#parser.add_argument("--label", type=int) # only relevant for singletask models?
#parser.add_argument("--tag", type=str, choices = ['val', 'test'])
#parser.add_argument('-train_test_splits', type=str,
#                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
#                    help='path to pickle file containing a dictionary with train, val, and test IDs')
parser.add_argument('--batch_size', type=int, default = 16)
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
parser.add_argument('--output_dir', type=str, default = None)
args = parser.parse_args()

# assertions
assert args.explicit_dataframe_path is not None # must provide a dataframe
assert args.explicit_model_path is not None # must provide a model pattern
assert args.explicit_model_path_label_index is not None # must provide a label index for the model path
assert args.mri_id is not None # must provide an MRI ID
assert args.output_dir is not None # must provide an output directory

assert torch.cuda.is_available()
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



# load mapping dataframes and label converter
df_hcp = pd.read_csv(args.df_starting_mapping_path)
df_desired = pd.read_csv(args.df_desired_path)
label_converter = LabelConverter(df_hcp, df_desired)

# load dataframe

    


# iterate on labels if single-task
total_combine = None
labels_to_iter = range(1,103)
if args.single_label is not None:
    labels_to_iter = [args.single_label]

if args.model_type == 'singletask_unet':

    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    segs_for_each_label = []
    for label in tqdm(labels_to_iter):

        # load dataframe
        df = load_data_from_label_and_index(model_type=args.model_type, 
            label=label,
            label_index=args.explicit_model_path_label_index, 
            explicit_dataframe_path=args.explicit_dataframe_path,
            explicit_yolo_bbox_dataframe_path=args.explicit_yolo_bbox_dataframe_path, 
            mri_id=args.mri_id
        )
        print(f'After filtering for ID, on label {label}: {df.shape[0]} slices')

        print(f'Path pattern for searching: {args.explicit_model_path}')
        model_path = get_path_using_label_and_index(glob(args.explicit_model_path), label, args.explicit_model_path_label_index)
        print(f'For label {label}, Model path is {model_path}')

        model = load_model(args.model_type, model_path, num_classes=1)
        dataset = MRIDataset_Imgs(df, label_id = label, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=2, as_one_hot=True, pool_labels=False, preprocess_fn=preprocess_input)
        res = run_model_over_dataset_for_segmentations(model, dataset, args.model_type)
        segs_for_each_label.append(res)
    total_combine = np.concatenate(segs_for_each_label, axis = 1)
elif args.model_type in ['singletask_unprompted', 'singletask_yolov7_prompted', 'singletask_oracle']:
    segs_for_each_label = []
    for label in tqdm(labels_to_iter):

        # load dataframe
        df = load_data_from_label_and_index(model_type=args.model_type, 
            label=label,
            label_index=args.explicit_model_path_label_index, 
            explicit_dataframe_path=args.explicit_dataframe_path,
            explicit_yolo_bbox_dataframe_path=args.explicit_yolo_bbox_dataframe_path, 
            mri_id=args.mri_id
        )
        print(f'After filtering for ID, on label {label}: {df.shape[0]} slices')

        model_path = get_path_using_label_and_index(glob(args.explicit_model_path), label, args.explicit_model_path_label_index)
        print(f'For label {label}, Model path is {model_path}')

        model = load_model(args.model_type, model_path, num_classes=1)
        dataset = MRIDataset(df, label_id = label, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=2, as_one_hot=True, pool_labels=False)
        res = run_model_over_dataset_for_segmentations(model, dataset, args.model_type)
        segs_for_each_label.append(res)
    total_combine = np.concatenate(segs_for_each_label, axis = 1)
elif args.model_type == 'pooltask_yolov7_prompted':
    raise NotImplementedError
else:
    raise NotImplementedError

# prep save path
save_path = os.path.join(args.output_dir, str(args.mri_id))
if not os.path.exists(save_path):
    os.makedirs(save_path)

# total_combine now has size (B,102,H,W) where B should be the number of rows in args.df_starting_mapping_path
# add back the "background" class to make it (B, 103, H, W)
bg_prediction = 1 - np.max(total_combine, axis = 1, keepdims = True) # will be 1 if no other class is predicted, and 0 if another class is predicted. Size is now (B, 1, H, W)
total_combine = np.concatenate((bg_prediction, total_combine), axis = 1)

# if only single label specified, just grab the binary (256,256,256) for that label
if args.single_label is not None: # special handling for this case
    total_combine_label_idxs = total_combine[:, 1, :, :] * args.single_label # replace 1's with the actual class number
    np.save(os.path.join(save_path, f'singletask_seg_{args.single_label}.npy'), total_combine_label_idxs)
else:
    # if all labels are desired, return the (B,103,H,W) 
    print('using all labels')
    total_combine = total_combine[:, :, :, :]
    np.save(os.path.join(save_path, f'singletask_seg_all.npy'), total_combine)
    # total_combine_label_idxs = np.argmax(total_combine, axis = 1) # now (B,H,W) <- dont do this


#np.save(os.path.join(save_path, 'singletask_seg_all.npy'), total_combine_label_idxs)