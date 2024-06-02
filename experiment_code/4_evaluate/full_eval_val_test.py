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
import scipy
import statannot
import argparse
import sys
sys.path.append('./modified_medsam_repo')
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.dataset import *

# analyzes the following models:
# singletask unprompted
# multitask unprompted
# singletask medsam-prompted
# singletask yolov7-prompted
# singletask yolov7-longer-prompted
# pooltask yolov7-prompted

# for each model: compute its predictions on the validation and the test set
# the following metrics are calculated for each class:

# sensitivity dice: of all the slices where this class is present in at least one pixel, what is the mean dice score?
# specificity dice: of all slices where this class is not present, what is our mean dice score (this can only be 0 or 1)
# normal dice: of all slices, what is the mean dice score (if the class is not present, this can only be 0 or 1)

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
        print('model path in sam readable format already')

    if model_type == 'multitask_unprompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path).to('cuda')
    elif model_type == 'pooltask_yolov7_prompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path).to('cuda')
    else:
        # singletask model
        model = build_sam_vit_b_multiclass(3, checkpoint=model_path).to('cuda')



    model.eval()
    return model
def load_model_from_label_and_type(model_type, label):
    assert model_type in ['singletask_unprompted', 'multitask_unprompted',
                'singletask_medsam_prompted', 'singletask_yolov7_prompted',
                'singletask_yolov7_longer_prompted', 'pooltask_yolov7_prompted']
    
    if model_type == 'singletask_unprompted':
        raise NotImplementedError
    elif model_type == 'multitask_unprompted':
        model_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/ce_only_resume_training_from_checkpoint_8-9-23/MedSAM_finetune_hcp_ya_constant_bbox_all_tasks-20230810-115803/medsam_model_best.pth'
        num_classes = 103
    elif model_type == 'singletask_medsam_prompted':
        raise NotImplementedError
    elif model_type == 'singletask_yolov7_prompted':
        model_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/second_round_w_bbox_yolov7_finetunes_longer_8-17-23/label{label}/*/medsam_model_best.pth'
        listo = glob(model_path)
        assert len(listo) == 1
        model_path = listo[0]
        num_classes = 3 # note we have to pass in 3 so that we get the singletask sam model, which predicts 3 masks, even though the more accurate number would be 2
    elif model_type == 'singletask_yolov7_longer_prompted':
        model_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/second_round_w_bbox_yolov7_finetunes_60epochs_8-20-23/label{label}/*/medsam_model_best.pth'
        listo = glob(model_path)
        assert len(listo) == 1
        model_path = listo[0]
        num_classes = 3
    elif model_type == 'pooltask_yolov7_prompted':
        model_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/pooled_labels_ckpt_continue_8-22-23/model_best_20230822-115028.pth'
        num_classes = 103 # have to pass in 103 here unfortunately because this model was accidentally trained to output 103 masks, even though only the first one is actually used and loss-propagated through

    return load_model(model_type, model_path, num_classes)
def load_data_from_label_and_type(model_type, label, tag = 'val', args):
    df_hcp = pd.read_csv(args.df_starting_mapping_path)
    if model_type in ['multitask_unprompted', 'pooltask_yolov7_prompted']:
        df_desired = pd.read_csv(args.df_desired_path)
    else:
        df_desired = pd.read_csv(f'/gpfs/home/kn2347/MedSAM/class_mappings/label{label}_only_name_class_mapping.csv')
    NUM_CLASSES = len(df_desired)
    label_converter = LabelConverter(df_hcp, df_desired)

    # train val test split
    train_test_splits_path = args.train_test_splits
    dicto = pickle.load(open(train_test_splits_path, 'rb'))
    ids = dicto[tag] # selects val or test ids, this should be a list

    if args.world_size is not None:
        assert args.node_rank is not None

        total_len = len(ids)
        my_start = int((args.node_rank / args.world_size) * total_len)
        my_end = int(((args.node_rank + 1) / args.world_size) * total_len)
        if args.node_rank == args.world_size - 1:
            assert my_end == total_len
        
        ids = ids[my_start:my_end]
        print(f'only operating on indices {my_start} to {my_end - 1} inclusive')
    
    df = None
    label_id = None
    pool_labels = None


    # now, load data
    if model_type in ['multitask_unprompted']:
        # multi task
        df = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv')
        label_id = None
        pool_labels = False

    elif model_type in ['pooltask_yolov7_prompted']:
        # pool task
        df_all_samples = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv')
        df_all_samples = df_all_samples[df_all_samples['id'].isin(ids)].reset_index(drop=True)

        # we should replicate all rows with the column label_number ranging from 1...102
        labels_nums = list(range(1,103))
        df_all_samples = pd.concat([df_all_samples]*len(labels_nums),keys = labels_nums, names = ['label_number']).reset_index(level=0)
        if tag == 'val':
            df_box_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/yolov7/path_df_pooled_labels_only_with_bbox_yolov7.csv'
        elif tag == 'test':
            df_box_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/yolov7/test/path_df_pooled_labels_only_with_bbox_yolov7_TEST.csv'

        


        df_boxes = pd.read_csv(df_box_path,
                               index_col=0)
        df_boxes = df_boxes[df_boxes['id'].isin(ids)].reset_index(drop=True)
        
        df_all_samples = df_all_samples.drop(columns = ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3'])
        df = df_all_samples.merge(df_boxes, how='left', on=['id','slice','image_embedding_slice_path', 'segmentation_slice_path', 'image_path', 'label_number'])
        label_id = 1
        pool_labels = True

    elif model_type in ['singletask_unprompted']:
        # single task unprompted
        df = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv')
        label_id = label
        pool_labels = False

    elif model_type in ['singletask_medsam_prompted', 'singletask_yolov7_prompted',
                'singletask_yolov7_longer_prompted']:
        # single task prompted
        if model_type == 'singletask_medsam_prompted':
            this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/medsam/path_df_label{label}_only_with_bbox.csv'
        elif model_type in ['singletask_yolov7_prompted', 'singletask_yolov7_longer_prompted']:
            if tag == 'val':
                this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/yolov7/path_df_label{label}_only_with_bbox_yolov7.csv'
            elif tag == 'test':
                this_path = f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/yolov7/test/path_df_label{label}_only_with_bbox_yolov7_TEST.csv'
        
        df_all_samples = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv')
        df_all_samples = df_all_samples[df_all_samples['id'].isin(ids)].reset_index(drop=True)

        df_boxes = pd.read_csv(this_path)
        df_boxes = df_boxes[df_boxes['id'].isin(ids)].reset_index(drop=True)

        df_all_samples = df_all_samples.drop(columns = ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3'])
        df = df_all_samples.merge(df_boxes, how='left', on=['id','slice','image_embedding_slice_path', 'segmentation_slice_path', 'image_path'])

        label_id = 1
        pool_labels = False

    
    df = df[df['id'].isin(ids)].reset_index(drop=True)

    if model_type =='pooltask_yolov7_prompted':
        dataset = MRIDatasetForPooled(df, label_id = label_id, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=True, pool_labels=pool_labels)
    else:
        dataset = MRIDataset(df, label_id = label_id, bbox_shift=0, label_converter = label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=True, pool_labels=pool_labels)
    
    return df_hcp, df_desired, NUM_CLASSES, label_converter, dataset

def run_model_over_dataset(model, dataset, model_type):
    batch_sz = 16   
    dataloader = DataLoader(
        dataset,
        batch_size = batch_sz,
        shuffle = False,
        num_workers = 0,
        pin_memory = True
    )
    num_classes = 1
    if model_type=='multitask_unprompted':
        num_classes = 103
    collector = {'dice_sensitivity':[], 'dice_specificity':[], 'overall_dice':[], 'label_numbers':[]}
    for step, tup in enumerate(tqdm(dataloader)):
        if isinstance(dataset, MRIDatasetForPooled):
            image_embedding, gt2D, boxes, slice_names, label_nums = tup
        else:
            image_embedding, gt2D, boxes, slice_names = tup
        
        image_embedding, gt2D, boxes = image_embedding.cuda(), gt2D.cuda(), boxes.cuda()
        medsam_pred = torch.as_tensor(
            medsam_inference(model, image_embedding, boxes, 256, 256, as_one_hot=True,
            model_trained_on_multi_label=(model_type=='multitask_unprompted'), num_classes = num_classes),
            dtype=torch.uint8
        ).cuda()


        if model_type == 'multitask_unprompted':
            assert len(medsam_pred.shape) == 4 and medsam_pred.shape[1] == 103
            assert len(gt2D.shape) == 4 and gt2D.shape[1] == 103
        else:
            assert len(medsam_pred.shape) == 4 and medsam_pred.shape[1] == 1
            assert len(gt2D.shape) == 4 and gt2D.shape[1] == 1
        
        dices_no_mask = dice_scores_multi_class(medsam_pred, gt2D, eps=1e-6, mask_empty_class_images_with_nan = False)
        collector['dice_sensitivity'].append(dice_scores_multi_class(medsam_pred, gt2D, eps=1e-6, mask_empty_class_images_with_nan = True))
        collector['overall_dice'].append(dices_no_mask)
        B, classes, H, W = gt2D.shape
        gt2D_flattened = gt2D.view(B, classes, -1)
        is_negative_examples = (gt2D_flattened == 0).all(dim=2) # size (B,C)


        r, c = torch.where(is_negative_examples)
        r = r.cpu()
        c = c.cpu()
        res = torch.full((B, classes), torch.nan)
        res[r, c] = dices_no_mask[r, c]
        collector['dice_specificity'].append(res)

        if model_type == 'pooltask_yolov7_prompted':
            collector['label_numbers'].append(label_nums)
        #xd = collector['dice_specificity'][-1]
    
    if model_type != 'pooltask_yolov7_prompted':
        collector['overall_dice'] = torch.concat(collector['overall_dice']).nanmean(dim=0)
        collector['dice_sensitivity'] = torch.concat(collector['dice_sensitivity']).nanmean(dim=0)
        collector['dice_specificity'] = torch.concat(collector['dice_specificity']).nanmean(dim=0)
    else:
        collector['overall_dice'] = torch.concat(collector['overall_dice'])
        collector['dice_sensitivity'] = torch.concat(collector['dice_sensitivity'])
        collector['dice_specificity'] = torch.concat(collector['dice_specificity'])
        collector['label_numbers'] = torch.concat(collector['label_numbers'])
    return collector


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices = ['singletask_unprompted', 'multitask_unprompted',
                                                          'singletask_medsam_prompted', 'singletask_yolov7_prompted',
                                                          'singletask_yolov7_longer_prompted', 'pooltask_yolov7_prompted'])
parser.add_argument("--label", type=str) # only relevant for singletask models?
parser.add_argument("--tag", type=str, choices = ['val', 'test'])
parser.add_argument('-train_test_splits', type=str,
                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
                    help='path to pickle file containing a dictionary with train, val, and test IDs')
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
parser.add_argument("--world_size", type=int, default=None)
parser.add_argument("--node_rank", type=int, default=None)
parser.add_argument('--output_dir', type=str, default = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/eval_results_test_10-13-23')
args = parser.parse_args()

df_hcp = pd.read_csv(args.df_starting_mapping_path)
df_desired = pd.read_csv(args.df_desired_path)
NUM_CLASSES = len(df_desired)
label_converter = LabelConverter(df_hcp, df_desired)

model_type = args.model_type
label = args.label
tag = args.tag
df_hcp, df_desired, num_classes, label_converter, dataset = load_data_from_label_and_type(model_type, label, tag = tag)
model = load_model_from_label_and_type(model_type, label)
collector = run_model_over_dataset(model, dataset, model_type)

if args.world_size is not None: # make sure to save with node number included
    file_name = os.path.join(args.output_dir, f'eval_{model_type}_{tag}_label{label}_node{args.node_rank}.pkl')
else:
    file_name = os.path.join(args.output_dir, f'eval_{model_type}_{tag}_label{label}.pkl')
with open(file_name, 'wb') as fp:
    pickle.dump(collector, fp)
    print('done!')