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
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
import nibabel as nib
import pickle
import time
from PIL import Image
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import sam_model_registry
from MedSAM_HCP.dataset import MRIDataset_Imgs, load_datasets
from MedSAM_HCP.MedSAM import MedSAM
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.base import Loss


import wandb

class dice_ce_loss(Loss):
    def __init__(self, lambda_dice = 1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_dice = lambda_dice

    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W)
        # y_true: (B, C, H, W)

        # assert that these are all probabilities post-sigmoid
        assert torch.all(y_pred >= 0) and torch.all(y_pred <= 1)

        ce_loss_fn = nn.BCELoss(reduction='mean')
        dice_loss = monai.losses.DiceLoss(include_background = True, sigmoid=False, squared_pred=True, reduction='mean',
            batch = True)
        return dice_loss(y_pred, y_true) * self.lambda_dice + ce_loss_fn(y_pred, y_true.float()) * (1-self.lambda_dice)

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_frame_path', type=str,
                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv',
                    help='path to pandas dataframe with all paths for training')
parser.add_argument('-train_test_splits', type=str,
                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
                    help='path to pickle file containing a dictionary with train, val, and test IDs')
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
parser.add_argument('-label_id', type=int, default=1,
                    help='Label number for training')
parser.add_argument('-num_classes', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-num_workers', type=int, default=2)
parser.add_argument('-lr', type=float, default=1e-4 * 32)
parser.add_argument('-epochs', type=int, default=40)
parser.add_argument('-freeze_encoder', type=bool, default=True)
parser.add_argument('-project_name', type=str, default='test')
parser.add_argument('-wandb_run_name', type=str, default=None)
parser.add_argument('-work_dir', type=str, default='./work_dir')
parser.add_argument('--lambda_dice', type=float, default=0, help='What fraction of the total loss should the dice loss contribute to? (default: 0.0)')
parser.add_argument('--log_val_every', type=int, default=None)

parser.add_argument('--early_stop_delta', type=float, default=None,
                    help='early stopping delta, as a percent of the prior loss (e.g. at least 1 percent improvement in loss each epoch')

parser.add_argument('--early_stop_patience', type=int, default=None,
                    help='number of epochs without sufficient delta until exiting')

#parser.add_argument('-output_folder_name', type=str, default=None)

args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = join(args.work_dir, args.project_name + '-' + args.wandb_run_name + '-' + run_id)


label_id = args.label_id
num_classes = args.num_classes
activation = 'sigmoid'
batch_sz = args.batch_size
num_workers = args.num_workers
lr = args.lr
freeze_encoder = args.freeze_encoder

def init_wandb():
    wandb.login()
    wandb.init(project = args.project_name, 
                dir = '/gpfs/home/kn2347/wandb',
                name = args.wandb_run_name + '-' + run_id,
                config = {
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'label_id': args.label_id,
                    'freeze_encoder': args.freeze_encoder
                })

    wandb.define_metric('epoch')
    wandb.define_metric('train_dice_loss', step_metric='epoch')
    wandb.define_metric('train_iou_score', step_metric='epoch')
    wandb.define_metric('train_dice_score', step_metric='epoch')
    wandb.define_metric('val_dice_loss', step_metric='epoch')
    wandb.define_metric('val_iou_score', step_metric='epoch')
    wandb.define_metric('val_dice_score', step_metric='epoch')

init_wandb()

df_hcp = pd.read_csv(args.df_starting_mapping_path)
df_desired = pd.read_csv(args.df_desired_path)
label_converter = LabelConverter(df_hcp, df_desired)

preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

train, val, test = load_datasets(
            args.data_frame_path,
            args.train_test_splits,
            label_id = label_id, bbox_shift=0, 
                sample_n_slices = None, label_converter=label_converter, NUM_CLASSES=num_classes+1, 
                as_one_hot=True, pool_labels=False, preprocess_fn = preprocess_input, dataset_type = MRIDataset_Imgs)

train_loader = DataLoader(train, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
valid_loader = DataLoader(val, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

model = smp.Unet(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=num_classes,            # model output channels (number of classes in your dataset)
    activation = activation
)

if freeze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

#loss = DiceLoss()
loss = dice_ce_loss(lambda_dice = args.lambda_dice)
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5)
]

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
     lr=lr)

# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device='cuda',
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device='cuda',
    verbose=True,
)

# Training loop:
max_score = 0

early_stop_ctr = 0
early_stop_prev_loss = 9e9
early_stop_cutoff = args.early_stop_patience

for i in range(0, args.epochs):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)

    if i == 25:
        optimizer.param_groups[0]['lr'] = lr / 10
        print('Decrease decoder learning rate 10-fold!')

    wandb.log({
        'epoch': i,
        'train_dice_loss': train_logs['dice_ce_loss'],
        'train_iou_score': train_logs['iou_score'],
        'train_dice_score': train_logs['fscore']
    })
    
    skip_val = args.log_val_every is not None and i % args.log_val_every != 0
    if skip_val:
        continue

    valid_logs = valid_epoch.run(valid_loader)

    wandb.log({
        'epoch': i,
        'val_dice_loss': valid_logs['dice_ce_loss'],
        'val_iou_score': valid_logs['iou_score'],
        'val_dice_score': valid_logs['fscore']
    })

    print(valid_logs)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, model_save_path + '-best_model.pth')
        print('Model saved!')

    early_stop_pct = (early_stop_prev_loss - valid_logs['dice_ce_loss']) / early_stop_prev_loss
    if args.early_stop_delta is not None and early_stop_pct < args.early_stop_delta:
        # if it doesn't decrease by at least 0.001, increment counter
        early_stop_ctr += 1
        if early_stop_ctr > early_stop_cutoff:
            # need to terminate
            print('Early stop criterion reached')
            break
    else:
        early_stop_ctr = 0
    early_stop_prev_loss = valid_logs['dice_ce_loss']


 