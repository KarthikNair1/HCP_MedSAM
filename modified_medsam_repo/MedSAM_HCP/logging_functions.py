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
from segment_anything import sam_model_registry
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import nibabel as nib
from typing import Callable
from .MedSAM import *
from .dataset import *
import wandb

def init_wandb(args):
    wandb.login()
    wandb.init(project=args.task_name, 
        name = args.wandb_run_name,
        dir = args.wandb_dir,
        config={"lr": args.lr, 
                "batch_size": args.batch_size,
                "data_path": args.data_frame_path,
                "splits_path": args.train_test_splits,
                "model_type": args.model_type,
                "as_one_hot": args.as_one_hot,
                "loss_switching": args.loss_switching,
                "loss_reweighted": args.loss_reweighted,
                "keep_zero_weight": args.keep_zero_weight,
                "lambda_dice": args.lambda_dice,
                "label": args.label_id
                })
    
    # Define metrics for wandb
    wandb.define_metric('num_training_samples')
    wandb.define_metric('epoch')
    wandb.define_metric('train_step_loss', step_metric='num_training_samples')
    wandb.define_metric('train_epoch_loss', step_metric='num_training_samples')
    wandb.define_metric('val_epoch_loss', step_metric='num_training_samples')
    wandb.define_metric('label_1/*', step_metric='num_training_samples')
    wandb.define_metric('val_dice_scores/*', step_metric='num_training_samples', summary='max')

def print_cuda_memory(gpu):
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)

    print(f'[GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb')
    print(f'[GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb')

def log_losses_step(loss, class_losses, dice_class_losses, ce_class_losses, medsam_pred, total_number_of_training_examples_seen, args):
    # logs losses, class losses (if applicable), and the sum of predictions for a single class

    wandb.log({"train_step_loss": loss.item(),
                'num_training_samples': total_number_of_training_examples_seen})

    # log loss on label 1 (or just the only label if single-task)
    label_idx = 1 if args.label_id is None else 0
    if class_losses is not None:
        wandb.log({"label_1/label1_loss": class_losses[label_idx].item()})
        wandb.log({"label_1/label1_DICE_loss": dice_class_losses[label_idx].item()})
        wandb.log({"label_1/label1_CE_loss": ce_class_losses[label_idx].item()})
    
    # print sum predictions per class
    if args.as_one_hot:
        class_1_sum = (torch.sigmoid(medsam_pred) > 0.5).int().sum(dim=(0,2,3))[label_idx].item() # C
    else:
        class_1_sum = (torch.argmax(medsam_pred, dim=1) == 1).int().sum().item()

    wandb.log({"label_1/label1_sum": class_1_sum})

def log_predicted_probabilities():
    preds = logits_to_pred_probs(medsam_pred, args.as_one_hot) # (B,C,H,W) of predicted probabilities
    fig, _ = plot_prediction_distribution(preds[:,label_idx,:,:]) # filter for probs for label_idx only
    wandb.log({"label_1/label1_avg_prediction": wandb.Image(fig)})
    plt.close()    


def log_class_losses_as_barplots(fig_class_loss, fig_top_worst):
    # plot losses for each class as barplot
    wandb.log({f'class_loss_training_barplot': wandb.Image(fig_class_loss)})

    # plot losses for the best and worst 5 classes
    wandb.log({f'top5_worst5_losses_training_barplot': wandb.Image(fig_top_worst)})

    # close any plots
    plt.close()