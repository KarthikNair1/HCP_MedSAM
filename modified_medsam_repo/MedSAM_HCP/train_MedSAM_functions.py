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
from datetime import datetime
import pandas as pd
import nibabel as nib

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, logits_to_pred_probs
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass, resume_model_optimizer_and_epoch_from_checkpoint, save_model_optimizer_and_epoch_to_checkpoint
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *
from MedSAM_HCP.logging_functions import init_wandb, print_cuda_memory, log_losses_step, log_predicted_probabilities, log_class_losses_as_barplots
import wandb

def retrieve_class_weights_tensor(NUM_CLASSES_FOR_LOSS, label_converter, args):
    if args.class_weights_dict_path is not None and args.loss_reweighted:
        # load from file, then populate tensor by mapping indices
        class_weights = pickle.load(open(args.class_weights_dict_path, 'rb'))
        class_weights_tensor = torch.zeros((NUM_CLASSES_FOR_LOSS)).cuda()
        for key in class_weights.keys():
            compressed_idx = label_converter.hcp_to_compressed(key)
            if compressed_idx == 0: # unknown class
                if args.keep_zero_weight:
                    if key == 0: # if this is the true unknown class and not just another class that didn't map properly
                        class_weights_tensor[compressed_idx] = class_weights[key]
                else:
                    class_weights_tensor[compressed_idx] = 0
            else:
                class_weights_tensor[compressed_idx] = class_weights[key]
    else:
        # weight each class equally
        class_weights_tensor = torch.ones((NUM_CLASSES_FOR_LOSS)).cuda() 
    
    return class_weights_tensor

def train_step(medsam_model, optimizer, scaler, loss_type, class_weights_tensor, lambda_dice, image_embedding, gt2D, boxes, args):

    optimizer.zero_grad()
    boxes_np = boxes.detach().cpu().numpy()
    image_embedding, gt2D = image_embedding.cuda(), gt2D.cuda()
    if args.use_amp:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            medsam_pred = medsam_model(image_embedding, boxes_np)
            loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = args.focal_loss_set_alpha)
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        medsam_pred = medsam_model(image_embedding, boxes_np)
        loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = args.focal_loss_set_alpha)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return loss, class_losses, dice_class_losses, ce_class_losses, medsam_pred

def validate_step(medsam_model, optimizer, scaler, loss_type, class_weights_tensor, lambda_dice, image_embedding, gt2D, boxes, args):
    with torch.no_grad():
        boxes_np = boxes.detach().cpu().numpy()
        image_embedding, gt2D = image_embedding.cuda(), gt2D.cuda()
        
        if args.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                medsam_pred = medsam_model(image_embedding, boxes_np)
                loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = args.focal_loss_set_alpha)

                # generate predictions for validation dice scores
                medsam_binary_predictions_as_onehot = torch.from_numpy(convert_logits_to_preds_onehot(medsam_pred, args.as_one_hot, H=256, W=256))
                dice_scores_multiclass = dice_scores_multi_class(medsam_binary_predictions_as_onehot, gt2D)
        else:
            medsam_pred = medsam_model(image_embedding, boxes_np)
            loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = args.focal_loss_set_alpha)

            # generate predictions for validation dice scores
            medsam_binary_predictions_as_onehot = torch.from_numpy(convert_logits_to_preds_onehot(medsam_pred, args.as_one_hot, H=256, W=256))
            dice_scores_multiclass = dice_scores_multi_class(medsam_binary_predictions_as_onehot, gt2D)
        
    return loss, class_losses, dice_class_losses, ce_class_losses, dice_scores_multiclass


def log_stuff_at_step(loss, class_losses, dice_class_losses, ce_class_losses, medsam_pred, 
                    total_number_of_training_examples_seen, medsam_model, val_dataset,
                    epoch, label_converter, args):
    # assumes this is the host process
    if not args.use_wandb:
        return

    # log losses
    log_losses_step(
        loss, class_losses, dice_class_losses, ce_class_losses, medsam_pred, total_number_of_training_examples_seen, args
    )

    label_idx = 1 if args.label_id is None else 0

    # plot distribution of predicted probabilities for the class 1
    #preds = logits_to_pred_probs(medsam_pred, args.as_one_hot) # (B,C,H,W) of predicted probabilities
    #fig, _ = plot_prediction_distribution(preds[:,label_idx,:,:]) # filter for probs for label_idx only
    #wandb.log({"label_1/label1_avg_prediction": wandb.Image(fig)})

    # plot class-wise losses as barplots if multitask
    if class_losses is not None and args.is_multitask: 
        log_class_losses_as_barplots(
            fig_class_loss = plot_losses_for_classes(class_losses[1:]),
            fig_top_worst = plot_worst_k_best_k(class_losses[1:], k = 5, label_converter = label_converter)
        )
    
    # plot 3 random examples from label 1 to see how the model is doing mid-epoch
    if not args.suppress_train_debug_imgs:
        fig, _, _ = plot_random_example(val_dataset, model_list=[medsam_model.module],
                        names_list=[f'Label: 1, Epoch: {epoch}'],
                        label_to_viz = label_idx,
                        n_ex = 3,
                        seed = 182,
                        as_one_hot = args.as_one_hot,
                        model_trained_on_multi_label = args.is_multitask
        )
        if fig is not None:
            wandb.log({f'train_images_debug_on_val/class_1': fig})
            plt.close()
        
    return

def log_stuff_at_epoch(train_epoch_loss, val_epoch_loss, val_class_losses, class_weights_tensor,
                        val_dice_scores, 
                        epoch, total_number_of_training_examples_seen,
                        medsam_model, val_dataset, 
                        NUM_CLASSES, NUM_CLASSES_FOR_LOSS,
                        label_converter, args):
    if not args.use_wandb:
        return

    wandb.log({"train_epoch_loss": train_epoch_loss,
                "num_training_samples": total_number_of_training_examples_seen, 
                "epoch":epoch}, commit=True)

    due_for_val_logging = not (args.log_val_every is not None and epoch % args.log_val_every!=0)
    if due_for_val_logging:
        wandb.log({"val_epoch_loss": val_epoch_loss}, commit = True)

    # log example segmentations on validation images
    if not args.fast_dev_run and args.log_val_every is None: # if not dev-running

        if args.is_multitask: # if multitask
            labels_to_track = list(range(NUM_CLASSES))
        else: # singletask
            labels_to_track = [0]
        
        print('plotting validation examples')
        for i, label in enumerate(tqdm(labels_to_track)):
            text_label = label_converter.compressed_to_name(label)
            fig, _, _ = plot_random_example(val_dataset, model_list=[medsam_model.module],
                                names_list=[f'Label: {text_label}, Epoch: {epoch}'],
                                label_to_viz = label,
                                n_ex = 3,
                                seed = 182,
                                as_one_hot=args.as_one_hot,
                                model_trained_on_multi_label = args.is_multitask)

            if fig is not None:
                wandb.log({f'val_images/class_{text_label}': fig})
                plt.close()
    
    # if multitask, plot class losses vs class weights
    if args.is_multitask: 
            fig, _ = plot_class_losses_vs_weights(val_class_losses, class_weights_tensor)
            wandb.log({'val_class_loss_vs_weights': wandb.Image(fig)})
            plt.close()

    # log validation dice scores
    if due_for_val_logging: # if this epoch is due for logging, perform the logging
        for class_num in range(NUM_CLASSES_FOR_LOSS):
            text_label = label_converter.compressed_to_name(class_num)
            wandb.log({f'val_dice_scores/class_{text_label}': val_dice_scores[class_num].item()})

    # for pooled model, plot random bbox outputs every 5 epochs
    if args.pool_labels and epoch % 5 == 0:
        
        fig, ax = plot_random_bboxes(val_dataset, medsam_model.module, dev='cuda', n_ex = 5, seed=182)
        if fig is not None:
            wandb.log({f'val_images/random_bbox': wandb.Image(fig)})
            plt.close()