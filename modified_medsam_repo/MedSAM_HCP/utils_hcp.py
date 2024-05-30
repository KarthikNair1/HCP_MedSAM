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

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax, color = 'blue'):
    # assumes xmin, ymin, xmax, ymax
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))    

def load_and_preprocess_slice(ds, img_idx, dev='cuda'):
    image_data = ds.load_image(img_idx)
    img_embedding, img_seg, img_box, slice_name = ds[img_idx]
    img_embedding, img_seg, img_box = img_embedding.to(dev), img_seg.to(dev), img_box.to(dev)

    # reshape for computation
    img_embedding = img_embedding[None, :, :, :]
    img_box = img_box.reshape((1,4))
    img_box_1024 = img_box * 4 # for ori model
    img_seg = img_seg.cpu().numpy()

    return image_data, img_embedding, img_box, img_box_1024, img_seg
def is_slice_blank(image_data):
    return np.abs(image_data).sum() == 0
def seg_get_class_indices(preds): # preds should be of shape (B, H, W) or (B, C, H, W)
    if not torch.is_tensor(preds):
        preds = torch.IntTensor(preds)
    assert not torch.is_floating_point(preds)  # must be class numbers

    if len(preds.shape) == 3:
        # this is 0's and 1's already, so good as is
        return preds[:, None, :, :]
    elif len(preds.shape) == 4:
        raise NotImplementedError

def dice_score_single_class(truth, pred, eps=1e-6):
    # B, 1, Rows, Columns for both truth and pred
    # values are 0 or 1: calculate overlap w.r.t. the 1
    B, classes, r, c = truth.size()
    assert classes == 1
    truth = truth.view(B, -1).cuda()
    pred = pred.view(B, -1).cuda()

    overlap_term = (truth * pred).sum() * 2 + eps
    union_term = (truth + pred).sum() + eps

    return (overlap_term / union_term).cpu().item()

def dice_scores_multi_class(pred, truth, eps=1e-6, mask_empty_class_images_with_nan = True):
    # B, C, H, W tensors
    # pred should be binarized already

    assert truth.dtype == torch.long
    assert pred.dtype == torch.uint8
    
    B, classes, H, W = truth.size()
    truth = truth.view(B, classes, -1).cuda()
    pred = pred.view(B, classes, -1).cuda()
    overlap_term = (truth * pred).sum(dim = 2) * 2 + eps # (B, C)
    
    union_term = (truth + pred).sum(dim = 2 ) + eps # (B, C)

    individual_dice_scores = overlap_term / union_term # (B, C)
    if mask_empty_class_images_with_nan:
        masker = (truth == 0).all(dim=2).float() # B, C of bool
        individual_dice_scores = torch.where(masker == 1, np.nan, individual_dice_scores) # B, C of nan or dice score
    
    
    return individual_dice_scores.cpu() # (B,C) tensor of type float


    
    #dice_scores_batch_averaged = torch.nanmean(dim=0)
    #assert dice_scores_batch_averaged.shape == torch.Size([classes])
    #return dice_scores_batch_averaged.cpu()

def plot_random_example(test_ds, model_list, names_list, label_to_viz, as_one_hot, dev='cuda', dest_H=256, dest_W=256, n_ex = 1, seed=182, restrict_to_label_present = True, debug=False,
                        model_trained_on_multi_label=True):

    np.random.seed(seed)
    # set up the plot
    fig, axs = plt.subplots(n_ex, len(model_list)+2, figsize=(25, 25), squeeze=False) # 0 index is ground truth segmentation
    
    for iter in range(n_ex):
        num_times_tried = 0
        #print(f'trying {label_to_viz}')
        while True:
            num_times_tried+=1
            if num_times_tried > 10:
                return None, None, None
            img_idx = np.random.randint(len(test_ds))

            # load image, embedding, segmentation, and input mask
            image_data, img_embedding, img_box, img_box_1024, img_seg = load_and_preprocess_slice(test_ds, img_idx, dev)
            C = img_seg.shape[0]
            if img_seg.shape == 2:
                # class-index based
                img_seg = (img_seg == label_to_viz).long() # (H,W) of binary numbers
            else:
                # one-hot-encoded based
                img_seg = img_seg[label_to_viz,:,:]

            # break if blank
            if is_slice_blank(image_data):
                continue
            
            seg_list = []

            # run inference
            for model in model_list:
                

                this_seg_result = medsam_inference(model, img_embedding, img_box, H=dest_H, W=dest_W, as_one_hot=as_one_hot,
                                                   model_trained_on_multi_label=model_trained_on_multi_label)
                this_seg_result = np.reshape(this_seg_result, (C, dest_H, dest_W)) #(classes, dest_H,dest_W))
                #if debug:
                    #print(f'seg shape: {this_seg_result.shape}')
                    #print(f'seg sum: {this_seg_result.sum()}')
                    #print(f'seg sums over classes: {this_seg_result.sum(axis=(1,2))}')
                    #print(f'{np.where(this_seg_result[2,:,:] == 1)}')
                this_seg_result = this_seg_result[label_to_viz,:,:] # (H, W) of 1s and 0s
                
                seg_list.append(this_seg_result)

            segmentation_mask_sum = sum([x.sum() for x in seg_list])
            total_sum = img_seg.sum() + segmentation_mask_sum
            if restrict_to_label_present and total_sum == 0: # neither ground truth nor segmentations have any predicted True's
                continue

            break
            
        # plot 
        
        image_data_3c = np.repeat(image_data[:,:,None], 3, axis=-1)

        axs[iter, 0].imshow(image_data_3c)
        if iter == 0:
            axs[iter, 0].set_title('Input Image', fontsize=20)
        axs[iter, 0].axis('off')

        axs[iter, 1].imshow(image_data_3c)
        
        show_mask(img_seg>0, axs[iter, 1])
        if iter == 0:
            axs[iter, 1].set_title('Ground Truth', fontsize=20)
        axs[iter, 1].axis('off')

        for i in range(len(model_list)):
            axs[iter, i+2].imshow(image_data_3c)
            show_mask(seg_list[i], axs[iter, i+2])
            show_box(img_box.flatten().cpu().numpy(), axs[iter, i+2])
            if iter==0:
                axs[iter, i+2].set_title(f'{names_list[i]} Segmentation', fontsize=20)
            axs[iter, i+2].axis('off')

    fig.show()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    return fig, axs, seg_list

def convert_medsam_checkpt_to_readable_for_sam(checkpt, to_save_dir = None):
    result = torch.load(checkpt)['model']

    # now remove t{he "module." prefix
    result_dict = {}
    for k,v in result.items():
        new_k = '.'.join(k.split('.')[1:])
        result_dict[new_k] = v
    if to_save_dir is not None:
        torch.save(result_dict, to_save_dir)
    return result_dict

def plot_losses_for_classes(class_loss_tensor):
    # class_loss_tensor has size (num_classes-1) (because we exclude 0'th class)
    
    num_classes = class_loss_tensor.shape[0]
    x_list = list(range(1, num_classes+1))
    y_list = class_loss_tensor.cpu().tolist()

    fig, ax = plt.subplots()
    ax.bar(x_list, y_list, log=True)
    ax.set_xlabel('Region Number (compressed)')
    ax.set_ylabel('Batch Loss')
    
    ax.set_ylim([1e-8, 1e3])

    return fig, ax

def plot_worst_k_best_k(class_loss_tensor, k = 5, label_converter = None):
    losses_npy = class_loss_tensor.cpu().detach().numpy()
    idx_sort = np.argsort(losses_npy)
    xs = list(range(2*k))
    x_ticks = []
    ys = []
    for i in range(k):
        # best i
        this_idx = idx_sort[i]
        ys.append(losses_npy[this_idx])

        if label_converter is not None:
            x_ticks.append(label_converter.compressed_to_name(this_idx + 1)) # add 1 because we removed 0-index-class when passing in
    N = losses_npy.shape[0]
    assert (2*k <= N)
    for i in range(N - k, N):
        this_idx = idx_sort[i]
        ys.append(losses_npy[this_idx])

        if label_converter is not None:
            x_ticks.append(label_converter.compressed_to_name(this_idx + 1)) # add 1 because we removed 0-index-class when passing in
        # -k, -(k-1), ... -1    - > k total entries
    
    fig, ax = plt.subplots()
    ax.bar(xs, ys, log=True)
    ax.set_ylabel('Batch Loss')
    #ax.xticks(xs, x_ticks, rotation='vertical')
    ax.set_xticks(xs)
    ax.set_xticklabels(x_ticks, rotation=65, ha='right')
    fig.tight_layout()
    return fig, ax

def plot_class_losses_vs_weights(val_class_losses, class_weights_tensor):
    x_list = class_weights_tensor[1:].cpu().tolist()
    y_list = torch.log10(val_class_losses[1:].cpu()).tolist()

    fig, ax = plt.subplots()
    ax.scatter(x_list, y_list)
    ax.set_xlabel('class weight')
    ax.set_ylabel('log10 class loss')
    
    return fig, ax
def plot_prediction_distribution(predicted_probs):
    fig, ax = plt.subplots()
    ax.hist(predicted_probs.cpu().detach().numpy().flatten(), range=[0, 1], bins=20, log=True)
    ax.set_xlabel('predicted probability')
    ax.set_ylabel('counts')
    return fig, ax

def proc_pattern_for_eval_result_load(pattern):
    # e.g. pattern: '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/eval_results_val_8-26-23/eval_singletask_yolov7_longer_prompted_val_label*.pkl'
    files = glob(pattern)
    listo = []
    for file in files:
        label_num = int(file.split('/')[-1].split('label')[1].split('.pkl')[0])
        stuff = pd.read_pickle(file)
        this_list = [label_num, stuff['dice_sensitivity'].item(), stuff['dice_specificity'].item(), stuff['overall_dice'].item()]
        this_arr = np.array(this_list).reshape((1, 4))
        listo.append(this_arr)
    combined = np.concatenate(listo, axis = 0)
    df = pd.DataFrame(combined, columns = ['label_number', 'dice_sensitivity', 'dice_specificity', 'overall_dice'])
    df = df.sort_values('label_number').reset_index(drop=True)

    return df

def plot_random_bboxes(test_ds, model, dev='cuda', n_ex = 1, seed=182):
    #np.random.seed(seed)
    # set up the plot
    fig, axs = plt.subplots(n_ex, n_ex, figsize=(25,25), squeeze=False) # 0 index is ground truth segmentation
    
    for iter in range(n_ex*n_ex):
        img_idx = np.random.randint(len(test_ds))

        # load image, embedding, segmentation, and input mask
        image_data, img_embedding, img_box, img_box_1024, img_seg = load_and_preprocess_slice(test_ds, img_idx, dev)
        C = img_seg.shape[0]

        random_box = torch.normal(mean = 128, std = torch.Tensor([128/3.0, 128/3.0, 128/3.0, 128/3.0])).view(1,4).to(dev).int()
        random_box = torch.clip(random_box, 0, 256)
        if random_box[0,0] > random_box[0,2]:
            random_box = random_box[:,[2,1,0,3]]
        if random_box[0,1] > random_box[0,3]:
            random_box = random_box[:,[0,3,2,1]]
        this_seg_result = medsam_inference(model, img_embedding, random_box, H=256, W=256, as_one_hot=True,
                                                   model_trained_on_multi_label=False, num_classes=1)
        #this_seg_result = B,1,H,W
        seg_result = this_seg_result[0,0,:,:] # now H,W

        image_data_3c = np.repeat(image_data[:,:,None], 3, axis=-1)

        iter_r = iter // n_ex
        iter_c = iter % n_ex

        axs[iter_r, iter_c].imshow(image_data_3c)
        axs[iter_r, iter_c].axis('off')
        show_mask(seg_result, axs[iter_r, iter_c])
        show_box(random_box.flatten().cpu().numpy(), axs[iter_r, iter_c])
    
    fig.show()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    
    return fig, axs






