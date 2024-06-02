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
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
import nibabel as nib
import pickle
from typing import List, Tuple, Type
import time

def my_forward_mask_decoder(
    model,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
):  
    print(f'input image embedding shape: {input.shape}')
    print(f'sparse prompt shape: {sparse_prompt_embeddings.shape}')
    print(f'dense prompt shape: {dense_prompt_embeddings.shape}')
    print(f'{model.num_mask_tokens=}')

    print('==========================================================================================')

    output_tokens = torch.cat([model.iou_token.weight, model.mask_tokens.weight], dim=0)
    print(f'{model.iou_token.weight.shape=}')
    print(f'{model.mask_tokens.weight.shape=}')
    print(f'{output_tokens.shape=}')

    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    print(f'after expanding to sparse prompt size: {output_tokens.shape=}')
    
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
    print(f'total token size: {tokens.shape=}')

    # Expand per-image data in batch direction to be per-mask
    if image_embeddings.shape[0] != tokens.shape[0]:
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    else:
        src = image_embeddings
    print(f'after expanding per-image data in batch direction to be per-mask: {src.shape=}')
    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape
    print(f'positional src shape: {pos_src.shape=}')

    # Run the transformer
    hs, src = model.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + model.num_mask_tokens), :]
    print('==========================================================================================')
    print('post transformer!!')
    print(f'{hs.shape=}')
    print(f'{src.shape=}')
    print(f'{iou_token_out.shape=}')
    print(f'{mask_tokens_out.shape=}')

    print('==========================================================================================')
    print('upscale mask embeddings and predict masks')

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    print(f'{src.shape=}')
    upscaled_embedding = model.output_upscaling(src)
    print(f'{upscaled_embedding.shape=}')
    hyper_in_list: List[torch.Tensor] = []
    for i in range(model.num_mask_tokens):
        hyper_in_list.append(model.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        shapo = model.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]).shape
        print(f'shape of class {i}th hypernetwork MLP output: {shapo}')
    hyper_in = torch.stack(hyper_in_list, dim=1)
    print(f'shape of hyper_in: {hyper_in.shape}')
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
    print(f'masks shape: {masks.shape}')
    # Generate mask quality predictions
    iou_pred = model.iou_prediction_head(iou_token_out)
    print(f'mask quality predictions: {iou_pred.shape}')

    if multimask_output:
        mask_slice = slice(1, None)
    else:
        mask_slice = slice(0, 1)
    masks = masks[:, mask_slice, :, :]
    iou_pred = iou_pred[:, mask_slice]

    print('==========================================================================================')
    print('post-multimask slicing!')
    print(f'{masks.shape=}')
    print(f'{iou_pred.shape=}')

    # Prepare output
    return masks, iou_pred


path = '/gpfs/home/kn2347/MedSAM/medsam_vit_b.pth'
model = sam_model_registry['vit_b'](checkpoint=path)

B = 2
N = 3
embed_num_filters = 256
embed_H = 64
embed_W = 64
box_torch = torch.as_tensor(np.array([[0, 0, embed_H, embed_W],
                                      [0, 0, embed_H, embed_W]]))
input = torch.zeros((B, embed_num_filters, embed_H, embed_W))

sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

low_res_logits, _ = my_forward_mask_decoder(model.mask_decoder, 
        image_embeddings=input, # (B, 256, 64, 64)
        image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=True,
)