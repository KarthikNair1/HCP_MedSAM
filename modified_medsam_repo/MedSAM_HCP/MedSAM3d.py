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

class MedSAM3D(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                neighborhood_dim):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder


        self.neighborhood_average_linear = []
        self.neighborhood_dim = neighborhood_dim
        for i in range(256):
            self.neighborhood_average_linear.append(nn.Linear(in_features = self.neighborhood_dim, out_features=1, bias=False))
        self.neighborhood_average_linear = nn.ModuleList(self.neighborhood_average_linear)
        
        
        with torch.no_grad():
            for i in range(256):
                tens = torch.full((self.neighborhood_dim,), 1e-4)
                tens[self.neighborhood_dim//2] = 1
                self.neighborhood_average_linear[i].weight = nn.Parameter(tens)
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # also freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
    def forward(self, image_embedding, box):
        # image_embedding has shape (B, 256, 64, 64, 5)
        # box has shape (B, 103, 4)
        

        B, F, H, W, N = image_embedding.size()
        assert B == 1
        C = box.size()[1]
        compressed_embedding = torch.zeros((B, F, H, W)).to(image_embedding.device)
        for filter in range(256):
            compressed_embedding[:,filter,:,:] = self.neighborhood_average_linear[filter](image_embedding[:,filter,:,:,:]).view(B, H, W) # now (B,F,64,64)

        # now reshape so that class becomes the batch dimension, and we can run everything through the model once without a for loop
        compressed_embedding = torch.repeat_interleave(compressed_embedding, repeats = C, dim = 0) # now (C,F,64,64)
        box = box.squeeze().view(C, 1, 4) # box goes from B, C, 4 -> C, 1, 4

        nan_mask = torch.any(torch.isnan(box), dim=2) # (C, 1) tensor
        if torch.all(nan_mask):
            desired_len = C * 1 * 256 * 256
            ret = 0 * compressed_embedding.flatten()[:desired_len].view(C, 1, 256, 256)
            return ret

        box[nan_mask, :] = 0 # replace nan masks with filler values for now, and mask them out later

        with torch.no_grad():

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
        
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=compressed_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False
        ) # this output has shape (C, 1, H', W')

        # mask out
        low_res_masks[nan_mask, :, :] = -1e4
        return low_res_masks # is (C, 1, 256, 256)
    












def convert_logits_to_preds_onehot_3d(logits, H, W):
    low_res_pred = torch.sigmoid(logits)  # (B, C, H, W)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (B, C, H, W)
    low_res_pred = low_res_pred.cpu().detach().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8) # (B, C, H, W)
    return medsam_seg

def medsam_inference_3d(medsam_model, img_embed, box_1024, H, W, as_one_hot, model_trained_on_multi_label=True):
    low_res_logits = medsam_model(img_embed, box_1024)
    return convert_logits_to_preds_onehot_3d(low_res_logits, H, W)