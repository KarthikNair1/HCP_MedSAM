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

class MedSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                multimask_output):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # also freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        self.multimask_output = multimask_output

    def forward(self, image_embedding, box):
        
        # do not compute gradients for prompt encoder and image encoder
        with torch.no_grad():
            #image_embedding = self.image_encoder(image) # (B, 256, 64, 64), don't need to run this b/c precomputed embeddings
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image_embedding.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=self.multimask_output
          )

        #ori_res_masks = F.interpolate(low_res_masks, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
        #return ori_res_masks
        return low_res_masks

def logits_to_pred_probs(logits, as_one_hot):
    if as_one_hot:
        pred_probs = torch.sigmoid(logits)  # (B, C, H, W)
    else:
        pred_probs = torch.softmax(logits, dim = 1) # (B, C, H, W)
    return pred_probs

def convert_logits_to_preds_onehot(logits, as_one_hot, H, W):
    low_res_pred = logits_to_pred_probs(logits, as_one_hot)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (B, C, H, W)

    if as_one_hot:
        low_res_pred = low_res_pred.cpu().detach().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8) # (B, C, H, W)
    else:
        C = low_res_pred.shape[1]
        medsam_seg = torch.argmax(low_res_pred, dim = 1).long() # (B, H, W)
        medsam_seg = torch.nn.functional.one_hot(medsam_seg, num_classes=C) # (B, H, W, C)
        medsam_seg = torch.permute(medsam_seg, (0, 3, 1, 2)) # (B, C, H, W)
        medsam_seg = medsam_seg.cpu().detach().numpy().astype(np.uint8)
    return medsam_seg

def medsam_inference(medsam_model, img_embed, box_1024, H, W, as_one_hot, model_trained_on_multi_label=True,
                    num_classes=103):
    
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    nan_mask = torch.any(torch.isnan(box_torch), dim=1) # (B) tensor
    if torch.all(nan_mask): # skips over empty batches
        return torch.zeros((box_torch.shape[0], num_classes, H, W))
    
    # replace bbox values of nan_mask with 0s for now, and at the end we'll mask them out with zeros
    box_torch[nan_mask, :] = 0
    
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (B, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output= model_trained_on_multi_label,
        ) # output of this will be (B, C, H, W)
    
    low_res_logits[nan_mask, :, :, :] = -1e8
    
    return convert_logits_to_preds_onehot(low_res_logits, as_one_hot, H, W)