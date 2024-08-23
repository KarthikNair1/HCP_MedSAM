import monai
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

def weighted_ce_loss(pred, gt, weights, as_one_hot=True):
    # pred must be pre-sigmoid values
    # weights must be tensor of shape (NUM_CLASSES)
    assert weights.shape[0] == pred.shape[1]
    C = weights.shape[0]
    if as_one_hot:
        assert len(gt.shape) == 4 # B, C, H, W
        gt = gt.float()
        loss_fn = nn.BCEWithLogitsLoss(reduction='none') 
        losses = loss_fn(pred, gt) # B, C, H, W
        losses = losses.mean(dim = (2, 3)) # B, C
        class_losses = (weights * losses).mean(dim=0).view(C) # mean over batch -> C
        if C > 1:
            overall_loss = class_losses[1:].mean() # mean over classes -> 1, also remove unknown channel
        else:
            overall_loss = class_losses

        # note that class losses does not remove unknown class
    else:
        assert len(gt.shape) == 4 # B, C, H, W, one hot encoded
        gt = gt.float()
        loss_fn = nn.CrossEntropyLoss(reduction='none', weight = weights)
        loss_tensor = loss_fn(pred, gt) # (B, H, W)
        overall_loss = loss_tensor.mean()

        zero_tensor = torch.zeros((C)).cuda()
        class_idxs = gt.argmax(dim=1) # now (B, H, W)
        class_idx_flatten = class_idxs.flatten()
        counts = zero_tensor.index_add(0, class_idx_flatten, torch.ones_like(class_idx_flatten).float())
        class_losses = zero_tensor.index_add(0, class_idx_flatten, loss_tensor.flatten())
        class_losses /= (counts+1e-6) # this line averages the class losses over the number of pixels in that class encountered
        
    return overall_loss, class_losses
        
def weighted_dice_loss(pred, gt, weights, as_one_hot=True):
    # pred must be pre-sigmoid values of shape (B,C,H,W)
    # gt has same shape but binary values (0 or 1)
    # weights must be tensor of shape (NUM_CLASSES)

    assert weights.shape[0] == pred.shape[1]
    if as_one_hot:
        B, C, H, W = pred.shape
        loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='none') 
        
    else:
        #assert len(gt.shape) == 3 # B, H, W
        B, C, H, W = gt.shape
        #gt = gt.view(B, 1, H, W)
        loss_fn = monai.losses.DiceLoss(softmax=True, squared_pred=True, reduction='none') 
    
    losses = loss_fn(pred, gt) # B, C, 1, 1 
    losses = losses.view(B, C) # (B, C)
    class_losses = (weights * losses).mean(dim=0).view(C) # mean over batch -> C
    overall_loss = class_losses.mean() # mean over classes
    #overall_loss = class_losses[1:].mean() # mean over classes -> 1, also remove unknown channel
    # note that class losses does not remove unknown class

    
    return overall_loss, class_losses

def weighted_ce_dice_loss(pred, gt, weights, lambda_dice = 0.5, as_one_hot=True):
    

    dice_weight = 2 * lambda_dice
    ce_weight = 2 * (1 - lambda_dice)

    dice_loss, dice_class_loss = weighted_dice_loss(pred, gt, weights, as_one_hot)
    ce_loss, ce_class_loss = weighted_ce_loss(pred, gt, weights, as_one_hot)

    total_loss = ce_weight * ce_loss + dice_weight * dice_loss
    total_class_loss = ce_weight * ce_class_loss + dice_weight * dice_class_loss
    return total_loss, total_class_loss, dice_weight * dice_class_loss, ce_weight * ce_class_loss 

def unweighted_ce_loss(pred, gt):
    return nn.BCEWithLogitsLoss(reduction="mean")(pred, gt.float())

def unweighted_dice_loss(pred, gt):
    return monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")(pred, gt)

def unweighted_ce_dice_loss(pred, gt, lambda_dice = 0.5):
    dice_weight = 2 * lambda_dice
    ce_weight = 2 * (1 - lambda_dice)

    total_loss = ce_weight * unweighted_ce_loss(pred, gt) + dice_weight * unweighted_dice_loss(pred, gt)
    return total_loss

def edt_penalized_loss(preds, gt):
    # https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py
    # gt has shape (B, C, H, W)
    # same with preds
    B, C, H, W = gt.shape
    result = np.zeros((B,C))

    for batch_num in range(B):
        for class_num in range(C):
            posmask = gt[batch_num, class_num, :, :] # (B, H, W) of 0s and 1s
            negmask = ~posmask
            pos_edt = distance_transform_edt(posmask)
            pos_edt = (np.max(pos_edt) - pos_edt) * posmask
            neg_edt = distance_transform_edt(negmask)
            neg_edt = (np.max(neg_edt) - neg_edt) * negmask

            result[batch_num, class_num] = pos_edt / np.max(pos_edt) + neg_edt/np.max(neg_edt)
            # above line normalizes so that different shapes have comparable loss

    # TODO
    return result

def unweighted_focal_loss(pred, gt, set_alpha = False):
    # gt is (B,C,H,W)
    if set_alpha:
        # set to 1 / total area of 1's in the image
        # note, this calculation really only makes sense if using one hot encoded tensors I think
        alpha = 1.0 / (gt == 1).sum(dim = (1,2,3)).float().mean() # mean's over batch to get a per-sample value
    else:
        alpha = None
    return monai.losses.FocalLoss(alpha = alpha, use_softmax=False, reduction='mean')(pred, gt) # use_softmax=False causes sigmoid to be called
            
def loss_handler(loss_type, pred, gt, weights, lambda_dice = 0.5, as_one_hot=True, focal_alpha = None): # use this for more general function
    loss_types = ['weighted_ce_dice_loss', 'focal_loss']
    if loss_type == 'weighted_ce_dice_loss':
        return weighted_ce_dice_loss(pred, gt, weights, lambda_dice, as_one_hot) # loss, class_losses, dice_class_losses, ce_class_losses
    elif loss_type == 'focal_loss':
        return unweighted_focal_loss(pred, gt, focal_alpha), None, None, None
            

