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

from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.dataset3d import MRIDataset3D, load_datasets_3d
from MedSAM_HCP.MedSAM3d import MedSAM3D, convert_logits_to_preds_onehot_3d

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

    print(result_dict)    
    if model_type == 'multitask_unprompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path, strict=False).to('cuda')
    elif model_type == 'pooltask_yolov7_prompted':
        model = build_sam_vit_b_multiclass(num_classes, checkpoint=model_path, strict=False).to('cuda')
    else:
        # singletask model
        model = build_sam_vit_b_multiclass(3, checkpoint=model_path, strict=False).to('cuda')



    model.eval()
    return model, result_dict

df_hcp = pd.read_csv('/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv')
df_desired = pd.read_csv('/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
NUM_CLASSES = len(df_desired)
label_converter = LabelConverter(df_hcp, df_desired)


#checkpoint = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/pooled_labels_8-20-23/model_best_20230820-215516.pth'
checkpoint = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/3d_model_8-31-23/MedSAM_finetune_3D-20230904-223957/medsam_model_best.pth'
device = 'cuda:0'
#sam_model = build_sam_vit_b_multiclass(NUM_CLASSES, checkpoint=checkpoint).to(device)
sam_model, result_dict = load_model('pooltask_yolov7_prompted', '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/3d_model_8-31-23/MedSAM_finetune_3D-20230904-223957/medsam_model_best.pth', 3)
#sam_model = sam_model.cuda()
medsam_model = MedSAM3D(image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        neighborhood_dim=5
                    )
medsam_model.load_state_dict(result_dict, strict = True)

medsam_model = medsam_model.cuda()

path_df_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_with_yolov7_box_path.csv'
train_test_splits_path = '/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle'
train_dataset, val_dataset, test_dataset = load_datasets_3d(path_df_path, 
                                                                train_test_splits_path, 
                                                                bbox_shift=0, 
                                                                label_converter = label_converter,
                                                                neighborhood_dim=5)




#lab_nums = torch.Tensor(val_dataset.data_frame['label_number'].to_numpy()).long().cuda()
batch_sz = 1
dataloader = DataLoader(
        val_dataset,
        batch_size = batch_sz,
        shuffle = False,
        num_workers = 4,
        pin_memory = True)
        
class_dice_collector = []

for step, (image_embedding, gt2D, boxes, slice_names) in enumerate(tqdm(dataloader)):

        image_embedding, gt2D, boxes = image_embedding.cuda(), gt2D.cuda(), boxes.cuda()
        medsam_pred = medsam_model(image_embedding,  boxes)
        medsam_pred = medsam_pred.permute((1,0,2,3))
        medsam_pred = (torch.sigmoid(medsam_pred) > 0.5).to(dtype=torch.uint8)
        
        dice_scores = dice_scores_multi_class(medsam_pred, gt2D, eps=1e-6, mask_empty_class_images_with_nan=True) # (B=1, C)
        dice_scores = dice_scores.detach().cpu() # extraneous but do it anyway
        class_dice_collector.append(dice_scores)

dices = torch.cat(class_dice_collector, dim=0).cpu().detach().numpy() # (N, C) tensor
dices_avg_per_class = np.nanmean(dices, axis = 0) # (C)

np.save('/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/eval_3D_pooled_labels_9-4-23/val_dices.npy', dices_avg_per_class)



#medsam_inference(medsam_model, img_embed, box_1024, H, W, as_one_hot, model_trained_on_multi_label=True)
