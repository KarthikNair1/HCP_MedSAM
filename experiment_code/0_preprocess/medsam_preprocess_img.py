# python /gpfs/home/kn2347/test/medsam_preprocess_img.py --start_idx 0 --end_idx 1113

import nibabel as nib
# %% environment and functions
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
args = parser.parse_args()

start_idx = args.start_idx
end_idx = args.end_idx

#%% load model and image


MedSAM_CKPT_PATH = "/gpfs/home/kn2347/MedSAM/medsam_vit_b.pth"
#device = "cuda:0"
device='cpu'
#medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
#medsam_model = medsam_model.to(device)
#medsam_model.eval()

from segment_anything.utils.transforms import ResizeLongestSide
import glob

file_pattern = '/gpfs/data/luilab/karthik/brats_dataset/BraTSReg_Training_Data_v3/*/*_t1.nii.gz'
paths = glob.glob(file_pattern)
'/gpfs/data/cbi/hcp/hcp_seg/data_orig/100206/mri/aparc+aseg.mgz'
paths = sorted(paths)

#root = '/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/pretrained_image_encoded_slices'
#segment_root_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/segmentation_slices'


segment_root_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/brats_slices_npy/segmentation_slices'
img_encoding_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/brats_slices_npy/pretrained_image_encoded_slices'

ctr = 0

for p in paths[start_idx:min(end_idx, len(paths))]:
    id = p.split('/')[-3]

    print (f'On {ctr}/{min(end_idx, len(paths)) - start_idx}')
    ctr+=1

    folder_dir = f'{root}/{id}'
    segment_folder_dir = f'{segment_root_dir}/{id}'
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    if not os.path.exists(segment_folder_dir):
        os.makedirs(segment_folder_dir)
    
    
    data_mri = nib.load(p).get_fdata()  
    seg_path = os.path.join('/'.join(p.split('/')[:-1]), 'aparc+aseg.mgz')
    data_segmentation = nib.load(seg_path).get_fdata()

    for slice in range(data_mri.shape[1]):
        
        slice_path = f'{folder_dir}/{slice}.npy'

        if not os.path.exists(slice_path): # save image encoding
            x = data_mri[:,slice,:].astype('uint8')
            assert len(x.shape)==2
            x = np.repeat(x[:, :, None], 3, axis=-1)
            x = transform.resize(x, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            x_tensor_preproc = (x - x.min()) / np.clip(
                x.max() - x.min(), a_min=1e-8, a_max=None
            )
            x_tensor_preproc = torch.tensor(x_tensor_preproc).float().permute(2,0,1).unsqueeze(0).to('cuda')

            with torch.no_grad():
                embedding = medsam_model.image_encoder(x_tensor_preproc) # shape is (1, 256, 64, 64)

            # save embedding to a file
            np.save(slice_path, embedding.cpu().numpy()[0, :, : ,:]) # saved as (256, 64, 64)
            

        seg_slice_path = f'{segment_folder_dir}/seg_{slice}.npy'
        if not os.path.exists(seg_slice_path): # save freesurfer mask
            y = data_segmentation[:, slice, :].astype(int) # 256 x 256 now
            assert y.shape == (256, 256)

            np.save(seg_slice_path, y) # saved as (256, 256)


        