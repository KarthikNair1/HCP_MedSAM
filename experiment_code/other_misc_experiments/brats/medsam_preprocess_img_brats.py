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
from PIL import Image

import argparse

# https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
def symmetric_pad_array(input_array: np.ndarray, target_shape: tuple, pad_value: int) -> np.ndarray:

    for dim_in, dim_target in zip(input_array.shape, target_shape):
        if dim_target < dim_in:
            raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

    pad_width = []
    for dim_in, dim_target  in zip(input_array.shape, target_shape):
        if (dim_in-dim_target)%2 == 0:
            pad_width.append((int(abs((dim_in-dim_target)/2)), int(abs((dim_in-dim_target)/2))))
        else:
            pad_width.append((int(abs((dim_in-dim_target)/2)), (int(abs((dim_in-dim_target)/2))+1)))
    
    return np.pad(input_array, pad_width, 'constant', constant_values=pad_value)


parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
args = parser.parse_args()

start_idx = args.start_idx
end_idx = args.end_idx

#%% load model and image


MedSAM_CKPT_PATH = "/gpfs/home/kn2347/MedSAM/medsam_vit_b.pth"
device = "cuda:0"
#device='cpu'
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

from segment_anything.utils.transforms import ResizeLongestSide
import glob

file_pattern = '/gpfs/data/luilab/karthik/brats_dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_*/*_t1ce.nii.gz'
#file_pattern_seg = '/gpfs/data/luilab/karthik/brats_dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_*/*_seg.nii.gz'
paths = glob.glob(file_pattern)
paths = sorted(paths)

#root = '/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/pretrained_image_encoded_slices'
#segment_root_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/segmentation_slices'

# e.g. /gpfs/data/luilab/karthik/brats_dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
segment_root_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/brats_slices_npy/segmentation_slices'
img_encoding_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/brats_slices_npy/pretrained_image_encoded_slices'

ctr = 0

for p in paths[start_idx:min(end_idx, len(paths))]:
    id = p.split('/')[-2].split('BraTS20_Training_')[1]

    print (f'On {ctr}/{min(end_idx, len(paths)) - start_idx}')
    ctr+=1

    folder_dir = f'{img_encoding_dir}/{id}'
    segment_folder_dir = f'{segment_root_dir}/{id}'
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    if not os.path.exists(segment_folder_dir):
        os.makedirs(segment_folder_dir)
    

    data_mri = nib.load(p).get_fdata()
    data_mri = np.transpose(data_mri, (0,2,1))
    data_mri = np.flip(data_mri, axis=2)
    data_mri = np.flip(data_mri, axis=1)
    data_mri = symmetric_pad_array(data_mri, (256,256,256), 0)

    seg_path = os.path.join('/'.join(p.split('/')[:-1]), f'BraTS20_Training_{id}_seg.nii.gz')
    
    data_segmentation = nib.load(seg_path).get_fdata()
    data_segmentation = np.transpose(data_segmentation, (0,2,1))
    data_segmentation = np.flip(data_segmentation, axis=2)
    data_segmentation = np.flip(data_segmentation, axis=1)
    data_segmentation = symmetric_pad_array(data_segmentation, (256,256,256), 0)


    for slice in range(data_mri.shape[1]):
        
        slice_path = f'{folder_dir}/{slice}.npy'
        seg_slice_path = f'{segment_folder_dir}/seg_{slice}.npy'

        if not os.path.exists(slice_path): # save image encoding

            #x = data_mri[:,slice,:].astype('uint8')
            x = data_mri[:,slice,:]
            if x.max() > 0:
                x = (x / x.max() * 255).astype('uint8')
            else:
                x = x.astype('uint8')
            assert len(x.shape)==2
            x = np.repeat(x[:, :, None], 3, axis=-1)
            my_img = x
            x = transform.resize(x, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            x_tensor_preproc = (x - x.min()) / np.clip(
                x.max() - x.min(), a_min=1e-8, a_max=None
            )
            x_tensor_preproc = torch.tensor(x_tensor_preproc).float().permute(2,0,1).unsqueeze(0).to('cuda')
            
            with torch.no_grad():
                embedding = medsam_model.image_encoder(x_tensor_preproc) # shape is (1, 256, 64, 64)

            # save embedding to a file
            np.save(slice_path, embedding.cpu().numpy()[0, :, : ,:]) # saved as (256, 64, 64)

            # save raw image as well
            img_save_dir = '/gpfs/data/luilab/karthik/pediatric_seg_proj/brats_slices_npy/dir_structure_for_yolov7/train/images/'

            img_pil = Image.fromarray(my_img.astype('uint8'))
            img_pil.save(img_save_dir + f'{id}_slice{slice}.png')

        if not os.path.exists(seg_slice_path):
            x = data_segmentation[:,slice,:].astype('uint8')
            assert len(x.shape)==2

            np.save(seg_slice_path, x)



        