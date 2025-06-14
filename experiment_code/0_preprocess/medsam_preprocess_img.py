# python /gpfs/home/kn2347/test/medsam_preprocess_img.py --start_idx 0 --end_idx 1113

import nibabel as nib
# %% environment and functions
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
import glob
import argparse
from PIL import Image



parser = argparse.ArgumentParser()

# accepts start_idx and end_idx for processing whole dataset across multiple nodes
parser.add_argument("--start_idx", type=int, default = 0)
parser.add_argument("--end_idx", type=int, default = 1113)
parser.add_argument("--mri_file_pattern", type=str, help = 'String pattern for all .mgz/.gz MRI files. For example, /gpfs/data/cbi/hcp/hcp_seg/data_orig/*/mri/T1.mgz')
parser.add_argument("--MedSAM_checkpoint_path", type = str, default = "/gpfs/home/kn2347/MedSAM/medsam_vit_b.pth")
parser.add_argument("--dest_image_encoding_dir", type = str, help = "Target directory for saving all image encodings, e.g. /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/pretrained_image_encoded_slices")
parser.add_argument("--dest_seg_dir", type = str, default = None, help = "Target directory for saving all image segmentations, e.g. /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/segmentation_slices")
parser.add_argument("--dest_img_dir", type = str, help = "Target directory for saving all raw images, e.g. /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/images")
parser.add_argument("--id_index_in_path", type=int, default = -3, help = 'In the path to an mri, at what directory level is the id for the mri that will be used for file names? Should be a python index, e.g negative references back to front')
parser.add_argument("--seg_suffix", type=str, default = "aparc+aseg.mgz", help = 'What is the suffix for the segmentation file?')
parser.add_argument("--image_norm_max", type=float, default = None, help = 'What should be the png file max value after normalized and rescaling?')
parser.add_argument("--slice_axis", type=int, default = 1, help = 'Which axis to slice the MRI on?')

parser.add_argument("--device", type = str, default = "cuda", help = 'cuda or cpu?')

args = parser.parse_args()


start_idx = args.start_idx
end_idx = args.end_idx
file_pattern = args.mri_file_pattern
MedSAM_CKPT_PATH = args.MedSAM_checkpoint_path
device = args.device
img_encoding_dir = args.dest_image_encoding_dir
segment_root_dir = args.dest_seg_dir

# load model and image
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# get all paths to MRI files and sort them so that the order is reproducible
paths = glob.glob(file_pattern)
paths = sorted(paths)

ctr = 0

# for every MRI found according to the pattern:
for p in paths[start_idx:min(end_idx, len(paths))]:
    id = p.split('/')[args.id_index_in_path]

    print (f'On {ctr}/{min(end_idx, len(paths)) - start_idx}')
    ctr+=1

    folder_dir = f'{img_encoding_dir}/{id}'
    segment_folder_dir = f'{segment_root_dir}/{id}'
    img_dir = os.path.join(args.dest_img_dir, id)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    if not os.path.exists(segment_folder_dir):
        os.makedirs(segment_folder_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # load the MRI file and the segmentation file using nibabel
    data_mri = nib.load(p).get_fdata()
    print(data_mri.shape)

    # get max value for purposes of normalization
    max_value = data_mri.max()

    # for every slice, calculate the image encoding and save it, as well as the segmentation file as .npy's
    for slice in range(data_mri.shape[args.slice_axis]):
        
        slice_path = f'{folder_dir}/{slice}.npy'

        if not os.path.exists(slice_path): # save image encoding
            #x = data_mri[:,slice,:]
            x = np.take(data_mri, slice, axis=args.slice_axis)
            if x.max() > 0:
                if args.image_norm_max is None:
                    x = (x / max_value * 255).astype('uint8')
                else:
                    x = (x / max_value * args.image_norm_max).astype('uint8')
            else:
                x = x.astype('uint8')
            assert len(x.shape)==2
            x = np.repeat(x[:, :, None], 3, axis=-1)
            my_img = x
            x = transform.resize(x, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            x_tensor_preproc = (x - x.min()) / np.clip(
                x.max() - x.min(), a_min=1e-8, a_max=None
            )
            x_tensor_preproc = torch.tensor(x_tensor_preproc).float().permute(2,0,1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = medsam_model.image_encoder(x_tensor_preproc) # shape is (1, 256, 64, 64)

            # save embedding to a file
            np.save(slice_path, embedding.cpu().numpy()[0, :, : ,:]) # saved as (256, 64, 64)
            
        # save raw image as well
        img_dest_dir = os.path.join(img_dir, f'{id}_slice{slice}.png')

        if not os.path.exists(img_dest_dir):
            #x = data_mri[:,slice,:]
            x = np.take(data_mri, slice, axis=args.slice_axis)
            if x.max() > 0:
                if args.image_norm_max is None:
                    x = (x / max_value * 255).astype('uint8')
                else:
                    x = (x / max_value * args.image_norm_max).astype('uint8')
            else:
                x = x.astype('uint8')
            assert len(x.shape)==2
            x = np.repeat(x[:, :, None], 3, axis=-1)

            my_img = x
            img_pil = Image.fromarray(my_img.astype('uint8'))
            img_pil.save(img_dest_dir)

        
        
        if segment_root_dir is not None:
            seg_path = os.path.join('/'.join(p.split('/')[:-1]), '*' + args.seg_suffix)
            seg_path = glob.glob(seg_path)[0]
            data_segmentation = nib.load(seg_path).get_fdata()
            seg_slice_path = f'{segment_folder_dir}/seg_{slice}.npy'

            if not os.path.exists(seg_slice_path): # save freesurfer mask

                #y = data_segmentation[:, slice, :].astype(int) # 256 x 256 now
                y = np.take(data_segmentation, slice, axis=args.slice_axis).astype(int)
                #assert y.shape == (256, 256)

                np.save(seg_slice_path, y) # saved as (256, 256)


        