import plotly.express as px
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from glob import glob
import argparse

import sys
sys.path.append('./modified_medsam_repo')
from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, medsam_inference
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *

def generate_colors(seed = 2024, num_classes = 103, force_color = False):
    np.random.seed(seed)
    rand_nums = np.random.random((num_classes, 3))
    color_label_mapper = dict()
    for i in range(num_classes):
        if force_color:
            # use a fixed color for all regions, which should have high contrast with the MRI itself
            color_label_mapper[i] = np.array((201, 203, 52))
        else:
            color_label_mapper[i] = (rand_nums[i, :] * 255).astype('uint8')

    return color_label_mapper

def get_animation(mri_img_pattern, is_png, seg_arr, region_number, out_path, label_converter, color_label_mapper):
    # load the MRI
    img_paths = glob(mri_img_pattern)
    color_vec = color_label_mapper[region_number] # (np array of 3)

    if not is_png:
        img = nib.load(mri_file_path).get_fdata().astype(np.uint8)

    # generate slice by slice
    img_list = []
    for slice_num in range(50,220):
        # first, start with the source mri image
        if is_png:
            list_match = [s for s in img_paths if f'slice{slice_num}' in s]
            assert len(list_match) == 1
            this_slice_img = np.asarray(Image.open(list_match[0]))
        else:
            list_match = img_paths[0] # just the MRI itself
            this_slice_img = np.repeat(img[:,slice_num,:][:,:,None], 3, axis=-1)
        
        if region_number <= 0:
            # only output the raw scan as the image
            final_img = Image.fromarray(this_slice_img.astype('uint8'), 'RGB')
            img_list.append(final_img)
            continue
        
        # next, create a mask for the specific region number
        this_slice_mask = seg_arr[slice_num, :, :] == region_number
        this_slice_mask = np.repeat(this_slice_mask[:,:,None], 3, axis=-1) # 256,256,3

        color_portion = np.zeros((this_slice_mask.shape[0], this_slice_mask.shape[1], 3))
        #mask_uint8 = np.dstack((color_portion, this_slice_mask*0.5*255)).astype('uint8')
        
        for i in range(3):
            color_portion[:,:,i] = color_vec[i]
        

        mask_colored = this_slice_mask * color_portion # has 0s if not active at that pixel, and the right color value if it is
        # but we need to grab all the 0s and set them to the source value

        mask_colored[np.where(this_slice_mask==0)] = this_slice_img[np.where(this_slice_mask==0)]

        # form a linear combination
        delta = 0.7
        final = delta * mask_colored + (1-delta) * this_slice_img

        final_img = Image.fromarray(final.astype('uint8'), 'RGB')

        img_list.append(final_img)
    
    if not os.path.exists(out_path):  
        os.makedirs(out_path)

    img_list[0].save(os.path.join(out_path, f'{label_converter.compressed_to_name(region_number)}.gif'), format='GIF', save_all=True, append_images = img_list[1:],
                    duration=100, loop=0)

# fuse together all regions to get an animation
def get_animation_fused(mri_img_pattern, is_png, seg_arr, out_path, color_label_mapper):
    # load the MRI
    img_paths = glob(mri_img_pattern)

    if not is_png:
        img = nib.load(mri_file_path).get_fdata().astype(np.uint8)

    # 256,256,1 * 1,3

    # generate slice by slice
    img_list = []
    delta = 0.7
    for slice_num in range(50,220):
        # first, start with the source mri image
        if is_png:
            list_match = [s for s in img_paths if f'slice{slice_num}' in s]
            assert len(list_match) == 1
            this_slice_img = np.array(Image.open(list_match[0]))
        else:
            list_match = img_paths[0] # just the MRI itself
            this_slice_img = np.repeat(img[:,slice_num,:][:,:,None], 3, axis=-1)        

        # next, create a mask for each specific region number
        num_regions = len(np.unique(seg_arr[slice_num, :, :]))
        color_portion = this_slice_img.copy() # 256,256,3
        for region_number in np.unique(seg_arr[slice_num, :, :]):
            if region_number==0:
                continue
            color_vec = color_label_mapper[region_number]
            this_slice_mask = seg_arr[slice_num, :, :] == region_number
            this_slice_mask = np.repeat(this_slice_mask[:,:,None], 3, axis=-1) # 256,256,3

            #mask_uint8 = np.dstack((color_portion, this_slice_mask*0.5*255)).astype('uint8')
            color_tmper = np.zeros((256,256,3))
            for i in range(3):
                color_tmper[:,:,i] = color_vec[i]
            

            mask_colored = this_slice_mask * color_tmper # has 0s if not active at that pixel, and the right color value if it is
            # but we need to grab all the 0s and set them to the source value
            color_portion[np.where(this_slice_mask==1)] = mask_colored[np.where(this_slice_mask==1)]
            #mask_colored[np.where(this_slice_mask==0)] = this_slice_img[np.where(this_slice_mask==0)]

        final_array = (1-delta)*this_slice_img + delta * color_portion
        final_img = Image.fromarray(final_array.astype('uint8'), 'RGB')

        img_list.append(final_img)
    if not os.path.exists(out_path):  
        os.makedirs(out_path)
    img_list[0].save(os.path.join(out_path, 'fused.gif'), format='GIF', save_all=True, append_images = img_list[1:],
                    duration=100, loop=0)



parser = argparse.ArgumentParser()
parser.add_argument("--mri_path", type=str, default=None, help='Either a regex pattern pointing to slice pngs, or a single path to an nii file')
parser.add_argument("--seg_path", type=str, default=None, help='Path to a .npy holding segmentation for the whole MRI i.e. size 256x256x256 of class indices')
#parser.add_argument("--input_mode", choices = ['png', 'nii'], help = 'Input png if passing in a directory of pngs, or input nii if passing in a single nii file.')
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv')
parser.add_argument('--output_dir', type=str, default = '/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/eval_results_test_10-13-23')
parser.add_argument('--do_region_gifs', action='store_true')
parser.add_argument('--do_fuse_gif', action='store_true')
parser.add_argument('--single_label', type=int, default=None, help='Label to use if only one label should be outputted in the segmentation')
parser.add_argument('--force_color', action='store_true', help='Should segmentations all be plotted with the same color for high contrast')

args = parser.parse_args()
if not args.do_region_gifs and not args.do_fuse_gif:
    print("Use either --do_region_gifs or --do_fuse_gif (or both) for this script to do anything")

df_hcp = pd.read_csv(args.df_starting_mapping_path)
df_desired = pd.read_csv(args.df_desired_path)
NUM_CLASSES = len(df_desired)
label_converter = LabelConverter(df_hcp, df_desired)

is_png_pattern = args.mri_path.endswith('.png')

print(f'mri pattern: {args.mri_path}')
print(f'seg path: {args.seg_path}')
seg_arr = np.load(args.seg_path, allow_pickle=True)

assert seg_arr.shape == (256,256,256)

# generate colors
color_label_mapper = generate_colors(seed = 2024, num_classes = 103, force_color = args.force_color)



mri_img_pattern = args.mri_path

if args.do_region_gifs:

    labels_to_iter = range(0,103)
    if args.single_label is not None:
        labels_to_iter = [args.single_label]

    for i in labels_to_iter:
        if i in np.unique(seg_arr):
            get_animation(mri_img_pattern, is_png_pattern, seg_arr, i, out_path = args.output_dir, label_converter = label_converter, color_label_mapper = color_label_mapper)
        elif args.single_label is not None:
            # force create the animation even if the label has no positive pixels
            get_animation(mri_img_pattern, is_png_pattern, seg_arr, i, out_path = args.output_dir, label_converter = label_converter, color_label_mapper = color_label_mapper)

if args.do_fuse_gif:
    get_animation_fused(mri_img_pattern, is_png_pattern, seg_arr, out_path = args.output_dir, color_label_mapper = color_label_mapper)