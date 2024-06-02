'''
In this analysis, I wanted to look at what the MedSAM embeddings were capturing. I calculated embeddings from the pre-trained MedSAM model before any finetuning
for many MRI's, and then for some slices from these MRI's, I looked at their nearest neighbors in embedding space and displayed them
'''

import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from skimage import io
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from monai.networks import one_hot
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
from skimage import io, transform
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import nibabel as nib
import pickle
from torch.utils.data import RandomSampler
import scipy
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

def get_image_slice_from_row(row):
    img = nib.load(row['image_path']).get_fdata()[:,row['slice'],:].astype(np.uint8)
    img = np.repeat(img[:,:,None], 3, axis=-1)
    return img
def get_image_embedding_from_row(row):
    embedding = np.load(row['image_embedding_slice_path'])
    return embedding
def plot_from_row(row):
    img_np = get_image_slice_from_row(row)
    plt.imshow(img_np)
def get_nearest_slices_in_dataframe(input_embedding, df_to_search, top_k = 1, num_limit = 100):

    input_embedding = input_embedding.flatten()
    num_limit = min(num_limit, df_to_search.shape[0])

    # randomly permute the rows of df_to_search to disrupt the slice-by-slice analysis
    df_to_search = df_to_search.sample(frac=1).reset_index(drop=True)
    list_content = []
    for idx in tqdm(range(num_limit)):
        r = df_to_search.iloc[idx, :]
        this_embedding = get_image_embedding_from_row(r).flatten()

        
        distance = scipy.spatial.distance.cosine(input_embedding, this_embedding)
        list_content.append((distance, r))
    
    list_content.sort(key=lambda x: x[0])
    srted_list = list_content
    srted_list = srted_list[:top_k]
    srted_rows = [x[1] for x in srted_list] # get only the rows, not the distance
    return srted_rows


df = pd.read_csv('/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df.csv')
df = df[(df['slice'] > 30) & (df['slice'] < 220)] # remove padded portions
df = df.reset_index(drop=True)
print(df.shape)

# window of +/-10
while True:
    idx = np.random.randint(df.shape[0]) # the "source" image
    src_row = df.iloc[idx, :]
    src_embed = get_image_embedding_from_row(src_row) # "source" embedding
    src_slice = src_row['slice']
    if src_slice <= 30 or src_slice >= 220:
        continue
    
    df_slice_window = df[abs(df['slice'] - src_slice) <= 10].reset_index(drop=True) # only select slices nearby to limit search space for speed
    df_slice_window = df_slice_window[df_slice_window['id'] != src_row['id']].reset_index(drop=True)
    closest_rows = get_nearest_slices_in_dataframe(src_embed, df_slice_window, top_k = 5, num_limit = 50000)
    
    src_img = get_image_slice_from_row(df.loc[idx, :])
    closest_imgs = [get_image_slice_from_row(x) for x in closest_rows] # list of (256,256,3) np arrays
    # these are both (256, 256, 3)

    # now create the figure
    #fig, axs = plt.subplots(1, len(closest_imgs)+1)
    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(src_img)
    axs[0,0].set_title(f"slice:{src_row['slice']}, id:{src_row['id']}")

    for i in range(len(closest_imgs)):
        rr = (i+1) // 3
        cc = (i+1) % 3
        axs[rr,cc].imshow(closest_imgs[i])
        axs[rr,cc].set_title(f"slice:{closest_rows[i]['slice']}, id:{closest_rows[i]['id']}")

    rt = '/gpfs/home/kn2347/results/medsam_embeddings_nearest_neighbor_plots/results_slicewindow10_deep_differentid'
    ext = f"{src_row['id']}_{src_row['slice']}_nearest_neighbors_img_embedding.png"
    joined = os.path.join(rt, ext)
    fig.savefig(joined, dpi = 300)
    plt.close(fig)


