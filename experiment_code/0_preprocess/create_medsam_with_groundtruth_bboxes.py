import pandas as pd
import tqdm
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label", type=int)
args = parser.parse_args()

label=args.label
dfo = pd.read_csv(f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/{label}/0.1/isolated_path_df_constant_bbox.csv')
b1 = []
b2 = []
b3 = []
b4 = []
for i, r in tqdm.tqdm(dfo.iterrows()):
    nam = f'{r["id"]}_slice{r["slice"]}.txt'
    pattern = -1
    for tag in ['train', 'val', 'test']:

        pat = f'/gpfs/data/cbi/hcp/hcp_ya/hcp_ya_slices_npy/dir_structure_for_yolov7/{tag}/labels/{nam}'
        if os.path.exists(pat):
            pattern = pat
            break
    assert pattern != -1


    xx = pd.read_csv(pattern, sep = ' ', header=None, names = ['label', 'x_cent', 'y_cent', 'width', 'height'])
    xx['x1'] = ((xx['x_cent'] - xx['width']/2.0) * 256).astype(int)
    xx['x2'] = ((xx['x_cent'] + xx['width']/2.0) * 256).astype(int)

    xx['y1'] = ((xx['y_cent'] - xx['height']/2.0) * 256).astype(int)
    xx['y2'] = ((xx['y_cent'] + xx['height']/2.0) * 256).astype(int)

    subdf = xx[xx['label'] == label]

    if subdf.shape[0] == 0:
        b1.append(np.nan)
        b2.append(np.nan)
        b3.append(np.nan)
        b4.append(np.nan)
    elif subdf.shape[0] == 1:
        b1.append(subdf['x1'].item())
        b2.append(subdf['y1'].item())
        b3.append(subdf['x2'].item())
        b4.append(subdf['y2'].item())
    else:
        assert False

zz = dfo.copy()
zz['bbox_0'] = b1
zz['bbox_1'] = b2
zz['bbox_2'] = b3
zz['bbox_3'] = b4

zz.to_csv(f'/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/{label}/0.1/isolated_path_df_bboxes_from_ground_truth.csv', index=False)
