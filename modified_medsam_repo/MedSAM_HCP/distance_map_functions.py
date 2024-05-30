import argparse
from pathlib import Path
from operator import add
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import partial

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import torch.sparse
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from skimage.io import imsave
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt as eucl_distance

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    # seg must be a one-hot encoded np array of shape (C, H, W)
    num_classes: int = seg.shape[0]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(num_classes):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res
