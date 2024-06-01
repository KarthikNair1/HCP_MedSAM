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
from segment_anything import sam_model_registry
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
import nibabel as nib
from typing import Callable
from .MedSAM import *
from .dataset import *

def print_cuda_memory(gpu):
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)

    print(f'[GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb')
    print(f'[GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb')