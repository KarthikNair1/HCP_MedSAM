#%% setup environment
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
import monai

import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import pandas as pd
import nibabel as nib
import pickle
import time
import sys
sys.path.append('./modified_medsam_repo')
from segment_anything import sam_model_registry
from MedSAM_HCP.dataset import MRIDataset, MRIDataset_Imgs_MedSAM, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, logits_to_pred_probs
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass, resume_model_optimizer_and_epoch_from_checkpoint, save_model_optimizer_and_epoch_to_checkpoint
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *
from MedSAM_HCP.logging_functions import init_wandb, print_cuda_memory, log_losses_step, log_predicted_probabilities, log_class_losses_as_barplots
from MedSAM_HCP.train_MedSAM_functions import retrieve_class_weights_tensor, train_step, validate_step, log_stuff_at_step, log_stuff_at_epoch
