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
from segment_anything import sam_model_registry
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

from MedSAM_HCP.dataset import MRIDataset, load_datasets
from MedSAM_HCP.MedSAM import MedSAM, logits_to_pred_probs
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass, resume_model_optimizer_and_epoch_from_checkpoint, save_model_optimizer_and_epoch_to_checkpoint
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *
from MedSAM_HCP.logging_functions import init_wandb, print_cuda_memory, log_losses_step, log_predicted_probabilities, log_class_losses_as_barplots
from MedSAM_HCP.train_MedSAM_functions import retrieve_class_weights_tensor, train_step, validate_step, log_stuff_at_step, log_stuff_at_epoch

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_frame_path', type=str,
                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv',
                    help='path to pandas dataframe with all paths for training')
parser.add_argument('-train_test_splits', type=str,
                    default='/gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle',
                    help='path to pickle file containing a dictionary with train, val, and test IDs')
parser.add_argument('-class_weights_dict_path', type=str,
                    default=None,
                    help='path to pickle file containing a dictionary with keys as class numbers and values as weights for the loss function. class numbers should be according to HCP mapping')
parser.add_argument('-task_name', type=str, default='MedSAM-ViT-B')
parser.add_argument('-label_id', type=int, default=None)
parser.add_argument('-model_type', type=str, default='vit_b')
parser.add_argument('-checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth')
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument('--load_pretrain', type=bool, default=True, 
                    help='')
parser.add_argument('-pretrain_model_path', type=str, default='')
parser.add_argument('-work_dir', type=str, default='./work_dir')
parser.add_argument('--as_one_hot', action = 'store_false', help='Should each pixel be allowed to have multiple segmentation classes?')
parser.add_argument('--pool_labels', action='store_true', default = False, help='treat each (mri, class) pair as a single example to a single-task model and train across all of these')

# train
parser.add_argument('-num_epochs', type=int, default=1000)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-num_workers', type=int, default=8)
parser.add_argument('-sample_n_slices', type=int, default=None)
parser.add_argument('--loss_reweighted', action='store_true', default=False)
parser.add_argument('--lambda_dice', type=float, default=0.5, help='What fraction of the total loss should the dice loss contribute to? (default: 0.5)')
parser.add_argument('--loss_switching', action='store_true', default=False)
parser.add_argument('--keep_zero_weight', action='store_true', default=False, help='Should we keep the classweight for class 0 (background class) instead of replacing with 0 for loss calculation? (default: False)')
parser.add_argument('--bbox_shift', type=int, default=0)
# Optimizer parameters
parser.add_argument('-weight_decay', type=float, default=0.01,
                    help='weight decay (default: 0.01)')
parser.add_argument('-lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('-use_wandb', type=bool, default=False, 
                    help='use wandb to monitor training')        
parser.add_argument('-wandb_dir', type=bool, default='/gpfs/home/kn2347/wandb', 
                    help='Directory of wandb')    
parser.add_argument('-use_amp', action='store_true', default=False, 
                    help='use amp')           
## Distributed training args
parser.add_argument('--world_size', type=int, help='world size')
parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
parser.add_argument('--bucket_cap_mb', type = int, default = 25,
                    help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
parser.add_argument('--grad_acc_steps', type = int, default = 1,
                    help='Gradient accumulation steps before syncing gradients for backprop')
parser.add_argument('--resume', type = str, default = '',
                    help="Resuming training from checkpoint")
parser.add_argument('--init_method', type = str, default = "env://")
parser.add_argument('--fast_dev_run', action='store_true', default=False, help='runs a single batch of training and validation')
parser.add_argument('--df_starting_mapping_path', type=str, default = '/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv', help = 'Path to dataframe holding the integer labels in the segmentation numpy files and the corresponding text label, prior to subsetting for only the labels we are interested in.')
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
parser.add_argument('--wandb_run_name', type=str, default = None)
parser.add_argument('--suppress_train_debug_imgs', action='store_true', default = False)
parser.add_argument('--focal_loss', action='store_true', default = False)
parser.add_argument('--focal_loss_set_alpha', action='store_true', default = False, help='should we use the inverse class frequency to weight the 0 and 1s in focal loss?')
parser.add_argument('--log_val_every', type=int, default=None)

args = parser.parse_args()

if args.pool_labels and args.label_id is None: # this should not happen
    assert False

# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = join(args.work_dir, args.task_name + '-' + run_id)

def main():
    
    debug = True
    if debug:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'


    ngpus_per_node = torch.cuda.device_count()
    print("Spawning processes")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):

    # Determine the rank of the current process in the distributed training setup
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")

    # Check if the current process is the main host (rank )
    is_main_host = rank == 0

    # If the current process is the main host, set up the model save directory and initialize wandb
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(__file__, join(model_save_path, run_id + '_' + os.path.basename(__file__)))

        # initialize wandb project if specified
        if args.use_wandb and is_main_host:
            import wandb
            init_wandb(args)
    
    # Set the current GPU for PyTorch
    torch.cuda.set_device(gpu)

    # Initialize the distributed process group
    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = args.init_method,
        rank = rank,
        world_size = world_size
    )

    # load names of regions and corresponding label number per FreeSurfer
    df_hcp = pd.read_csv(args.df_starting_mapping_path)

    # load names of desired subset of regions and corresponding label number in the range 0...102
    df_desired = pd.read_csv(args.df_desired_path)


    is_multitask = args.label_id is None
    NUM_CLASSES = len(df_desired)
    NUM_CLASSES_FOR_LOSS = NUM_CLASSES
    if is_multitask:
        NUM_CLASSES_FOR_LOSS = 1

    if args.pool_labels: # so that we only load the architecture to produce 3 masks, not 103 masks
       NUM_CLASSES = 1

    

    label_converter = LabelConverter(df_hcp, df_desired) # compresses label numbers to range 0...102

    # build SAM model from checkpoint
    sam_model = build_sam_vit_b_multiclass(num_classes=max(NUM_CLASSES, 3), checkpoint=args.checkpoint) # if single class, load original SAM model

    # initialize MedSAM model object using the loaded SAM model
    medsam_model = MedSAM(image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        multimask_output= is_multitask # 2 because unknown class is also present in single-task case
                    ).cuda()

    # memory before model initialized
    print(f'[RANK {rank}] Before DDP initialized:')
    print_cuda_memory(gpu)

    # Initialize the model for distributed training
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids = [gpu],
        output_device = gpu,
        gradient_as_bucket_view = True,
        find_unused_parameters = True,
        bucket_cap_mb = args.bucket_cap_mb ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )
    
    # memory after model initialized
    print(f'[RANK {rank}] After DDP initialized:')
    print_cuda_memory(gpu)

    # Set the model to training mode
    medsam_model.train()

    print('Number of total parameters: ', sum(p.numel() for p in medsam_model.parameters())) 
    print('Number of trainable parameters: ', sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))

    # Setting up optimiser and loss func
    mask_dec_params = list(
            medsam_model.module.mask_decoder.parameters() # only optimize the parameters of mask decoder, do not update prompt encoder or image_encoder
    )
    optimizer = torch.optim.AdamW(
        mask_dec_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print('Number of mask decoder parameters: ', sum(p.numel() for p in mask_dec_params if p.requires_grad)) # 93729252

    train_dataset, val_dataset, test_dataset = load_datasets(
        args.data_frame_path, args.train_test_splits, args.label_id, 
        bbox_shift=args.bbox_shift, sample_n_slices = args.sample_n_slices, 
        label_converter=label_converter, NUM_CLASSES=NUM_CLASSES, 
        as_one_hot=args.as_one_hot, pool_labels=args.pool_labels
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # get class weights if specified, or populate with tensor of 1's if not specified
    class_weights_tensor = retrieve_class_weights_tensor(NUM_CLASSES_FOR_LOSS, label_converter, args)
    
    print('Number of training samples: ', len(train_dataset))
    print('Number of validation samples: ', len(val_dataset))

    # Distributed sampler does shuffling intrinsically,
    # So no need to shuffle in dataloader if sampler is defined
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = (train_sampler is None),
        num_workers = args.num_workers,
        pin_memory = True,
        sampler = train_sampler
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        shuffle = (val_sampler is None),
        num_workers = args.num_workers,
        pin_memory = True,
        sampler = val_sampler
    )

    # if checkpoint specified, make sure to load model and optimizer from this checkpoint and resume from that epoch
    start_epoch = 0
    if args.resume is not None:
        medsam_model, optimizer, start_epoch = resume_model_optimizer_and_epoch_from_checkpoint(args, rank, gpu, medsam_model, optimizer)
        torch.distributed.barrier()
    
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")
    else:
        scaler = None

    if not args.loss_reweighted:
        print('running WITHOUT reweighted loss')

    # set up train parameters and loss trackers
    num_epochs = args.num_epochs
    train_losses = []
    val_losses = []
    best_val_loss = 1e10
    start_time = time.time()
    total_number_of_training_examples_seen = 0
    lambda_dice = args.lambda_dice
    loss_type = 'weighted_ce_dice_loss'
    if args.focal_loss:
        loss_type = 'focal_loss'
    focal_loss_set_alpha = args.focal_loss_set_alpha
    
    # training loop
    for epoch in range(start_epoch, num_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
        val_class_losses = torch.zeros((NUM_CLASSES_FOR_LOSS)) # calculated loss by class
        val_dice_scores_collected_list = []

        if epoch >= 1 and args.loss_switching:
            lambda_dice = 1
        
        # train from all TRAINING examples in this epoch
        for step, (image_embedding, gt2D, boxes, _) in enumerate(tqdm(train_dataloader, desc = f"[RANK {rank}: GPU {gpu}]")):
            
            # if fast_dev_run specified, exit training loop early
            if args.fast_dev_run and step == 5:
                break

            loss, class_losses, dice_class_losses, ce_class_losses, medsam_pred = train_step(
                medsam_model, optimizer, scaler, class_weights_tensor, 
                lambda_dice, image_embedding, gt2D, boxes, args
            )
            
            total_number_of_training_examples_seen += image_embedding.shape[0]

            # update running training loss
            train_epoch_loss += loss.item()

            if is_main_host:
                if step>10 and step % 100 == 0: # if we reach a "checkpoint" step

                    # then save checkpoint
                    save_model_optimizer_and_epoch_to_checkpoint(
                        args=args, medsam_model=medsam_model, optimizer=optimizer, epoch=epoch, 
                        filename = os.path.join(model_save_path, 'medsam_model_latest_step.pth')
                    )

                    # also log metrics and example images
                    log_stuff_at_step(
                        loss=loss, class_losses=class_losses, 
                        dice_class_losses=dice_class_losses, ce_class_losses=ce_class_losses,
                        medsam_pred=medsam_pred, 
                        total_number_of_training_examples_seen=total_number_of_training_examples_seen,
                        medsam_model=medsam_model, val_dataset = val_dataset,
                        epoch=epoch, args=args
                    )

        # validate on all VALIDATION samples in this epoch
        for step, (image_embedding, gt2D, boxes, _) in enumerate(tqdm(val_dataloader, desc = f"[RANK {rank}: GPU {gpu}]")):
            if args.log_val_every is not None and epoch % args.log_val_every != 0:
                break

            if args.fast_dev_run and step > 1:
                break

            loss, class_losses, dice_class_losses, ce_class_losses, dice_scores_multiclass = validate_step(
                medsam_model, optimizer, scaler, 
                class_weights_tensor, lambda_dice, image_embedding, gt2D, boxes, args
            )

            if class_losses is not None: # if there are multiple classes
                val_class_losses += class_losses.cpu() # collect losses for each class

            val_dice_scores_collected_list.append(dice_scores_multiclass)
            val_epoch_loss += loss.item()
        
        # calculate losses for this epoch and update the global train and val losses
        # Note that loss calculation divides by the number of batches, thus sensitive to batch size
        train_epoch_loss /= step+1
        train_losses.append(train_epoch_loss)
        val_epoch_loss /= step+1
        val_losses.append(val_epoch_loss)
        val_class_losses /= step+1

        val_dice_scores = torch.cat(val_dice_scores_collected_list, dim=0).nanmean(dim=0) # stack list of (B, C) tensors by dim=0 and nanmean by dim=0

        # Check CUDA memory usage
        print_cuda_memory(gpu)

        if is_main_host: 
            # save epoch checkpoint
            save_model_optimizer_and_epoch_to_checkpoint(
                args=args, medsam_model=medsam_model, optimizer=optimizer, epoch=epoch, 
                filename = os.path.join(model_save_path, 'medsam_model_latest.pth')
            )
            
            # save the best model
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                save_model_optimizer_and_epoch_to_checkpoint(
                    args=args, medsam_model=medsam_model, optimizer=optimizer, epoch=epoch, 
                    filename = os.path.join(model_save_path, 'medsam_model_best.pth')
                )
            
            # log epoch-level metrics and example images
            log_stuff_at_epoch(
                train_epoch_loss=train_epoch_loss, val_epoch_loss=val_epoch_loss,
                val_class_losses=val_class_losses, class_weights_tensor=class_weights_tensor,
                val_dice_scores=val_dice_scores, epoch=epoch, 
                total_number_of_training_examples_seen=total_number_of_training_examples_seen,
                medsam_model=medsam_model, val_dataset=val_dataset, 
                NUM_CLASSES=NUM_CLASSES, NUM_CLASSES_FOR_LOSS=NUM_CLASSES_FOR_LOSS,
                label_converter=label_converter, args=args
            )

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {train_epoch_loss}, Val Loss: {val_epoch_loss}')
        torch.distributed.barrier()
    total_time = time.time() - start_time
    print('Training loop took %s seconds ---' % total_time)
    if args.use_wandb and is_main_host:
        wandb.finish()

if __name__ == "__main__":
    main()


# %%
