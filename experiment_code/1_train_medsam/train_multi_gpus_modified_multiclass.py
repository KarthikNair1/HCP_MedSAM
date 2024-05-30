"""
train the image encoder and mask decoder
freeze prompt image encoder
"""


'''
python /gpfs/home/kn2347/MedSAM/train_multi_gpus_modified.py \
    --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    -task_name MedSAM_finetune_hcp_ya_constant_bbox \
    -label_id 2 \
    -checkpoint /gpfs/home/kn2347/MedSAM/medsam_vit_b.pth \
    -work_dir /gpfs/home/kn2347/results/medsam_finetuning_model_checkpoints_7-18-23 \
    -num_epochs 10 -batch_size 8 -num_workers 0 -lr 1e-5 -use_wandb True -use_amp \
    --world_size 1 --node_rank 0 -sample_n_slices 30

'''
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
from MedSAM_HCP.MedSAM import MedSAM
from MedSAM_HCP.build_sam import build_sam_vit_b_multiclass
from MedSAM_HCP.utils_hcp import *
from MedSAM_HCP.loss_funcs_hcp import *

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
parser.add_argument('--df_desired_path', type=str, default = '/gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv')
parser.add_argument('--wandb_run_name', type=str, default = None)
parser.add_argument('--suppress_train_debug_imgs', action='store_true', default = False)
parser.add_argument('--focal_loss', action='store_true', default = False)
parser.add_argument('--focal_loss_set_alpha', action='store_true', default = False, help='should we use the inverse class frequency to weight the 0 and 1s in focal loss?')
parser.add_argument('--log_val_every', type=int, default=None)

args = parser.parse_args()

if args.pool_labels and args.label_id is None: # this should not happen
    assert False

# %% set up model for fine-tuning
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = join(args.work_dir, args.task_name + '-' + run_id)

def main():
    
    debug = True
    if debug:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'


    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(__file__, join(model_save_path, run_id + '_' + os.path.basename(__file__)))

        # initialize wandb project
        if args.use_wandb and is_main_host:
            import wandb
            wandb.login()
            wandb.init(project=args.task_name, 
                name = args.wandb_run_name,
                dir = '/gpfs/home/kn2347/wandb',
                config={"lr": args.lr, 
                        "batch_size": args.batch_size,
                        "data_path": args.data_frame_path,
                        "splits_path": args.train_test_splits,
                        "model_type": args.model_type,
                        "as_one_hot": args.as_one_hot,
                        "loss_switching": args.loss_switching,
                        "loss_reweighted": args.loss_reweighted,
                        "keep_zero_weight": args.keep_zero_weight,
                        "lambda_dice": args.lambda_dice,
                        "label": args.label_id
                        })
            
            wandb.define_metric('num_training_samples')
            wandb.define_metric('epoch')
            wandb.define_metric('train_step_loss', step_metric='num_training_samples')
            wandb.define_metric('train_epoch_loss', step_metric='num_training_samples')
            wandb.define_metric('val_epoch_loss', step_metric='num_training_samples')
            wandb.define_metric('label_1/*', step_metric='num_training_samples')
            wandb.define_metric('val_dice_scores/*', step_metric='num_training_samples', summary='max')
    
    torch.cuda.set_device(gpu)
    #device = torch.device("cuda:{}".format(gpu))
    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = args.init_method,
        rank = rank,
        world_size = world_size
    )
    
    #sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)


    df_hcp = pd.read_csv('/gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv')
    df_desired = pd.read_csv(args.df_desired_path)
    NUM_CLASSES = len(df_desired)
    NUM_CLASSES_FOR_LOSS = NUM_CLASSES
    if args.label_id is not None:
        NUM_CLASSES_FOR_LOSS = 1

    if args.pool_labels: # so that we only load the architecture to produce 3 masks, not 103 masks
       NUM_CLASSES = 1

    label_converter = LabelConverter(df_hcp, df_desired)
    sam_model = build_sam_vit_b_multiclass(num_classes=max(NUM_CLASSES, 3), checkpoint=args.checkpoint) # if single class, load original SAM model
    print(sam_model)
    medsam_model = MedSAM(image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        multimask_output= args.label_id is None # 2 because unknown class is also present in single-task case
                    ).cuda()
    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
    print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb')
    print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb')
    if rank % ngpus_per_node == 0:
        print('Before DDP initialization:')
        os.system('nvidia-smi')


    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids = [gpu],
        output_device = gpu,
        gradient_as_bucket_view = True,
        find_unused_parameters = True,
        bucket_cap_mb = args.bucket_cap_mb ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
    print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb')
    print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb')
    if rank % ngpus_per_node == 0:
        print('After DDP initialization:')
        os.system('nvidia-smi')

    medsam_model.train()

    print('Number of total parameters: ', sum(p.numel() for p in medsam_model.parameters())) # 93735472
    print('Number of trainable parameters: ', sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)) # 93729252

    ## Setting up optimiser and loss func
    # only optimize the parameters of mask decoder, do not update prompt encoder or image_encoder
    #img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(medsam_model.mask_decoder.parameters())
    mask_dec_params = list(
            medsam_model.module.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        mask_dec_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print('Number of mask decoder parameters: ', sum(p.numel() for p in mask_dec_params if p.requires_grad)) # 93729252
    
    #%% train
    num_epochs = args.num_epochs
    iter_num = 0
    train_losses = []
    val_losses = []
    best_val_loss = 1e10

    train_dataset, val_dataset, test_dataset = load_datasets(args.data_frame_path, args.train_test_splits, args.label_id, bbox_shift=args.bbox_shift, sample_n_slices = args.sample_n_slices, label_converter=label_converter, NUM_CLASSES=NUM_CLASSES, as_one_hot=args.as_one_hot, pool_labels=args.pool_labels)
    
    if args.class_weights_dict_path is not None and args.loss_reweighted:
        class_weights = pickle.load(open(args.class_weights_dict_path, 'rb'))
        class_weights_tensor = torch.zeros((NUM_CLASSES_FOR_LOSS)).cuda()
        for key in class_weights.keys():
            compressed_idx = label_converter.hcp_to_compressed(key)
            if compressed_idx == 0: # unknown class
                if args.keep_zero_weight:
                    if key == 0: # if this is the true unknown class and not just another class that didn't map properly
                        class_weights_tensor[compressed_idx] = class_weights[key]
                else:
                    class_weights_tensor[compressed_idx] = 0
            else:
                class_weights_tensor[compressed_idx] = class_weights[key]
    else:
        class_weights_tensor = torch.ones((NUM_CLASSES_FOR_LOSS)).cuda()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader

    print('Number of training samples: ', len(train_dataset))
    print('Number of validation samples: ', len(val_dataset))

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

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location = loc)
            start_epoch = checkpoint['epoch'] + 1
            medsam_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        torch.distributed.barrier()
    
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")

    if not args.loss_reweighted:
        print('running WITHOUT reweighted loss')

    start_time = time.time()
    total_number_of_training_examples_seen = 0
    lambda_dice = args.lambda_dice
    loss_type = 'weighted_ce_dice_loss'
    if args.focal_loss:
        loss_type = 'focal_loss'
    focal_loss_set_alpha = args.focal_loss_set_alpha
    
    for epoch in range(start_epoch, num_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
        val_class_losses = torch.zeros((NUM_CLASSES_FOR_LOSS))
        val_dice_scores_collected_list = []

        if epoch >= 1 and args.loss_switching:
            lambda_dice = 1
        
        for step, (image_embedding, gt2D, boxes, _) in enumerate(tqdm(train_dataloader, desc = f"[RANK {rank}: GPU {gpu}]")):
            
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            #image, gt2D = image.to(device), gt2D.to(device)
            image_embedding, gt2D = image_embedding.cuda(), gt2D.cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    medsam_pred = medsam_model(image_embedding, boxes_np)
                    loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = focal_loss_set_alpha)
            
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image_embedding, boxes_np)
                loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = focal_loss_set_alpha)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_number_of_training_examples_seen += image_embedding.shape[0]

            if is_main_host and args.use_wandb:
                wandb.log({"train_step_loss": loss.item(),
                            'num_training_samples': total_number_of_training_examples_seen})

                # log loss on label 1 (or just the only label if single-task)
                label_idx = 1 if args.label_id is None else 0
                if class_losses is not None:
                    wandb.log({"label_1/label1_loss": class_losses[label_idx].item()})
                    wandb.log({"label_1/label1_DICE_loss": dice_class_losses[label_idx].item()})
                    wandb.log({"label_1/label1_CE_loss": ce_class_losses[label_idx].item()})
                

                # print sum predictions per class
                if args.as_one_hot:
                    class_1_sum = (torch.sigmoid(medsam_pred) > 0.5).int().sum(dim=(0,2,3))[label_idx].item() # C
                else:
                    class_1_sum = (torch.argmax(medsam_pred, dim=1) == 1).int().sum().item()

                wandb.log({"label_1/label1_sum": class_1_sum})
                

            if step>10 and step % 100 == 0:
                if is_main_host and not args.suppress_train_debug_imgs:
                    checkpoint = {'model': medsam_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
                    torch.save(checkpoint, join(model_save_path, 'medsam_model_latest_step.pth'))
                    label_idx = 1 if args.label_id is None else 0
                    fig, _, _ = plot_random_example(val_dataset, model_list=[medsam_model.module],
                                    names_list=[f'Label: 1, Epoch: {epoch}'],
                                    label_to_viz = label_idx,
                                    n_ex = 3,
                                    seed = 182,
                                    as_one_hot = args.as_one_hot,
                                    model_trained_on_multi_label = args.label_id is None)
                    if fig is not None:
                        wandb.log({f'train_images_debug_on_val/class_1': fig})
                        plt.close()
                    
                    fig, _ = plot_prediction_distribution(torch.sigmoid(medsam_pred)[:,label_idx,:,:])
                    wandb.log({"label_1/label1_avg_prediction": wandb.Image(fig)})

                    if class_losses is not None and args.label_id is None: # multitask with calculated class losses
                        fig, _ = plot_losses_for_classes(class_losses[1:])
                        wandb.log({f'class_loss_training_barplot': wandb.Image(fig)})
                        plt.close()

                        fig, _ = plot_worst_k_best_k(class_losses[1:], k = 5, label_converter = label_converter)
                        wandb.log({f'top5_worst5_losses_training_barplot': wandb.Image(fig)})
                        plt.close()

                    
                    



            train_epoch_loss += loss.item()
            iter_num += 1

            if args.fast_dev_run and step == 4:
                break

            # if rank % ngpus_per_node == 0:
            #     print('\n')
            #     os.system('nvidia-smi')
            #     print('\n')
        train_epoch_loss /= step+1
        train_losses.append(train_epoch_loss)

        for step, (image_embedding, gt2D, boxes, _) in enumerate(tqdm(val_dataloader, desc = f"[RANK {rank}: GPU {gpu}]")):
            if args.log_val_every is not None and epoch % args.log_val_every != 0:
                break
            with torch.no_grad():
                boxes_np = boxes.detach().cpu().numpy()
                #image, gt2D = image.to(device), gt2D.to(device)
                image_embedding, gt2D = image_embedding.cuda(), gt2D.cuda()
                

                if args.use_amp:
                    ## AMP
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        medsam_pred = medsam_model(image_embedding, boxes_np)
                        loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = focal_loss_set_alpha)
                        if class_losses is not None:
                            val_class_losses += class_losses.cpu()

                        # generate predictions for validation dice scores
                        medsam_binary_predictions_as_onehot = torch.from_numpy(convert_logits_to_preds_onehot(medsam_pred, args.as_one_hot, H=256, W=256))
                        val_dice_scores_collected_list.append(dice_scores_multi_class(medsam_binary_predictions_as_onehot, gt2D))
                else:
                    medsam_pred = medsam_model(image_embedding, boxes_np)
                    loss, class_losses, dice_class_losses, ce_class_losses = loss_handler(loss_type, medsam_pred, gt2D, class_weights_tensor, lambda_dice, args.as_one_hot, focal_alpha = focal_loss_set_alpha)
                    if class_losses is not None:
                        val_class_losses += class_losses.cpu()

                    # generate predictions for validation dice scores
                    medsam_binary_predictions_as_onehot = torch.from_numpy(convert_logits_to_preds_onehot(medsam_pred, args.as_one_hot, H=256, W=256))
                    val_dice_scores_collected_list.append(dice_scores_multi_class(medsam_binary_predictions_as_onehot, gt2D))

            val_epoch_loss += loss.item()

            if args.fast_dev_run and step > 0:
                break

        val_epoch_loss /= step+1
        val_losses.append(val_epoch_loss)
        val_class_losses /= step+1
        print('Val Class Losses:')
        print(val_class_losses)
        # i think the loss calculation is a little funky right now because it divides by the number of batches

        # Check CUDA memory usage
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
        print('\n')
        print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb')
        print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb')
        print(f'[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb')
        print('\n')

        if args.use_wandb and is_main_host: 
            wandb.log({"train_epoch_loss": train_epoch_loss,
                        "val_epoch_loss": val_epoch_loss,
                        "num_training_samples": total_number_of_training_examples_seen, 
                        "epoch":epoch}, commit=True)

            labels_to_track = list(range(NUM_CLASSES))
            # log example segmentations on validation images

            if not args.fast_dev_run and args.label_id is None and args.log_val_every is None: # if not dev-running and we are doing multi-task
                print('plotting validation examples')
                for i, label in enumerate(tqdm(labels_to_track)):
                    text_label = label_converter.compressed_to_name(label)
                    fig, _, _ = plot_random_example(val_dataset, model_list=[medsam_model.module],
                                        names_list=[f'Label: {text_label}, Epoch: {epoch}'],
                                        label_to_viz = label,
                                        n_ex = 3,
                                        seed = 182,
                                        as_one_hot=args.as_one_hot)

                    if fig is not None:
                        wandb.log({f'val_images/class_{text_label}': fig})
                        plt.close()
                    
                    fig, _ = plot_class_losses_vs_weights(val_class_losses, class_weights_tensor)
                    wandb.log({'val_class_loss_vs_weights': wandb.Image(fig)})
                    plt.close()
            elif args.label_id is not None:
                label_idx = 0 # single task so the "1" is actually at index 0
                fig, _, _ = plot_random_example(val_dataset, model_list=[medsam_model.module],
                                names_list=[f'Label: 1, Epoch: {epoch}'],
                                label_to_viz = label_idx,
                                n_ex = 3,
                                seed = 182,
                                as_one_hot = args.as_one_hot,
                                model_trained_on_multi_label = args.label_id is None)
                if fig is not None:
                    wandb.log({f'val_images_examples/class_1': fig})
                    plt.close()


            # log validation dice scores
            if not (args.log_val_every is not None and epoch%args.log_val_every!=0):
                val_dice_scores = torch.cat(val_dice_scores_collected_list, dim=0).nanmean(dim=0) # stack list of (B, C) tensors by dim=0 and nanmean by dim=0
                for class_num in range(NUM_CLASSES_FOR_LOSS):
                    text_label = label_converter.compressed_to_name(class_num)
                    wandb.log({f'val_dice_scores/class_{text_label}': val_dice_scores[class_num].item()})

            # for pooled model, plot random bbox outputs
            if args.pool_labels and epoch % 5 == 0:
                
                fig, ax = plot_random_bboxes(val_dataset, medsam_model.module, dev='cuda', n_ex = 5, seed=182)
                if fig is not None:
                    wandb.log({f'val_images/random_bbox': wandb.Image(fig)})
                    plt.close()

                
            
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Loss: {train_epoch_loss}, Val Loss: {val_epoch_loss}')
        # save the model checkpoint
        if is_main_host:
            checkpoint = {'model': medsam_model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, join(model_save_path, 'medsam_model_latest.pth'))
            
            ## save the best model
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                torch.save(checkpoint, join(model_save_path, 'medsam_model_best.pth'))
        torch.distributed.barrier()
    total_time = time.time() - start_time
    print('Training loop took %s seconds ---' % total_time)
    if args.use_wandb and is_main_host:
        wandb.finish()

if __name__ == "__main__":
    main()


# %%
