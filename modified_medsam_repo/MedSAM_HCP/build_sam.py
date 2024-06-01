from functools import partial
from pathlib import Path
import urllib.request
import torch

from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

def build_sam_vit_b_multiclass(num_classes, checkpoint=None):

    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

        
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        if not f'mask_decoder.output_hypernetworks_mlps.{num_classes}.layers.0.weight' in state_dict.keys():
            print('filling weights for additional classes')
            for class_num in range(4, num_classes+1):
                for idx in range(3):
                    state_dict[f'mask_decoder.output_hypernetworks_mlps.{class_num}.layers.{idx}.weight'] = state_dict[f'mask_decoder.output_hypernetworks_mlps.0.layers.{idx}.weight']
                    state_dict[f'mask_decoder.output_hypernetworks_mlps.{class_num}.layers.{idx}.bias'] = state_dict[f'mask_decoder.output_hypernetworks_mlps.0.layers.{idx}.bias']
                state_dict['mask_decoder.mask_tokens.weight'] = torch.repeat_interleave(state_dict['mask_decoder.mask_tokens.weight'][0:1,:], repeats = num_classes+1, dim=0)
                state_dict['mask_decoder.iou_prediction_head.layers.2.weight'] = torch.repeat_interleave(state_dict['mask_decoder.iou_prediction_head.layers.2.weight'][0:1,:], repeats = num_classes+1, dim=0)
                state_dict['mask_decoder.iou_prediction_head.layers.2.bias'] = torch.repeat_interleave(state_dict['mask_decoder.iou_prediction_head.layers.2.bias'][0:1], repeats = num_classes+1, dim=0)

        sam.load_state_dict(state_dict)
    return sam

def resume_model_optimizer_and_epoch_from_checkpoint(args, rank, gpu, medsam_model, optimizer):
    if os.path.isfile(args.resume):
        print(rank, "=> loading checkpoint '{}'".format(args.resume))
        ## Map model to be loaded to specified single GPU
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location = loc)
        start_epoch = checkpoint['epoch'] + 1
        medsam_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        return medsam_model, optimizer, start_epoch
    else:
        print('Not a valid resume path')
        return None, None, None