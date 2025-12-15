import os
import torch
#from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
#from transformers import SamModel, SamProcessor
#import ptp_utils
#from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion
#from attention_store import AttentionStore
from PerMIRS.visualization_utils import bbox_from_mask
#from sam_utils import show_points_on_image, show_masks_on_image, show_single_mask_on_image
import torch.nn.functional as F
from util.commons import setup_logging, make_deterministic, resume_from_checkpoint
import argparse
from models.sansa.sansa import build_sansa
import opts
from torch.utils.data import DataLoader
import util.misc as utils
from util.metrics import db_eval_iou
from util.promptable_utils import build_prompt_dict
import sys


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mIoU = eval_permis(model, args)
    return mIoU



def eval_permis(model: torch.nn.Module, args: argparse.Namespace) -> float:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    from ds import build_dataset
    validation_ds = 'permis' if args.dataset_file == 'multi' else args.dataset_file 
    print(f'Evaluating {validation_ds}')
    ds = build_dataset(validation_ds, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    runn_avg, runn_acc = 0.0, 0.0
    # create a dictionary to store IoU for each class
    class_iou, class_acc = {}, {}
    pbar = tqdm(dataloader, ncols=100, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    print(f"Number of batches: {len(dataloader)}")
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']
        class_name = batch['class_name'][0]

        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            outputs = model(imgs, prompt_dict)

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()

        iou, acc = db_eval_iou((query_mask.numpy() > 0), pred_masks[-1:].numpy())
        iou, acc = iou.item(), acc.item()
        runn_avg += iou
        runn_acc += acc

        # update class_iou dictionary
        if class_name not in class_iou:
            class_iou[class_name] = []
            class_acc[class_name] = []
        class_iou[class_name].append(iou)
        class_acc[class_name].append(acc)

        if (idx + 1) % 50 == 0:
            pbar.set_description(f"runn. avg = {(runn_avg / (idx + 1)) * 100:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=pred_masks[-1].numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                iou=iou,
            )
    for class_name in class_iou:
        class_iou[class_name] = sum(class_iou[class_name]) / len(class_iou[class_name])
        class_acc[class_name] = sum(class_acc[class_name]) / len(class_acc[class_name])
        print(f"Class {class_name}, IoU= {class_iou[class_name] * 100:.1f}, Acc= {class_acc[class_name] * 100:.1f}")

    mAcc = runn_acc / len(dataloader)
    print(f"mAcc = {mAcc * 100:.1f}")
    mIoU = runn_avg / len(dataloader)
    print(f"mIoU = {mIoU * 100:.1f}")
    return mIoU


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    device = "cuda"
    main(args)
