# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.spine_plot import *
from torchvision.transforms import functional as F
from scipy.optimize import linear_sum_assignment

def mean(l):
    return sum(l) / len(l) if len(l) > 0 else 0

def plot_images(writer, step, samples, outputs, targets, indices, epoch, i, tag='train', folder='def'):
    folder = f'spine_plot/{folder}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for d in range(len(samples)):
        img = spine_to_pil(samples[d])
        img = spine_plot_centers(img, outputs['pred_boxes'][d])
        img = spine_plot_connection(img, targets[d][:,1:3][indices[d][1]], outputs['pred_boxes'][d][indices[d][0]])
        img = spine_class(img, outputs['pred_logits'][d])
        img.save(f'{folder}/{epoch:03d}_{i:03d}_{d:02d}_{tag}_all.jpg')
        writer.add_image(f'{tag}/all_points', F.to_tensor(img), global_step=step + d)

        img = spine_to_pil(samples[d])
        img = spine_plot_centers(img, targets[d][:,1:3][indices[d][1]], color=(0, 255, 0))
        img = spine_plot_centers(img, outputs['pred_boxes'][d], threshold=0.5, logits=outputs['pred_logits'][d])
        img.save(f'{folder}/{epoch:03d}_{i:03d}_{d:02d}_{tag}_out.jpg')
        writer.add_image(f'{tag}/out_points', F.to_tensor(img), global_step=step + d)

def spine_evaluation(src_outputs, logits, src_targets, threshold=0.5, r=10):
    outputs, targets = torch.clone(src_outputs).detach(), torch.clone(src_targets).detach()
    outputs = outputs[logits.squeeze(-1) > threshold]
    C = torch.cdist(outputs, targets, p=2).cpu()
    idx_out, idx_tar = linear_sum_assignment(C)
    outputs, targets = outputs[idx_out], targets[idx_tar]
    dist = torch.cdist(outputs, targets, p=2).diag()

    dist *= 360 # which is the random crop size
    per_pix_dist = 0.2929687786102323 # mm per pixel
    dist *= per_pix_dist # actual distance in mm

    in_dist = dist[dist < r]
    out_dist = dist[dist >= r]

    FN = len(src_targets) - len(in_dist)
    FP = len(out_dist)
    TP = len(in_dist)
    return FN, FP, TP, in_dist


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer = None, args = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    losses_items = []

    FNs, FPs, TPs, AVGs, TAR = [], [], [], [], []

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(samples)

        # import numpy as np
        # couples = []
        # for x in np.arange(1/10, 1, 1/5):
        #     for y in np.arange(1/12, 1, 1/6):
        #         couples.append(torch.tensor([x, y]))
        # outputs['pred_boxes'][0] = torch.cat(couples).view(-1, 2)

        loss_dict, indices = criterion(outputs, targets)

        if epoch % 50 == 0 or epoch == (args.epochs - 1):
            step = (epoch * len(data_loader) + i) * args.batch_size
            plot_images(writer, step, samples, outputs, targets, indices, epoch, i, tag='train', folder=args.comment)

        for i in range(len(samples)):
            FN, FP, TP, in_dist = spine_evaluation(outputs['pred_boxes'][i], outputs['pred_logits'][i], targets[i][:, 1:3])
            FNs.append(FN)
            FPs.append(FP)
            TPs.append(TP)
            TAR.append(len(targets[i]))
            AVGs.append(in_dist)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict).float()

        not_used_keys = [k for k in loss_dict.keys() if k not in weight_dict.keys()]
        if len(not_used_keys) > 0 and i == 0:
            print(f'[WARNING] these key are not used to calculate the loss: {not_used_keys}')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        losses_items.append(loss_value)
        print(f"{epoch:03d}_{i:03d} loss_value: {loss_value:.04f} mean {mean(losses_items):.04f} loss_centers {loss_dict['loss_centers'].item():.04f} loss_bce {loss_dict['loss_bce'].item():.04f} loss_spine_l1 {loss_dict['loss_spine_l1'].item():.04f}")
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    writer.add_scalar('train_metric/FN', sum(FNs) / sum(TAR), global_step=epoch)
    writer.add_scalar('train_metric/FP', sum(FPs) / sum(TAR), global_step=epoch)
    writer.add_scalar('train_metric/TP', sum(TPs) / sum(TAR), global_step=epoch)
    if len(torch.cat(AVGs)) > 0:
        writer.add_scalar('train_metric/avg_dist', torch.cat(AVGs).mean(), global_step=epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, epoch, writer = None, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator = None
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    FNs, FPs, TPs, AVGs, TAR = [], [], [], [], []

    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(samples)
        loss_dict, indices = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        if epoch % 50 == 0 or (args.epochs - 1):
            step = (epoch * len(data_loader) + i) * args.batch_size
            plot_images(writer, step, samples, outputs, targets, indices, epoch, i, tag='test', folder=args.comment)

        for i in range(len(samples)):
            FN, FP, TP, in_dist = spine_evaluation(outputs['pred_boxes'][i], outputs['pred_logits'][i], targets[i][:, 1:3])
            FNs.append(FN)
            FPs.append(FP)
            TPs.append(TP)
            TAR.append(len(targets[i]))
            AVGs.append(in_dist)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    writer.add_scalar('test_metric/FN', sum(FNs) / sum(TAR), global_step=epoch)
    writer.add_scalar('test_metric/FP', sum(FPs) / sum(TAR), global_step=epoch)
    writer.add_scalar('test_metric/TP', sum(TPs) / sum(TAR), global_step=epoch)
    if len(torch.cat(AVGs)) > 0:
        writer.add_scalar('test_metric/avg_dist', torch.cat(AVGs).mean(), global_step=epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
