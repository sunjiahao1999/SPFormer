import argparse
import os.path as osp

import gorilla
import torch
import numpy as np

from spformer.dataset import build_dataloader, build_dataset
from spformer.evaluation import ScanNetEval
from spformer.model import SPFormer
from spformer.utils import get_root_logger, save_gt_instances, save_pred_instances
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    model = SPFormer(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            result = model(batch, mode="predict")
            results.append(result)
            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not cfg.data.test.prefix == "test":
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES, dataset.NYU_ID)
        scannet_eval.evaluate(pred_insts, gt_insts)

    # save output
    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.prefix == "test":
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
