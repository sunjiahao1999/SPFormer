import argparse
import datetime
import gorilla
import os
import os.path as osp
import shutil
import time
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from spformer.dataset import build_dataloader, build_dataset
from spformer.evaluation import ScanNetEval
from spformer.model import SPFormer
from spformer.utils import AverageMeter, get_root_logger


def get_args():
    parser = argparse.ArgumentParser('SPFormer')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epoch, model, dataloader, optimizer, lr_scheduler, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    meter_dict = {}
    end = time.time()

    for i, batch in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)
        loss, log_vars = model(batch, mode='loss')

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # time and print
        remain_iter = len(dataloader) * (cfg.train.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']
        if i % 10 == 0:
            log_str = f'Epoch [{epoch}/{cfg.train.epochs}][{i}/{len(dataloader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, '
            log_str += f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)

    # update lr
    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr']

    # log and save
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    save_file = osp.join(cfg.work_dir, 'lastest.pth')
    gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler)


@torch.no_grad()
def eval(epoch, model, dataloader, cfg, logger, writer):
    logger.info('Validation')
    pred_insts, gt_insts = [], []
    progress_bar = tqdm(total=len(dataloader))
    val_dataset = dataloader.dataset

    model.eval()
    for batch in dataloader:
        result = model(batch, mode='predict')
        pred_insts.append(result['pred_instances'])
        gt_insts.append(result['gt_instances'])
        progress_bar.update()
    progress_bar.close()

    # evaluate
    logger.info('Evaluate instance segmentation')
    scannet_eval = ScanNetEval(val_dataset.CLASSES)
    eval_res = scannet_eval.evaluate(pred_insts, gt_insts)
    writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
    writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
    writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
    logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(eval_res['all_ap'], eval_res['all_ap_50%'],
                                                                  eval_res['all_ap_25%']))

    # save
    save_file = osp.join(cfg.work_dir, f'epoch_{epoch:04d}.pth')
    gorilla.save_checkpoint(model, save_file)


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./exps', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'config: {args.config}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # seed
    gorilla.set_random_seed(cfg.train.seed)

    # model
    model = SPFormer(**cfg.model).cuda()
    count_parameters = gorilla.parameter_count(model)['']
    logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')

    # optimizer and scheduler
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)

    # pretrain or resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = gorilla.resume(args.resume, logger, model, optimizer, lr_scheduler)
    elif cfg.train.pretrain:
        logger.info(f'Load pretrain from {cfg.train.pretrain}')
        gorilla.load_checkpoint(model, cfg.train.pretrain, strict=False)

    # train and val dataset
    train_dataset = build_dataset(cfg.data.train, logger)
    train_loader = build_dataloader(train_dataset, **cfg.dataloader.train)
    if not args.skip_validate:
        val_dataset = build_dataset(cfg.data.val, logger)
        val_loader = build_dataloader(val_dataset, **cfg.dataloader.val)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train(epoch, model, train_loader, optimizer, lr_scheduler, cfg, logger, writer)
        if not args.skip_validate and (epoch % cfg.train.interval == 0):
            eval(epoch, model, val_loader, cfg, logger, writer)
        writer.flush()


if __name__ == '__main__':
    main()
