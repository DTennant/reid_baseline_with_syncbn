# Creator: Tennant
# Email: Tennant_1999@outlook.com

import os
import os.path as osp

# PyTorch as the main lib for neural network
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision as tv

# Use visdom for moniting the training process
import visdom
from utils import Visualizer
from utils import setup_logger

# Use yacs for training config management
# argparse for overwrite
from config import cfg
import argparse

# import losses and model
from losses import make_loss
from model import build_model, convert_model
from trainer import BaseTrainer

# dataset
from dataset import make_dataloader

from optim import make_optimizer, WarmupMultiStepLR


from evaluate import eval_func, euclidean_dist, re_rank
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="ReID training")
    parser.add_argument('-c', '--config_file', type=str,
                        help='the path to the training config')
    parser.add_argument('-t', '--test', action='store_true',
                        default=False, help='Model test')
    parser.add_argument('opts', help='overwriting the training config' 
                        'from commandline', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.test:
        test(args)
    else:
        train(args)

def train(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_gpus = torch.cuda.device_count()

    logger = setup_logger('reid_baseline', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info(args)
    logger.info('Running with config:\n{}'.format(cfg))

    train_dl, val_dl, num_query, num_classes = make_dataloader(cfg, num_gpus) 

    model = build_model(cfg, num_classes)
    if num_gpus > 1:
        # convert to use sync_bn
        model = nn.DataParallel(model)
        model = convert_model(model)
        logger.info('More than one gpu used, convert model to use SyncBN.')

    loss_func = make_loss(cfg, num_classes)

    optim = make_optimizer(cfg, model, num_gpus)
    scheduler = WarmupMultiStepLR(optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    trainer = BaseTrainer(cfg, model, train_dl, val_dl,
                          optim, scheduler, loss_func, num_query)

    for epoch in range(trainer.epochs):
        for batch in trainer.train_dl:
            trainer.step(batch)
            trainer.handle_new_batch()
        trainer.handle_new_epoch()

def test(args):
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger('reid_baseline.eval', cfg.OUTPUT_DIR, 0, train=False)

    logger.info('Running with config:\n{}'.format(cfg))
    
    _, val_dl, num_query, num_classes = make_dataloader(cfg)

    model = build_model(cfg, num_classes)
    if cfg.TEST.MULTI_GPU:
        model = nn.DataParallel(model)
        model = convert_model(model)
        logger.info('Use multi gpu to inference')
    para_dict = torch.load(cfg.TEST.WEIGHT)
    model.load_state_dict(para_dict)
    model.cuda()
    model.eval()

    feats, pids, camids = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_dl, total=len(val_dl),
                         leave=False):
            data, pid, camid = batch
            data = data.cuda()
            feat = model(data).detach().cpu()
            feats.append(feat)
            pids.append(pid)
            camids.append(camid)
    feats = torch.cat(feats, dim=0)
    pids = torch.cat(pids, dim=0)
    camids = torch.cat(camids, dim=0)

    query_feat = feats[:num_query]
    query_pid = pids[:num_query]
    query_camid = camids[:num_query]

    gallery_feat = feats[num_query:]
    gallery_pid = pids[num_query:]
    gallery_camid = camids[num_query:]
    
    distmat = euclidean_dist(query_feat, gallery_feat)

    cmc, mAP = eval_func(distmat.numpy(), query_pid.numpy(), gallery_pid.numpy(), 
                         query_camid.numpy(), gallery_camid.numpy(),
                         use_cython=True)
    logger.info('Validation Result:')
    logger.info('CMC Rank-1: {:.2%}'.format(cmc[1 - 1]))
    logger.info('CMC Rank-5: {:.2%}'.format(cmc[5 - 1]))
    logger.info('CMC Rank-10: {:.2%}'.format(cmc[10 - 1]))
    logger.info('mAP: {:.2%}'.format(mAP))
    logger.info('-' * 20)

    distmat = re_rank(query_feat, gallery_feat)
    cmc, mAP = eval_func(distmat, query_pid.numpy(), gallery_pid.numpy(),
                         query_camid.numpy(), gallery_camid.numpy(),
                         use_cython=True)

    logger.info('ReRanking Result:')
    logger.info('CMC Rank-1: {:.2%}'.format(cmc[1 - 1]))
    logger.info('CMC Rank-5: {:.2%}'.format(cmc[5 - 1]))
    logger.info('CMC Rank-10: {:.2%}'.format(cmc[10 - 1]))
    logger.info('mAP: {:.2%}'.format(mAP))
    logger.info('-' * 20)


if __name__ == '__main__':
    main()


