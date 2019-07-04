# Creator: Tennant
# Email: Tennant_1999@outlook.com

import os
import os.path as osp

# PyTorch as the main lib for neural network
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
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

# import losses and model
from losses import make_loss
from model import build_model, convert_model
from trainer import BaseTrainer

# dataset
from dataset import make_dataloader

from optim import make_optimizer, WarmupMultiStepLR

def main():
    args = parse_args()
    if args.test:
        pass
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
        for it in range(len(trainer.train_dl)):
            trainer.step()



if __name__ == '__main__':
    main()


