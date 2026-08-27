import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.DataLoaders_spec import make_dataloader
from model import RSSNet
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_RSSNet
import os
import numpy as np
import random

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)    
    random.seed(seed)    
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True

def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train(seed = 42):
    parser = argparse.ArgumentParser(
        description='Parameters for training RSSNet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of RSSNet")
    net = RSSNet.Net(**opt['RSSNet'])
    logger.info("Building the optimizer of RSSNet")
    optimizer = make_optimizer(net.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of RSSNet')
    # train_dataloader, val_dataloader = make_dataloader(opt)
    train_dataloader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['train_batch_size'], upsample=opt['datasets']['upsample'])
    val_dataloader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['val_batch_size'], upsample=opt['datasets']['upsample'])
    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        factor=opt['scheduler']['factor'],
        patience=opt['scheduler']['patience'],
        verbose=True, min_lr=opt['scheduler']['min_lr'])
    
    # build trainer
    logger.info('Building the Trainer of RSSNet')
    trainer = trainer_RSSNet.Trainer(train_dataloader, val_dataloader, net, optimizer, scheduler, opt, seed=seed)
    trainer.run()
    # trainer.validation(100)


if __name__ == "__main__":
    seed = 42
    setup_seed(seed)
    train(seed)
