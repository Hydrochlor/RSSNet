import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.DataLoaders_spec import make_dataloader as make_train
from dataloader.DataLoaders_spec import make_dataloader as make_val
from model import RSSNet
from model import AFRCNN
from model import Conv_TasNet
from model import TDANet
from model import model_rnn
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_RSSNet_test
import os
import random
import numpy as np

def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def test():
    parser = argparse.ArgumentParser(
        description='Parameters for testing RSSNet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--ref', type=str, help='Path to ref_spec file.')
    parser.add_argument('--list', type=str, help='Path to list_spec file.')
    parser.add_argument('--mix', type=str, help='Path to mix_spec file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)

    opt['resume']['state'] = True
    opt['datasets']['val']['ref_path'] = args.ref
    opt['datasets']['val']['list_path'] = args.list
    opt['datasets']['val']['mix_path'] = args.mix

    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of RSSNet")
    # Net = model_rnn.Dual_RNN_model(**opt['Dual_Path_RNN'])
    # Net = AFRCNN.AFRCNN(**opt['AFRCNN'])
    # Net = TDANet.TDANet_2(**opt['TDANet'])
    # Net = Conv_TasNet.ConvTasNet(**opt['ConvTasNet'])
    Net = RSSNet.Net(**opt['RSSNet'])
    # build optimizer
    logger.info("Building the optimizer of RSSNet")
    optimizer = make_optimizer(Net.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of RSSNet')
    # train_dataloader, val_dataloader = make_dataloader(opt)
    train_dataloader = make_train(is_train=False, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['train_batch_size'], upsample=opt['datasets']['upsample'])
    val_dataloader = make_val(is_train=False, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
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
    trainer = trainer_RSSNet_test.Trainer(train_dataloader, val_dataloader, Net, optimizer, scheduler, opt)
    trainer.validation(100)


if __name__ == "__main__":
    #setup_seed(42)
    test()
