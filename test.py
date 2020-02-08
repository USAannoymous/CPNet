
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from dataloader import NYUlabel13
from models import CPNet

def parse_args():
    parser = argparse.ArgumentParser(description='Train CPNet')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./config/mono.yaml',
                        type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    #Model loading code
    model = CPNet.get_depth_net(cfg)
    model.load_state_dict(torch.load('./nyuv2.pth'))
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # Data loading code
    valid_loader = NYUlabel13.getTestingData()

    # evaluate on validation set
    loss_indicator = validate(
        valid_loader, model
    )

if __name__ == '__main__':
    main()
