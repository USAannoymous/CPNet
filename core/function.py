from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
import math
import numpy as np
import torch
import torch.nn as nn
import skimage
import skimage.io
import sys
import scipy.io


def validate(val_loader, model):
    sums = 0
    ct = 0
    # switch to evaluate mode
    model.eval()
    
    totalNumber = 0
    with torch.no_grad():
        end = time.time()
        for i, sample_batched in enumerate(val_loader):
            input, target = sample_batched['image'], sample_batched['depth']
            # compute output
            output = model(input)
            output_depth = output[0]
            
            target = target.cuda(non_blocking=True)
            h, w = target.size(2), target.size(3)

            scale_output_depth = nn.functional.interpolate(input=output_depth, size=(h, w),mode='bilinear', align_corners=True)

            num_images = input.size(0)
            input = input.data.cpu().numpy()
            pred = scale_output_depth.data.cpu().numpy()
            gt = target.data.cpu().numpy()

            for j in range(num_images):
                rmse = (gt[j,0]-pred[j,0])**2
                rmse = np.sqrt(rmse.mean())
                sums = sums + rmse
                ct = ct+1

    avg_rmse = sums/ct 
    print('Test Rmse: ',avg_rmse)
    return avg_rmse


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

