# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import time

import moxing.mxnet as mox

def create_logger(root_output_path, cfg, image_set):
    # set up logger
    # if not os.path.exists(root_output_path):
    #     os.makedirs(root_output_path)
    # assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)
    ## support obs
    if not mox.file.exists(root_output_path):
        mox.file.make_dirs(root_output_path)
    assert mox.file.exists(root_output_path), '{} does not exist'.format(root_output_path)
    ## support obs

    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    # if not os.path.exists(config_output_path):
    #     os.makedirs(config_output_path)
    ## support obs
    if not mox.file.exists(config_output_path):
        mox.file.make_dirs(config_output_path)
    ## support obs

    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    # if not os.path.exists(final_output_path):
    #     os.makedirs(final_output_path)
    ## support obs
    if not mox.file.exists(final_output_path):
        mox.file.make_dirs(final_output_path)
    ## support obs

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path

