# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
def setup_libs():
    current_dir = os.path.dirname(__file__)
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'libs')):
            libs_dir = os.path.join(current_dir, 'libs')
            if libs_dir not in sys.path:
                sys.path.insert(0, libs_dir)
            return
        current_dir = os.path.dirname(current_dir)
setup_libs()

from yacs.config import CfgNode as CN

# high_resoluton_net related params for segmentation
HIGH_RESOLUTION_NET = CN()
HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
HIGH_RESOLUTION_NET.STEM_INPLANES = 64
HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
HIGH_RESOLUTION_NET.WITH_HEAD = True

HIGH_RESOLUTION_NET.STAGE2 = CN()
HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

HIGH_RESOLUTION_NET.STAGE3 = CN()
HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

HIGH_RESOLUTION_NET.STAGE4 = CN()
HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'seg_hrnet': HIGH_RESOLUTION_NET,
}
