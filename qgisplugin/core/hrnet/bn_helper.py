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

import torch
import functools


BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
relu_inplace = True
# if torch.__version__.startswith('0'):
#     from .sync_bn.inplace_abn.bn import InPlaceABNSync
#     BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
#     BatchNorm2d_class = InPlaceABNSync
#     relu_inplace = False
# else:
#     BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
#     relu_inplace = True