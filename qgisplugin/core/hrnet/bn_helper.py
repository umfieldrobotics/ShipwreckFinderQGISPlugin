from ...safe_libs_setup import setup_libs, safe_import_ml_libraries

setup_libs()
libs = safe_import_ml_libraries()

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