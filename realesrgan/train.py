# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline
# import numpy.testing
# import scipy.stats
#
# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models


import warnings



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # Filter warnings from torchvision about the deprecated functional_tensor module
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional_tensor')

    train_pipeline(root_path)
