import numpy as np
import argparse
import importlib
import torch
import os
import  sys
BASE_DIR = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))
sys.path.append( BASE_DIR  ) 

from fedAvg.utils.worker_utils import read_data
from main import read_options, set_options, printArgs
from fedAvg.config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def test_va(options):
    options['algo'] = 'sa'
    options['noaverage'] = True
    options['noprint'] = True
    # options['eval_mem'] = True
    options['eval_mem'] = False

    options['dataset'] = 'randomvector'
    options['model'] = 'aggregation'
    
    options['noprintargs'] = False

    return options

def main():
    # default arguments in main.py
    options = read_options()
    # Parse command line arguments
    options = test_va(options)
    # set arguments
    options, trainer_class, dataset_name, sub_data = set_options(options)
    # # print arguments
    # printArgs(options)

    all_data_info = None
    options['traindata_info'] = (None, None)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
