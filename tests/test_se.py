import numpy as np
import argparse
import importlib
import torch
import os
import  sys
BASE_DIR = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))
sys.path.append( BASE_DIR  ) 

from fedAvg.utils.worker_utils import read_data, read_verify_data
from main import read_options, set_options, printArgs
from fedAvg.config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS



def test_se(options):
    options['algo'] = 'se'
    options['noaverage'] = True
    options['noprint'] = True
    options['eval_mem'] = False
    # options['dataset'] = 'randomvector'
    # options['model'] = 'aggregation'

    return options



def main():
    # default arguments in main.py
    options = read_options()
    # Parse command line arguments
    options = test_se(options)
    # set arguments
    options, trainer_class, dataset_name, sub_data = set_options(options)
    # # print arguments
    # printArgs(options)


    # print('dataset_name:', dataset_name)
    all_data_info, verify_data = None, None
    options['traindata_info'] = (None, None)
    if dataset_name != 'randomvector':
        train_path = os.path.join('./Datasets', dataset_name, 'Train')
        test_path = os.path.join('./Datasets', dataset_name, 'Test')
        verify_path = os.path.join('./Datasets', dataset_name, 'Verify')
        options['traindata_info'] = (train_path, sub_data)
        # `dataset` is a tuple like (cids, groups, train_data, test_data)
        all_data_info = read_data(train_path, test_path, sub_data, dataset_name=dataset_name, model_name=options['model'])
        verify_data = read_verify_data(verify_path, sub_data, dataset_name=dataset_name, model_name=options['model'])

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info, verify_data=verify_data)
    trainer.train()


if __name__ == '__main__':
    main()
