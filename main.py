import numpy as np
import argparse
import importlib
import torch
import os

from fedAvg.utils.worker_utils import read_data, read_verify_data
from fedAvg.config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def read_options(options = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='sa')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: False)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=48)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')

    # for PFL
    parser.add_argument('--clients_num',
                        help='number of clients;',
                        type=int,
                        default=100)
    parser.add_argument('--servers_num',
                        help='number of server(s) (only support 1 in this version);',
                        type=int,
                        default=1)
    parser.add_argument('--clients_threshold_ratios',
                        help='threshold of clients to recover serects;',
                        type=float,
                        default=0.5)
    parser.add_argument('--dropout_ratio',
                        help='dropout ratio of clients per round;',
                        type=float,
                        default=0.4)
    parser.add_argument('--eval_client',
                        action='store_true',
                        default=False,
                        help='whether to evaluate clients\' local inputs (default: False)')
    parser.add_argument('--eval_method',
                        help='method for local input evaluation',
                        type=str,
                        default='l2')
    # for PFL data saving
    parser.add_argument('--nosave_pfl',
                        action='store_true',
                        default=False,
                        help='whether to save the setup_data into file (default: False)')
    parser.add_argument('--nobuild_from_file',
                        action='store_true',
                        default=False,
                        help='whether to build the clients and the server from setup_data file (default: False)')
    parser.add_argument('--nokey_files',
                        action='store_true',
                        default=False,
                        help='whether to save the clients keys into file (default: False)')
    parser.add_argument('--setup_data_dir',
                        help='Dir of setup_data files',
                        type=str,
                        default='./pfl_setup_datas/')
    parser.add_argument('--eval_mem',
                        action='store_true',
                        default=False,
                        help='whether to evaluate memory cost by saving a client & the server data to file (default: False)')


    # for random vector
    parser.add_argument('--random_verctor',
                        action='store_true',
                        default=False,
                        help='whether input vector is random vector (default: False)')
    parser.add_argument('--input_length',
                        help='length of input_vectors in [RV] dataset;',
                        type=int,
                        default=2**19)
    parser.add_argument('--weight_min',
                        help='minimum weight of a client in [RV] dataset;',
                        type=int,
                        default=400)
    parser.add_argument('--weight_max',
                        help='maximum weight of a client in [RV] dataset;',
                        type=int,
                        default=700)

    # for print
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noprintargs',
                        action='store_true',
                        default=False,
                        help='whether to print arguments (default: False)')


    parsed = parser.parse_args()
    options = parsed.__dict__
    return options

def set_options(options):
    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    # options.update(MODEL_PARAMS(dataset_name, options['model']))
    options, dataset_name = MODEL_PARAMS(options, dataset_name)

    # options setting
    options['save_pfl'] = not options['nosave_pfl']
    options['build_from_file'] = not options['nobuild_from_file']
    options['key_files'] = not options['nokey_files']
    del options['nosave_pfl'], options['nobuild_from_file'], options['nokey_files']
    options['sv_num'] = 1
    options['clients_threshold'] = round(options['clients_threshold_ratios']*options['clients_num'])
    setup_mark = 'dis_'+str(options['clients_threshold'])+'_outof_'+str(options['clients_num'])
    # model_mark = dataset_name+'_'+options['model']
    if dataset_name == 'randomvector':
        model_mark = dataset_name+'_'+options['algo']
    else:
        model_mark = dataset_name+'_'+options['model']+'_'+options['algo']
    options['setup_mark'] = setup_mark
    options['model_mark'] = model_mark
    options['trainer_pfl_info_file'] = options['setup_data_dir']+setup_mark
    options['svkeys_dir'] = options['setup_data_dir']+'svkeys/'
    options['dikeys_dir'] = options['setup_data_dir']+'dikeys/'
    options['diskshares_dir'] = options['setup_data_dir']+'diskshares/'+setup_mark+'/'
    options['stat_dir'] = './result/'+model_mark+'/'+setup_mark+'/'
    options['local_data_dir'] = './local_datas/'+model_mark+'/'+setup_mark+'/'


    # Load selected trainer
    #trainer_path = 'fedAvg.trainers.%s' % options['algo']
    trainer_path = 'mkTPFL.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    # print('mod:', mod)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    return options, trainer_class, dataset_name, sub_data


def printArgs(options):
    # Print arguments and return
    print_flag = not options['noprintargs']
    del options['noprintargs']
    if print_flag:
        max_length = max([len(key) for key in options.keys()])
        fmt_string = '\t%' + str(max_length) + 's : %s'
        print('>>> Arguments:')
        for keyPair in sorted(options.items()):
            print(fmt_string % keyPair)


def main():
    # Parse command line arguments
    # options, trainer_class, dataset_name, sub_data = read_options()
    options = read_options()
    options, trainer_class, dataset_name, sub_data = set_options(options)
    printArgs(options)

    all_data_info, verify_data = None, None
    options['traindata_info'] = (None, None)
    if dataset_name != 'randomvector':
        train_path = os.path.join('./Datasets', dataset_name, 'Train')
        test_path = os.path.join('./Datasets', dataset_name, 'Test')
        verify_path = os.path.join('./Datasets', dataset_name, 'Verify')
        options['traindata_info'] = (train_path, sub_data)
        options['dataset_name'] = dataset_name
        # `dataset` is a tuple like (cids, groups, train_data, test_data)
        all_data_info = read_data(train_path, test_path, sub_data, dataset_name=dataset_name, model_name=options['model'])
        verify_data = read_verify_data(verify_path, sub_data, dataset_name=dataset_name, model_name=options['model'])

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info, verify_data=verify_data)
    trainer.train()


if __name__ == '__main__':
    main()
