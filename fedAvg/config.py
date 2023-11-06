# GLOBAL PARAMETERS
DATASETS = ['randomvector', 'mnist', 'cifar10']

TRAINERS = {'fedavg': 'FedAvgTrainer',
            'sa': 'SATrainer',
            'se': 'SecEvalTester',
            'sa_basic': 'BasicSATrainer',
            }

OPTIMIZERS = TRAINERS.keys()


def MODEL_PARAMS(options, dataset):
    model = options['model']
    dataset = dataset.split('_')[0]
    dataset_name = dataset

    if options['random_verctor'] or dataset_name in ['randomvector', 'RV', 'rv']:
        del options['wd'], options['device'], options['eval_every'], options['lr'], options['num_epoch'], options['gpu']
        options['model'] = 'aggregation'
        options['dataset'] = 'randomvector'
        dataset_name = 'randomvector'
        del options['random_verctor']
        
    else:
        for k in ['input_length', 'weight_min','weight_max']:
            options.pop(k, None)
        
        if dataset == 'mnist' or dataset == 'nist':
            if model in ['logistic', '2nn', 'cnn']:
               options.update( {'input_shape': 784, 'num_class': 10} )
            # elif model == 'cnn':
            #    options.update( {'input_shape': (64, 
            #     1, 28, 28), 'num_class': 10} )
            else:
                options.update( {'input_shape': (1, 28, 28), 'num_class': 10} )
        elif dataset == 'cifar10':
            options.update( {'input_shape': (3, 32, 32), 'num_class': 10} )

        else:
            raise ValueError('Not support dataset {}!'.format(dataset))

    return options, dataset_name

