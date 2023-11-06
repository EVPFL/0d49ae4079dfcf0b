import os
import sys

num_round = 1
clients_per_round = 1

# variables = { 
#     'dataset': ['mnist_all_data_0_equal_niid','cifar10_all_data_0_equal_niid'],
#     'model': ['logistic', '2nn', 'lenet'],
#     'eval_method': ['l2','ln','zeno','cos']
# }

variables = { 
    'dataset': ['randomvector'],
    'input_length': [ 2**i for i in range(10, 20, 1) ],
    'eval_method': ['l2','ln','zeno','cos']
}



def test_trainer(avgs_dict):
    command = 'python3.10 ./tests/test_se.py'
    command += ' --algo se'
    command += ' --noprintargs'
    command += ' --num_round {}'.format(num_round)
    command += ' --clients_per_round {}'.format(clients_per_round)
    # avgs_str = ''
    for k,v in avgs_dict.items():
        command += ' --{} {}'.format(k,v)
    print('command:', command)
    os.system(command)

def test():
    avgs_dict = {}
    keys = list( variables.keys() )

    repeat_times = 1
    count = 0

    for _ in range(repeat_times):
        for v0 in variables[keys[0]]:
            for v1 in variables[keys[1]]:
                for v2 in variables[keys[2]]:
                    avgs_dict[keys[0]] = v0
                    avgs_dict[keys[1]] = v1
                    avgs_dict[keys[2]] = v2
                    count += 1
                    avgs_dict ['seed'] = count
                    test_trainer(avgs_dict)

if __name__ == '__main__':
    test()

