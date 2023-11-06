import os
import sys

clients_num = 100
num_round = 1
variables = { 
    'clients_threshold_ratios': [0.5],
    'dropout_ratio': [0],
    'clients_per_round': [10, 20, 30, 40, 50],
    'input_length': [ 2**i for i in range(10, 20, 1) ]
}


def test_trainer(avgs_dict):
    command = 'python3.10 ./tests/test_sa_basic.py'
    command += ' --algo sa_basic'
    command += ' --random_verctor'
    # command += ' --noaverage --noprintargs --noprint --eval_mem'
    command += ' --noprintargs'
    command += ' --clients_num {}'.format(clients_num)
    command += ' --num_round {}'.format(num_round)
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
                    for v3 in variables[keys[3]]:
                        avgs_dict[keys[0]] = v0
                        avgs_dict[keys[1]] = v1
                        avgs_dict[keys[2]] = v2
                        avgs_dict[keys[3]] = v3
                        count += 1
                        avgs_dict ['seed'] = count
                        test_trainer(avgs_dict)

if __name__ == '__main__':
    test()

