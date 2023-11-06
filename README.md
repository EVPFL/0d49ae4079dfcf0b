# EVPFL 0.1

Note: We implement this version in Apple M1 (ARM64).


## Overview
This repository contains the codes for the paper
> [EVPFL](XXX): The paper is currently under review with the condition of maintaining anonymity.

Our code is based on the codes:
* [fedavgpy](https://github.com/bokunwang/fedavgpy) is a federated algorithm used in heterogeneous networks;
* [HEAAN-Python](https://github.com/Huelse/HEAAN-Python) is a Python wrapper for HEAAN library;
* [MKLHS] The Multi-key Linearly Homomorphic Signature Scheme is on the [paper](https://eprint.iacr.org/2019/830.pdf), which is implemented in the library [relic](https://github.com/relic-toolkit/relic);
* [LHH] Linearly Homomorphic Hash supports linear operations on its output. This scheme is based on the LHH function used in [paper](https://eprint.iacr.org/2022/1073) and the library [VeriFL](https://github.com/ErwinSCat/VeriFL).


## DIR TREE
``` shell
|-README.md
|-main.py
|-Datasets (based on [fedavgpy])
    |-cifar10
        |-generate_equal.py
    |-mnist
        |-generate_equal.py
|-fedAvg (based on [fedavgpy])
|-mkTPFL
    |-ckks (Classes based on MPHEAAN)
        |-ckks_decryptor.py
        |-ckks_encoder.py
        |-ckks_encryptor.py
        |-ckks_evaluator.py
        |-ckks_key_generator.py
        |-ckks_parameters.py
        |-util.py
    |-mklhs (Classes based on MKLHE)
        |-mklhs_encoder.py
        |-mklhs_parameters.py
        |-mklhs_verifier.py
    |-roles
        |-dataisland.py (Client Class)
        |-flserver.py (Server Class)
        |-model_evaluator.py (Encrypted Model Evaluator Class: to evaluate the ciphertext of local updates)
    |-srcs-cpython (tools, including LHH, MKLHS and MPHEAAN, have been removed in this version)
    |-trainers
        |-base.py (basic Trainer Class)
        |-fedavg.py (fedavg Trainer Class and corresponding train Function)
        |-sa.py (SA Trainer Class: for verifiable aggregation test)
        |-se.py (SE Trainer Class: for secure evaluation test)
    |-utils
|-pfl_setup_datas (public parameters and datas generated in the setup phase)
    |-dikeys (keys of each client)
    |-svkeys (keys of server)
    |-diskshares (sk shares held by each client)
|-result (logs)
|-tests
    |-test_se.py (basic secure evaluation test: a local update contains local gradients of LeNet5 training on a local MNIST dataset)
    |-tests_se.py (execute test_va.py multiple times: different evaluation methods and models training on different datasets)
    |-test_va.py (basic verifiable aggregation test: a local update is a random vector)
    |-tests_va.py (execute test_va.py multiple times: different lengths and numbers of aggregated inputs)
|-Instruction_Videos
    |-tools_compiling.mp4 (compile the tools: https://github.com/EVPFL/tools_AMD64)
    |-tests_basic.mp4 (run the basic tests of EVPFL)
    |-tests_100clients.mp4 (run the tests under 100 clients setup)

```


## Usage

### 1. Compile libraries: LHH, MKLHS, MPHEAAN

Please download and compile the EVPFL tools library.
* AMD64 version: https://github.com/EVPFL/tools_AMD64
* ARM64 version: https://github.com/EVPFL/tools_ARM64

###  2. Generate datasets. 
* ### CIFAR10
    ``` shell
    cd Datasets/cifar10
    python3 generate_equal.py
    ```
* ### MNIST
    ``` shell
    cd Datasets/mnist
    python3 generate_equal.py
    ```

### 3. Then start to train. You can run the tests or main.py.

* ### Arguments (show this help message and exit):
``` shell
python3 main.py -h
```
    ``` shell
    usage: main.py [-h] [--algo {fedavg,sa,se,sa_basic}] [--dataset DATASET]
                   [--model MODEL] [--wd WD] [--gpu] [--noaverage]
                   [--device DEVICE] [--num_round NUM_ROUND]
                   [--eval_every EVAL_EVERY]
                   [--clients_per_round CLIENTS_PER_ROUND]
                   [--batch_size BATCH_SIZE] [--num_epoch NUM_EPOCH] [--lr LR]
                   [--seed SEED] [--dis DIS] [--clients_num CLIENTS_NUM]
                   [--servers_num SERVERS_NUM]
                   [--clients_threshold_ratios CLIENTS_THRESHOLD_RATIOS]
                   [--dropout_ratio DROPOUT_RATIO] [--eval_client]
                   [--eval_method EVAL_METHOD] [--nosave_pfl]
                   [--nobuild_from_file] [--nokey_files]
                   [--setup_data_dir SETUP_DATA_DIR] [--eval_mem]
                   [--random_verctor] [--input_length INPUT_LENGTH]
                   [--weight_min WEIGHT_MIN] [--weight_max WEIGHT_MAX] [--noprint]
                   [--noprintargs]

    options:
      -h, --help            show this help message and exit
      --algo {fedavg,sa,se,sa_basic}
                            name of trainer;
      --dataset DATASET     name of dataset;
      --model MODEL         name of model;
      --wd WD               weight decay parameter;
      --gpu                 use gpu (default: False)
      --noaverage           whether to only average local solutions (default:
                            False)
      --device DEVICE       selected CUDA device
      --num_round NUM_ROUND
                            number of rounds to simulate;
      --eval_every EVAL_EVERY
                            evaluate every ____ rounds;
      --clients_per_round CLIENTS_PER_ROUND
                            number of clients trained per round;
      --batch_size BATCH_SIZE
                            batch size when clients train on data;
      --num_epoch NUM_EPOCH
                            number of epochs when clients train on data;
      --lr LR               learning rate for inner solver;
      --seed SEED           seed for randomness;
      --dis DIS             add more information;
      --clients_num CLIENTS_NUM
                            number of clients;
      --servers_num SERVERS_NUM
                            number of server(s) (only support 1 in this version);
      --clients_threshold_ratios CLIENTS_THRESHOLD_RATIOS
                            threshold of clients to recover serects;
      --dropout_ratio DROPOUT_RATIO
                            dropout ratio of clients per round;
      --eval_client         whether to evaluate clients\' local inputs (default:
                            False)
      --eval_method EVAL_METHOD
                            method for local input evaluation
      --nosave_pfl          whether to save the setup_data into file (default:
                            False)
      --nobuild_from_file   whether to build the clients and the server from
                            setup_data file (default: False)
      --nokey_files         whether to save the clients keys into file (default:
                            False)
      --setup_data_dir SETUP_DATA_DIR
                            Dir of setup_data files
      --eval_mem            whether to evaluate memory cost by saving a client &
                            the server data to file (default: False)
      --random_verctor      whether input vector is random vector (default: False)
      --input_length INPUT_LENGTH
                            length of input_vectors in [RV] dataset;
      --weight_min WEIGHT_MIN
                            minimum weight of a client in [RV] dataset;
      --weight_max WEIGHT_MAX
                            maximum weight of a client in [RV] dataset;
      --noprint             whether to print inner result (default: False)
      --noprintargs         whether to print arguments (default: False)
    ```

* ### test - verifiable aggregation
``` shell
python3 tests/test_va.py
```

* ### test - secure evaluation
``` shell
python3 tests/test_se.py
```

``` shell
python3 main.py
```



