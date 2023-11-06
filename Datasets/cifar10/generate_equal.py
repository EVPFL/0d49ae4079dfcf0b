import torch
import numpy as np
import pickle
import os
import torchvision
import random
from math import ceil
from collections import defaultdict
from torchvision.transforms import ToTensor
cpath = '.' #os.path.dirname(__file__)

NUM_USER = 100
RATIO_VER = 0.01
SAVE = True
DATASET_FILE = os.path.join(cpath, 'Raw')
IMAGE_DATA = False
NIID_FLAG = False
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        self.data = np.array(images)
        self.target = np.array(labels)

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst


def main():
    # Get CIFAR10 data, normalize, and divide by level
    print('>>> Get CIFAR10 data.')
    trainset = torchvision.datasets.CIFAR10(DATASET_FILE, download=True, train=True, transform=ToTensor())
    testset = torchvision.datasets.CIFAR10(DATASET_FILE, download=True, train=False, transform=ToTensor())

    train_cifar10 = ImageDataset(trainset.data, trainset.targets)
    test_cifar10 = ImageDataset(testset.data, testset.targets)

    # get and split the total train and test dataset
    # all types (i.e., '1' to '10') have the same number of samples: traindata & testdata
    # the dataset saves into a list, a element of which is 20 samples: split_traindata & split_testdata
    traindata = []
    for number in range(10):
        idx = train_cifar10.target == number
        traindata.append(train_cifar10.data[idx])
    min_number = min([len(dig) for dig in traindata])
    for number in range(10):
        traindata[number] = traindata[number][:min_number-1]
    split_traindata = []
    for digit in traindata:
        split_traindata.append(data_split(digit, 20))
    testdata = []
    for number in range(10):
        idx = test_cifar10.target == number
        testdata.append(test_cifar10.data[idx])
    split_testdata = []
    for digit in testdata:
        split_testdata.append(data_split(digit, 20))

    # print the infomation of total train and test dataset
    data_distribution = np.array([len(v) for v in traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))
    digit_count = np.array([len(v) for v in split_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))
    digit_count = np.array([len(v) for v in split_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train and test samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    train_y_dict = [defaultdict(list) for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]
    for user in range(NUM_USER):
        #print(user, np.array([len(v) for v in split_traindata]))
        cnt = 0
        # random choose 2 type
        for d in choose_two_digit(split_traindata):
            l = len(split_traindata[d][-1])
            train_X[user] += split_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()
            train_y_dict[user][d] += range(cnt,cnt+l,1)
            cnt += l
            l = len(split_testdata[d][-1])
            test_X[user] += split_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()
    print(">>> Train & Test Data is non-i.i.d. distributed")
    print(">>> Train & Test Data is balanced")

    # Assign verify samples to each user
    verify_X = []
    verify_y = []
    verify_inxs = [[] for _ in range(NUM_USER)]
    ver_num = ceil( RATIO_VER * len(train_y[-1]) )
    for user in range(NUM_USER):
        for d,inxs in train_y_dict[user].items():
            num_trains = len(inxs)
            num_samples = int(num_trains*RATIO_VER) if d!=list(train_y_dict[user].keys())[-1] else ver_num-len(verify_inxs[user])
            samples_inxs = random.sample(range(0, num_trains), num_samples)
            ver_inxs = [ inxs[i] for i in samples_inxs ]
            # verify_inxs[user] += ver_inxs
            verify_X += [ train_X[user][i] for i in ver_inxs ]
            verify_y += [ train_y[user][i] for i in ver_inxs ]
    print(">>> Verification Data sampled from Train Data is balanced")

    # Setup directory for train/test data
    print('>>> Set data path for MNIST.')
    niid = 1 if NIID_FLAG else 0
    train_path = '{}/Train/all_data_{}_equal_niid.pkl'.format(cpath, niid)
    test_path = '{}/Test/all_data_{}_equal_niid.pkl'.format(cpath, niid)
    verify_path = '{}/Verify/all_data_{}_equal_niid.pkl'.format(cpath, niid)
    for dpath in [train_path, test_path, verify_path]:
        dir_path = os.path.dirname(dpath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    
    # Setup 100 users
    for i in range(NUM_USER):
        uname = i
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    verify_data = {
        'users': [ 0 ], 
        'user_data': { 0:{'x': verify_X, 'y': verify_y} }, 
        'num_samples': [ len(verify_y) ] }

    print('>>> User train data distribution: {}'.format(train_data['num_samples']))
    print('>>> User verify data distribution: {}'.format(verify_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))
    print('>>> Total verifying size: {}'.format(sum(verify_data['num_samples'])))

    # Save user data
    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)
        with open(verify_path, 'wb') as outfile:
            pickle.dump(verify_data, outfile)
        print('>>> Save data.')


if __name__ == '__main__':
    main()

