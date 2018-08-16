import os
import shutil
import time

import csv
import simplejson

import numpy as np

import torch
import torch.optim
import torchvision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, history=10):
        self.history = history
        self.values = np.zeros(self.history, dtype=np.float32)
        self.num_recorded = 0

    def push(self, value):
        assert np.isscalar(value)
        if self.num_recorded == 0:
            self.values.fill(value)
        else:
            self.values[:-1] = self.values[1:]
            self.values[-1] = value
        self.num_recorded += 1

    def last(self):
        return self.values[-1]

    def average(self):
        num_avg = min(self.history, self.num_recorded)
        return np.mean(self.values[-num_avg:])

    def reset(self):
        self.values = np.zeros(self.history, dtype=np.float32)
        self.num_recorded = 0


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

####################################################################
####################################################################

def print_config(config):
    print('#'*60)
    print('Training configuration:')
    for k,v  in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#'*60)

def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))

def output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    subdir = "{}_{}_{}_lr{:.3f}".format(prefix, config.dataset, config.model, config.learning_rate)
    return os.path.join(config.save_dir, subdir)

def prepare_output_dirs(config):
    # Set output directories
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    return config

####################################################################
####################################################################

def get_optimizer(config, params):
    if config.optimizer == 'SGD':
        return torch.optim.SGD(params, config.learning_rate, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        return torch.optim.Adam(params, config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError('Chosen optimizer is not supported, please choose from (SGD | adam | rmsprop)')

def current_learning_rate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def current_weight_decay(optimizer):
    return optimizer.state_dict()['param_groups'][0]['weight_decay']

def save_checkpoint(save_file_path, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer':  optimizer_state_dict}
    torch.save(states, save_file_path)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def write_clips_as_grid(step, clips, targets, dataset, output_dir='./output/', num_examples=10, n_row=4):
    for i in range(0,num_examples):
        clip = clips[i].permute((1,0,2,3))
        grid = torchvision.utils.make_grid(clip, nrow=n_row, normalize=True)
        class_label = dataset.class_names[int(targets[i].numpy())]
        torchvision.utils.save_image(grid, os.path.join(output_dir, 'step{:04d}_example{:02d}_{}.jpg'.format(step+1, i, class_label)))