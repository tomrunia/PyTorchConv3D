import csv
import numpy as np

import torch
import torch.optim


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

def get_optimizer(params, optimizer_type, learning_rate, weight_decay, momentum=None):
    if optimizer_type == 'SGD':
        return torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError('Chosen optimizer is not supported, please choose from (SGD | adam | rmsprop)')
