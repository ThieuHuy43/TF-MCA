import os
import torch
import numpy as np
import random
import torch.nn.functional as F

def set_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(gpu:str):
    gpu_list = [str(x) for x in gpu.split(';') if x]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_list)
    return gpu_list


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def print_nested_dict(d, indent=0):
    """ Recursively prints nested dictionaries with indentation for clear structure """
    for key, value in d.items():
        print('    ' * indent + str(key) + ':', end='')
        if isinstance(value, dict):
            print()  # Move to the next line before printing nested dictionary
            print_nested_dict(value, indent + 1)
        else:
            print(' ' + str(value))

            
class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    

def calculate_batch_entropy(probabilities):
    probabilities = probabilities.float()
    log_probabilities = torch.log(probabilities + 1e-9)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropy

