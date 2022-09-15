# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch.nn as nn
from CNN_model import Generator
from SetParams import set_parameter_requires_grad, get_params_to_update, ChangeParams

# pytest ModelParams.py -W ignore::FutureWarning

def test_model_parameters():
    """
    Instantiate Generator, test number of trainable parameters matches the number
    of trainable parameters from torch.summary. No change in requires_grad in all layers.
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    params_to_update = get_params_to_update(model=gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 960357 # total params

def test_one_layer():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last layer.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=1)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 577

def test_two_layers():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last 2 layers.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=2)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 37569 # 577+128+36864

def test_three_layers():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last 3 layers.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=3)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 168769 # 577+128+36864+128+131072

def test_four_layers():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last 4 layers.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=4)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 431169 # 577+128+36864+128+131072+256+262144

def test_five_layers():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last 5 layers.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=5)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 955713 # 577+128+36864+128+131072+256+262144+256+524288

def test_six_layers():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last 6 layers.
    Test number of trainable parameters matches the number of trainable parameters from torch.summary. 
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=6)
    params_to_update = get_params_to_update(gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 960321 # 577+128+36864+128+131072+256+262144+256+524288+512+4096
    
def test_reset_params():
    """
    Instantiate Generator, set all model parameters to requires_grad=False. Re-initialise last layer.
    Re-set all model parameters to requires_grad=True. Test number of trainable parameters matches 
    the number of trainable parameters from torch.summary. No change in requires_grad in all layers.
    """
    gen = Generator(nr_input_var=3, nr_filter=16)
    set_parameter_requires_grad(model=gen)
    ChangeParams(gen=gen, num_layers=1)
    set_parameter_requires_grad(model=gen, requires_grad=True)
    params_to_update = get_params_to_update(model=gen)
    total = 0
    for params in params_to_update:
        total += params.numel()
    assert total == 960357 # total params
