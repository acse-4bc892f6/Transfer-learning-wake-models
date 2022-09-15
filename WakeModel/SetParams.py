from CNN_model import Generator
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn as nn

def set_parameter_requires_grad(model: Generator, requires_grad:bool=False)->None:
    """
    Change requires_grad attribute of model parameters. Taken from Machine Learning module implementation lecture 4.
    https://github.com/ese-msc-2021/ML_module/blob/master/implementation/4_CNN/4_CNN_afternoon/Transfer_Learning_Solutions.ipynb

    Args:
        model (Generator): CNN model to be altered.
        requires_grad (bool, optional): torch.autograd record operations if True.
            Defaults to False.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad
    return None

def get_params_to_update(model: Generator)->list:
    """
    Returns list of model parameters that have required_grad=True. Taken from Machine Learning module implementation lecture 4.
    https://github.com/ese-msc-2021/ML_module/blob/master/implementation/4_CNN/4_CNN_afternoon/Transfer_Learning_Solutions.ipynb

    Args:
        model (Generator): CNN model to be iterated over.

    Returns:
        params_to_update (list): list of model parameters that have requires_grad=True.
    """
    params_to_update = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update

def SplitList(param_list:list, ratio:float, seed:int=1)->list:
    """
    Reduce the size of param_list while preserving the proportion of values over the 
    uniform distribution. Used to maintain consistency of input data for CNN models
    when the input dataset size is changed.

    Args:
        param_list (list): list of uniformly distributed values
        ratio (float): new size / original size
        seed (int, optional): Random number seed. Default to 1.

    Returns:
        indices (list): list of indices from param_list to construct the new list.
            Has length of new size.
    """
    X = []
    for _ in range(len(param_list)):
        X.append([param_list[_]])
    X = np.array(X)
    y = np.round(param_list)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    for _, index in sss.split(X, y):
        indices = index

    return indices

def LoadModel(model_path:str, model_name:str)->Generator:
    """
    Load a previously saved CNN model to an instantiated Generator object.

    Args:
        model_path (str): Path to directory to store the saved model.
        model_name (str): Name of the trained saved model (needs be .pt).

    Returns:
        gen (Generator): Previously saved CNN model.
    """
    gen = Generator(nr_input_var=3, nr_filter=16) # instantiate Generator
    gen.load_model(model_path+model_name) # load model
    return gen

def ChangeParams(gen:Generator, num_layers:int=1)->None:
    """
    Change model parameters num_layers from the last layer to requires_grad=True for transfer learning.

    Args:
        gen (Generator): CNN model to be modified.
        num_layers (int, optional): number of layers from the last layer which model parameters will be changed.
            Defaults to 1.
    """

    gen.last_layer = nn.ConvTranspose2d(16 * 4, 1, kernel_size=3,
                        stride=3, padding=1)
    # re-initialise last two layers
    if num_layers==2:
        gen.second_last_layer = gen.layer(16 * 4, 16 * 4, 3, 2, 1)
    # re-initialise last three layers
    elif num_layers==3:
        gen.second_last_layer = gen.layer(16 * 4, 16 * 4, 3, 2, 1)
        gen.net4 = gen.layer(16 * 8, 16 * 4, 4, 2, 1)
    # re-initialise last four layers
    elif num_layers==4:
        gen.second_last_layer = gen.layer(16 * 4, 16 * 4, 3, 2, 1)
        gen.net4 = gen.layer(16 * 8, 16 * 4, 4, 2, 1)
        gen.net3 = gen.layer(16 * 8, 16 * 8, 4, 2, 1)
    # re-initialise last five layers
    elif num_layers==5:
        gen.second_last_layer = gen.layer(16 * 4, 16 * 4, 3, 2, 1)
        gen.net4 = gen.layer(16 * 8, 16 * 4, 4, 2, 1)
        gen.net3 = gen.layer(16 * 8, 16 * 8, 4, 2, 1)
        gen.net2 = gen.layer(16 * 16, 16 * 8, 4, 1, 1)
    # re-initialise all except first layer
    elif num_layers==6:
        gen.second_last_layer = gen.layer(16 * 4, 16 * 4, 3, 2, 1)
        gen.net4 = gen.layer(16 * 8, 16 * 4, 4, 2, 1)
        gen.net3 = gen.layer(16 * 8, 16 * 8, 4, 2, 1)
        gen.net2 = gen.layer(16 * 16, 16 * 8, 4, 1, 1)
        gen.net1 = gen.layer(1, 16 * 16, 4, 2, 1)
    elif num_layers!=1:
        raise KeyError("Not implemented")

    return None
