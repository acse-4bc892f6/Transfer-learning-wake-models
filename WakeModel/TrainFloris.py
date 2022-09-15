from CNN_model import Generator
from SetParams import get_params_to_update, set_parameter_requires_grad, ChangeParams
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import floris.tools as wfct

""" 
Title: train_CNN.py
Author: Jens Bauer
Date: 25 June 2022
Code version: 1.0
Availability: https://github.com/soanagno/wakenet/tree/master/Code/CNNWake
"""

def floris_file_v2_to_v3(floris_path='./inputs/', floris_name='FLORIS_input_gauss.json'):
    """
    Converts FLORIS file from json to yaml so that it can be loaded
    into FlorisInterface in FLORIS version 3.

    Args:
        floris_path (str, optional): Path to directory storing FLORIS json file.
            This is be the same path that the yaml file will be saved.
            Defaults to "./inputs".
        floris_name (str, optional): name of the FLORIS json file.
            Defaults to "FLORIS_input_gauss.json".
    """
    # load json file using legacy function
    fi = wfct.FlorisInterfaceLegacyV2(floris_path+floris_name+'.json')
    # convert to yaml and save
    fi.floris.to_file(floris_path+floris_name+'.yaml')
    return None

def train_CNN_floris(
        nr_epochs, learning_rate, batch_size,
        train_size, val_size, u_range,
        ti_range, yaw_range, nr_workers=0, ML_model=None,
        x_train=None, y_train=None, x_eval=None, y_eval=None,
        change_params=False, num_layers=1,
        floris_path="./inputs/", floris_name="FLORIS_input_jensen",
        legacy=False, model_path="./TrainedModels/",
        model_name="FlorisJensen.pt",
        save_model=True, plot_fig=True,
        train_fig_path="./TrainingFigures/", 
        train_fig_name="FlorisJensen.png",
        seed=1, nr_filters=16, image_size=163):
    """
    Create a new model and train it for a certain number of epochs using a
    newly generated dataset. Dataset is generated from FLORIS. Hyper-parameters 
    such as model size or lr can be changed as input to the function. 
    After training the model error for all epochs is plotted and the model
    performance will be evaluated on a test set. Finally, the model
    will saved as the model_name which needs to add as .pt file

    Args:
        nr_epochs (int): Nr. of training epochs
        learing_rate (float): Model learning rate
        batch_size (int): Training batch size
        train_size (int): Size of the generated training set
        val_size (int): Size of the generated validation set
        u_range (list): Bound of u values [u_min, u_max] used
        ti_range (list): Bound of TI values [TI_min, TI_max] used
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used
        nr_workers (int, optional): Nr. of worker to load data. Defaults to 0.
        ML_model (Generator, optional): CNN model passed as an argument. 
            Defaults to None.
        x_train (torch.tensor, optional): Tensor of size (train_size, 1, 3) 
            which includes the flow conditons of the correspoding flow field.
            To be used for training.
        y_train (torch.tensor, optional): Tensor of size (train_size, image_size, image_size)
            which includes all the generated normalised flow fields. To be used for training.
        x_eval (torch.tensor, optional): Tensor of size (val_size, 1, 3) 
            which includes the flow conditons of the correspoding flow field.
            To be used for validation.
        y_eval (torch.tensor, optional): Tensor of size (val_size, image_size, image_size)
            which includes all the generated normalised flow fields. To be used for validation.
        change_params (bool, optional): Set requires_grad=False in CNN layers if True. 
            Defaults to False.
        num_layers (int, optional): Number of layers left as requires_grad=True. 
            Defaults to 1.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Defaults to "./inputs/".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
        model_path (str, optional): Path to directory to store the saved model.
        model_name (str, optional): Name of the trained saved model (needs be .pt).
        save_model (bool, optional): Save trained model if True.
            Defaults to True.
        plot_fig (bool, optional): Create plot of validation over epochs.
            Defaults to True.
        train_fig_path (str, optional): Path to directory to store plot of
            validation errors over epochs during training.
        train_fig_name (str, optional): File name of the saved plot.
        seed (int, optional): Random number seed. Default to 1.
        nr_filters (int, optional): Nr. of filters used for the conv layers.
            Defaults to 16.
        image_size (int): Size of the data set images, needs to match the
            model output size for the current model this is 163 x 163.

    Returns:
        gen (Generator): Trained CNN model.
        loss (float): Training loss defined by the loss function.
        val_error (float): Percentage error on the validation set.
        error_list (list): List of percentage errors on validation set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The current inputs are: u, ti and yaw. If more are
    # used please change this input var
    nr_input_var = 3

    # set_seed(seed)

    if ML_model is None:
        # create a generator of the specified size
        gen = Generator(nr_input_var, nr_filters)
    else:
        # generator passed in as argument
        gen = ML_model
        # ensure requires_grad=True for all model parameters
        set_parameter_requires_grad(model=gen, requires_grad=True)

    # generate training and validation sets if not passed as argument
    
    if x_train is None and y_train is None:
        x_train, y_train = gen.create_floris_dataset(size=train_size,
            image_size=image_size, seed=seed, legacy=legacy,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            floris_init_path=floris_path, floris_name=floris_name)

    if x_eval is None and y_eval is None:
        x_eval, y_eval = gen.create_floris_dataset(size=val_size,
            image_size=image_size, seed=seed, legacy=legacy,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            floris_init_path=floris_path, floris_name=floris_name)

    dataset = TensorDataset(x_train.float(), y_train.float())
    # generate dataload for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=nr_workers)

    # init the weights of the generator
    gen.initialize_weights()

    # change model parameters to requires_grad=False if change_params=True
    if change_params:
        # change all model parameters to requires_grad=False
        set_parameter_requires_grad(gen)
        # model parameters of layer defaults to requires_grad=True when re-initialised
        # re-initialise last layer
        ChangeParams(gen=gen, num_layers=num_layers)
        
        # create optimizer based on model parameters with requires_grad=True
        optimizer = optim.Adam(get_params_to_update(gen), lr=learning_rate)

    else:
        optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
    
    # send generator to device
    gen = gen.to(device)
    
    # set up learning rate scheduler using hyperparameters
    scheduler_gen = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.6, patience=4, verbose=True)

    # use L2 norm as criterion
    criterion = nn.MSELoss()

    # init to list to store error
    error_list = []

    for _ in range(nr_epochs):  # train model

        gen.train()  # set model to training mode

        # use method to train for one epoch
        loss = gen.epoch_training(criterion, optimizer, dataloader, device)

        gen.eval()  # set model to evaluation
        # evaluation on validation set
        val_error = gen.error(x_eval.float(), y_eval.float(),
                              device, image_size=image_size)

        # if error has not decreased over the past 4
        # epochs decrease the lr by a factor of 0.6
        scheduler_gen.step(val_error)
        error_list.append(val_error)

        print(f" Epoch: {_:.0f},"
              f" Training loss: {loss:.4f},"
              f" Validation error: {val_error:.2f}")

    print("Finished training")
    # save model
    if save_model:
        gen.save_model(model_name, model_path)

    # plot the validation error over epochs
    if plot_fig:
        plt.plot(range(nr_epochs), error_list)
        plt.xlabel("epoch")
        plt.ylabel("validation error")
        plt.savefig(train_fig_path+train_fig_name)

    return gen, loss, val_error, error_list

if __name__ == '__main__':

    # Train a new model with the given parameters
    gen, loss, val_error, error_list = train_CNN_floris(
        nr_filters=16, nr_epochs=25, learning_rate=0.003, 
        batch_size=11, train_size=110, val_size=22,
        u_range=[3, 12], ti_range=[0.015, 0.25], yaw_range=[-30, 30],
        floris_path='./inputs/', floris_name='FLORIS_input_jensen', legacy=False, save_model=False,
        model_path='./TrainedModels/', model_name='FlorisJensen.pt',
        train_fig_path='./TrainingFigures/', train_fig_name='FlorisJensen.png')

    print(f"Final training loss: {loss:.4f},\n"
    f"Final validation error: {val_error:.2f}")