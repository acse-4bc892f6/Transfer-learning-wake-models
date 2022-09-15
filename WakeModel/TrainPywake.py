from CNN_model import Generator
from SetParams import get_params_to_update, set_parameter_requires_grad, ChangeParams
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time

def train_CNN_pywake(
    nr_epochs, learning_rate, train_size, val_size, batch_size,
    u_range, ti_range, yaw_range, u_list=None, ti_list=None, yaw_list=None,
    x_train=None, y_train=None, x_eval=None, y_eval=None, x_list=[0.], y_list=[0.], nr_input_vars=3,
    nr_filters=16, image_size=163, nr_workers=0, ML_model=None, change_params=False, num_layers=1,
    load_floris=True, floris_path='./inputs/', floris_name='FLORIS_input_jensen_1x3', legacy=False,
    model="Jensen", model_path="./TrainedModels/", model_name="PywakeJensen.pt", save_model=True, 
    plot_fig=True, train_fig_path="./TrainingFigures/", train_fig_name="PywakeJensen.png", seed=1):
    """
    Create a new model and train it for a certain number of epochs using a
    newly generated dataset. Dataset is generated using py_wake. Hyper-parameters 
    such as model size or learning rate can be changed as input to the function.
    After training the model error for all epochs is plotted and the model
    performance will be evaluated on a test set. Finally, the model
    will saved as the model_name which needs to add as .pt file

    Args:
        nr_epochs (int): Nr. of training epochs
        learing_rate (float): CNN learning rate.
        train_size (int): Size of the generated training set
        val_size (int): Size of the generated validation set
        batch_size (int): Training batch size
        u_range (list): Bound of wind speed values [u_min, u_max] used.
        ti_range (list): Bound of turbulence intensity values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        u_list (list, optional): Uniform distribution of wind speed values.
        ti_list (list, optional): Uniform distribution of turbulence intensity values.
        yaw_list (list, optional): Uniform distribution of yaw angle values.
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
        x_list (list, optional): list of x coordinates of non-FLORIS wind turbines.
            Defaults to [0.].
        y_list (list, optional): list of y coordinates of non-FLORIS wind turbines. 
            Defaults to [0.].
        nr_input_vars (int, optional): Nr. of input variables. Defaults to 3.
        nr_filters (int, optional): Nr. of filters used for the conv layers. Defaults to 16.
        image_size (int, optional): Size of the data set images, needs to match the
            model output size for the current model, which is 163 x 163.
        nr_workers (int, optional): Nr. of worker to load data. Defaults to 0.
        ML_model (Generator, optional): CNN model passed as an argument. Defaults to None.
        change_params (bool, optional): Set requires_grad=False in CNN layers if True. Defaults to False.
        num_layers (int, optional): Number of layers left as requires_grad=True. Defaults to 1.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Defaults to "./inputs/".
        floris_name (str, optional): Name of the FLORIS file without .json or .yaml.
            Defaults to "FLORIS_input_jensen".
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        model (str, optional): Wake deficit model. Default to "Jensen".
        model_path (str, optional): Path to directory to store the saved model.
        model_name (str, optional): Name of the trained saved model (needs be .pt).
        save_model (bool, optional): Save the trained model to model_path with model_name if True.
            Defaults to True.
        plot_fig (bool, optional): Plot validation error over epochs and save plot. Defaults to True.
        train_fig_path (str, optional): Path to directory to store plot of
            validation errors over epochs during training.
        train_fig_name (str, optional): File name of the saved plot of validation error over epochs.
        seed (int, optional): Random number seed. Default to 1.

    Returns:
        gen (Generator): Trained CNN model.
        loss (float): Training loss defined by the loss function.
        val_error (float): Percentage error on the validation set.
        error_list (list): List of percentage errors on validation set.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # The current inputs are: u, ti and yaw. If more are
    # # used please change this input var
    # nr_input_var = 3

    # set_seed(seed)

    if ML_model is None:
        # create a generator of the specified size
        gen = Generator(nr_input_vars, nr_filters)
    else:
        # generator passed in as argument
        gen = ML_model
        # ensure requires_grad=True for all model parameters
        set_parameter_requires_grad(model=gen, requires_grad=True)

    # generate training and validation sets if not passed as argument

    if x_train is None and y_train is None:
        x_train, y_train = gen.create_pywwake_dataset(
            model=model, size=train_size, image_size=image_size, seed=seed,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range, 
            u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
            load_floris=load_floris, x_list=x_list, y_list=y_list,
            floris_init_path=floris_path, floris_name=floris_name, legacy=legacy)

    if x_eval is None and y_eval is None:
        x_eval, y_eval = gen.create_pywwake_dataset(
            model=model, size=val_size, image_size=image_size, seed=seed,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
            load_floris=load_floris, x_list=x_list, y_list=y_list,
            floris_init_path=floris_path, floris_name=floris_name, legacy=legacy)
    
    # create tensor dataset from tensors
    dataset = TensorDataset(x_train.float(), y_train.float())
    # generate dataloader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=nr_workers)

    # init the weights of the generator
    gen.initialize_weights()

    # change model parameters to requires_grad=False if change_params=True
    if change_params:
        # change all model parameters to requires_grad=False
        set_parameter_requires_grad(gen)
        # model parameters of layer defaults to requires_grad=True when re-initialised
        # re-initialise num_layers
        ChangeParams(gen=gen, num_layers=num_layers)
        
        # create optimizer based on model parameters with requires_grad=True
        optimizer = optim.Adam(get_params_to_update(gen), lr=learning_rate)

    else:
        optimizer = optim.Adam(gen.parameters(), lr=learning_rate)

    # send generator to device
    gen = gen.to(device)
    
    # reduce learning rate by factor 0.6 
    # if validation error has not improved after 4 iterations
    scheduler_gen = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.6, patience=4, verbose=True)

    # use L2 norm as criterion
    criterion = nn.MSELoss()

    # init to list to store error
    error_list = []

    start = time.time() # start of training

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
    end = time.time() # end of training
    print("Training time: %.2f seconds" % (end-start)) # print training time
    # save model to model_path using model_name
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

    # range of values for each input parameter
    u_range=[3,12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]
    
    # hyperparameters for training
    train_size=400
    val_size=80
    batch_size=40
    nr_epochs=25
    learning_rate=0.003
    model='Fuga'

    # paths and file names
    floris_name='FLORIS_input_jensen_1x3'
    model_name='PywakeFuga.pt'
    train_fig_name='PywakeFuga.png'

    floris_path = './inputs/'
    model_path = './TrainedModels/FLORIS_input_jensen_1x3/'
    train_fig_path = './TrainingFigures/FLORIS_input_jensen_1x3/'

    gen, loss, val_error, val_error_list = train_CNN_pywake(
        nr_epochs=nr_epochs, learning_rate=learning_rate,
        train_size=train_size, val_size=val_size, batch_size=batch_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        # load_floris=False, x_list=[0., 630., 1260., 0., 630., 1260.], y_list=[100., 100., 100., -100., -100., -100.], # py_wake turbine
        floris_path=floris_path, floris_name=floris_name,
        model=model, model_name=model_name, model_path=model_path,
        train_fig_name=train_fig_path, train_fig_path=train_fig_name)


    print(f"Final training loss: {loss:.4f},\n"
    f"Final validation error: {val_error:.2f}")