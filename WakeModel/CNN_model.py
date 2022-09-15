from Pywake import PywakeWorkflow
from SetSeed import set_seed
import floris.tools as wfct
import torch
import torch.nn as nn
import numpy as np
import random
import time

def uniform_distribution(size:int, param_range:list, seed:int=1):
    """
    Return list of values following uniform distribution given the minimum and maximum values

    Args:
        size (int): Size of the uniformly distributed list
        param_range (list): Minimum and maximum values of the uniformly distributed list
        seed (int, optional): Random number seed. Default to 1.

    Returns:
        param_list (list): Uniformly distributed values
    """
    set_seed(seed=seed)
    param_list = [round(random.uniform(param_range[0], param_range[1]), 2) for
                i in range(0, size)]
    return param_list

""" 
Title: CNN_model
Author: Jens Bauer
Date: 25 June 2022
Code version: 1.0
Availability: https://github.com/soanagno/wakenet/tree/master/Code/CNNWake 
"""

class Generator(nn.Module):
    """
    The class is the Neural Network that generates the flow field around one
    or more wind turbines. The network uses the pytorch framwork and uses fully
    connected and transpose convolutional layers.
    The methods of this class include the training of the network,
    testing of the accuracy and generaton of the training data.
    """

    def __init__(self, nr_input_var, nr_filter):
        """
        init method that generates the network architecture using pytroch's
        ConvTranspose2d and Sequential layers. The number of input varibles
        and size of the given network can be changed. The output size will not
        change and it set at 163 x 163 pixels.

        Args:
            nr_input_var (int): Nr. of inputs, usually 3 for u, ti and yaw
            nr_filter (int): Nr. filters used in deconv layers, more filters
                             means that the network will have more parameters
        """
        super(Generator, self).__init__()
        # linear layer
        self.FC_Layer = nn.Sequential(nn.Linear(in_features=nr_input_var,
                                                out_features=9))

        self.net1 = self.layer(1, nr_filter * 16, 4, 2, 1)
        self.net2 = self.layer(nr_filter * 16, nr_filter * 8, 4, 1, 1)
        self.net3 = self.layer(nr_filter * 8, nr_filter * 8, 4, 2, 1)
        self.net4 = self.layer(nr_filter * 8, nr_filter * 4, 4, 2, 1)

        self.second_last_layer = self.layer(nr_filter * 4, nr_filter * 4, 3, 2, 1)

        self.last_layer = nn.ConvTranspose2d(nr_filter * 4, 1, kernel_size=3,
                            stride=3, padding=1)

    def layer(self, in_filters, out_filters, kernel_size, stride, padding):
        """
        One layer of the CNN which consits of ConvTranspose2d,
        a batchnorm and LRelu activation function.
        Function is used to define one layer of the network

        Args:
            in_filters (int): Nr. of filters in the previous layer
            out_filters (int): Nr. of output filters
            kernel_size (int): Size of the ConvTranspose2d layer
            stride (int): Stride of the ConvTranspose2d layer
            padding (int): Padding used in this layer

        Returns:
            nn.Sequential: Pytroch Sequential container that defines one layer
        """
        # One layer of the network uses:
        # Deconvolutional layer, then batch norm and leakyrelu
        # activation function
        single_layer = nn.Sequential(nn.ConvTranspose2d(in_filters,
                                                        out_filters,
                                                        kernel_size,
                                                        stride,
                                                        padding,
                                                        bias=False),
                                     nn.BatchNorm2d(out_filters),
                                     nn.LeakyReLU(0.2))

        return single_layer

    def initialize_weights(self):
        """
        Initilize weights using a normal distribution with mean = 0,std2 = 0.02
        which has helped training. Loop over all modules, if module is
        convolutional layer or batchNorm then initialize weights.

        Args:
            model (torch model): Neural network model defined using Pytorch
        """
        # for ever layer in model
        for m in self.modules():
            # check if it deconvolutional ot batch nrom layer
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                # initialize weights using a normal distribution
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        """
        Functions defines a forward pass though the network. Can be used for
        a single input or a batch of inputs

        Args:
            x (torch.tensor): input tensor, to be passed through the network

        Returns:
            flow_fields (torch.tensor): Output of network
        """
        # first the fully connected layer takes in the input, and outputs
        # 9 neurons which are reshaped into a 3x3 array
        x = self.FC_Layer(x).view(-1, 1, 3, 3)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.second_last_layer(x)
        x = self.last_layer(x)
        # the Conv layers take in the 3x3 array and output a 163x163 array
        return x

    @staticmethod
    def create_floris_dataset(
        size, u_range, ti_range, yaw_range,
        u_list=None, ti_list=None, yawn_list=None, TimeGen=True,
        floris_init_path="./", floris_name="FLORIS_input_jensen",
        legacy=False, normalisation=12, image_size=163, seed=1):
        """
        Function to generate the dataset needed for training using FLORIS.
        The flowfield around a turbine is generated for a range of wind
        speeds, turbulent intensities and yaw angles. The 2d array and
        correspoding init conditions are saved for training.

        Function can be used to generated training, validation and test sets.

        Args:
            size (int): Size of the dataset
            image_size (int): Size of the flow field outputs that
                are generated, this depends on the neural network used,
                should be 163.
            u_range (list): Bound of u values [u_min, u_max] used.
            ti_range (list): Bound of TI values [TI_min, TI_max] used.
            yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
            u_list (list, optional): Uniform distribution of wind speed values.
            ti_list (list, optional): Uniform distribution of turbulence intensity values.
            yawn_list (list, optional): Uniform distribution of yaw angle values.
            TimeGen (bool, optional): print out data generation time in seconds if True.
                Defaults to True.
            floris_init_path (str, optional): Path to the FLORIS file.
                Defaults to "./".
            floris_name (str, optional): name of the FLORIS file without
                .json or .yaml. Defaults to "FLORIS_input_jensen".
            legacy (bool, optional): True if FLORIS file is json file.
                False if FLORIS file is yaml file.
                Defaults to False.
            normalisation (int, optional): Normalise CNN output so that
                wind speed lies between 0 and 1. Defaults to 12.
            image_size (int): Size of the flow field outputs that are generated,
            this depends on the neural network used. Defaults to 163.
            seed (int, optional): Random number seed. Default to 1.

        Returns:
            y (torch.tensor): Tensor of size (size, image_size, image_size)
                which includes all the generated flow fields. The flow fields
                are normalised to help training
            x (torch.tensor): Tensor of size (size, 1, 3) which includes the
                flow conditons of the correspoding flow field in the x tensor.
        """

        # uniform distribution of inflow conditions
        if u_list is None:
            u_list = uniform_distribution(size=size, param_range=u_range, seed=seed)
        if ti_list is None:
            ti_list = uniform_distribution(size=size, param_range=ti_range, seed=seed)
        if yawn_list is None:
            yawn_list = uniform_distribution(size=size, param_range=yaw_range, seed=seed)

        # create FlorisInterface from input file
        if legacy:
            floris_turbine = wfct.FlorisInterfaceLegacyV2(
                floris_init_path + floris_name + ".json"
            )
        else:
            floris_turbine = wfct.FlorisInterface(
                floris_init_path + floris_name + ".yaml"
            )

        # initialize empty numpy array to store 2d arrays and
        # corresponding u, ti and yawn values
        y = np.zeros((size, image_size, image_size))
        x = np.zeros((size, 3))

        # create train examples
        print("generate FLORIS data")
        if TimeGen:
            start = time.time()
        for _ in range(0, size):
            if _ % 100 == 0:
                print(f"{_}/{size}")
            # set wind speed, ti and yawn angle for FLORIS model

            floris_turbine.reinitialize(
                wind_speeds=[u_list[_]],
                turbulence_intensity=ti_list[_]
                )

            # calculate the wakefield
            to_shape = np.shape(floris_turbine.floris.grid.x)[:3]
            yaw_angle = yawn_list[_] * np.ones((to_shape))
            floris_turbine.calculate_wake(yaw_angles=yaw_angle)

            # extract horizontal plane at hub height
            cut_plane = floris_turbine.calculate_horizontal_plane(
                height=90,
                x_resolution=image_size,
                y_resolution=image_size,
                x_bounds=[0, 3000],
                y_bounds=[-200, 200]
            ).df.u.values.reshape(image_size, image_size)

            # save the wind speed values of the plane at hub height and
            # the corresponding turbine stats
            y[_] = cut_plane
            x[_] = u_list[_], ti_list[_], yawn_list[_]

        print("Finished generating data")
        if TimeGen:
            end = time.time()
            print("Data generation time is %.2f seconds" % (end-start))

        # turn numpy array into a pytroch tensor
        x_tensor = torch.from_numpy(x)
        # The wind speeds are normalised
        # i.e. every value will be between 0-1 which helps training
        y /= normalisation
        y_tensor = torch.from_numpy(y).view(-1, 1, np.shape(y)[1], np.shape(y)[2])

        return x_tensor, y_tensor
    
    @staticmethod
    def create_pywwake_dataset(
        size, u_range, ti_range, yaw_range,
        u_list=None, ti_list=None, yawn_list=None,
        load_floris=True, x_list=[0.], y_list=[0.],
        floris_init_path="./inputs/",
        floris_name="FLORIS_input_jensen",
        model="Jensen", TimeGen=True, legacy=False,
        image_size=163, normalisation=12, seed=1):
        """
        Function to generate the dataset needed for training using py_wake.
        Wind turbine object in py_wake is created from FLORIS input file.
        Another option is to create in-built py_wake wind turbine object.
        The flowfield around a turbine is generated for a range of wind
        speeds, turbulent intensities and yaw angles. The 2d array and
        correspoding init conditions are saved for training.
        Squared sum superposition model, Jiminez wake deflection model, 
        and Crespo Hernandez turbulence model are fixed in the wind farm model.

        Function can be used to generated training, validation and test sets.

        Args:
            size (int): Size of the dataset.
            u_range (list): Bound of wind speed values [u_min, u_max] used.
            ti_range (list): Bound of turbulence intensity values [TI_min, TI_max] used.
            yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
            u_list (list, optional): Uniform distribution of wind speed values.
            ti_list (list, optional): Uniform distribution of turbulence intensity values.
            yawn_list (list, optional): Uniform distribution of yaw angle values.
            load_floris (bool, optional): Load turbine information from FLORIS file if True.
                Defaults to True.
            x_list (list, optional): list of x coordinates of non-FLORIS wind turbines.
                Defaults to [0.].
            y_list (list, optional): list of y coordinates of non-FLORIS wind turbines. 
                Defaults to [0.].
            floris_init_path (str, optional): Path to the FLORIS file.
                Defaults to "./inputs/".
            floris_name (str, optional): name of the FLORIS file without .json or .yaml.
                Defaults to "FLORIS_input_jensen".
            legacy (bool, optional): True if FLORIS file is json file. 
                False if FLORIS file is yaml file.
                Defaults to False.
            model (str, optional): wake deficit model. Default to "Jensen".
            TimeGen (bool, optional): print out data generation time in seconds if True.
                Defaults to True.
            image_size (int): Size of the flow field outputs that are generated,
            this depends on the neural network used. Defaults to 163.
            normalisation (int, optional): Normalise CNN output so that
                wind speed lies between 0 and 1. Defaults to 12.
            seed (int, optional): Random number seed. Default to 1.

        Returns:
            y (torch.tensor): Tensor of size (size, image_size, image_size)
                which includes all the generated flow fields. The flow fields
                are normalised to help training
            x (torch.tensor): Tensor of size (size, 1, 3) which includes the
                flow conditons of the correspoding flow field in the x tensor.
        """

        # uniform distribution of inflow conditions
        if u_list is None:
            u_list = uniform_distribution(size=size, param_range=u_range, seed=seed)
        if ti_list is None:
            ti_list = uniform_distribution(size=size, param_range=ti_range, seed=seed)
        if yawn_list is None:
            yawn_list = uniform_distribution(size=size, param_range=yaw_range, seed=seed)
        
        # initialize empty numpy array to store 2d arrays and
        # corresponding u, ti and yawn values
        y = np.zeros((size, image_size, image_size))
        x = np.zeros((size, 3))

        if TimeGen:
            start = time.time()

        # create train examples
        print("Generate py_wake data using " + model)
        for _ in range(0, size):
            if _ % 100 == 0:
                print(f"{_}/{size}")

            ews = PywakeWorkflow(
                ws=u_list[_], ti=ti_list[_], yaw_angle=yawn_list[_],
                model=model, load_floris=load_floris, x_list=x_list, y_list=y_list, 
                floris_path=floris_init_path, floris_name=floris_name,
                legacy=legacy, show_fig=False)

            # save the wind speed values of the plane at hub height and
            # the corresponding turbine stats
            y[_] = ews
            x[_] = u_list[_], ti_list[_], yawn_list[_]

        print("Finished generating data")

        if TimeGen:
            end = time.time()
            print("Generating data using py_wake %s model takes %.2f seconds" % (model, end-start))

        # turn numpy array into a pytroch tensor
        x_tensor = torch.from_numpy(x)
        # The wind speeds are normalised
        # i.e. every value will be between 0-1 which helps training
        y /= normalisation
        y_tensor = torch.from_numpy(y).view(-1, 1, np.shape(y)[1], np.shape(y)[2])

        return x_tensor, y_tensor

    def error(self, x_eval, y_eval, device, image_size=163):
        """
        Calculate the average pixel wise percentage error of the model on
        a evaluation set. The error function calculates:

        .. math:: 
            1/set\_size * \sum_{n=0}^{set\_size}(1/image\_size^2 *
            \sum_{i=0}^{image\_size^2}(100*abs(y\_eval_{n,i} - prediction_{n,i})/
            max(y\_eval_{n,i})))

        Args:
            x_eval (torch.tensor): Tensor of size (set_size, 1, 3) which includes the
                flow conditons of the correspoding flow field in the x_eval tensor.
                set_size is the size of the evaluation set.
            y_eval (torch.tensor): Tensor of size (set_size, image_size, image_size)
                which includes all the generated normalised flow fields.
                set_size is the size of the evaluation set.
            device (torch.device): Device to store and run the neural network on,
                either cpu or cuda.
            image_size (int, optional): Size of the flow field outputs that
                are generated. Default to 163.

        Returns:
            mean_error (float): average pixel wise percentage error
        """
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        
        error_list = []
        # Use model to predict the wakes for the given conditions in x
        model_predict = self.forward(x_eval)

        for n in range(0, len(x_eval)):
            # Calculate the mean error between CNNwake output and FLORIS
            # for a given flow field using the function given above
            
            # pixel_error = np.sum(abs(
            #         y_eval.detach().cpu().numpy()[n] -
            #         model_predict.squeeze(1)[n].detach().cpu().numpy()) /
            #         (torch.max(y_eval.detach()[n]).cpu().numpy()))

            denominator = torch.max(y_eval.detach()[n]).cpu().numpy()
            if (np.allclose(denominator, 0)):
                pixel_error = 0.
            else:
                pixel_error = np.sum(abs(
                        y_eval.detach().cpu().numpy()[n] -
                        model_predict.squeeze(1)[n].detach().cpu().numpy()) / denominator)
            # divide by number of pixels in array for an mean value
            pixel_error /= image_size * image_size
            error_list.append(pixel_error * 100)

        # return mean error
        mean_error = np.mean(error_list)
        return mean_error

    def epoch_training(self, criterion, optimizer, dataloader, device):
        """
        Trains the model for one epoch data provided by dataloader. The model
        will be updated after each batch and the function will return the
        train loss of the last batch

        Args:
            criterion (torch.nn.criterion): Loss function used to train model
            optimizer (torch.optim.Optimizer): Optimizer for gradient descent
            dataloader (torch.utils.DataLoader): Dataloader to store dataset
            device (torch.device): Device on which model/data is stored, cpu or cuda

        Returns:
            training loss (float): Loss of training set defined by criterion
        """
        # For all training data in epoch
        for X, y in dataloader:
            # move data to device
            X = X.to(device)
            y = y.to(device)
            # images need to be in correct shape: batch_size x 1 x 1 x 3


            optimizer.zero_grad()  # Zero gradients of previous step
            
            # compute reconstructions of flow-field using the CNN
            outputs = self.forward(X).to(device)

            # compute training reconstruction loss using the
            # loss function set
            train_loss = criterion(outputs, y)

            train_loss.backward()  # compute accumulated gradients
            optimizer.step()  # Do optimizer step

        # return training loss
        return train_loss.item()

    def load_model(self, path='.', device='cpu'):
        """
        Function to load model from a pt file into this class.

        Args:
            path (str, optional): path to saved model. Default to '.'.
            device (torch.device, optional): Device to load onto,
                cpu or cuda. Default to 'cpu'.

        """
        # load the pretrained model
        self.load_state_dict(torch.load(path, map_location=device))

    def save_model(self, name, path):
        """
        Function to save current model paramters so that it can
        be used again later. Needs to be saved with as .pt file

        Args:
            name (str): name of .pt file that stores the model.
            path (str): path to directory to which the .pt file is saved.
        """
        # Save current model parameters
        torch.save(self.state_dict(), path+name)
