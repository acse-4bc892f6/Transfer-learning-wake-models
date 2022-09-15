import matplotlib.pyplot as plt
from CNN_model import Generator
import numpy as np
import pandas as pd
import torch

""" Compute pwpe for a test dataset generated from Fuga, create and save pwpe dataset """

def PixelWisePercentageError(gen, x_eval, y_eval, device, image_size=163):
    """
    Calculate the pixel wise percentage error of the model on
    a evaluation set.

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
        error_list (list): pixel wise percentage error.
    """
    x_eval = x_eval.to(device)
    y_eval = y_eval.to(device)
    
    error_list = []
    # Use model to predict the wakes for the given conditions in x
    model_predict = gen(x_eval.float())

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

    return error_list

def test_trained_CNN_single(
    model_path, test_size, pywake_model,
    u_range, ti_range, yaw_range,
    seed=123, device='cpu',
    load_floris=True, legacy=False,
    floris_path='./inputs/',
    floris_name='FLORIS_input_jensen_1x3'):
    """
    Compute average pixel-wise percentage error (pwpe) for predictions generated from
    a CNN that has been trained using data generated from a single py_wake model.

    Args:
        model_path (str): Path to directory of the saved CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        pywake_model (str): py_wake model used to train the CNN.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        seed (int, optional): Random number seed. Default to 123.
        device (torch.device, optional): Device on which model/data is stored, cpu or cuda.
            Default to cpu.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".

    Returns:
        pwpe (list): Pixel-wise percentage error of CNN computed from dataset.
    """

    # instantiate Generator object
    gen = Generator(nr_input_var=3, nr_filter=16)

    # create dataset
    x_test, y_test = gen.create_pywwake_dataset(size=test_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        load_floris=load_floris, floris_init_path=floris_path,
        floris_name=floris_name, legacy=legacy,
        model=pywake_model, image_size=163, seed=seed, TimeGen=False)

    # load parameters to Generator object
    model_name = 'Pywake'+pywake_model+'.pt'
    # print("Loading from", model_name)
    gen.load_model(path=model_path+model_name, device=device)
    # compute pwpe
    pwpe = PixelWisePercentageError(gen=gen, x_eval=x_test, y_eval=y_test, device=device)
    
    return pwpe

def test_trained_CNN_mftl(model_path, test_size,
    fuga_model_path, LF_list,
    u_range, ti_range, yaw_range, seed=123,
    device='cpu', HF_model='Fuga',
    load_floris=True, legacy=False,
    floris_path='./inputs/',
    floris_name='FLORIS_input_jensen_1x3'):
    """
    Compute average pixel-wise percentage error (pwpe) for predictions generated from
    a CNN that has undergone transfer learning with multi-fidelity data, and
    the data is generated from py_wake models.

    Args:
        model_path (str): Path to directory of the saved CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        fuga_model_path (str): Path to directory of saved benchmark CNN trained only on Fuga.
        LF_list (list): List of saved CNN trained using low-fidelity model data.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        seed (int, optional): Random number seed. Default to 123.
        device (torch.device, optional): Device on which model/data is stored, cpu or cuda.
            Default to cpu.
        HF_model (str, optional): high-fidelity py_wake model used during transfer learning.
            Default to Fuga.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".

    Returns:
        pwpe_list (list): Pixel-wise percentage error of each CNN loaded from file.
        fuga_pwpe (list): Pixel-wise percentage error of Fuga-trained CNN.
    """

    # instantiate Generator object
    gen = Generator(nr_input_var=3, nr_filter=16)
    # create dataset
    x_test, y_test = gen.create_pywwake_dataset(size=test_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        load_floris=load_floris, floris_init_path=floris_path,
        floris_name=floris_name, legacy=legacy,
        model=HF_model, image_size=163, seed=seed, TimeGen=False)

    # store pixel-wise percentage error of each CNN
    pwpe_list = []
    for LF_model in LF_list:
        model_name = 'Pywake'+LF_model+HF_model+'.pt'
        print("Loading from ", model_name)
        gen.load_model(model_path+model_name, device=device)
        pwpe = PixelWisePercentageError(gen=gen, x_eval=x_test, y_eval=y_test, device=device)
        pwpe_list.append(pwpe)

    print("Loading from PywakeFuga.pt")
    gen.load_model(fuga_model_path+'PywakeFuga.pt', device=device)
    fuga_pwpe = PixelWisePercentageError(gen=gen, x_eval=x_test, y_eval=y_test, device=device)

    
    return pwpe_list, fuga_pwpe

def test_trained_LF_CNN_HF(
    LF_model_path, test_size, LF_list,
    u_range, ti_range, yaw_range, seed=123,
    device='cpu', HF_model='Fuga',
    load_floris=True, legacy=False,
    floris_path='./inputs/',
    floris_name='FLORIS_input_jensen_1x3'):
    """
    Compute average pixel-wise percentage error (pwpe) for predictions generated from
    a CNN that has only been trained with LF data. pwpe computed using HF data.

    Args:
        LF_model_path (str): Path to directory of the saved LF CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        fuga_model_path (str): Path to directory of saved benchmark CNN trained only on Fuga.
        LF_list (list): List of saved CNN trained using low-fidelity model data.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        seed (int, optional): Random number seed. Default to 123.
        device (torch.device, optional): Device on which model/data is stored, cpu or cuda.
            Default to cpu.
        HF_model (str, optional): high-fidelity py_wake model used during transfer learning.
            Default to Fuga.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
    """

    # instantiate Generator object
    gen = Generator(nr_input_var=3, nr_filter=16)
    # create dataset
    x_test, y_test = gen.create_pywwake_dataset(size=test_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        load_floris=load_floris, floris_init_path=floris_path,
        floris_name=floris_name, legacy=legacy,
        model=HF_model, image_size=163, seed=seed, TimeGen=False)

    # store pixel-wise percentage error of each CNN
    LF_pwpe_list = []
    for LF_model in LF_list:
        LF_model_name = 'Pywake'+LF_model+'.pt'
        print("Loading from ", LF_model_name)
        gen.load_model(LF_model_path+LF_model_name)
        LF_pwpe = PixelWisePercentageError(gen=gen, x_eval=x_test, y_eval=y_test, device=device)
        LF_pwpe_list.append(LF_pwpe)

    # LF_pwpe_list = np.array(LF_pwpe_list)
    
    return LF_pwpe_list

def CreateDatasetAndBoxPlot(data_list, LF_list,
    results_path, dataset_name='pwpe_dataset.csv', boxplot_name='pwpe_BoxPlot.png', ymax=10):
    """
    Create pixel-wise percentage error (pwpe) dataset and pwpe box plot,
    then save them to specified directory.

    Args:
        data_list (list): pwpe computed for each trained CNN.
        LF_list (list): xticklist for box plot, contains name of LF models.
        results_path (str): Path to directory storing the dataset and box plot.
        dataset_name (str, optional): File name of saved dataset. Default to 'pwpe_dataset.csv'.
        boxplot_name (str, optional): File name of saved box plot. Default to 'pwpe_BoxPlot.png'.
        ymax (int, optional): Maximum value on the y-axis of the box plot. Default to 10.
    """

    data_dict = {}
    for i in range(len(LF_list)):
        data_dict[LF_list[i]] = data_list[i]

    data_df = pd.DataFrame.from_dict(data_dict)
    data_df.to_csv(results_path+dataset_name)

    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.boxplot(data_df.values)
    ax.set_xticklabels(LF_list)
    ax.set_xlabel('py_wake models', fontsize=14)
    ax.set_ylabel('pixel-wise percentage error', fontsize=14)
    ax.set_ylim(ymin=0, ymax=ymax)
    plt.savefig(results_path+boxplot_name)

    return None

def LF_CNN_error(LF_list, model_path, test_size, u_range, ti_range, yaw_range,
    load_floris=True, legacy=False, floris_path='./inputs/', floris_name='FLORIS_input_jensen',
    dataset_name='pwpe_dataset.csv', boxplot_name='pwpeBoxPlot.png',
    results_path='./results/', ymax=10, seed=123):
    """
    Compute pixel-wise percentage error (pwpe) for CNNs trained with low-fidelity (LF) data only.
    pwpe is computed based on the LF data which the CNN is trained on. 
    pwpe dataset and box plots are created and saved.

    Args:
        LF_list (list): xticklist for box plot, contains name of LF models.
        model_path (str): Path to directory of the saved CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        dataset_name (str, optional): File name of saved dataset. Default to 'pwpe_dataset.csv'.
        boxplot_name (str, optional): File name of saved box plot. Default to 'pwpe_BoxPlot.png'.
        results_path (str, optional): Path to directory storing the results. Default to './results/'.
        ymax (int, optional): Maximum value on the y-axis of the box plot. Default to 10.
        seed (int, optional): Random number seed. Default to 123.
    """

    pwpe_list = []

    for i in range(len(LF_list)):
        model_name = 'Pywake'+LF_list[i]+'.pt'
        print("Loading from", model_name)

        pwpe = test_trained_CNN_single(model_path=model_path,
        test_size=test_size, pywake_model=LF_list[i], seed=seed,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        device='cpu', load_floris=load_floris, legacy=legacy,
        floris_path=floris_path, floris_name=floris_name)

        pwpe_list.append(pwpe)

    
    CreateDatasetAndBoxPlot(data_list=pwpe_list, LF_list=LF_list, results_path=results_path,
                            dataset_name=dataset_name, boxplot_name=boxplot_name, ymax=ymax)

    return None

def LF_CNN_error_Fuga(LF_list, LF_model_path, test_size, u_range, ti_range, yaw_range, seed=123,
    load_floris=True, legacy=False, floris_path='./inputs/', floris_name='FLORIS_input_jensen',
    dataset_name='LF_pwpe_dataset.csv', boxplot_name='LF_pwpeBoxPlot.png', results_path='./results/', ymax=10):
    """
    Compute pixel-wise percentage error (pwpe) for CNNs trained with low-fidelity (LF) data only.
    pwpe is computed based on the Fuga data. pwpe dataset and box plots are created and saved.

    Args:
        LF_list (list): xticklist for box plot, contains name of LF models.
        LF_model_path (str): Path to directory of the saved CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        seed (int, optional): Random number seed. Default to 123.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        dataset_name (str, optional): File name of saved dataset. Default to 'pwpe_dataset.csv'.
        boxplot_name (str, optional): File name of saved box plot. Default to 'pwpe_BoxPlot.png'.
        results_path (str, optional): Path to directory storing the results. Default to './results/'.
        ymax (int, optional): Maximum value on the y-axis of the box plot. Default to 10.
    """

    LF_pwpe_list = test_trained_LF_CNN_HF(
        test_size=test_size, LF_model_path=LF_model_path, LF_list=LF_list,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range, seed=seed,
        load_floris=load_floris, legacy=legacy, 
        floris_path=floris_path, floris_name=floris_name)

    CreateDatasetAndBoxPlot(data_list=LF_pwpe_list, LF_list=LF_list,
        results_path=results_path, dataset_name=dataset_name, boxplot_name=boxplot_name, ymax=ymax)

    return None


def MFTL_CNN_error(LF_list, model_path, test_size, u_range, ti_range, yaw_range, seed=123,
    load_floris=True, legacy=False, floris_path='./inputs/', floris_name='FLORIS_input_jensen',
    dataset_name='pwpe_dataset.csv', boxplot_name='pwpeBoxPlot.png', results_path='./results/', ymax=10):
    """
    Compute pixel-wise percentage error (pwpe) for CNNs undergone transfer learning using multi-fidelity data.
    pwpe is computed based on the Fuga data. pwpe dataset and box plots are created and saved.

    Args:
        LF_list (list): xticklist for box plot, contains name of LF models.
        model_path (str): Path to directory of the saved CNN.
        test_size (int): Size of the dataset used to compute average pwpe.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        seed (int, optional): Random number seed. Default to 123.
        load_floris (bool, optional): Load turbine information from FLORIS file if True.
            Defaults to True.
        legacy (bool, optional): True if FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        dataset_name (str, optional): File name of saved dataset. Default to 'pwpe_dataset.csv'.
        boxplot_name (str, optional): File name of saved box plot. Default to 'pwpe_BoxPlot.png'.
        results_path (str, optional): Path to directory storing the results. Default to './results/'.
        ymax (int, optional): Maximum value on the y-axis of the box plot. Default to 10.
    """

    pwpe_list, fuga_pwpe = test_trained_CNN_mftl(
        model_path=model_path, test_size=test_size, seed=seed,
        fuga_model_path=fuga_model_path, LF_list=LF_list,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        load_floris=load_floris, legacy=legacy, 
        floris_path=floris_path, floris_name=floris_name)

    pwpe_list.append(fuga_pwpe)
    LF_list.append('Fuga')

    CreateDatasetAndBoxPlot(data_list=pwpe_list, LF_list=LF_list,
    results_path=results_path, dataset_name=dataset_name, boxplot_name=boxplot_name, ymax=ymax)

    return None


if __name__ == '__main__':

    # path and file name of floris input file
    load_floris=True
    floris_path='./inputs/'
    floris_name='FLORIS_input_jensen_1x3'
    legacy = False

    # path to directory storing multi-fidelity trained CNNs
    model_path='./TrainedModels/' + floris_name + '/Experiment2/LF1320HF400/'

    # path to directory storing LF trained CNNs
    LF_model_path='./TrainedModels/' + floris_name + '/Experiment1/LF2500/'

    # path to directory that stores the results
    results_path = './results/' + floris_name + '/Experiment2/LF1320HF400/'

    # path to directory storing benchmark CNN trained only with Fuga
    fuga_model_path = './TrainedModels/' + floris_name + '/Size5000/'

    LF_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss']

    # mftl
    LF_list1 = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss', 'Fuga']


    # size of test dataset
    test_size = 500

    # range of values for each input parameter
    u_range=[3, 12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]

    ### re-adjust y-axis ###

    # data_df = pd.read_csv(results_path+'pwpe_dataset_100.csv', index_col=0)
    # _, ax = plt.subplots(1, 1, figsize=(12, 6))
    # ax.boxplot(data_df.values)
    # ax.set_xticklabels(LF_list1)
    # ax.set_xlabel('py_wake models', fontsize=14)
    # ax.set_ylabel('pixel-wise percentage error', fontsize=14)
    # ax.set_ylim(ymin=0, ymax=5)
    # plt.savefig(results_path+'pwpeBoxPlot_10.png')


    ## Create box plots for mftl CNN ###

    MFTL_CNN_error(LF_list=LF_list, model_path=model_path, test_size=test_size,
    u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
    load_floris=load_floris, legacy=legacy, results_path=results_path,
    floris_path=floris_path, floris_name=floris_name, seed=10, 
    dataset_name='pwpe_dataset.csv', boxplot_name='pwpeBoxPlot.png')


    ### LF trained CNN, test error based on LF model ###

    # LF_CNN_error(LF_list=LF_list, model_path=LF_model_path, test_size=test_size,
    #     u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
    #     load_floris=load_floris, legacy=legacy, results_path=results_path,
    #     floris_path=floris_path, floris_name=floris_name)


    ### LF trained CNN, test error based on Fuga ###

    # LF_CNN_error_Fuga(LF_list=LF_list, LF_model_path=LF_model_path, test_size=test_size,
    # u_range=u_range, ti_range=ti_range, yaw_range=yaw_range, results_path=results_path,
    # load_floris=load_floris, legacy=legacy, floris_path=floris_path, floris_name=floris_name)


    ### benchmark, test error based on LF model ###

    # LF_CNN_error(LF_list=LF_list1, model_path=LF_model_path, test_size=test_size,
    #     u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
    #     load_floris=load_floris, legacy=legacy, results_path=results_path,
    #     floris_path=floris_path, floris_name=floris_name)

    ### benchmark, test error based on Fuga ###

    # LF_CNN_error_Fuga(LF_list=LF_list1, LF_model_path=LF_model_path, test_size=test_size,
    # u_range=u_range, ti_range=ti_range, yaw_range=yaw_range, results_path=results_path,
    # load_floris=load_floris, legacy=legacy, floris_path=floris_path, floris_name=floris_name,
    # ymax=14)
