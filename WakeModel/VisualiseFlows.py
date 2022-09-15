from CNN_model import Generator
from Pywake import PywakeWorkflow
import floris.tools as wfct
import torch
import matplotlib.pyplot as plt
import numpy as np

def CompareFlows(
    u, ti, yaw_angle, package="py_wake", pywake_model="Jensen",
    nr_input_var=3, nr_filters=16, image_size=163,
    model_path="./TrainedModels/", model_name="FlorisJensen.pt",
    load_floris=True, x_list=[0.], y_list=[0.],
    floris_path="./inputs/", floris_name="FLORIS_input_jensen",
    legacy=False, fig_path="./results/", fig_name="FlorisJensen.png",
    show_fig=False, save_fig=True):

    """
    Loads a trained model from a .pt file, use the model and selected package
    to predict the flow field around the wind turbine. 
    Contour plots of flow fields from package, model and relative difference are shown.

    Args:
        u (float): wind speed in m/s.
        ti (float): turbulence intensity.
        yaw_angle (float): yaw angle.
        package (str): Package used for training the model. 
            Either 'FLORIS' or 'py_wake'. Default to 'py_wake'.
        pywake_model (str, optional): model used to generate flow field in py_wake.
        nr_input_var (int, optional): Nr. of inpu variables for the model.
            Default to 3.
        nr_filters (int, optional): Nr. of filters used for the conv layers.
            Default to 16.
        image_size (int): Size of the data set images, needs to match the
            model output size which is 163 x 163
        model_path (str, optional): Path to directory to store the saved model
        model_name (str): Name of the trained saved model (needs be .pt)
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
        fig_path (str, optional): Path to directory to store the figure
        fig_name (str, optional): File name of the stored figure
        show_fig (bool, optional): Show figure on window if True.
            Defaults to False.
        save_fig (bool, optional): Save figure to fig_path with fig_name.
            Defaults to True.

    Returns:
        y, prediction (numpy.ndarray, numpy.ndarray): flow field prediction
            from FLORIS and model respectively
    """

    # instantiate and load model
    model = Generator(nr_input_var, nr_filters)
    model.load_model(model_path+model_name)

    x = np.array([u, ti, yaw])
    x = torch.from_numpy(x)

    # use FLORIS to compute flow field
    if package == "FLORIS":

        if legacy:
            fi = wfct.FlorisInterfaceLegacyV2(floris_path+floris_name+".json")
        else:
            fi = wfct.FlorisInterface(floris_path+floris_name+".yaml")

        # compute flow field from floris
        fi.reinitialize(wind_speeds=[u], turbulence_intensity=ti)
        to_shape = np.shape(fi.floris.grid.x)[:3]
        yaw_angle = yaw * np.ones((to_shape))
        fi.calculate_wake(yaw_angles=yaw_angle)
        y = fi.calculate_horizontal_plane(
                    height=90,
                    x_resolution=image_size,
                    y_resolution=image_size,
                    x_bounds=[0, 3000],
                    y_bounds=[-200, 200]
                ).df.u.values.reshape(image_size, image_size)

    # use py_wake to compute flow field
    elif package == "py_wake":

        # compute flow field from py_wake
        y = PywakeWorkflow(ti=ti, ws=u, yaw_angle=yaw, show_fig=False,
            load_floris=load_floris, x_list=x_list, y_list=y_list,
            floris_path=floris_path, floris_name=floris_name, model=pywake_model)

    # compute flow field from CNN
    prediction = model(x.float())
    prediction = prediction[0,0,:,:].detach().numpy()
    prediction *= 12

    # plot both flow fields and absolute difference
    fig = plt.figure(figsize=(12,6))
    fig.add_subplot(3,1,1)
    plt.imshow(y, cmap='coolwarm', extent=[0, 3000, -200, 200])
    if package == "FLORIS":
        plt.title("FLORIS (u=%.2f, ti=%.2f, yaw=%.2f)" % (u, ti, yaw))
    elif package == "py_wake":
        plt.title("py_wake %s (u=%.2f, ti=%.2f, yaw=%.2f)" % (pywake_model, u, ti, yaw))
    plt.colorbar(label='wind speed [m/s]')
    fig.add_subplot(3,1,2)
    plt.imshow(prediction, cmap='coolwarm', extent=[0, 3000, -200, 200])
    plt.title("CNN_model (u=%.2f, ti=%.2f, yaw=%.2f)" % (u, ti, yaw))
    plt.colorbar(label='wind speed [m/s]')
    difference = np.abs(prediction-y)
    fig.add_subplot(3,1,3)
    plt.imshow(difference, cmap='coolwarm', extent=[0, 3000, -200, 200])
    plt.title("Absolute difference")
    plt.colorbar()
    if show_fig:
        plt.show()
    if save_fig:
        # save plot
        plt.savefig(fig_path+fig_name)

    return y, prediction

def LF_trained_CNN(model_list, u, ti, yaw, model_path, floris_path, floris_name, fig_path):
    """
    Create figures for CNNs that has been trained using a single model py_wake package.
    Compare flow fields generated by those CNNs with flow fields generated from py_wake.
    Assume turbines are loaded from FLORIS input files. Used for processing experiment runs.

    Args:
        LF_list (list): low-fidelity py_wake models.
        u (float): wind speed in m/s.
        ti (float): turbulence intensity.
        yaw_angle (float): yaw angle.
        model_path (str, optional): Path to directory to store the saved model.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        fig_path (str): Path to directory to store the figure.
    """
    # go over all saved CNNs and create figure
    for model in model_list:

        y, prediction = CompareFlows(
            u=u, ti=ti, yaw_angle=yaw,
            pywake_model= model,
            model_path=model_path,
            model_name='Pywake'+model+'.pt',
            floris_path=floris_path,
            floris_name=floris_name,
            fig_path=fig_path,
            fig_name='Pywake'+model+'.png'
        )

    return y, prediction

def mftl_CNN(LF_list, u, ti, yaw, model_path, floris_path, floris_name, fig_path, HF_model='Fuga'):
    """
    Create figures for CNNs that has undergone transfer learning using py_wake package.
    Compare flow fields generated by those CNNs with flow fields generated from py_wake Fuga.
    Assume turbines are loaded from FLORIS input files. Used for processing experiment runs.

    Args:
        LF_list (list): low-fidelity py_wake models.
        u (float): wind speed in m/s.
        ti (float): turbulence intensity.
        yaw_angle (float): yaw angle.
        model_path (str, optional): Path to directory to store the saved model.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        fig_path (str): Path to directory to store the figure.
        HF_model (sstr, optional): high-fidelity model used in transfer learning.
            Defaults to Fuga.
    """
    # go over all saved MFTL CNNs and create figure
    for LF_model in LF_list:
        y, prediction = CompareFlows(
            u=u, ti=ti, yaw_angle=yaw,
            pywake_model=HF_model,
            model_path=model_path,
            model_name='Pywake'+LF_model+'Fuga.pt',
            floris_path=floris_path,
            floris_name=floris_name,
            fig_path=fig_path,
            fig_name='Pywake'+LF_model+'Fuga.png',
        )

    return y, prediction


if __name__ == '__main__':

    # range of values for each input parameter
    u_range=[3, 12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]

    # randomly generate data from range of values
    # import random
    # u = random.uniform(u_range[0], u_range[1])
    # ti = random.uniform(ti_range[0], ti_range[1])
    # yaw = random.uniform(yaw_range[0], yaw_range[1])

    u = 11.3
    ti = 0.1
    yaw = 18.

    # path and file name of floris input file
    floris_path='./inputs/'
    floris_name='FLORIS_input_jensen_1x3'

    # path and file name of saved CNN
    model_path='./TrainedModels/' + floris_name + '/Experiment3/LF1375HF250/'
    # path and file name of results from CNN and py_wake
    fig_path='./results/' + floris_name + '/Experiment3/LF1375HF250/'

    y, prediction = CompareFlows(u=u, ti=ti, yaw_angle=yaw,
    package="py_wake", pywake_model='BGauss',
    model_path=model_path, model_name='PywakeBGaussFuga.pt',
    # load_floris=False, x_list=[0., 630., 1260., 0., 630., 1260.], y_list=[100., 100., 100., -100., -100., -100.],
    floris_name=floris_name, legacy=False, fig_path='./results/',
    fig_name='PywakeBGauss.png', show_fig=False, save_fig=True)


    ### time Pywake vs CNN ###
    # import time

    # pywake_model = 'Fuga'

    # x = np.array([u, ti, yaw])
    # x = torch.from_numpy(x)

    # gen = Generator(nr_input_var=3, nr_filter=16)
    # gen.load_model(path=model_path+'PywakeJensenFuga.pt')
    # start = time.time()
    # ews = gen(x.float())
    # end = time.time()
    # print("CNN time using  = ", end-start)

    # start = time.time()
    # ews = PywakeWorkflow(ti=ti, ws=u, yaw_angle=yaw, load_floris=True, floris_name=floris_name, model=pywake_model, show_fig=False)
    # end = time.time()
    # print("py_wake time = ", end-start)


    # ### Benchmark
    # model_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss', 'Fuga']
    # LF_trained_CNN(model_list=model_list, u=u, ti=ti, yaw=yaw, model_path=model_path,
    #     floris_path=floris_path, floris_name=floris_name, fig_path=fig_path)
    
    ### LF trained
    # model_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss']

    # LF_trained_CNN(model_list=model_list, u=u, ti=ti, yaw=yaw, model_path=model_path,
    #     floris_path=floris_path, floris_name=floris_name, fig_path=fig_path)

    ### plot results from CNN undergone transfer learning ###
    
    # LF_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss']

    # mftl_CNN(LF_list=LF_list, u=u, ti=ti, yaw=yaw, model_path=model_path,
    #     floris_path=floris_path, floris_name=floris_name, fig_path=fig_path)
