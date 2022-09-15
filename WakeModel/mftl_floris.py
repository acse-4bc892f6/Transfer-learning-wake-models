from CNN_model import Generator, uniform_distribution
from SetParams import SplitList, LoadModel
from TrainFloris import train_CNN_floris
import matplotlib.pyplot as plt


def MFTL_FLORIS(HF_nr_epochs, HF_learning_rate, HF_batch_size, HF_train_size, HF_val_size,
    u_range, ti_range, yaw_range, load_model, LF_train_size,
    LF_nr_epochs=25, LF_learning_rate=0.003, LF_batch_size=50, LF_val_size=30,
    nr_workers=0, image_size=163, nr_filters=16, nr_input_vars=3,
    floris_path="./inputs/", LF_floris_name="FLORIS_input_jensen", LF_legacy=False,
    HF_floris_name="FLORIS_input_cc", HF_legacy=False, save_LF_model=False, save_HF_model=True,
    LF_model_name="FLORISJensen.pt", HF_model_name="FLORISJensenFuga.pt",
    train_fig_name="FLORISJensenFuga.png", train_fig_path="./TrainingFigures/",
    model_path="./TrainedModels/", load_model_path="./TrainedModels/", seed=1):
    """
    Implementation of multi-fidelity transfer learning: train a CNN model with low-fidelity data,
    then re-train the same CNN model with high-fidelity data. Both datasets are generated using FLORIS.
    The final trained model and the plot of validation error over epochs is saved.

    Args:
        HF_nr_epochs (int): Nr. of training epochs for high-fidelity data.
        HF_learning_rate (float): Model learning rate while training using high-fidelity data.
        HF_batch_size (int): Batch size while training using high-fidelity data.
        HF_train_size (int): Size of the generated high-fidelity training set.
        HF_val_size (int): Size of the generated high-fidelity validation set.
        u_range (list): Bound of u values [u_min, u_max] used.
        ti_range (list): Bound of TI values [TI_min, TI_max] used.
        yaw_range (list): Bound of yaw angles [yaw_min, yaw_max] used.
        load_model (bool): True if model is loaded from a .pt file.
        LF_train_size (int): Size of the generated low-fidelity training set.
        LF_nr_epochs (int, optional): Nr. of training epochs for low-fidelity data.
            Defaults to 25.
        LF_learning_rate (float, optional): Model learning rate while training using low-fidelity data.
            Defaults to 0.003.
        LF_batch_size (int, optional): Batch size while training using low-fidelity data.
            Defaults to 50.
        LF_val_size (int, optional): Size of the generated low-fidelity validation set.
            Defaults to 30.
        nr_workers (int, optional): Nr. of worker to load data. Defaults to 0.
        image_size (int, optional): Size of the data set images, needs to match the
            model output size for the current model, which is 163 x 163.
        nr_filters (int, optional): Nr. of filters used for the conv layers. Defaults to 16.
        nr_input_vars (int, optional): Nr. of input variables. Defaults to 3.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Defaults to "./inputs/".
        LF_floris_name (str, optional): Name of the low-fidelity FLORIS file without .json or .yaml.
            Defaults to "FLORIS_input_jensen".
        LF_legacy (bool, optional): True if low-fidelity FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        HF_floris_name (str, optional): Name of the high-fidelity FLORIS file without .json or .yaml.
            Defaults to "FLORIS_input_cc".
        HF_legacy (bool, optional): True if high-fidelity FLORIS file is json. False if FLORIS file is yaml.
            Defaults to False.
        save_LF_model (bool, optional): Save the low-fidelity model if True. Defaults to False.
        save_HF_model (bool, optional): Save the high-fidelity model if True. Defaults to True.
        LF_model_name (str, optional): Name of the saved low-fidelity model (needs to be .pt).
        HF_model_name (str, optional): Name of the saved high-fidelity model (needs to be .pt).
        train_fig_name (str, optional): File name of the saved plot of validation error over epochs during training.
        train_fig_path (str, optional): Path to directory to store plot of validation errors over epochs.
        model_path (str, optional): Path to directory to store the saved CNN model.
        load_model_path (str, optional): Path to directory to the saved CNN model to be loaded.
        seed (int, optional): Random number seed. Default to 1.

    Returns:
        LFgen (Generator): CNN model trained with low-fidelity data.
        HFgen (Generator): CNN model undergone multi-fidelity transfer learning.
    """

    # generate uniform distribution of wind speed, turbulence intensity and yaw angle
    
    u_list = uniform_distribution(size=LF_train_size, param_range=u_range, seed=seed)
    ti_list = uniform_distribution(size=LF_train_size, param_range=ti_range, seed=seed)
    yaw_list = uniform_distribution(size=LF_train_size, param_range=yaw_range, seed=seed)

    # instantiate Generator object for data generation
    gen = Generator(nr_input_var=nr_input_vars, nr_filter=nr_filters)
    
    plt.figure() # instantiate plot of validation error against epochs

    # train low-fidelity model if not loaded from .pt file
    if not load_model:
        LF_x_train, LF_y_train = gen.create_floris_dataset(size=LF_train_size,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
            floris_init_path=floris_path, floris_name=LF_floris_name,
            legacy=LF_legacy, seed=seed)

        LF_x_eval, LF_y_eval = gen.create_floris_dataset(size=LF_val_size,
            u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            # u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
            floris_init_path=floris_path, floris_name=LF_floris_name,
            legacy=LF_legacy, seed=seed)

        LFgen, LFloss, LFval_error, LFval_error_list = train_CNN_floris(
            nr_epochs=LF_nr_epochs, learning_rate=LF_learning_rate, batch_size=LF_batch_size, 
            train_size=LF_train_size, val_size=LF_val_size, u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
            nr_workers=nr_workers, x_train=LF_x_train, y_train=LF_y_train, x_eval=LF_x_eval, y_eval=LF_y_eval,
            floris_path=floris_path, floris_name=LF_floris_name, legacy=LF_legacy, image_size=image_size, seed=seed,
            model_path=model_path, model_name=LF_model_name, save_model=save_LF_model, plot_fig=False)

        print("Final LF training loss %.4f, final LF validation error %.2f" % (LFloss, LFval_error))

        plt.plot(range(LF_nr_epochs), LFval_error_list, label='LF model')

    else:
        LFgen = LoadModel(model_path=load_model_path, model_name=LF_model_name)

    # reduce size of uniform distribution list proportionally 
    indices = SplitList(u_list, float(HF_train_size/LF_train_size), seed)
    u_list = [u_list[idx] for idx in indices]
    ti_list = [ti_list[idx] for idx in indices]
    yaw_list = [yaw_list[idx] for idx in indices]
 
    # create high-fidelity training and validation sets from uniform distribution

    HF_x_train, HF_y_train = gen.create_floris_dataset(size=HF_train_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
        floris_init_path=floris_path, floris_name=HF_floris_name,
        legacy=LF_legacy, seed=seed)

    HF_x_eval, HF_y_eval = gen.create_floris_dataset(size=HF_val_size,
        u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        # u_list=u_list, ti_list=ti_list, yawn_list=yaw_list,
        floris_init_path=floris_path, floris_name=HF_floris_name,
        legacy=LF_legacy, seed=seed)

    # only train last layer
    print("First stage retraining:")

    temp_gen, temp_loss, temp_val_error, temp_val_error_list = train_CNN_floris(ML_model=LFgen,
        nr_epochs=HF_nr_epochs, learning_rate=HF_learning_rate, batch_size=HF_batch_size,
        train_size=HF_train_size, val_size=HF_val_size, u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        nr_workers=nr_workers, x_train=HF_x_train, y_train=HF_y_train, x_eval=HF_x_eval, y_eval=HF_y_eval,
        change_params=True, num_layers=1, floris_path=floris_path, floris_name=HF_floris_name, legacy=HF_legacy,
        model_path=model_path, model_name=HF_model_name, plot_fig=False, save_model=False,
        seed=seed, nr_filters=nr_filters, image_size=image_size)

    print("Temp final training loss %.4f, temp final validation error %.2f" % (temp_loss, temp_val_error))
    plt.plot(range(HF_nr_epochs), temp_val_error_list, label='temp')

    # retrain whole network using same training and validation sets
    print("Second stage retraining:")
    HFgen, HFloss, HFval_error, HFval_error_list = train_CNN_floris(ML_model=temp_gen,
        nr_epochs=HF_nr_epochs, learning_rate=HF_learning_rate, batch_size=HF_batch_size,
        train_size=HF_train_size, val_size=HF_val_size, u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
        nr_workers=nr_workers, x_train=HF_x_train, y_train=HF_y_train, x_eval=HF_x_eval, y_eval=HF_y_eval,
        change_params=False, floris_path=floris_path, floris_name=HF_floris_name, legacy=HF_legacy,
        model_path=model_path, model_name=HF_model_name, plot_fig=False, save_model=save_HF_model,
        seed=seed, nr_filters=nr_filters, image_size=image_size)

    print("Final HF training loss %.4f, final HF validation error %.2f" % (HFloss, HFval_error))

    # plot validation error over epochs and save plot
    plt.plot(range(HF_nr_epochs), HFval_error_list, label='cc')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('validation error')
    plt.savefig(train_fig_path+train_fig_name)

    return LFgen, HFgen

if __name__ == '__main__':

    # range of values for each input parameter
    u_range=[3,12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]
    
    # low-fidelity training hyperparameters
    LF_train_size=110
    LF_val_size=22
    LF_batch_size=11
    LF_nr_epochs=30
    LF_learning_rate=0.003 

    # high-fidelity training parameters
    HF_train_size=100
    HF_val_size=20
    HF_batch_size=10
    HF_nr_epochs=30
    HF_learning_rate=0.003

    # path to floris input file
    floris_path = './inputs/'
    # path to CNN
    load_model_path = './TrainedModels/'
    model_path = './TrainedModels/'
    # path to training figures
    train_fig_path = './TrainingFigures/'

    LFgen, HFgen = MFTL_FLORIS(HF_nr_epochs=HF_nr_epochs, HF_learning_rate=HF_learning_rate,
    HF_batch_size=HF_batch_size, HF_train_size=HF_train_size, HF_val_size=HF_val_size,
    u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
    load_model=False, load_model_path=load_model_path,
    LF_train_size=LF_train_size, LF_nr_epochs=LF_nr_epochs, LF_learning_rate=LF_learning_rate,
    LF_batch_size=LF_batch_size, LF_val_size=LF_val_size, floris_path=floris_path,
    LF_floris_name='FLORIS_input_jensen', HF_floris_name='FLORIS_input_cc',
    save_LF_model=True, LF_model_name='FLORISJensen.pt',
    save_HF_model=True, HF_model_name='FLORISJensenCurl.pt',
    model_path=model_path, train_fig_path=train_fig_path,
    train_fig_name='FlorisJensenCurl.png')
