import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

""" Create box and whisker plots for MFTL CNNs """

def FindDataByModel(LF_model, pwpe_path, pwpe_name='pwpe_dataset.csv'):
    """
    Return array of average pixel-wise percentage error (pwpe) computed from TestTrainedCNN.py
    for a specified py_wake wake deficit model from a saved dataset.

    Args:
        LF_model (str): Wake deficit model.
        pwpe_path (str): Path to the dataset storing pwpe.
        pwpe_name (str, optional): File name of dataset.

    Returns:
        col (numpy.ndarray): Array of average pwpe.
    """
    data_df = pd.read_csv(pwpe_path+pwpe_name)
    col = data_df[LF_model].values
    return col

def GetAllCols(LF_list, dir_list, LF_pwpe_path=None, include_base=True,
    base_pwpe_path='./results/FLORIS_input_jensen/LF5000/',
    path_to_dirs='./results/FLORIS_input_jensen/',
    add_fuga=True, fuga_path='./results/FLORIS_input_jensen/Size5000/'):
    """
    Return a list of arrays of average pixel-wise percentage error (pwpe) based on the list of wake deficit models.
    Pwpe arrays are loaded from a list of directories and are computed from TestTrainedCNN.py.

    Args:
        LF_list (list): Wake deficit models.
        dir_list (list): Directories storing pwpe datasets.
        LF_pwpe_path (str, optional): Path to directory storing CNN only trained with LF data.
        include_base (bool, optional): True if include benchmark LF CNN trained with 4000 LF samples.
        base_pwpe_path (str, optional): Path to dataset storing pwpe of benchmark LF CNNs.
        path_to_dirs (str, optional): Path to list of directories.
        add_fuga (bool, optional): Add pwpe of CNN trained using fuga data if True. Defaults to True.
        fuga_path (str, optional): Path to file that contains fuga CNN pwpe.

    Returns:
        plot_col (list): Arrays of pwpe loaded from the list of directory, one array for each wake deficit model.
    """
    plot_col = []
    for LF_model in LF_list:
        all_cols = []
        if include_base:
            all_cols.append(FindDataByModel(LF_model=LF_model, pwpe_path=base_pwpe_path, pwpe_name='LF_pwpe_dataset.csv'))
        if LF_pwpe_path is not None:
            all_cols.append(FindDataByModel(LF_model=LF_model, pwpe_path=LF_pwpe_path, pwpe_name='LF_pwpe_dataset.csv'))
        # load array of pwpe from directories and add to list
        for dir in dir_list:
            path_to_pwpe = path_to_dirs + dir + '/'
            all_cols.append(FindDataByModel(LF_model=LF_model, pwpe_path=path_to_pwpe))

        if add_fuga:
            fuga_col = FindDataByModel(LF_model='Fuga', pwpe_path=fuga_path)
            all_cols.append(fuga_col)

        # add to final list for each wake deficit model
        plot_col.append(all_cols)

    return plot_col

def CreateBoxPlots(plot_cols, xtick_list, x_label, y_label, LF_list, results_path, fig_name, ymax=10):
    """
    Create box plots for a range of dataset sizes.

    Args:
        plot_cols (list): Arrays of pwpe.
        xtick_list (list): xticklabels in matplotlib.
        x_label (str): x-axis label.
        y_label (str): y-axis label.
        results_path (str): path to store the box plots.
        fig_name (str): file name of figure of box plots.
    """
    # nine wake deficit models
    _, ax = plt.subplots(3, 3, figsize=(16,20), constrained_layout=True, sharey=True)
    idx = 0
    for i in range(3):
        for j in range(3):
            ax[i,j].boxplot(plot_cols[idx])
            ax[i,j].set_xticklabels(xtick_list, fontsize=15, rotation=90)
            if j%3==0:
                ax[i,j].set_ylabel(y_label, fontsize=20)
            ax[i,j].set_xlabel(x_label, fontsize=20)
            ax[i,j].set_title(LF_list[idx], fontsize=25)
            ax[i,j].set_ylim(ymin=0, ymax=ymax)
            ax[i,j].set_yticks(np.arange(0, ymax, 2), fontsize=15)
            idx+=1

    plt.savefig(results_path+fig_name)

    return None


if __name__ == '__main__':

    # floris input file
    floris_name = 'FLORIS_input_jensen_1x3'
    
    # all LF wake deficit models
    LF_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss']
    # path to store the box plots
    results_path = './results/' + floris_name + '/'
    fuga_path='./results/' + floris_name + '/Size5000/'

    # experiment 1 directories
    dir_list = ['Experiment1/LF2500HF100', 'Experiment1/LF2500HF200', 'Experiment1/LF2500HF300', 'Experiment1/LF2500HF400']
    LF_pwpe_path = results_path + 'Experiment1/LF2500/'
    # arrays of pwpe of all wake deficits model for experiment 1
    plot_cols = GetAllCols(LF_list=LF_list, LF_pwpe_path=LF_pwpe_path, dir_list=dir_list,
        base_pwpe_path=results_path+'Size5000/', path_to_dirs=results_path, fuga_path=fuga_path)
    # xticklabels of box plots for experiment 1
    xtick_list = ['5000/0', '2500/0', '2500/100', '2500/200', '2500/300', '2500/400', '0/5000']
    # create box plots of all wake deficit models for experiment 1
    CreateBoxPlots(plot_cols=plot_cols, xtick_list=xtick_list,
    x_label='LF dataset size / HF dataset size',
    y_label='pixel-wise percentage error',
    LF_list=LF_list, results_path=results_path,
    fig_name='mftl_BoxPlots1.png', ymax=19)

    # experiment 2 directories
    dir_list = ['Experiment1/LF2500HF400', 'Experiment2/LF2200HF400', 'Experiment2/LF1320HF400', 'Experiment2/LF440HF400']
    # arrays of pwpe of all wake deficits model for experiment 2
    plot_cols = GetAllCols(LF_list=LF_list, dir_list=dir_list, path_to_dirs=results_path,
        base_pwpe_path=results_path+'Size5000/', fuga_path=fuga_path)
    # xticklabels of box plots for experiment 2
    xtick_list = ['5000/0', '2500/400', '2200/400', '1320/400', '440/400', '0/5000']
    # create box plots of all wake deficit models for experiment 2
    CreateBoxPlots(plot_cols=plot_cols, xtick_list=xtick_list,
    x_label='LF dataset size / HF dataset size',
    y_label='pixel-wise percentage error',
    LF_list=LF_list, results_path=results_path,
    fig_name='mftl_BoxPlots2.png', ymax=15)

    # experiment 3 directories
    dir_list = ['Experiment3/LF1375HF250', 'Experiment3/LF825HF250', 'Experiment3/LF275HF250']
    # arrays of pwpe of all wake deficits model for experiment 5
    plot_cols = GetAllCols(LF_list=LF_list, dir_list=dir_list, path_to_dirs=results_path,
        base_pwpe_path=results_path+'Size5000/', fuga_path=fuga_path)
    # xticklabels of box plots for experiment 3
    xtick_list = ['5000/0', '1375/250', '825/250', '275/250', '0/5000']
    # create box plots of all wake deficit models for experiment 3
    CreateBoxPlots(plot_cols=plot_cols, xtick_list=xtick_list,
    x_label='LF dataset size / HF dataset size',
    y_label='pixel-wise percentage error',
    LF_list=LF_list, results_path=results_path,
    fig_name='mftl_BoxPlots3.png', ymax=15)

    # experiment 4 directories
    dir_list = ['Experiment1/LF2500HF100', 'Experiment4/LF550HF100', 'Experiment4/LF330HF100', 'Experiment4/LF110HF100']
    # arrays of pwpe of all wake deficits model for experiment 5
    plot_cols = GetAllCols(LF_list=LF_list, dir_list=dir_list, path_to_dirs=results_path,
        base_pwpe_path=results_path+'Size5000/', fuga_path=fuga_path)
    # xticklabels of box plots for experiment 3
    xtick_list = ['5000/0', '2500/100', '550/100', '330/100', '110/100', '0/5000']
    # create box plots of all wake deficit models for experiment 3
    CreateBoxPlots(plot_cols=plot_cols, xtick_list=xtick_list,
    x_label='LF dataset size / HF dataset size',
    y_label='pixel-wise percentage error',
    LF_list=LF_list, results_path=results_path,
    fig_name='mftl_BoxPlots4.png', ymax=25)
