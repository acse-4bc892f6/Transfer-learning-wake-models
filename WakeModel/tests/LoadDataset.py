# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from BoxPlots import GetAllCols
import pandas as pd
import numpy as np

def test_GetAllCols():
    """Test GetAllCols function loads the correct dataset"""

    LF_list = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss']
    dir_list = ['csv_directory1', 'csv_directory2']

    all_cols1 = GetAllCols(LF_list=LF_list, dir_list=dir_list, path_to_dirs='./', add_fuga=False, include_base=False)

    all_cols2 = []
    
    for LF_model in LF_list:
        temp = []
        for dir in dir_list:
            df = pd.read_csv('./'+dir+'/pwpe_dataset.csv')
            temp.append(df[LF_model].values)
        all_cols2.append(temp)

    assert np.allclose(np.shape(all_cols1), np.shape(all_cols2))
    assert np.allclose(all_cols1, all_cols2)
