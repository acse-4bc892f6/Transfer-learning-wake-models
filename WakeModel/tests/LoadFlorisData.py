# https://stackoverflow.com/questions/16981921/relative-imports-in-python-3
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from Pywake import GetWindfarmLayout, PywakeWorkflow
import yaml
import json
import numpy as np
import random

# pytest LoadFlorisData.py -W ignore::FutureWarning

def test_load_floris_layout():
    """Load turbine layout from json and yaml FLORIS file, test same layout is obtained."""

    floris_path = '../inputs/'
    floris_name = 'FLORIS_input_gauss'

    x_load_yaml, y_load_yaml = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=False)
    x_load_json, y_load_json = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=True)

    with open(floris_path+floris_name+".yaml") as stream:
        parsed_yaml = yaml.safe_load(stream)
    x_pos_yaml = parsed_yaml['farm']['layout_x']
    y_pos_yaml = parsed_yaml['farm']['layout_y']

    with open(floris_path+floris_name+".json") as legacy_dict_file:
        configuration_v2 = json.load(legacy_dict_file)
    x_pos_json = configuration_v2['farm']['properties']['layout_x']
    y_pos_json = configuration_v2['farm']['properties']['layout_y']

    assert np.allclose(x_pos_json, x_load_json)
    assert np.allclose(y_pos_json, y_load_json)
    assert np.allclose(x_pos_yaml, x_load_yaml)
    assert np.allclose(y_pos_yaml, y_load_yaml)
    


def test_PywakeWorkflow():
    """Load from json and yaml FLORIS file, test same flow field is produced."""
    
    u_range=[3, 12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]
    
    u = round(random.uniform(u_range[0], u_range[1]))
    ti = round(random.uniform(ti_range[0], ti_range[1]))
    yaw = round(random.uniform(yaw_range[0], yaw_range[1]))

    floris_path = '../inputs/'
    floris_name = 'FLORIS_input_gauss'

    ews_json = PywakeWorkflow(ti=ti, ws=u, yaw_angle=yaw, load_floris=True,
    floris_path=floris_path, floris_name=floris_name, legacy=False, show_fig=False)

    ews_yaml = PywakeWorkflow(ti=ti, ws=u, yaw_angle=yaw, load_floris=True,
    floris_path=floris_path, floris_name=floris_name, legacy=True, show_fig=False)

    assert np.allclose(ews_json, ews_yaml)


