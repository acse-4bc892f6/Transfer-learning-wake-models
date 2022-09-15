import json
from floris.tools.floris_interface_legacy_reader import (
            _convert_v24_dictionary_to_v3
        )
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os

import py_wake
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformSite
from py_wake import HorizontalGrid
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import NOJDeficit, TurboNOJDeficit,\
    FugaDeficit, NiayifarGaussianDeficit, ZongGaussianDeficit,\
    CarbajofuertesGaussianDeficit, TurboGaussianDeficit, GCLDeficit,\
    BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit
from py_wake.turbulence_models import CrespoHernandez,\
    STF2017TurbulenceModel, GCLTurbulence
from py_wake.superposition_models import SquaredSum, LinearSum, MaxSum
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.examples.data.iea37 import IEA37_WindTurbines


def GetTurbineFromFloris(floris_path="inputs/", floris_name="FLORIS_input_jensen", legacy=False):
    """
    Create a py_wake WindTurbine object from FLORIS file.
    Assume FLORIS file contains power thrust table, rotor diameter, and hub height.

    Args:
        floris_path (str): Path to directory storing FLORIS file.
            Defaults to "inputs/".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_jensen".
        legacy (bool, optional): True if FLORIS file is json file.
            False if FLORIS file is yaml file. Defaults to False.
    Returns:
        wt (WindTurbine): py_wake WindTurbine object with physical properties from FLORIS file
    """
    # load from json file
    if legacy:
        with open(floris_path+floris_name+".json") as legacy_dict_file:
            configuration_v2 = json.load(legacy_dict_file)
        _, turb_dict = _convert_v24_dictionary_to_v3(configuration_v2)
        power = turb_dict["power_thrust_table"]["power"]
        ct = turb_dict["power_thrust_table"]["thrust"]
        u = turb_dict["power_thrust_table"]["wind_speed"]
        D = turb_dict["rotor_diameter"]
        hh = turb_dict["hub_height"]

    # load from yaml file
    else:
        with open(floris_path+floris_name+".yaml") as stream:
            parsed_yaml = yaml.safe_load(stream)
        power = parsed_yaml['farm']['turbine_type'][0]['power_thrust_table']['power']
        ct = parsed_yaml['farm']['turbine_type'][0]['power_thrust_table']['thrust']
        u = parsed_yaml['farm']['turbine_type'][0]['power_thrust_table']['wind_speed']
        D = parsed_yaml['farm']['turbine_type'][0]['rotor_diameter']
        hh = parsed_yaml['farm']['turbine_type'][0]['hub_height']

    # create WindTurbine object with extracted physical properties
    wt = WindTurbine(name='MyWT',
                    diameter=D,
                    hub_height=hh,
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))

    return wt

def GetWindfarmLayout(floris_path="inputs/", floris_name="FLORIS_input_jensen", legacy=False):
    """
    Load x and y coordinates of wind turbines in wind farm from FLORIS file.
    Assume layout is specified in FLORIS file.

    Args:
        floris_path (str): Path to directory storing FLORIS file.
            Defaults to "inputs/".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_jensen".
        legacy (bool, optional): True if FLORIS file is json file.
            False if FLORIS file is yaml file. Defaults to False.

    Returns:
        x_pos, y_pos (list, list): x and y coordinates of wind turbines respectively.
    """
    # load from json file
    if legacy:
        with open(floris_path+floris_name+".json") as legacy_dict_file:
            configuration_v2 = json.load(legacy_dict_file)
        x_pos = configuration_v2['farm']['properties']['layout_x']
        y_pos = configuration_v2['farm']['properties']['layout_y']

    # load from yaml file
    else:
        with open(floris_path+floris_name+".yaml") as stream:
            parsed_yaml = yaml.safe_load(stream)
        x_pos = parsed_yaml['farm']['layout_x']
        y_pos = parsed_yaml['farm']['layout_y']
    
    return x_pos, y_pos

def GetHorizontalGrid(wt, domain_x=[0,3000], domain_y=[-200,200], dim=163):
    """
    Return the domain as HorizontalGrid object. 
    
    Default to :math:`x \in [0,3000]` and :math:`y \in [-200,200]`.

    Args:
        wt (WindTurbine): wind turbine object.
        domain_x (list, optional): domain of x. Default to [0, 3000].
        domain_y (list, optional): domain of y. Default to [-200, 200].
        dim (int, optional): dimension of horizontal grid. Default to 163.

    Returns:
        grid (HorizontalGrid): domain of x=[0, 3000], y=[-200,200]
            at dimension 163x163.
    """
    grid = HorizontalGrid(
        x=np.linspace(domain_x[0], domain_x[1], dim),
        y=np.linspace(domain_y[0], domain_y[1], dim),
        h=wt.hub_height()*np.ones(dim))
    return grid

def GetUniformSite(ti, ws, p_wd=[1]):
    """
    Create a py_wake UniformSite object, a wind turbine site that has
    constant, uniform speed.

    Args:
        ti (float): turbulence intensity.
        ws (float): wind speed.
        p_wd (list): wind speed probability distribution.
            Default is [1].
    """
    site = UniformSite(p_wd=p_wd, ti=ti, ws=ws)
    return site

def GetWindFromFlowMap(flow_map):
    """
    Extract effective wind speed from flow_map.

    Args:
        flow_map (FlowMap): effective local wind speed and effective local turbulence intensity
            at all grid points, generated from simulation.

    Returns:
        ews (numpy.ndarray): effective local wind speed in m/s at all grid points.

    """
    # calculate effective wind speed based on the source code of flow_map.plot_wake_map function
    # https://topfarm.pages.windenergy.dtu.dk/PyWake/_modules/py_wake/flow_map.html#FlowMap.plot_wake_map
    u = (flow_map.WS_eff * flow_map.P / flow_map.P.sum(['wd', 'ws'])).sum(['wd', 'ws'])
    ews = u.isel(h=0).data
    return ews

def PlotFlowMapContour(flow_map):
    """
    Create contour plot of effective wind speed from FlowMap object. Uses cmap=coolwarm.

    Args:
        flow_map (FlowMap): effective local wind speed and effective local turbulence intensity
            at all grid points, generated from simulation.
    """
    plt.figure(figsize=(8,3))
    flow_map.plot_wake_map(cmap='coolwarm',
                        plot_colorbar=True,
                        plot_windturbines=False,
                        ax=None)

    plt.show()

    return None

def TestWakeDeficit(
    ti, ws, yaw_angle, wd=270, floris_path="./inputs/",
    floris_name="FLORIS_input_jensen", legacy=False):
    """
    Generate flow field plots for different wake deficit models. 
    Squared sum superposition model, Jiminez wake deflection model,
    and Crespo Hernandez turbulence model are fixed.

    For more details on wake deficit models, please see:
    https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html#Wake-deficit-models

    Args:
        ti (float): turbulence intensity.
        ws (float): wind speed.
        yaw_angle (float): yaw angle
        wd (float, optional): wind direction. Default to 270.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs/".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
    """
    wt = GetTurbineFromFloris(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    x_pos, y_pos = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    uniform_site = GetUniformSite(ti=ti, ws=ws)
    lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

    WakeDeficitList = [NOJDeficit(), TurboNOJDeficit(), FugaDeficit(LUT_path=lut_path),
    BastankhahGaussianDeficit(), IEA37SimpleBastankhahGaussianDeficit(), NiayifarGaussianDeficit(),
    ZongGaussianDeficit(), CarbajofuertesGaussianDeficit(), TurboGaussianDeficit(), GCLDeficit()]

    TitleList = ["NOJDeficit", "TurboJensen", "FugaDeficit", "BastankhahGaussianDeficit",
        "IEA37SimpleBastankhahGaussianDeficit", "NiayifarGaussianDeficit", "ZongGaussianDeficit",
        "CarbajofuertesGaussianDeficit", "TurboGaussianDeficit", "GCLDeficit"]

    FlowMapList=[]

    for model in WakeDeficitList:

        WakeDeficitModel = All2AllIterative(
            site=uniform_site,
            windTurbines=wt,
            wake_deficitModel=model,
            turbulenceModel=CrespoHernandez(),
            deflectionModel=JimenezWakeDeflection(),
            superpositionModel=SquaredSum())
            
        sim_res = WakeDeficitModel(x=x_pos, y=y_pos, h=wt.hub_height(), yaw=yaw_angle)
        grid = GetHorizontalGrid(wt=wt)
        flow_map = sim_res.flow_map(grid=grid, ws=ws, wd=wd)
        FlowMapList.append(flow_map)

    fig, ax = plt.subplots(5, 2, figsize=(14, 8))
    fig.tight_layout()
    idx = 0
    for i in range(5):
        for j in range(2):
            flow_map = FlowMapList[idx]
            ews = GetWindFromFlowMap(flow_map=flow_map)
            ax[i,j].imshow(ews, cmap='coolwarm', extent=[0, 3000, -200, 200])
            ax[i,j].set_title(TitleList[idx]+'(u=%.2f, ti=%.2f, yaw=%.2f)' % (ws, ti, yaw_angle))
            idx += 1
    plt.show()
    
    return None

def TestTurbulence(
    ti, ws, yaw_angle, wd=270, floris_path="inputs/",
    floris_name="FLORIS_input_jensen", legacy=False):
    """
    Generate flow field plots for different turbulence models. 
    Squared sum superposition model, Jiminez wake deflection model,
    and Fuga wake deficit model are fixed.

    For more details on turbulence models, please see:
    https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html#Turbulence-models

    Args:
        ti (float): turbulence intensity.
        ws (float): wind speed.
        yaw_angle (float): yaw angle
        wd (float, optional): wind direction. Default to 270.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
    """
    wt = GetTurbineFromFloris(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    x_pos, y_pos = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    uniform_site = GetUniformSite(ti=ti, ws=ws)
    lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

    TurbulenceList = [CrespoHernandez(), STF2017TurbulenceModel(), GCLTurbulence()]

    TitleList = ["CrespoHernandez", "STF2017TurbulenceModel", "GCLTurbulence"]

    FlowMapList=[]

    for model in TurbulenceList:

        WakeDeficitModel = All2AllIterative(
            site=uniform_site,windTurbines=wt,
            wake_deficitModel=FugaDeficit(LUT_path=lut_path),
            turbulenceModel=model,
            deflectionModel=JimenezWakeDeflection(),
            superpositionModel=SquaredSum())
            
        sim_res = WakeDeficitModel(x=x_pos, y=y_pos, h=wt.hub_height(), yaw=yaw_angle)
        grid = GetHorizontalGrid(wt=wt)
        flow_map = sim_res.flow_map(grid=grid, ws=ws, wd=wd)
        FlowMapList.append(flow_map)

    fig, ax = plt.subplots(3, 1, figsize=(8, 5))
    fig.tight_layout()
    idx = 0
    for i in range(3):
        flow_map = FlowMapList[idx]
        ews = GetWindFromFlowMap(flow_map=flow_map)
        ax[i].imshow(ews, cmap='coolwarm', extent=[0, 3000, -200, 200])
        ax[i].set_title(TitleList[idx]+'(u=%.2f, ti=%.2f, yaw=%.2f)' % (ws, ti, yaw_angle))
        idx += 1
    plt.show()
    
    return None

def TestSuperposition(
    ti, ws, yaw_angle, wd=270, floris_path="inputs/",
    floris_name="FLORIS_input_jensen", legacy=False):
    """
    Generate flow field plots for different superposition models. 
    STF2017 turbulence model, Jiminez wake deflection model,
    and Fuga wake deficit model are fixed.

    For more details on superposition models, please see:
    https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/EngineeringWindFarmModels.html#Superposition-models

    Args:
        ti (float): turbulence intensity.
        ws (float): wind speed.
        yaw_angle (float): yaw angle
        wd (float, optional): wind direction. Default to 270.
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
    """

    wt = GetTurbineFromFloris(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    x_pos, y_pos = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
    uniform_site = GetUniformSite(ti=ti, ws=ws)
    lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

    SuperpositionList = [SquaredSum(), LinearSum(), MaxSum()]

    TitleList = ["SquaredSum", "LinearSum", "MaxSum"]

    FlowMapList=[]

    for model in SuperpositionList:

        WakeDeficitModel = All2AllIterative(
            site=uniform_site,windTurbines=wt,
            wake_deficitModel=FugaDeficit(LUT_path=lut_path),
            turbulenceModel=STF2017TurbulenceModel(),
            deflectionModel=JimenezWakeDeflection(),
            superpositionModel=model)
            
        sim_res = WakeDeficitModel(x=x_pos, y=y_pos, h=wt.hub_height(), yaw=yaw_angle)
        grid = GetHorizontalGrid(wt=wt)
        flow_map = sim_res.flow_map(grid=grid, ws=ws, wd=wd)
        FlowMapList.append(flow_map)

    fig, ax = plt.subplots(3, 1, figsize=(8, 5))
    fig.tight_layout()
    idx = 0
    for i in range(3):
        flow_map = FlowMapList[idx]
        ews = GetWindFromFlowMap(flow_map=flow_map)
        ax[i].imshow(ews, cmap='coolwarm', extent=[0, 3000, -200, 200])
        ax[i].set_title(TitleList[idx]+'(u=%.2f, ti=%.2f, yaw=%.2f)' % (ws, ti, yaw_angle))
        idx += 1
    plt.show()
    
    return None


def PywakeWorkflow(
    ti, ws, yaw_angle, wd=270, load_floris=True, x_list=[0.], y_list=[0.],
    floris_path="inputs/", floris_name="FLORIS_input_jensen", legacy=False, 
    model="Jensen", show_fig=True):
    """
    Full workflow from start to finish: load data from FLORIS file to create wind turbine(s),
    create a site for the wind turbine(s), simulate airflow and create contour plot.

    Args:
        ti (float): turbulence intensity.
        ws (float): wind speed.
        yaw_angle (float): yaw angle
        wd (float): wind direction. Default to 270.
        load_floris (bool, optional): Load physical properties of wind turbine from FLORIS file if True.
            Defaults to True.
        x_list (list, optional): list of x coordinates of non-FLORIS wind turbines. Defaults to [0.].
        y_list (list, optional): list of y coordinates of non-FLORIS wind turbines. Defaults to [0.].
        floris_path (str, optional): Path to directory storing FLORIS file.
            Default to "./inputs".
        floris_name (str, optional): name of the FLORIS file.
            Defaults to "FLORIS_input_gauss".
        legacy (bool, optional): True if FLORIS file is json.
            False if FLORIS file is yaml. Defaults to False.
        stf (bool, optional): switch turbulence model to STF2017. Default to False.
        model (str, optional): wake deficit model. Default to Jensen.
        show_fig (bool, optional): show contour plot of effective wind speed. Default to True.

    Returns:
        ews (numpy.ndarray): effective local wind speed in m/s at all grid points.
    """
    
    # load data from FLORIS file to create wind turbine
    if load_floris:
        wt = GetTurbineFromFloris(floris_path=floris_path, floris_name=floris_name, legacy=legacy)
        x_pos, y_pos = GetWindfarmLayout(floris_path=floris_path, floris_name=floris_name, legacy=legacy)

    # otherwise create pre-defined wind turbine from py_wake
    else:
        assert len(x_list) == len(y_list), "len(x_list) must be equal to len(y_list)"
        wt = IEA37_WindTurbines() # chose this turbine type because of uniform site
        x_pos, y_pos = x_list, y_list # pass coordinates of wind turbines as argument

    # create uniform site from turbulence intensity and wind speed
    uniform_site = GetUniformSite(ti=ti, ws=ws)
    # path to look up table for Fuga
    lut_path = os.path.dirname(py_wake.__file__)+'/tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/'

    # wake deficit models
    if model == "Jensen":
        wdm = NOJDeficit()
    elif model == "TurboJensen":
        wdm = TurboNOJDeficit()
    elif model == "BGauss":
        wdm = BastankhahGaussianDeficit()
    elif model == "SBGauss":
        wdm = IEA37SimpleBastankhahGaussianDeficit()
    elif model == "NGauss":
        wdm = NiayifarGaussianDeficit()
    elif model == "ZGauss":
        wdm = ZongGaussianDeficit()
    elif model == "CGauss":
        wdm = CarbajofuertesGaussianDeficit()
    elif model == "TurboGauss":
        wdm = TurboGaussianDeficit()
    elif model == "Larsen":
        wdm = GCLDeficit()
    elif model == "Fuga":
        wdm = FugaDeficit(LUT_path=lut_path)

    else:
        raise KeyError("Not Implemented")

    # create wake deficit model with fixed turbulence model, deflection model and superposition model
    WakeDeficitModel = All2AllIterative(
        site=uniform_site,
        windTurbines=wt,
        wake_deficitModel=wdm,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
        superpositionModel=SquaredSum())
        
    # run model to create simulation
    sim_res = WakeDeficitModel(x=x_pos, y=y_pos, h=wt.hub_height(), yaw=yaw_angle)

    grid = GetHorizontalGrid(wt=wt) # create horizontal grid for flow map
    flow_map = sim_res.flow_map(grid=grid, ws=ws, wd=wd) # create flow map
    ews = GetWindFromFlowMap(flow_map=flow_map) # extract effective wind speed

    if show_fig:
        # plot figure with cmap=coolwarm to be consistent with FLORIS
        PlotFlowMapContour(flow_map=flow_map)
        ### use py_wake in-built plot function ###
        # flow_map.plot_wake_map()
        # plt.show()
    
    return ews
