<!-- | workflow | status |
| - | - |
| `flake8` | [![flake8](https://github.com/ese-msc-2021/irp-myw1618/actions/workflows/flake8.yml/badge.svg)](https://github.com/ese-msc-2021/irp-myw1618/actions/workflows/flake8.yml) |
| `tests` | [![final-report](https://github.com/ese-msc-2021/irp-myw1618/actions/workflows/tests.yml/badge.svg)](https://github.com/ese-msc-2021/irp-myw1618/actions/workflows/tests.yml) | -->

# How to run WakeModel

## Prerequisite
Make sure all open-source libraries needed have been installed, which are listed in `requirements.txt` in the previous directory. 

## Train CNN using data generated from single PyWake model
In `TrainPywake.py`:
- Set hyperparameters, wake deficit model, floris_input_file, CNN model file name, train figure name and their paths as appropriate
- To use in-built PyWake turbine instead of the turbine defined in FLORIS input file, uncomment the line starting with `load_floris=False`, change `x_list` and `y_list` to the configuration of the wind farm.
- Execute the command `python TrainPywake.py`.

## Train CNN using multi-fidelity transfer leraning from two PyWake models
In `mftl_pywake.py`:
- Set hyperparameters for training, low-fidelity and high-fidelity wake deficit models, CNN model file names, train figure name and their paths as appropriate
- To use in-built PyWake turbine instead of the turbine defined in FLORIS input file, uncomment the line starting with `load_floris=False`, change `x_list` and `y_list` to the configuration of the wind farm
- Execute the command `python mftl_pywake.py`.

## Train CNN using data generated from single FLORIS model
In `TrainFloris.py`:
- Set hyperparameters, floris input file, CNN model file name, train figure name and their paths as appropriate
- Execute the command `python TrainFloris.py`.

## Train CNN using multi-fidelity transfer learning from two FLORIS models
In `mftl_floris.py`:
- Set hyperparameters for training, low-fidelity and high-fidelity floris files, CNN model file names, train figure name, and the paths as appropriate.
- Execute the command `python mftl_floris.py`.

## Compare PyWake/FLORIS model and CNN results
In `VisualiseFlows.py`:
- Set the package (`FLORIS` or `py_wake`), PyWake wake deficit model, floris_input_file, name of saved CNN model, figure name, and their paths as appropriate
- If the CNN model is trained using in-built PyWake turbine instead of the turbine defined in FLORIS input file, uncomment the line starting with `load_floris=False`, change `x_list` and `y_list` to the configuration of the wind farm.
- Execute the command `python VisualiseFlows.py`.

## Miscellaneous python scripts

- `TestTrainedCNN.py` is used to create datasets and box plots of pixel-wise percentage error of CNNs that are trained using wake deficit models from PyWake.

- `BoxPlots.py` uses datasets created from `TestTrainedCNN.py` to create box plots that shows the effect of different low-fidelity to high-fidelity dataset size ratios on the pixel-wise percentage error of the CNN.

- `histogram.py` shows how `SplitList` function in `SetParams.py` splits the uniform distribution proportionally by plotting histograms of the uniformly distributed input variables before and after splitting. An example can be seen in the `histograms` directory, where 4000 uniformly distributed inflow conditions is reduced to 100.

- `CNN_summary.py` prints out the `torchsummary` of the CNN. To save output as text file, change `write_to_file` to `True` before executing `python CNN_summary.py`. The output from this script is in `cnn_summary.txt`.


# Directories
- `TrainedModels` stores saved CNNs as `.pt` files.
- `TrainingFigures` stores plots of validation error over epochs during training.
- `results` contains datasets and box plots of pixel-wise percentage error of trained CNNs.
- `inputs` contains FLORIS input files for loading turbine information and for simulating flow fields using FLORIS.
- `tests` contains unit tests.
