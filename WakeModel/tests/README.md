This directory contains unit tests.

`LoadDataset` ensures the pixel-wise percentage error is extracted from the correct model and correct dataset. `csv_directory1` and `csv_directory2` contains the datasets for this unit test.

`LoadFlorisData.py` ensures turbine information is loaded from `yaml` and `json` FLORIS input files correctly.

`ModelParams.py` ensures the model parameters in the specified number layers of CNN have `requires_grad=True`.