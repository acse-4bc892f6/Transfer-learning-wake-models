All directories contain datasets and box plots of average pixel-wise percentage error (pwpe) of CNNs. pwpe is computed from a testing set containing 500 entries of inflow conditions and simulated flow fields.

CNNs in `Size5000` are trained using a dataset of size 5000 generated from a single PyWake model over 100 epochs.

Directories in `Experiment` are named following the syntax `LF+LF_set_size+HF+HF_set_size`. Directories with only `LF+LF_set_size` contain models that is only trained on data generated from a single model. LF stands for low-fidelity, and HF stands for high-fidelity. HF model used in all experiments is Fuga.

`mftl_BoxPlots` are box plots showing the error of CNNs that has undergone transfer learning over different LF set size to HF set size ratios. The title of each subplot shows the LF model used to generate training data.

| Directory | LF train set size | HF train set size | Epochs |
| - | - | - | - |
| Experiment1 | 2500 | 100, 200, 300, 400 | 30 |
| Experiment2 | 2200, 1320, 440 | 400 | 30 |
| Experiment3 | 1375, 825, 275 | 250 | 30 |
| Experiment4 | 550, 330, 110 | 100 | 30 |
