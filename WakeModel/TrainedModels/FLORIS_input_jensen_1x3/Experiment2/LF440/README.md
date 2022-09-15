All directories contain trained CNNs saved in .pt files. 

`MFTL` directories contain CNNs that have undergone transfer learning. Files are named following the syntax `Pywake + LF_model + HF_model + .pt`.

`Train` directories contain CNNs trained on data generated from a single model. Files are named following the syntax `Pywake + model + .pt`.

| Directory | Low-fidelity train set size | High-fidelity train set size | Epochs |
| - | - | - | - |
| `MFTL1` | 4000 | 100 | 100 |
| `MFTL2` | 2500 | 100 | 30 |
| `MFTL3` | 2500 | 200 | 30 |
| `MFTL4` | 2500 | 300 | 30 |
| `MFTL5` | 2500 | 400 | 30 |
| `MFTL6` | 2200 | 400 | 30 |
| `Train1` | 4000 | 0 | 100 |
| `Train2` | 2500 | 0 | 30 |
| `Train3` | 2200 | 0 | 30 |