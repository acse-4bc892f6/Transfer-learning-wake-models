`FLORIS_input_gauss.json` is available from https://github.com/soanagno/wakenet/tree/master/Code/CNNWake.

`FLORIS_input_gauss.yaml` is updated from `FLORIS_input_gauss.json` using in-built legacy reader in FLORIS.

`FLORIS_input_gauss_1x3.yaml` uses a 1x3 turbine layout instead of single turbine. The layout follows that in `jensen.yaml` from FLORIS GitHub page. The turbines are identical to `FLORIS_input_gauss.yaml`.

`FLORIS_input_jensen.yaml` contains identical turbine as the aforementioned files, but Jensen model is used instead. Physical constants are taken from `jensen.yaml` from FLORIS GitHub page.

`FLORIS_input_jensen_1x3.yaml` uses the 1x3 turbine layout from `jensen.yaml`. The turbines are identical to the aforementioned files.

`FLORIS_input_cc.yaml` contains identical turbine as the aforementioned files, but Curl model is used instead. Physical constants are taken from `cc.yaml` from FLORIS GitHub page.

Link to example inputs on FLORIS GitHub page: https://github.com/NREL/floris/tree/main/examples/inputs
