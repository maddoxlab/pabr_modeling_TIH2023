# pabr_modeling_TIH2023
Code used in the 2023 Trends in Hearing paper titled "Enhanced place specificity of the parallel auditory brainstem response: a modeling study."

First, run the `gen_pabr_stim.py` script to create the stimuli. This code and other scripts in this repository use [expyfun](https://github.com/LABSN/expyfun) for file i/o.

Next, run `get_impaired_thresholds.py` to interpolate the hearing impaired thresholds for all characteristic frequencies. Then run `calc_cohc_cihc.m` in MATLAB to get the hearing impaired parameters for the Zilany model. This code uses the function `carney2015_fitaudiogram` from the [Auditory Modeling Toolbox](https://amtoolbox.org/).

The model code can now be run using the scripts `run_pabr_carney.py` and `run_pabr_verhulst.py` to generate responses across all conditions for the Zilany et al. and Verhulst et al. models, respectively. These files take some time to run. On our system, the Zilany model took about 3 days while the Verhulst model took about one week. To run the Zilany model, [cochlea](https://github.com/mrkrd/cochlea) must be installed. To run the Verhulst model, the [CoNNear](https://github.com/HearingTechnology/CoNNear_IHC-ANF) version of the model must be installed. The Verhulst model code will need the path to the model folder updated at the top of the script. Figures for each condition will be generated as the code runs.

After running the models, the metric can be calculated by running the script `contours.py`, and summary figures will be generated. Again, the path to the Verhulst model directory will need to be updated.
