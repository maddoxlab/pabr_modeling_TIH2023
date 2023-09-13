addpath("amtoolbox-full-1.2.0\amtoolbox-full-1.2.0");
amt_start;
load("impaired_thresholds.mat");

species = 2; % use Shera (based on OAEs instead of behavior)​
% Get parameters for hearing impairment (default 2/3rds OHC loss)
[Cohc, Cihc, OHC_Loss]=carney2015_fitaudiogram(cfs, m9, species);
save("HI_vars_m9", "Cohc", "Cihc", "OHC_Loss", "cfs");
