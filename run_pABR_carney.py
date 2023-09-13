#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:47:42 2019

@author: Thomas Stoll
"""
import numpy as np
import matplotlib.pyplot as plt
import cochlea
from expyfun.io import read_hdf5, write_hdf5
from scipy.io import loadmat
from mne.filter import resample
from joblib import Parallel, delayed
from scipy.fftpack import fft, ifft
import datetime
import os


# %% Functions and filters
def doit(fi, cfi):
    stim_up = db_conv * stim_all_up[fi]  # scale to correct level

    if HI_tag == 'NH':  # No hearing loss
        cohc = np.ones(len(cf))
        cihc = np.ones(len(cf))
    else:
        HI_vars = loadmat('HI_vars_m9.mat')
        cohc = HI_vars['Cohc'][0]
        cihc = HI_vars['Cihc'][0]
        assert np.array_equal(cf, HI_vars['cfs'][0])

    if do_IHC:  # just run the IHC portion
        anf_rates_up = cochlea.zilany2014._zilany2014.run_ihc(
            stim_up,
            cf[cfi],
            fs_up,
            species='human',
            cohc=cohc[cfi],
            cihc=cihc[cfi]
        )
    else:  # get AN firing rates
        anf_rates_up = cochlea.run_zilany2014_rate(
            stim_up,
            fs_up,
            anf_types=anf_types,
            cf=cf[cfi],
            species='human',
            cohc=cohc[cfi],
            cihc=cihc[cfi]
        ).to_numpy()

    anf_rates = resample(anf_rates_up.T, fs_out, fs_up, npad='auto',
                         n_jobs=1)  # downsample
    anf_rates = anf_rates.reshape(len(anf_types), anf_rates.shape[-1])
    return anf_rates


def checkpath(s):
    if not os.path.isdir(s):
        os.makedirs(s)


# %% Setup
HI_tag = ['NH', 'm9'][0]
print('running %s' % HI_tag)
plt.ioff()
savefigs = True
saveresults = True

stim_pres_db_all = np.arange(30, 91, 5)  # levels to test
stim_rates = np.array([20, 40, 80, 120, 160, 200], dtype=str)
cf = cochlea.zilany2014.util.calc_cfs((125, 16e3, 201), 'human')

fs_up = int(100e3)  # required high sampling rate for the model
anf_types = ['hsr']  # only run high spont. rate fibers
n_jobs = 11  # n_jobs for the main parallel/delayed loop

stim_path = 'Stimuli/'
stim_fnames = ['pips_%s' % rate for rate in stim_rates]
stim_length = 60  # 60 seconds of stimuli
ear = 0
stim_gen_rms = 0.01
sine_rms_at_0db = 20e-6

# sampling rate and time range to save
fs_out = int(10e3)
t0 = -50e-3
t1 = 50e-3
t0_samp = int(t0 * fs_out)
t1_samp = int(t1 * fs_out)

t_plot = np.arange(t0_samp, t1_samp) * 1e3 / fs_out
fticks = (2 ** np.arange(np.log2(125),
                         np.log2(16e3 + 1), 1)).astype(int)

pad_time = 0.05
pad_samps = int(fs_up*pad_time)
total_start_time = datetime.datetime.now()
for ri, stim_rate in enumerate(stim_rates):
    fn_stim = stim_fnames[ri]
    s = read_hdf5(stim_path+fn_stim)  # load in stim
    fs_in = s['fs']
    f_band = s['f_band']
    n_freq = len(f_band)

    # arrange stim for use later
    stim_all = s['x'][:stim_length, ear, np.arange(n_freq, dtype=int)]
    stim_all = np.array([np.ravel(stim_all[:, fi]) for fi in range(n_freq)])
    stim_all = np.vstack((stim_all, stim_all.sum(0, keepdims=True)))

    # store stimulus timing for analysis later
    stim_pulse_in = np.zeros(stim_all.shape)
    for fi in range(n_freq):
        stim_pulse_in[fi, :] = s['x_pulse'][:stim_length, ear,
                                            fi, :].ravel()

    if not np.isclose(fs_in, fs_up):  # upsample
        stim_all_up = resample(stim_all, fs_up, fs_in, npad='auto',
                               n_jobs='cuda')
    else:
        stim_all_up = stim_all

    # pad the stimuli
    pad_before = s['x'][-1, ear, np.arange(n_freq, dtype=int),
                        :pad_samps]
    pad_before = np.vstack((pad_before, pad_before.sum(0, keepdims=True)))
    pad_after = s['x'][stim_length, ear, np.arange(n_freq, dtype=int),
                       -pad_samps:]
    pad_after = np.vstack((pad_after, pad_after.sum(0, keepdims=True)))
    stim_all_up = np.concatenate((pad_before, stim_all_up, pad_after),
                                 axis=-1)

    for dBi, stim_pres_db in enumerate(stim_pres_db_all):
        start_time = datetime.datetime.now()
        for do_IHC in [True, False]:
            if do_IHC:
                continue
                loc = 'IHC'
            else:
                loc = 'AN'
            print("Running %s: %s stim/s, %i dB" % (loc, stim_rate,
                                                    stim_pres_db))
            figsavepath = ('Figures/%s/rate_%s/%sdB/%s/' %
                           (HI_tag, stim_rate, stim_pres_db, loc))
            checkpath(figsavepath)

            # scalar to put in units of pascals
            db_conv = ((sine_rms_at_0db / stim_gen_rms) *
                       10 ** (stim_pres_db / 20.))

            # %% Run the model
            rates = np.array(Parallel(n_jobs=n_jobs)([
                delayed(doit)(fi, cfi)
                for fi in range(n_freq + 1) for cfi in range(len(cf))]))
            rates = rates.reshape(n_freq + 1, len(cf), len(anf_types),
                                  rates.shape[-1])

            # downsample the pulses to match fs_out (and account for padding)
            stim_pulse_inds = [(np.where(sp)[0]*float(fs_out) /
                               fs_in) + int(pad_time*fs_out) for
                               sp in stim_pulse_in]
            stim_pulse = np.zeros((n_freq, rates.shape[-1]))
            for fi in range(n_freq):
                stim_pulse[fi, stim_pulse_inds[fi].astype(int)] = 1

            # cross-correlate to find average response
            stim_fft = fft(stim_pulse) / stim_pulse.sum(-1, keepdims=True)
            stim_fft[:, 0] = 0

            h = np.zeros(np.r_[n_freq, rates.shape])
            for cfi in range(len(cf)):
                rate_fft = fft(rates[:, cfi])
                for fi, x_fft in enumerate(stim_fft):
                    h[fi, :, cfi] = ifft(rate_fft*x_fft.conj()).real

            h = np.concatenate((h[..., t0_samp:], h[..., :t1_samp]), -1)
            res_save_path = 'Results/Carney/%s/' % HI_tag
            checkpath(res_save_path)
            if saveresults:
                write_hdf5('%srate%s_%sdB_%s' %
                           (res_save_path, stim_rate, stim_pres_db, loc),
                           {'h': h, 'fs': fs_out}, overwrite=True)

            td = datetime.datetime.now() - start_time
            print('\nTotal time for %s, %s stim/s, %i dB: %s' %
                  (loc, stim_rate, stim_pres_db, td))

            # %% Plot the results
            for fi, freq in enumerate(f_band):
                plt.figure(figsize=(12, 8))
                h_ax = []

                r_multi = h[fi, -1, :, 0, :]
                r_single = h[fi, fi, :, 0, :]
                r_diff = r_multi - r_single

                # set color scaling and map
                clim = np.abs(r_single).max() * np.array([-1, 1])
                params = dict(vmin=clim[0], vmax=clim[1], cmap='seismic',
                              shading='auto')
                for plot_num, r_plot in enumerate((r_single, r_multi,
                                                   r_diff)):
                    plt.subplot(1, 3, plot_num + 1)
                    plt.pcolormesh(t_plot, cf, r_plot, **params)
                    plt.yscale('log')
                    plt.ylim([cf[0], cf[-1]])
                    plt.gca().yaxis.set_ticks(fticks)
                    plt.gca().yaxis.set_ticklabels(fticks)
                    plt.xlim([-10, 45])
                    if plot_num == 0:
                        plt.ylabel('Frequency (Hz)')
                        plt.title('Single band (%i Hz pip)' % (f_band[fi]))
                    elif plot_num == 1:
                        plt.xlabel('Time (ms)')
                        plt.title('Multi-band')
                    else:
                        plt.title('Difference (multi - single)')
                plt.suptitle('%s, %i dB, %s stim/s' % (loc, stim_pres_db,
                                                       stim_rate))
                plt.tight_layout()
                if savefigs:
                    plt.savefig('%s%04i_rate%s_%sdB_%s' %
                                (figsavepath, freq, stim_rate, stim_pres_db,
                                 loc))
            plt.close('all')
            del (h, stim_pulse_inds, stim_fft)
total_time = datetime.datetime.now() - total_start_time
print('Total run time for %s seconds of stimuli: %s' %
      (stim_length, total_time))
