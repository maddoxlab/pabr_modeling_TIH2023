#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:57:10 2021

@author: tom
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from expyfun.io import read_hdf5, write_hdf5
from mne.filter import resample
from scipy.fftpack import fft, ifft
model_path = ('/Data/Modeling/Verhulst/CoNNear_IHC-ANF-master/')
sys.path.append(model_path)
from extra_functions import load_connear_model


def checkpath(s):
    if not os.path.isdir(s):
        os.makedirs(s)


# %%  Define model specific variables
fs_connear = int(25e3)
CF_connear = np.loadtxt('%sconnear/cf.txt' % model_path)*1e3  # load CFs
# scaling values for the CoNNear models
cochlea_scaling = 1e6
ihc_scaling = 1e1
an_scaling = 1e-2
# CoNNear model directory
modeldir = '%sconnear/' % model_path
# reference model parameters
magic_constant = 0.118  # constant used for the estimation of the IHC output

# Define the CoNNear cochlea model hyperparameters
context_left_cochlea = 256
context_right_cochlea = 256
Nenc_cochlea = 4  # number of layers in the encoder - check for the input size
# Load the cochlea model - keep the uncropped output
cochlea = load_connear_model(modeldir, json_name="/cochlea.json",
                             weights_name="/cochlea.h5", name="cochlea_model",
                             crop=0)

# Define the IHC model hyperparameters
context_left_ihc = 256
context_right_ihc = 256
Nenc_ihc = 3  # number of layers in the encoder - check for the input size
# Load the 201-channel IHC model to simulate for all CFs
N_cf = 201
ihc = load_connear_model(modeldir, json_name="/ihc.json",
                         weights_name="/ihc.h5", name="ihc_model")

# Define the ANF model hyperparameters
context_left_an = 7936  # longer left-sided context for the ANF models
context_right_an = 256
Nenc_an = 14  # number of layers in the encoder - check for the input size
anfh = load_connear_model(modeldir, json_name="/anfh.json",
                          weights_name="/anfh.h5", name="anfh_model")


# %% ############ CoNNear cochlea ############
def comp_cochlea(stim):
    unpad = False
    stim = np.concatenate((np.zeros(context_left_cochlea), stim,
                           np.zeros(context_right_cochlea)))
    stim = np.expand_dims(stim, axis=[0, 2])  # make the stimulus 3D
    # check the time dimension size
    if stim.shape[1] % 2**Nenc_cochlea:  # check input is multiple of 16
        Npad = int(np.ceil(stim.shape[1] /
                           (2**Nenc_cochlea)))*(2**Nenc_cochlea)-stim.shape[1]
        stim = np.pad(stim, ((0, 0), (0, Npad), (0, 0)))  # zero-pad
        unpad = True
    # simulate the cochlear output
    connear_cochlea = cochlea.predict(stim, verbose=0)
    if unpad:
        connear_cochlea = connear_cochlea[:, :-Npad]
    return connear_cochlea


# %% ############ CoNNear IHC ################
def comp_ihc(connear_cochlea):
    unpad = False
    # check the time dimension size
    if connear_cochlea.shape[1] % 2**Nenc_ihc:  # check input is multiple of 8
        Npad = (int(np.ceil(connear_cochlea.shape[1] / (2**Nenc_ihc))) *
                (2**Nenc_ihc)-connear_cochlea.shape[1])
        connear_cochlea = np.pad(connear_cochlea, ((0, 0), (0, Npad), (0, 0)))
        unpad = True
    # simulate
    ihc_connear = ihc.predict(connear_cochlea)
    if unpad:
        ihc_connear = ihc_connear[:, :-Npad]
    ihc_connear = ihc_connear / ihc_scaling  # scaling for the IHC output
    return ihc_connear


# %% ############ CoNNear AN #################
def comp_an(ihc_connear):
    unpad = False
    ihc_connear_shape_in = np.array(ihc_connear.shape)
    left_pad = ihc_connear_shape_in.copy()
    left_pad[1] = context_left_an
    right_pad = ihc_connear_shape_in.copy()
    right_pad[1] = context_right_an
    ihc_connear = np.concatenate((np.zeros(left_pad), ihc_connear,
                                  np.zeros(right_pad)), axis=1)
    # check the time dimension size
    if ihc_connear.shape[1] % 2**Nenc_an:  # check input is multiple of 16384
        Npad = int(np.ceil(ihc_connear.shape[1] /
                           (2**Nenc_an)))*(2**Nenc_an)-ihc_connear.shape[1]
        ihc_connear = np.pad(ihc_connear, ((0, 0), (0, Npad), (0, 0)))  # 0-pad
        unpad = True
    # simulate
    anfh_connear = anfh.predict(ihc_connear, verbose=0)
    if unpad:
        anfh_connear = anfh_connear[:, :-Npad]
    anfh_connear = anfh_connear / an_scaling
    return anfh_connear


# %% Main function
def doit(fi):
    stim = db_conv * stim_all_freqs[fi]
    bm = comp_cochlea(stim)
    ihc = comp_ihc(bm)
    an = comp_an(ihc)
    return ihc, an


# %% Options
savefigs = True
saveresults = True
stim_length = 1  # how many seconds to run at once
plt.ioff()
ear = 0  # which ear's stimuli to load

stim_pres_db_all = np.arange(30, 91, 5)  # presentation levels
stim_pres_db_all = np.concatenate(([-999], stim_pres_db_all))
stim_gen_rms = 0.01
sine_rms_at_0db = 20e-6

t0 = -50e-3
t1 = 50e-3
t0_samp = int(t0 * fs_connear)
t1_samp = int(t1 * fs_connear)
t_plot = np.arange(t0_samp, t1_samp) * 1e3 / fs_connear
fticks = (2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1)).astype(int)

stim_rates = np.array([20, 40, 80, 120, 160, 200], dtype=str)
stim_file_names = ['pips_%s' % rate for rate in stim_rates]

stim_path = 'Stimuli/'
stim_list = []  # intialize a list to hold all stimuli (all rates)
print('Loading stim', end='')
for stim_rate in stim_rates:
    # load and resample all stimuli here to save time
    s = read_hdf5('%spips_%s' % (stim_path, stim_rate))
    if not np.isclose(s['fs'], fs_connear):
        s['x'] = resample(s['x'], fs_connear, s['fs'], npad='auto', n_jobs=-1,
                          verbose=False)
    stim_list += [s]
    print('.', end='')
print('')
f_band = stim_list[0]['f_band']
n_freq = len(f_band)
n_toks = 60  # number of second to run

pad_time = 0.05
pad_samps = int(fs_connear * pad_time)
for ri, stim_rate in enumerate(stim_rates):
    for stim_pres_db in stim_pres_db_all:
        print('\033[1;32m\x1B[4mRunning rate %s at %i dB\033[1;32m\x1B[0m' %
              (stim_rate, stim_pres_db))
        # scalar to put in units of pascals
        db_conv = ((sine_rms_at_0db / stim_gen_rms) *
                   10 ** (stim_pres_db / 20.))
        if stim_pres_db == -999:  # actually want to input zeros
            db_conv = 0
        h_ihc = np.zeros([n_freq, n_freq+1, len(CF_connear), len(t_plot)])
        h_an = np.zeros([n_freq, n_freq+1, len(CF_connear), len(t_plot)])
        for tok in range(n_toks):  # loop through stim in 1s chunks
            prend = ['\r', '\n'][tok == n_toks-1]
            print('\r\033[1;34m   Running tok %02i/%02i\033[1;34m\x1B[0m' %
                  (tok+1, n_toks), end=prend)
            s = stim_list[ri]
            fs_in = int(s['fs'])
            stim_all_freqs = s['x'][tok, ear, np.arange(5, dtype=int)]
            stim_pulse_in = np.zeros(stim_all_freqs.shape)
            stim_all_freqs = np.vstack((stim_all_freqs,
                                        stim_all_freqs.sum(0)))

            # pad the stimuli
            pad_before = s['x'][tok-1, ear,
                                np.arange(5, dtype=int)][..., -pad_samps:]
            pad_before = np.vstack((pad_before, pad_before.sum(0)))

            pad_after = s['x'][(tok+1) % 60, ear,
                               np.arange(5, dtype=int)][..., -pad_samps:]
            pad_after = np.vstack((pad_after, pad_after.sum(0)))

            stim_all_freqs = np.concatenate((pad_before, stim_all_freqs,
                                             pad_after), axis=-1)

            stim_pulse_in = s['x_pulse'][tok, ear, np.arange(5, dtype=int)]

            for fi in range(n_freq + 1):  # run the model
                if fi == 0:  # initialize vars to hold outputs
                    tmp_ihc, tmp_an = doit(fi)
                    tmp_ihc = tmp_ihc[0].T
                    rates_ihc = np.zeros(np.r_[n_freq+1, tmp_ihc.shape[-2:]])
                    rates_ihc[fi] = tmp_ihc.copy()
                    tmp_an = tmp_an[0].T
                    rates_an = np.zeros(np.r_[n_freq+1, tmp_an.shape[-2:]])
                    rates_an[fi] = tmp_an.copy()
                    del tmp_ihc, tmp_an
                else:
                    tmp = doit(fi)
                    rates_ihc[fi] = tmp[0][0].T
                    rates_an[fi] = tmp[1][0].T
                    del tmp

            #  downsample pulse train
            stim_pulse_inds = [(np.where(sp)[0] * float(fs_connear) /
                                fs_in) for sp in stim_pulse_in]

            stim_pulse = np.zeros((n_freq, fs_connear))
            for fi in range(n_freq):
                stim_pulse[fi,
                           stim_pulse_inds[fi].astype(int)] = 1

            # pad the regressor
            stim_pulse = np.pad(stim_pulse,
                                [[0, 0], [pad_samps, pad_samps]])

            stim_fft = fft(stim_pulse) / stim_pulse.sum(-1, keepdims=True)

            # calculate IHC response
            rates_fft = fft(rates_ihc)
            rates_fft[..., 0] = 0  # subtract mean

            h_tmp = np.array([np.real(ifft(rates_fft * np.conj(x_fft)))
                              for x_fft in stim_fft])  # cross-correlate
            h_tmp = np.concatenate((h_tmp[..., t0_samp:],
                                    h_tmp[..., :t1_samp]), -1)
            h_ihc += h_tmp
            # del h_tmp, rates_fft

            # calculate AN response
            rates_fft = fft(rates_an)
            rates_fft[..., 0] = 0  # subtract mean

            h_tmp = np.array([np.real(ifft(rates_fft * np.conj(x_fft)))
                              for x_fft in stim_fft])  # cross-correlate
            h_tmp = np.concatenate((h_tmp[..., t0_samp:],
                                    h_tmp[..., :t1_samp]), -1)
            h_an += h_tmp

        h_ihc /= n_toks
        h_an /= n_toks
        if stim_pres_db == -999:  # save AN response to no input
            write_hdf5('noinput_%s' % stim_rate, {'h_an': h_an,
                                                  'h_ihc': h_ihc,
                                                  'fs': fs_connear},
                       overwrite=True)
            continue
        else:  # subtract off oscillations
            h_an_nostim = read_hdf5('noinput_%s' % stim_rate)['h_an']
            h_ihc_nostim = read_hdf5('noinput_%s' % stim_rate)['h_ihc']
            h_an -= h_an_nostim
            h_ihc -= h_ihc_nostim

        print('plotting')
        for h, label in zip([h_ihc, h_an], ['IHC', 'AN']):
            figsavepath = ('Figures/rate_%s/%sdB/%s/' % (stim_rate,
                                                         stim_pres_db, label))
            checkpath(figsavepath)
            for fi, freq in enumerate(f_band):
                plt.figure(figsize=(12, 8))
                h_ax = []
                r_multi = h[fi, -1, :, :]
                r_single = h[fi, fi, :, :]
                r_diff = r_multi - r_single
                clim = np.abs(r_single).max() * np.array([-1, 1])
                params = dict(vmin=clim[0], vmax=clim[1], cmap='seismic',
                              shading='auto')
                for plot_num, r_plot in enumerate((r_single, r_multi,
                                                   r_diff)):
                    plt.subplot(1, 3, plot_num + 1)
                    plt.pcolormesh(t_plot, CF_connear, r_plot, **params)
                    plt.yscale('log')
                    plt.gca().yaxis.set_ticks(fticks)
                    plt.gca().yaxis.set_ticklabels(fticks)
                    plt.ylim([CF_connear[-1], CF_connear[0]])
                    plt.xlim([-10, 45])
                    if plot_num == 0:
                        plt.ylabel('Frequency (Hz)')
                        plt.title('Single band (%i Hz pip)' % (f_band[fi]))
                    elif plot_num == 1:
                        plt.xlabel('Time (ms)')
                        plt.title('Multi-band')
                    else:
                        plt.title('Difference (multi - single)')
                plt.suptitle('%s, %i dB, %s stim/s' % (label, stim_pres_db,
                                                       stim_rate))
                plt.tight_layout()
                if savefigs:
                    plt.savefig('%s%04i_rate%s_%sdB_%s' %
                                (figsavepath, freq, stim_rate, stim_pres_db,
                                 label))
                plt.close('all')

            res_save_path = 'Results/Verhulst/'
            checkpath(res_save_path)
            if saveresults:
                write_hdf5('%srate%s_%sdB_%s' %
                           (res_save_path, stim_rate, stim_pres_db, label),
                           {'h': h, 'fs': fs_connear}, overwrite=True)
