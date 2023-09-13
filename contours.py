#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:34:22 2023

@author: Tom Stoll
"""
import numpy as np
import matplotlib.pyplot as plt
from expyfun.io import read_hdf5
from cochlea.zilany2014.util import calc_cfs
from scipy.integrate import simpson
import matplotlib.colors
import cmocean
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


# General function to calculate PS metric
def calc_metric(r_single, r_multi, fidx):
    lags = np.argmax(r_single[..., t0:], axis=-1)+t0
    single = r_single[np.arange(len(cf)), lags]
    multi = r_multi[np.arange(len(cf)), lags]

    single /= single[fidx]
    multi /= multi[fidx]

    single_metric = simpson(single, cfs_oct)
    multi_metric = simpson(multi, cfs_oct)

    return single_metric, multi_metric


def checkpath(s):
    if not os.path.isdir(s):
        os.makedirs(s)


# path to Verhulst model directory
model_path = ('/Data/Modeling/Verhulst/CoNNear_IHC-ANF-master/')

# Make figure fonts bigger and Arial
BIGGER_SIZE = 18
font_bg = {'family': 'Arial',  # may need to add Arial font (or comment out)
           'size': BIGGER_SIZE}
plt.rc('font', **font_bg)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


stim_pres_db_all = np.arange(30, 91, 5)
stim_rates = np.array([20, 40, 80, 120, 160, 200])
freqs = np.array([500, 1000, 2000, 4000, 8000])
n_freq = len(freqs)

titles = ["Serial", "Parallel", "Ratio\n(Parallel/Serial)"]
for model in ['carney', 'verhulst']:
    if model == 'carney':
        HI_options = ['NH', 'm9']
        fpath = 'Results/Carney/%s/rate%s_%idB_AN'
        cf = calc_cfs((125, 16e3, 201), 'human')
    else:
        HI_options = ['']
        fpath = 'Results/Verhulst/%s/rate%s_%idB_AN'
        cf = np.loadtxt('%sconnear/cf.txt' % model_path)[::-1]*1e3
    cfs_oct = np.log2(cf)  # represent in octaves for later integration
    f_idx = [np.argmin(np.abs(cf-f)) for f in freqs]  # location of test freqs
    for HI_tag in HI_options:
        figsavepath = 'Figures/%s/contours' % HI_tag
        checkpath(figsavepath)
        single_metric = np.zeros([len(stim_rates), len(stim_pres_db_all),
                                  n_freq])
        multi_metric = np.zeros(single_metric.shape)
        for ri, rate in enumerate(stim_rates):
            for li, lvl in enumerate(stim_pres_db_all):
                s = read_hdf5(fpath % (HI_tag, rate, lvl))
                if model == 'carney':
                    h = s['h'][:, :, :, 0]
                else:  # verhulst model has cf in descending order, flip it
                    h = s['h'][:, :, ::-1]
                fs = s['fs']
                t = np.arange(-h.shape[-1]//2, h.shape[-1]//2) / fs
                t0 = np.argwhere(t == 0)[0][0]
                t1 = np.argwhere(t == 0.01)[0][0]
                h_sum = h.sum(2)  # sum across CF
                for foi in range(n_freq):
                    ser_r_max = np.max(h_sum[foi, foi, t0:t1])
                    par_n_std = np.std(h_sum[foi, -1, :t0])

                    # if hearing impaired, make sure a response is present
                    if (ser_r_max > par_n_std) or HI_tag in ['NH', '']:
                        r_single = h[foi, foi]
                        r_multi = h[foi, -1]

                        single_metric[ri, li, foi], multi_metric[ri, li, foi] \
                            = calc_metric(r_single, r_multi, f_idx[foi])

        # invert metric to be more similar to Q factor
        single_metric = 1/single_metric
        multi_metric = 1/multi_metric
        ratio = multi_metric/single_metric
        single_metric[np.isinf(single_metric)] = np.nan
        multi_metric[np.isinf(multi_metric)] = np.nan
        ratio[np.isinf(ratio)] = np.nan

        # consistent color scaling for NH models
        if model == 'carney' and HI_tag == 'NH':
            vmax_nh = np.max(np.r_[single_metric, multi_metric])*1.01
        if HI_tag == 'm9':
            vmax = 3
        else:
            vmax = vmax_nh
        fig, axs = plt.subplots(n_freq, 3, sharex=False, sharey=False,
                                figsize=(8, 12), gridspec_kw={'bottom': 0.05,
                                                              'top': 0.98,
                                                              'left': 0.05,
                                                              'right': 0.94,
                                                              'wspace': 0.15,
                                                              'hspace': 0.075})
        for resp_i, resp in enumerate([single_metric, multi_metric, ratio]):
            for foi in range(n_freq):
                ax = axs[n_freq-foi-1, resp_i]
                cm_lines, cmlb, cmub = cmocean.cm.phase, 1.0, 0.2
                cs = cm_lines(np.linspace(cmlb, cmub, 5))
                ncolors = 10
                gray = np.array([0.2, 0.2, 0.2, 1])

                if resp_i == 2:  # ratio plot, use log scale
                    colorbar_vals = [0.5, 1, 2]
                    clims_low = colorbar_vals[0]
                    clims_high = colorbar_vals[-1]
                    cvals = np.array(colorbar_vals)
                    norm = matplotlib.colors.LogNorm(clims_low, clims_high,
                                                     clip=False)
                    colors = [gray, cs[foi]/cs[foi], cs[foi]]
                    levels = np.logspace(np.log10(clims_low),
                                         np.log10(clims_high), ncolors)
                    cbar_ticks = [0.5, 1, 2]
                else:  # not ratio, don't use log scale
                    vmin = 0
                    colorbar_vals = [vmin, vmax]
                    cvals = np.array(colorbar_vals)
                    colors = [[1, 1, 1, 1], cs[foi]]
                    cbar_ticks = [0, 2]
                    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax,
                                                       clip=False)
                    levels = np.linspace(vmin, vmax, ncolors)

                tuples = list(zip(map(norm, cvals), colors))
                new_cm = (matplotlib.colors.
                          LinearSegmentedColormap.from_list("", tuples))

                contour = ax.contourf(stim_rates, stim_pres_db_all,
                                      resp[..., foi].T,
                                      cmap=new_cm,
                                      levels=levels, extend='both',
                                      norm=norm)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                if resp_i > 0:
                    cb = fig.colorbar(contour, cax=cax)
                    cb.ax.yaxis.set_tick_params(pad=0)
                    cbticks = cb.get_ticks()
                    cb.set_ticks([])
                    if resp_i == 1:
                        cbtickvals = np.array(["%g" % np.round(tick, 0)
                                              for tick in colorbar_vals])
                    else:
                        cb.ax.set_yscale('log')
                        cbtickvals = np.array(["%.2g" % np.round(tick, 1)
                                              for tick in colorbar_vals])
                    cb.set_ticks(cbtickvals.astype(float))
                    cb.set_ticklabels(cbtickvals)
                    cb.minorticks_off()
                else:
                    cax.yaxis.set_ticklabels([])
                    cax.xaxis.set_ticklabels([])
                    cax.axis('off')
                ax.set_box_aspect(1)
                if foi > 0:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(stim_rates, minor=False,
                                  labels=["20 ", "  40", "80",
                                          "120 ", "160", " 200"])
                if resp_i > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_yticks(stim_pres_db_all, minor=True)
                    ax.set_yticks(stim_pres_db_all[::4], minor=False,
                                  labels=['30\n', '50', '70', '90'])
        plt.savefig('Inverted_metric_%s%s' % (model, HI_tag), dpi=300)
        plt.close('all')
