#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 12:42:21 2023

@author: Tom Stoll
"""
import numpy as np
import matplotlib.pyplot as plt
import cochlea
from scipy.io import savemat


def plot_audiogram(freq, thesholds):
    fig, ax = plt.subplots(1)
    ax.plot(freq, thesholds, 'r')
    ax.set_ylim([100, 0])
    ax.grid(True)
    return fig, ax


def interp(x0, x1, y0, y1, x):
    y = y0 + (x-x0)*((y1-y0)/(x1-x0))
    return y


def interp_thresh(cfs, aud_freqs, aud_thresh):
    thresh = np.zeros(cfs.shape)
    # ensure aud_freqs is sorted
    assert all(a <= b for a, b in zip(aud_freqs, aud_freqs[1:]))
    for cfi, cf in enumerate(cfs):
        idx = np.searchsorted(aud_freqs, cf)
        if idx == 0:  # cfs below lowest measured freq
            thresh[cfi] = aud_thresh[0]
        elif idx == len(aud_freqs):  # cfs above highest measured freq
            thresh[cfi] = aud_thresh[-1]
        else:  # cfs in measured freq range
            thresh[cfi] = interp(aud_freqs[idx-1], aud_freqs[idx],
                                 aud_thresh[idx-1], aud_thresh[idx],
                                 cf)
    return thresh


# %%
cfs = cochlea.zilany2014.util.calc_cfs((125, 16e3, 201), 'human')

m9_freq = np.array([0.5, 1, 2, 3, 4, 6, 8])*1e3
m9_dBHL = [31, 37, 37, 39, 42, 45, 45]  # from Parthasarathy et al. (2020)
m9_interpd = interp_thresh(cfs, m9_freq, m9_dBHL)
fig3, ax3 = plot_audiogram(m9_freq, m9_dBHL)
ax3.plot(cfs, m9_interpd, 'k--')
ax3.set_title('M9')

# save values to be input to carney2015_fitaudiogram
mdict = {'cfs': cfs, 'm9': m9_interpd}
savemat('impaired_thresholds.mat', mdict)
