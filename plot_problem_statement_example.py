#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 00:14:32 2023

@author: louis
"""

import os
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE_BASE_UNIT = 8
DPI = 300
FONTSIZE = 30
plt.rcParams.update({"font.size": FONTSIZE, "figure.dpi": DPI, "lines.linewidth": 3, "lines.markersize": 16})

width = 14 / 0.72
height = width / 3
rouge_inria = "tab:red"

fs = int(16e3)
signal = np.load(os.path.join(".", "Data", "Example", "signal.npy"))
missing_indices = np.load(os.path.join(".", "Data", "Example", "missing_indices.npy"))

L = len(signal)
degraded_signal = signal.copy()
degraded_signal[missing_indices] = np.nan
restaured_signal = signal.copy()
known_indices = np.array(list(set(range(L)) - set(missing_indices)))
restaured_signal[known_indices] = np.nan
time = np.arange(L) / (fs / 1000)

if __name__ == "__main__":
    plt.close("all")
    plt.figure(figsize=(width, height))
    ax1 = plt.subplot(121)
    ax1.plot(time, degraded_signal, label=r"$x_v$")
    ax1.plot(time, restaured_signal, "--", label=r"$x_{\bar{v}}$", color=rouge_inria)
    ax1.set_ylabel("Amplitude")
    ax1.set_yticks([])
    ax1.set_xlabel("Time (ms)")
    ax1.legend()

    fourier_magnitudes = abs(np.fft.fft(signal))[: L // 2]
    freqs = np.fft.fftfreq(L, d=1 / 16e3)[: L // 2]
    ax2 = plt.subplot(122)
    ax2.grid()
    ax2.semilogx(freqs, fourier_magnitudes, color="tab:orange")
    ax2.set_ylabel(r"Fourier magnitudes")
    ax2.set_xlabel("Frequency (Hz)")

    plt.savefig("results/inpainting.pdf", bbox_inches="tight")
