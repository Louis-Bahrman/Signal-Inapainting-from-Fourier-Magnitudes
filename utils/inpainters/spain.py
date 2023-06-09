#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:28:16 2023

@author: louis
"""

import numpy as np
from tqdm.auto import tqdm


def zero_all_but_k_greatest_values(a, k):
    valuesToEliminate = np.argsort(np.abs(a))[:-k]
    b = a.copy()
    b[valuesToEliminate] = 0
    return b


def inpaint(degraded_signal, missing_indices, n_iter=4000, s=1, r=32, epsilon=0.1):
    # Slight modification of the original parameters n_iter=1000, r=1, s=1
    x_hat = degraded_signal.copy()
    u = np.zeros_like(degraded_signal)
    k = s
    for i in tqdm(range(1, n_iter + 1), leave=False):
        z_hat = zero_all_but_k_greatest_values(np.fft.fft(x_hat) + u, k)
        x_hat[missing_indices] = np.fft.ifft(z_hat - u).real[missing_indices]
        # if np.linalg.norm(np.fft.fft(x_hat - z_hat))**2 <= epsilon:
        if np.linalg.norm(np.fft.fft(x_hat) - z_hat) ** 2 <= epsilon:
            return x_hat
        else:
            u = u + np.fft.fft(x_hat) - z_hat
            if i % r == 0:
                k += s
    return x_hat
