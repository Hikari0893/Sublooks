import sys
from os import remove
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from sklearn.metrics import mean_squared_error

from Hamming_window_Az import hamming_window
from zero_pad_freq import zero_pad_freq
from custom_bytescale import custom_bytescale
from rebin import rebin
from Joint_Plots import calculate_1D_spectrum_joint
from Hist import display_histogram
from display_1D_spectrum_norm import display_1D_spectrum_norm
from utils import cos2mat

slc_path = "./data/koeln_slc.npy"
slc_raw = np.load(slc_path)
slc = slc_raw[0:500,0:500,0] + 1j* slc_raw[0:500,0:500,1]
fft_img = fftshift(fft(slc,axis=1))

sub_a_path = "sublook1.npy"
sub_b_path = "sublook2.npy"

sub_a = np.load(sub_a_path)
sub_b = np.load(sub_b_path)

maxplot = np.mean(np.abs(slc)*3)
plt.figure()
plt.imshow(np.abs(sub_a[0:500,0:500]), vmax=maxplot, vmin=0, cmap="gray")

plt.figure()
plt.imshow(np.abs(sub_b[0:500,0:500]), vmax=maxplot, vmin=0, cmap="gray")

plt.figure()
plt.imshow(np.abs(slc), vmax=maxplot, vmin=0, cmap="gray")

twolook_amp = np.sqrt((np.abs(sub_a[0:500,0:500])**2 + np.abs(sub_b[0:500,0:500])**2)/2)

plt.figure()
plt.imshow(twolook_amp, vmax=maxplot, vmin=0, cmap="gray")

plt.show()
