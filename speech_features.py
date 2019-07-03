# Implementing speech feature analysis in this tutorial:
# http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

#!/usr/bin/env python

import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.pyplot as plt

sample_rate, signal = wav.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
# signal = signal[:,0]    # keep only 1 column, i.e. channel
                        # pre-emphasis calculation below won't function properly with 2 channel signal
time_axis = np.linspace(0, len(signal)/sample_rate, num=len(signal))
# signal_dB_level = 20*np.log10(signal[0]) # Just take 1 channel of signal

pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

fig = plt.figure(num=1,figsize=(15,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(time_axis, signal)
ax1.set_title('Signal')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')

ax2.plot(time_axis, emphasized_signal)
ax2.set_title('Pre-Emphasised Signal')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')

plt.show()

#
# Framing
# Need to split signal into short time frames. Perform FT over these short frames as we can assume
# the signal is roughly constant over short time periods.
# We then concatenate FTs of short time frames to get frequency contours of the entire signal
#

frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

#
# Windowing: apply hamming window function
#

frames *= np.hamming(frame_length)
# frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

#
# FT and power spectrum
# n-point fft on each frame to calc. freq spectru,
#

NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#
# Filter Banks
# Apply triangular filter banks to power spectrum. Filter banks on a mel scale
#

nfilt = 40

low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

#
# Plot Spectrogram
#

time_axis = np.linspace(0, len(signal)/sample_rate, num=pow_frames.shape[0]) # num points = number of frames
freq_axis = np.linspace(0, sample_rate/2, num=filter_banks.shape[1]) # num points = number of
timebins, freqbins = np.shape(filter_banks)

fig2 = plt.figure(num=2, figsize=(15,10))
ax1 = fig2.add_subplot(111)
ax1.imshow(np.transpose(filter_banks), origin="lower", aspect="auto", interpolation="none")
# ax1.colorbar()

ax1.set_xlabel("time (s)")
ax1.set_ylabel("frequency (hz)")
plt.xlim([0, timebins])
plt.ylim([0, freqbins])
# plt.gca().set_xticklabels(time_axis)
# plt.gca().set_yticklabels(freq_axis)

# plt.specgram(emphasized_signal, NFFT=512, Fs=sample_rate)
plt.show()

#
# Calc MFCCs
#

num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

# (nframes, ncoeff) = mfcc.shape
# n = np.arange(ncoeff)
# lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
# mfcc *= lift  #*


#
# Plot spectrogram and MFCC
#

fig3 = plt.figure(num=3, figsize=(15,10))
ax1 = fig3.add_subplot(211)
ax1.imshow(np.transpose(filter_banks), origin="lower", aspect="auto", interpolation="none")
ax1.set_title('Spectrogram')
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency')

ax2 = fig3.add_subplot(212)
ax2.imshow(np.transpose(mfcc), origin="lower", aspect="auto", interpolation="none")
ax2.set_title('MFCCs')
ax2.set_xlabel('Time')
ax2.set_ylabel('Channel Index')

plt.show()


