# https://gist.github.com/boylea/1a0b5442171f9afbf372

#!/usr/bin/env python

import numpy as np
import pyqtgraph as pg
import pyaudio
from scipy.fftpack import dct
from PyQt5 import QtCore, QtGui

FS = 10000 # Hz
CHUNKSZ = 512 # samples
nfilt = 40 # number of mel filterbanks to use
nceps = 26

# Performs audio buffering from microphone
# Kept from original rolling spectrogram code.


class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=FS,
                            input=True,
                            frames_per_buffer=CHUNKSZ)

    def read(self):
        data = self.stream.read(CHUNKSZ)
        y = np.fromstring(data, 'int16')
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class MFCCWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(MFCCWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = np.zeros((100, nceps))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # setup the correct scaling for y-axis
        ceps = np.arange(start=0, stop=nceps)
        yscale = 1.0/(self.img_array.shape[1]/ceps[-1])
        self.img.scale((1./FS)*CHUNKSZ, yscale)

        self.setLabel('left', 'Cepstrum index')

        # prepare mel filterbanks for later use
        self.fbank = self.getFilterbanks()

        self.show()

    # Changed this code to produce MFCCs instead of spectrogram data
    # 1 slice of MFCCs are produced and then added onto the end of the image.
    def update(self, chunk):
        pre_emphasis = 0.97
        emphasized_chunk = np.append(chunk[0], chunk[1:] - pre_emphasis * chunk[:-1])
        emphasized_chunk *= np.hamming(len(chunk))

        magnitude_chunk = np.absolute(np.fft.rfft(emphasized_chunk, CHUNKSZ)) # Magnitude of the FFT
        power_chunk = ((1.0 / CHUNKSZ) * (magnitude_chunk ** 2))  # Power Spectrum

        filter_bank = np.dot(power_chunk, self.fbank.T) # Mel power spectrum
        filter_bank = np.where(filter_bank == 0, np.finfo(float).eps, filter_bank)  # Numerical Stability
        filter_bank = 20 * np.log10(filter_bank)  # dB

        mfcc = dct(filter_bank, norm='ortho')[1 : (nceps + 1)] # MFCCs
        mfcc = mfcc.T

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = mfcc
        self.img.setImage(self.img_array, autoLevels = False)

    def getFilterbanks(self, nfilt=40):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (FS / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
        bin = np.floor((CHUNKSZ + 1) * hz_points / FS)

        fbank = np.zeros((nfilt, int(np.floor(CHUNKSZ / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        return fbank


if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = MFCCWidget()
    w.read_collected.connect(w.update)

    mic = MicrophoneRecorder(w.read_collected)

    # time (seconds) between reads
    interval = FS/CHUNKSZ
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(1000/interval) #QTimer takes ms

    app.exec_()
    mic.close()