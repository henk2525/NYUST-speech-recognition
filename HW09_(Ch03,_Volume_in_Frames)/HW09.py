import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from DSPbox import frameMat
frameSize = 512
overlap = 128

if __name__ == '__main__':
    rate, signal = wav.read('HappyNewYear.wav')
    size = np.size(signal)
    time = np.linspace(0, size, size) / rate

    signal = signal / 2**15

    enframe = frameMat(signal, frameSize, overlap)
    absv = np.asarray([np.sum(np.abs(f)) for f in enframe.T])
    logv = np.asarray([10 * np.log10(np.sum(f ** 2)) for f in enframe.T])

    # framecount = enframe.shape[1]
    frameTime = (np.linspace(0, enframe.shape[1], enframe.shape[1]) * (frameSize-overlap)) / rate

    plt.subplots_adjust(hspace=1)
    plt.subplot(3, 1, 1)
    plt.plot(time, signal)
    plt.title("HappyNewYear.wav")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.subplot(3, 1, 2)
    plt.plot(frameTime, absv)
    plt.title("Abs-Sum Volume (Framesize = 512, Overlap = 128)")
    plt.xlabel("Time(s)")
    plt.ylabel("Volume Abs_Sum")
    plt.subplot(3, 1, 3)
    plt.plot(frameTime, logv)
    plt.title("Log-squared Sum Volume (Framesize = 512, Overlap = 128)")
    plt.xlabel("Time(s)")
    plt.ylabel("Volume decibels")
    plt.savefig('HW09.png')
    plt.show()
