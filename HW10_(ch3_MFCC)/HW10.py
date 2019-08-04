import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa as lb
label = ['1', '2', '3']

if __name__ == '__main__':
    rate, signal = wav.read('hello.wav')
    size = np.size(signal)
    time = np.linspace(0, size, size) / rate
    n_mfcc = 13
    n_fft = rate * 0.025
    hop_length = rate * 0.01

    signal = signal / 2**15
    mfcc = lb.feature.mfcc(signal, rate, n_mfcc=n_mfcc, n_fft=int(n_fft), hop_length=int(hop_length))
    mfcc_delta = lb.feature.delta(mfcc)
    mfcc_delta2 = lb.feature.delta(mfcc, order=2)
    frameCount = np.arange(np.size(mfcc[0]))

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(19.2, 10.8))
    axs[0].plot(frameCount, mfcc[0:3, :].T)
    axs[0].set(ylabel='amplitude', title='MFCC')
    axs[1].plot(frameCount, mfcc_delta[0:3, :].T)
    axs[1].set(ylabel='amplitude', title=r'MFCC-$\Delta$')
    lines = axs[2].plot(frameCount, mfcc_delta2[0:3, :].T)
    axs[2].set(xlabel='frame index', ylabel='amplitude', title=r'MFCC-$\Delta^2$')
    plt.legend(lines[0:3], ['1', '2', '3'])
    plt.savefig('HW10.png')
    plt.show()


