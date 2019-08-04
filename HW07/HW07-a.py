import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

rate, signal = wav.read('a.wav')
sigSize = np.size(signal)
time = np.linspace(0, sigSize, sigSize) / rate
normal = signal / 2**15
sample = normal[21000:21000+1024]
plt.subplot(2, 1, 1)
plt.subplots_adjust(hspace=0.5)
plt.plot(time, normal)
plt.plot([21000/rate, 21000/rate], [0.5, -0.5], 'r', lw=1)
plt.plot([22024/rate, 22024/rate], [0.5, -0.5], 'r', lw=1)
plt.title("a.wav")
plt.xlabel("time(seconds)")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, 1024, 1024), sample)
plt.xlabel("framesize=1024")
plt.ylabel("Amplitude")
plt.savefig('a.png')
plt.show()
