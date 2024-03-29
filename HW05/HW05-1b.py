import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

rate, signal = wav.read("tuningFork.wav")
Sigsize = np.size(signal)
time = np.linspace(0, Sigsize, Sigsize) / rate
normal = [(p-2**7)/2**7 for p in signal]
sample = normal[11000:11256]
plt.subplot(2, 1, 1)
plt.plot(time, normal)
plt.plot([11000/rate, 11000/rate], [0.3, -0.4], 'r', lw=1)
plt.plot([11256/rate, 11256/rate], [0.3, -0.4], 'r', lw=1)
plt.title("tuningFork.wav")
plt.xlabel("time(seconds)")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, 256, 256), sample)
plt.plot(5, sample[5], 'r', marker='o')
plt.plot(225, sample[225], 'r', marker='o')
plt.plot([5, 225], [sample[5], sample[225]], 'g', lw=1)
plt.xlabel("framesize=256")
plt.ylabel("Amplitude")
plt.savefig('HW05-1b.png')
plt.show()

