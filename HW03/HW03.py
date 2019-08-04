import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

rate, signal = wav.read('hide.wav')  # 雙聲道
signal = np.asarray([data[0] for data in signal])  # 取一聲道
sigSize = np.size(signal)
time = np.linspace(0, sigSize, sigSize) / rate

normal = signal.astype(float, copy=True) / 2**15

plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title("Original Signal")
plt.subplot(2, 1, 2)
plt.plot(time, normal)
plt.title("After Normalization")
plt.show()
