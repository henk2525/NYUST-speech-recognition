import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

rate, signal = wav.read('hello.wav')  # 單聲道
sigSize = np.size(signal)
time = np.linspace(0, sigSize, sigSize) / rate


encrypt = np.copy(signal)
for i in range(sigSize):
    if signal[i] > 0:
        encrypt[i] = 1 - signal[i]
    elif signal[i] < 0:
        encrypt[i] = -1 - signal[i]
encrypt = np.flipud(encrypt)          # 將矩陣倒序

wav.write('after.wav', rate, encrypt)

plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title("Original Signal")
plt.subplot(2, 1, 2)
plt.plot(time, encrypt)
plt.title("After Encrypt")
plt.show()
