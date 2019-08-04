from onesidespectra import One_sided_spectra
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

rate, signal = wav.read('./a.wav')
Size = np.size(signal)
time = np.linspace(0, Size, Size) / rate
after_signal = np.array(signal, copy=True)
for i in range(1, len(after_signal)):
    after_signal[i] = signal[i] - 0.98 * signal[i - 1]

wav.write('./AfHightPass.wav', rate, after_signal)
frame = signal[10000:10512]
after_frame = after_signal[10000:10512]
x, y = One_sided_spectra(frame, rate)
afx, afy = One_sided_spectra(after_frame, rate)
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Original Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sound Pressure Level (dB)")
plt.subplot(2, 1, 2)
plt.plot(afx, afy)
plt.title("After Pre-emphasis")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sound Pressure Level (dB)")
plt.savefig('HW08.png')
plt.show()