import numpy as np
from matplotlib import pyplot as plt

# make a time vector w/ 256 points
# and a source signal
N_cycles = 10.0
N_points = 256.0

# make a square wave w/ duty cycle <> 0.5
t = np.arange(0,N_cycles*np.pi,np.pi*N_cycles/N_points)
signal = np.sin(t)
threshold = 0.3
signal[np.where(signal>=threshold)]=1.0
signal[np.where(signal<threshold)]=-1.0

# normalize the signal (not technically
# necessary for this example, but required
# for measuring correlation of physically
# different signals)
signal = signal/signal.std()

# generate two samples of the signal
# with a temporal offset:
N = 128
offset = 5
sample_1 = signal[:N]
sample_2 = signal[offset:N+offset]

# determine the offset through cross-
# correlation
xc_num = np.abs(np.fft.ifft(np.fft.fft(sample_1)*np.fft.fft(sample_2).conjugate()))
offset_estimate = np.argmax(xc_num)
if offset_estimate>N//2:
    offset_estimate = offset_estimate - N


# for an approximate estimate of Pearson's
# correlation, we normalize by the RMS
# of individual autocorrelations:
autocorrelation_1 = np.abs(np.fft.ifft(np.fft.fft(sample_1)*np.fft.fft(sample_1).conjugate()))
autocorrelation_2 = np.abs(np.fft.ifft(np.fft.fft(sample_2)*np.fft.fft(sample_2).conjugate()))
xc_denom_approx = np.sqrt(np.max(autocorrelation_1))*np.sqrt(np.max(autocorrelation_2))
rho_approx = np.max(xc_num)/xc_denom_approx
print 'rho_approx',rho_approx

# this is an approximation because we've
# included autocorrelation of the whole samples
# instead of just the overlapping portion;
# using cropped versions of the samples should
# yield the correct correlation:
sample_1_cropped = sample_1[offset:]
sample_2_cropped = sample_2[:-offset]
# these should be identical vectors:
assert np.all(sample_1_cropped==sample_2_cropped)
autocorrelation_1_cropped = np.abs(np.fft.ifft(np.fft.fft(sample_1_cropped)*np.fft.fft(sample_1_cropped).conjugate()))
autocorrelation_2_cropped = np.abs(np.fft.ifft(np.fft.fft(sample_2_cropped)*np.fft.fft(sample_2_cropped).conjugate()))
xc_denom_exact_1 = np.sqrt(np.max(autocorrelation_1_cropped))*np.sqrt(np.max(autocorrelation_2_cropped))
rho_exact_1 = np.max(xc_num)/xc_denom_exact_1
print 'rho_exact_1',rho_exact_1

# alternatively we could try to use the
# whole sample autocorrelations and just
# scale by the number of pixels used to
# compute the numerator:
scaling_factor = float(len(sample_1_cropped))/float(len(sample_1))
print scaling_factor

plt.plot(sample_1_cropped)
plt.plot(sample_2_cropped)
plt.show()


sys.exit()




plt.plot(s1)
plt.plot(s2)
plt.show()
