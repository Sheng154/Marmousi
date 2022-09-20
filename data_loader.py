import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utilis import *
import segyio
from segysak.segy import get_segy_texthead
from segysak.segy import segy_loader

'''
dd = segy_loader('8-800_MIG_tide_LC.sgy')
d = np.asarray(dd.data)
'''

n_xl = 13601
dt = 0.004
t_max = 4.004
t = np.repeat(np.arange(0, t_max, dt)[None, :], n_xl, axis=0)  # [0, 4.0]
'''
clip_percentile = 98
vm = np.percentile(d, clip_percentile)
f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {d.max():.0f}'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
extent = [1, n_xl, t[0][-1] * 1000, t[0][0] * 1000]  # define extent [1, 13601, 4000, 0]
ax.imshow(d.T, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
ax.set_xlabel('CDP number')
ax.set_ylabel('TWT [ms]')
ax.set_title(f'Synthetic Seismic Section')
plt.show()
'''
marmousi = np.load('marmousi.npy')
d = integrate_trace(marmousi)
clip_percentile = 98
vm = np.percentile(d, clip_percentile)
f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {d.max():.0f}'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
extent = [1, n_xl, t[0][-1] * 1000, t[0][0] * 1000]  # define extent [1, 13601, 4000, 0]
ax.imshow(d, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
ax.set_xlabel('CDP number')
ax.set_ylabel('TWT [ms]')
ax.set_title(f'Synthetic Seismic Section')
plt.show()

clip_percentile = 98
vm = np.percentile(marmousi, clip_percentile)
f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {marmousi.max():.0f}'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
extent = [1, n_xl, t[0][-1] * 1000, t[0][0] * 1000]  # define extent [1, 13601, 4000, 0]
ax.imshow(marmousi, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
ax.set_xlabel('CDP number')
ax.set_ylabel('TWT [ms]')
ax.set_title(f'Synthetic Seismic Section')
plt.show()

envelope = np.zeros_like(synthetic)
iphase = np.zeros_like(synthetic)
ifrequency = np.ndarray((synthetic.shape[0] - 1, synthetic.shape[1]))
for i in range(synthetic.shape[1]):
    envelope[:, i], iphase[:, i], ifrequency[:, i] = inst_features(synthetic[:, i], 1 / dt)

clip_percentile = 98
vm = np.percentile(envelope, clip_percentile)
f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {envelope.max():.0f}'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
extent = [1, n_xl, t[0][-1] * 1000, t[0][0] * 1000]  # define extent [1, 13601, 4000, 0]
ax.imshow(envelope, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
ax.set_xlabel('CDP number')
ax.set_ylabel('TWT [ms]')
ax.set_title(f'Synthetic Seismic Section')
plt.show()


def butter_filter(data, cutoff, dt, highpass=True):
    order = 5
    nyq = 0.5 / dt
    normal_cutoff = cutoff / nyq
    if highpass is True:
        b, a = signal.butter(order, normal_cutoff, btype="highpass")
    else:
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    y = signal.filtfilt(b, a, data)
    return y

''' 
imp_tdom = np.load('imp_tdom.npy')    
imp_log = imp_tdom[:, 7700]
aa = imp_log[imp_log != 0]
imp_log_fir = butter_filter(aa, 5, dt, highpass=False)
plt.plot(aa)
plt.plot(imp_log_fir)
plt.show()
'''  # impedance log and filtered LFM

synthetic = np.load('marmousi.npy')
clip_percentile = 98
vm = np.percentile(synthetic, clip_percentile)
f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {synthetic.max():.0f}'
fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
extent = [1, n_xl, t[0][-1] * 1000, t[0][0] * 1000]  # define extent [1, 13601, 4000, 0]
ax.imshow(synthetic, cmap="RdBu", vmin=-vm, vmax=vm, aspect='auto', extent=extent)
ax.set_xlabel('CDP number')
ax.set_ylabel('TWT [ms]')
ax.set_title(f'Synthetic Seismic Section')
plt.show()
