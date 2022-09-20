import segyio
import segysak
import numpy as np
import matplotlib.pyplot as plt

den = segyio.open('MODEL_DENSITY_1.25m.segy', ignore_geometry=True)
vel = segyio.open('MODEL_P-WAVE_VELOCITY_1.25m.segy', ignore_geometry=True)

n_il = 1
n_xl = den.bin[segyio.BinField.Traces]
n_sam = den.bin[segyio.BinField.Samples]
del_D = den.bin[segyio.BinField.Interval]
print(den.bin)

den_tr = den.trace.raw[:]
vel_tr = vel.trace.raw[:]
imp = den_tr * vel_tr
np.save('P-impedance', imp.T)

'''
rc = np.zeros_like(imp)
for i in range(n_xl):
    for j in range(n_sam - 1):
        rc[i][j] = (imp[i][j+1] - imp[i][j]) / (imp[i][j+1] + imp[i][j])
'''

dt_interval = 2 * del_D/1000./vel_tr
twt = np.cumsum(dt_interval, axis=1)   # [0, 3.13]

dt = 0.004
t_max = 4.004
t = np.repeat(np.arange(0, t_max, dt)[None, :], n_xl, axis=0)  # [0, 4.0]
imp_tdom = np.array([np.interp(x=t[i], xp=twt[i], fp=imp[i]) for i in range(n_xl)])
imp_tdom1 = np.array([np.interp(x=t[i], xp=twt[i], fp=imp[i], right=0) for i in range(n_xl)])
np.save('imp_tdom', imp_tdom1.T)

rc_tdom = np.zeros_like(imp_tdom)
for i in range(n_xl):
    for j in range(np.int(t_max/dt)):
        rc_tdom[i][j] = (imp_tdom[i][j+1] - imp_tdom[i][j]) / (imp_tdom[i][j+1] + imp_tdom[i][j])


# define function of ricker wavelet
def ricker(f, length, dt):
    t0 = np.arange(-length/2, (length)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t0**2)) * np.exp(-(np.pi**2)*(f**2)*(t0**2))
    return t0, y


f = 20            #wavelet frequency
length = 0.512    #Wavelet vector length
t0, w = ricker(f, length, dt)
synthetic = np.zeros_like(rc_tdom)
for i in range(n_xl):
    synthetic[i] = np.convolve(w, rc_tdom[i], mode='same')
np.save('marmousi', synthetic.T)
