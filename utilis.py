import numpy as np
from scipy.signal import hilbert
import math

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs
signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)
'''


def inst_features(signal, fs):
    """
    Compute the instantaneous frequency, amplitude and phase for a univariate signal through time T = length of the signal.
    First, Hilbert transform is applied to the signal to extract the analytical form of the signal and then
    we obtain the instantaneous frequency, amplitude and phase of the signal as a function of time.
    Parameters
    ----------
    signal : numpy array
        Array depicts a signal sampled with specified frequency
    fs : int
        Number of generated samples.
        Specify this parameter based on the sampling frequency of the initial signal.
        It might be useful to visualise the data using different fs, e.g. if sampling frequency
        of the initial signal is 1/30 s then if we want to display the results daily, we should
        set this parameter as fs = 2*60*24 (as 1 point of our data corresponds to 30s).
    Returns
    -------
    instantaneous_frequency : numpy array
        Instantaneous frequency using fs defined by the user.
    amplitude_envelope : numpy array
        Instantaneous amplitude using fs defined by the user.
    instantaneous_phase : numpy array
        Instantaneous unwrapped phase using fs defined by the user.
    phase_wrap : numpy array
        Instantaneous wrapped phase using fs defined by the user.
    phase_angle : numpy array
        Instantaneous angle phase using fs defined by the user.
        np.angle ( analytical_signal )
    References
    ----------
    1. `Example from Scipy where the Hilbert transform is applied to determine the amplitude envelope and instantaneous frequency of an amplitude-modulated signal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html>`_.
    """

    # Compute the hilbert transform
    analytical_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytical_signal)
    instantaneous_phase = np.unwrap(np.angle(analytical_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
    return amplitude_envelope, instantaneous_phase, instantaneous_frequency


def integrate_trace(signal):
    if signal.ndim > 1:
        integrate = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            trace = signal[:, i]  # 统一取列向量
            integrate[:, i] = np.cumsum(trace)
    else:
        trace = signal
        integrate = np.cumsum(trace)
    return integrate
