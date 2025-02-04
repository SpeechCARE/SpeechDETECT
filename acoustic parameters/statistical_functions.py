import numpy as np
import scipy


def sma(signal, window_size=3):
    """Applies a simple moving average (SMA) smoothing filter."""
    signal = signal[~np.isnan(signal)]
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

def de(signal):
    """Calculates the delta (first order derivative) of a signal."""
    return np.diff(signal, n=1)

def max(frame):
    frame = frame[~np.isnan(frame)]
    return np.max(frame)

def min(frame):
    frame = frame[~np.isnan(frame)]
    return np.min(frame)

def span(frame):
    frame = frame[~np.isnan(frame)]
    return max(frame) - min(frame)

def maxPos(frame):
    frame = frame[~np.isnan(frame)]
    return np.argmax(frame)

def minPos(frame):
    frame = frame[~np.isnan(frame)]
    return np.argmin(frame)

def amean(frame):
    frame = frame[~np.isnan(frame)]
    return np.mean(frame)

def linregc1(frame):
    frame = frame[~np.isnan(frame)]
    x = np.arange(len(frame))
    try:
        m, _ = np.polyfit(x, frame, 1)
    except:
        return 0
    return m

def linregc2(frame):
    frame = frame[~np.isnan(frame)]
    x = np.arange(len(frame))
    try:
        _, t = np.polyfit(x, frame, 1)
    except:
        return 0
    return t

def linregerrA(frame):
    frame = frame[~np.isnan(frame)]
    x = np.arange(len(frame))
    try:
        m, t = np.polyfit(x, frame, 1)
    except:
        m = t = 0
    linear_fit = m * x + t
    return np.sum(np.abs(linear_fit - frame))

def linregerrQ(frame):
    frame = frame[~np.isnan(frame)]
    x = np.arange(len(frame))
    try:
        m, t = np.polyfit(x, frame, 1)
    except:
        m = t = 0
    linear_fit = m * x + t
    return np.sum((linear_fit - frame) ** 2)

def stddev(frame):
    frame = frame[~np.isnan(frame)]
    return np.std(frame)

def skewness(frame):
    frame = frame[~np.isnan(frame)]
    return scipy.stats.skew(frame)

def kurtosis(frame):
    frame = frame[~np.isnan(frame)]
    return scipy.stats.kurtosis(frame)

def quartile1(frame):
    frame = frame[~np.isnan(frame)]
    return np.percentile(frame, 25)

def quartile2(frame):
    frame = frame[~np.isnan(frame)]
    return np.percentile(frame, 50)

def quartile3(frame):
    frame = frame[~np.isnan(frame)]
    return np.percentile(frame, 75)

def iqr1_2(frame):
    frame = frame[~np.isnan(frame)]
    return quartile2(frame) - quartile1(frame)

def iqr2_3(frame):
    frame = frame[~np.isnan(frame)]
    return quartile3(frame) - quartile2(frame)

def iqr1_3(frame):
    frame = frame[~np.isnan(frame)]
    return quartile3(frame) - quartile1(frame)

def percentile1(frame):
    frame = frame[~np.isnan(frame)]
    return np.percentile(frame, 1)

def percentile99(frame):
    frame = frame[~np.isnan(frame)]
    return np.percentile(frame, 99)

def pctlrange0_1(frame):
    frame = frame[~np.isnan(frame)]
    return percentile99(frame) - percentile1(frame)

def upleveltime75(frame):
    frame = frame[~np.isnan(frame)]
    threshold = quartile3(frame)
    return np.mean(frame > threshold) * 100

def upleveltime90(frame):
    frame = frame[~np.isnan(frame)]
    min_val, max_val = np.min(frame), np.max(frame)
    threshold = min_val + 0.9 * (max_val - min_val)
    return np.mean(frame > threshold) * 100

function_names = [
    "max",
    "min",
    "span",
    "maxPos",
    "minPos",
    "amean",
    "linregc1",
    "linregc2",
    "linregerrA",
    "linregerrQ",
    "stddev",
    "skewness",
    "kurtosis",
    "quartile1",
    "quartile2",
    "quartile3",
    "iqr1_2",
    "iqr2_3",
    "iqr1_3",
    "percentile1",
    "percentile99",
    "pctlrange0_1",
    "upleveltime75",
    "upleveltime90"
]
