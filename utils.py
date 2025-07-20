from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import numpy as np
import pandas as pd

def features_dataframe(features_list):
    df = pd.DataFrame(features_list, columns=['Max','Min','Std','RMS', 'Kurtosis', 'Skewness', 'Crest Factor', 'Form Factor'])
    return df

def load_bearing_data(filepath):
    mat = loadmat(filepath)
    key = [k for k in mat.keys() if k.endswith('_DE_time')][0]
    signal = mat[key].flatten()
    features_list = []
    for i in range(0,len(signal)//1200):
        # Extract a segment of the signal
        segment = signal[i*1200:(i+1)*1200]
        # Extract features and append
        features = extract_features(segment)
        features_list.append(features)
    return features_dataframe(features_list)

def extract_features(signal):
    """Extracts statistical features from a signal's frequency spectrum."""
    N = len(signal)
    yf = fft(signal)
    # Get the magnitude of the spectrum
    spectrum = 2.0/N * np.abs(yf[0:N//2])
    
    # Statistical features
    max=np.max(spectrum)
    min=np.min(spectrum)
    std=np.std(spectrum)
    rms = np.sqrt(np.mean(spectrum**2))
    kurt = kurtosis(spectrum)
    skewness = skew(spectrum)
    crest_factor = np.max(np.abs(spectrum)) / rms
    form_factor = rms / np.mean(np.abs(spectrum))
    return [max, min, std, rms, kurt, skewness, crest_factor,form_factor]