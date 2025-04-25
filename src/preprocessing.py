import pandas as pd
import numpy as np
import numpy.matlib
import math
import scipy
import scipy.signal
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import os, sys, re
import matplotlib.pyplot as plt
import pywt
from matplotlib import mlab
np.float_ = np.float64
from skfda.preprocessing.smoothing.kernel_smoothers import LocalLinearRegressionSmoother
from skfda.representation.basis import BSpline, Fourier
from skfda.datasets import make_sinusoidal_process
from skfda import FDataGrid
from skfda.representation.interpolation import SplineInterpolation



################# COMERT PREPROCESSING #######################

'''Function [zero_matrix,zero_index] = get_zero_matrix(signal)
%% get_zero_matrix find the zero points. 
% Inputs:
%   signal: Non-stress test signal
% Outputs:
%   zero_matrix   : A matrix consists of zero point (start-end-count)
%   zero_index    : A vector consists of indexes that the zero points '''
def get_zero_matrix(signal):
    
    signal = pd.Series(signal)
    
    zero_index = np.where(signal == 0)[0]
    zero_matrix = []
    
    if zero_index.size != 0:
        i = 0
        
        zero_matrix.append([zero_index[0], 0, 0])
        for j in range(len(zero_index)-1):
            if zero_index[j+1] != zero_index[j]+1:
                zero_matrix[i][1] = zero_index[j]
                i += 1
                zero_matrix.append([zero_index[j+1], 0, 0])
        
        
        zero_matrix = np.array(zero_matrix)
        zero_matrix[i,1] = zero_index[-1]
        zero_matrix[:,2] = zero_matrix[:,1] - zero_matrix[:,0] + 1

    else:
        zero_index = 0
        zero_matrix = 0
    
    return zero_matrix, zero_index


'''Function signal = remove_long_gaps(signal)
%% remove long gaps (zeros) from signal. 
% Inputs:
%   signal: Non-stress test signal
% Outputs:
%   zero_matrix   : Signal without long gaps '''
#same code as matlab, with [] values to nan and drop
def remove_long_gaps(signal, secs=15):
    
    signal = pd.Series(signal)

    # long gaps > 15 secs
    freq = 4
    
    zero_matrix, zero_index = get_zero_matrix(signal)
    
    if np.isscalar(zero_matrix) == False:
        gap = zero_matrix[zero_matrix[:,2] >= secs*freq]
        (m,n) = gap.shape
        indices = []
        if m>=1:
            for j in range(m):
                indices.extend(range(gap[j][0], gap[j][1] + 1))
    
        indices = np.array(indices).flatten()
        signal[indices] = np.nan        #in matlab is = []
        

    signal = signal.dropna().reset_index(drop=True)

    return signal


'''Function my_wrcoef(X, coef_type, coeffs, wavename, level)
%% get coefficients of wave decomposition. 
% Inputs:
%   X: signal
%   coef_type: a or d
%   coeffs: coefficients
%   wavename: type of wave
%   level: level of wave decomposition
% Outputs:
%   coefficients'''
def my_wrcoef(X, coef_type, coeffs, wavename, level):
    
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))



'''Function wave_decomposition(signal)
%% wave decomposition. 
% Inputs:
%   signal: signal
% Outputs:
%   signal: signal after wave decomposition'''
#same code as matlab if wavedec and wrcoef are correct
def wave_decomposition(signal):
    
    level = 5
    db = "db5"
    X = signal
    coeffs = pywt.wavedec(X, db, level=level)
    
    A5 = my_wrcoef(X, 'a', coeffs, db, level)
    D5 = my_wrcoef(X, 'd', coeffs, db, level)
    D4 = my_wrcoef(X, 'd', coeffs, db, 4)
    D3 = my_wrcoef(X, 'd', coeffs, db, 3)
    D2 = my_wrcoef(X, 'd', coeffs, db, 2)
    D1 = my_wrcoef(X, 'd', coeffs, db, 1)
    signal = A5 + D5 + D4 + D3
    
    
    return signal


'''Function [missing_value] = missing_value_estimation(signal, ind_start, ind_stop, n)
% missing_value_estimation estimates the baseline values. 
% Inputs:
%   signal           : non-stress test signal
%   ind_start        : start index
%   ind_stop         : stop index
%   n                : product dot count
% Output:
%   missing_value    : return estimated missing value'''
#same code as matlab, except wave decomposition
def missing_value_estimation(signal, ind_start, ind_stop, n):
    
    np.random.seed(0)
    
    signal = pd.Series(signal)
    
    if ind_start > 1 and ind_stop < len(signal)-2:
        sig  = [signal[ind_start - 1], signal[ind_stop + 1]]
    else:
        sig = [signal.mean(), signal.mean() + signal.std()]
    
    sigma = signal.std()
    se = sigma / np.sqrt(n)
    
    
    # set the signal value depend on angle
    p1 = np.arange(0, np.pi/2, np.pi/n)  
    p1 = np.sin(p1)+1    # Up to signal value 
    p2 = np.arange(np.pi/2, np.pi, np.pi/n)
    p2 = np.sin(p2)+1    # Down to signal value
    xq = [*p1, *p2]         # Interpolation vector
    
    # if np.isnan(se) or se <= 0:
    #     r = np.zeros(len(np.arange(0, np.pi / 2, np.pi / n)) * 2)
    # else:
    #     r = np.random.randint(np.ceil(-se), np.ceil(se), size=len(xq)) + np.random.rand(1)
    r = np.random.randint(np.ceil(-se), np.ceil(se), size=len(xq)) + np.random.rand(1)
    
    xs = [1, 2]
    fit_object = PchipInterpolator(xs, sig)
    missing_value = fit_object(xq)
    missing_value = missing_value + r
    missing_value = missing_value[:n]
    
    #check
    if n>15*4:
        #print('wave decomposition')
        missing_value = wave_decomposition(missing_value) + np.random.rand(n)
    
    
    return missing_value



'''Function [signal] = zero_remove(signal, ind_start, ind_stop)
% ctgZeroRemove is isolated zero value from signal
% Inputs
%   signal         : Non-stress test signal 
%   ind_start      : Index start for zero value
%   ind_stop       : Index stop for zero  value
% Output
%   signal         : signal without zeros in that segment'''
def zero_remove(signal, ind_start, ind_stop):
    signal = pd.Series(signal)
    
    signal[ind_start:ind_stop+1] = missing_value_estimation(signal, ind_start, ind_stop, (ind_stop-ind_start+1))
    
    return signal


'''Function [signal] = all_zero_move(signal)
% all_zero_move removes all zero points
% Inputs
%   signal         : Non-stress test signal 
% Output
%   signal         : signal without zeros'''
def all_zero_move(signal, mins_cut=15):
    
    signal = pd.Series(signal)
    
    zero_matrix, zero_indices = get_zero_matrix(signal)
    if np.isscalar(zero_matrix) == False:
        m, n = zero_matrix.shape
        for i in range(m):
            signal = zero_remove(signal, zero_matrix[i,0], zero_matrix[i,1])
    

    
    #last X min of signal
    signal = signal[-mins_cut*60*4:]

    return signal


############################################################

'''function that calculates the preprocessing of a matrix of signals
    Inputs
        signals: matrix of signals
        type_signal: FHR or UC signal
        method: comert, comert_gaps, dahfi_gaps, dahfi_llrs, dahfi_bspline, dahfi_fourier, dahfi_llrs_gaps, dahfi_bspline_gaps or dahfi_fourier_gaps
        smoothing_parameter: for llrs method
        n_basis: for bspline and fourier method
        mins_cut: last number of minutes to consider
        secs_gap: number of max seconds to consider a small gap

    Output:
        signals: matrix of preprocessed signals
        '''
def preprocessing_signals(signals, mins_cut=15, secs_gap=15):
    # Expected length of the resulting signal
    signal_length = mins_cut * 60 * 4  # 4 samples per second

    #1) Remove long gaps (> 15 sec).
    signals = [remove_long_gaps(row, secs=secs_gap) for row in signals]
    #2) Zero remove and last 15 min of signal
    signals = [all_zero_move(row, mins_cut=mins_cut) for row in signals]


    # Add NaN padding is the signal is too short
    padded_signals = []
    for row in signals:
        if len(row) < signal_length:
            padded_row = np.pad(row, (0, signal_length - len(row)), constant_values=np.nan)
            print('padded:', len(row))
        else:
            padded_row = row

        padded_signals.append(padded_row)

    data = np.array(padded_signals)
    points = np.arange(0, data.shape[1])
    padded_signals = FDataGrid(grid_points=points, data_matrix=data)


    return padded_signals