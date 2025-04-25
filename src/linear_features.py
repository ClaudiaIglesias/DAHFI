import pandas as pd
import numpy as np
import numpy.matlib
import math
import scipy
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import os, sys, re
import matplotlib.pyplot as plt
from matplotlib import mlab


'''function LTV = ctgFeatureLTV(data,varargin)
% This function calculate the LTV feature. 
% LTV = Tmax - Tmin [ms]
% Input 
%       data    :   It represents the NST 
% Output
%       LTV     :   It represetn the Long Term Variability'''

def get_LTV(signal):
    
    sFHR = np.nanmean(np.reshape(signal, (-1,10)).T, axis=0)
    n = len(sFHR)
    j = 0
    LTV = 0
    for i in range(n-1):
        if(np.isnan(sFHR[i+1]) == True or np.isnan(sFHR[i]) == True):
            continue
        else:
            LTV = LTV + np.sqrt(sFHR[i+1]+sFHR[i])
            j += 1

    LTV = LTV/j
    
    return LTV 


'''function [delta, deltaTotal] = get_delta(data,varargin)
% This function is calculate delta value
% Input 
% data-> FHR
% Output
% delta 
% delta total
% delta = sigma(1:m)[ max(FHR(i))-min(FHR(i))/m]'''
def get_delta(signal):

    l = len(signal)
    freq = 4
    part = np.reshape(signal, (-1, 60 * freq))
    m, n = part.shape
    t = 0
    j = 0
    for i in range(m):
        max_i = np.nanmax(part[i, :])
        min_i = np.nanmin(part[i, :])

        if np.isnan(max_i) == True or np.isnan(min_i) == True:
            continue
        else:
            t = t + (max_i - min_i)
            j += 1
        
    #delta = t/m
    delta = t/j
    delta_total = np.nanmax(signal) - np.nanmin(signal)
    
    return delta, delta_total


'''Function [STV, II] = ctgFeatureSTV(data)
% This function calculates STV values of FHR
% Input
%   data    :   It represents the NST
% Output
%   STV     :   Short Time Varibility
%   II      :   Interval Index
'''
def get_STV_II(signal):

    #T24 = mean(vec2mat(nst,10)');
    sFHR = np.nanmean(np.reshape(signal, (-1,10)).T, axis=0)

    STV = 0;
    n = len(sFHR)
    j = 0
    for i in range(n-1):

        if(np.isnan(sFHR[i+1]) == True or np.isnan(sFHR[i]) == True):
            continue
        else:
            STV = STV + abs(sFHR[i+1] - sFHR[i])
            j += 1

    #STV = STV/n
    STV = STV/j
    II = STV/(np.nanstd(sFHR, axis=0))


    return STV, II


"""
This function calculates the linear features of a given set of signals."""
def get_linear_features(signals):

    mean_np = np.nanmean(signals, axis=1)
    std_np = np.nanstd(signals, axis=1)
    LTV_np = np.array([get_LTV(row) for row in signals])
    delta_np = np.array([get_delta(row) for row in signals])[:,0]
    STV_II = np.array([get_STV_II(row) for row in signals])
    STV_np = STV_II[:,0]
    II_np = STV_II[:,1]
    
    linear = np.array([mean_np, std_np, LTV_np, delta_np, STV_np, II_np])
    
    return linear