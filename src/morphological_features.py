import pandas as pd
import numpy as np
import numpy.matlib
import math
import scipy
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import os, sys, re
import matplotlib.pyplot as plt


'''function [fitresult, gof, baseline] = ctgBaselineFit(l, nst)
% Create a baseline fit.
% Inputs
%      l: the vector which start 1 and stop on length of the nst
%      nst: Non-stress test signal
%  Output:
%      fitresult : a fit object representing the fit.
%      baseline : the baseline value'''

def get_baseline_fit(l, signal):

	x_data = l
	y_data = signal

	# Fit model to data.
	fit_result = UnivariateSpline(x_data, y_data, s=5e5)
	#fit_result.set_smoothing_factor(5e5) #same param as s
	baseline = np.nanmean(fit_result(l))
	#print(fit_result(l), baseline)

	ret = np.array([fit_result, baseline], dtype="object")

	return ret


'''
% ctgBaseline estimates the baseline fetal heart rate. 
% [baseline, up_border, down_border,gps,dps,fit,t,n] = ctgBaseline(nst)
% Input
%   nst : non-stress test signal / fetal heart rate signal
% Output
%   baseline    : the baseline value 
%   up_border   : the up border value  
%   down_border : the down border value 
%   gps         : the green points represent reliable segments of NST
%   dps         : the dark points represent unreliable segments of NST 
%   fit         : the baseline curve 
%   t           : the vector includes samples of the nst (1:length(nst))
%   n           : the length of the nst length(nst)
%   ratio       : the data ratio used for baseline estimation
%   se          : standard error  
'''
def get_baseline(signal):

	signal = np.array(signal)
	#print(signal)
	#print('nans: ', np.count_nonzero(np.isnan(signal)))
	#print('not nans: ', np.count_nonzero(~np.isnan(signal)))
	if np.isnan(signal).all() == True:
		print('Empty signal')
		return np.nan

	#important: nan values at this step -> nan metrics
	t = range(len(signal))
	n = len(signal)
	mu = np.nanmean(signal)
	sigma = np.nanstd(signal)
	se = sigma/np.sqrt(n)
	up_border = mu + sigma + se
	down_border = mu - sigma - se

	gps = np.where((signal <= up_border) & (signal >= down_border))[0]
	dps = np.where(((signal > up_border) & (signal < down_border)))[0]

	mu2 = np.nanmean(signal[gps])
	sigma2 = np.nanstd(signal[gps])
	se2 = sigma2/np.sqrt(len(gps))
	fit = get_baseline_fit(gps,signal[gps])
	baseline = np.nanmean(signal[gps])
	ratio = len(gps)/len(signal)

	#print('mean', mu, 'baseline', baseline)

	ret = np.array([baseline, up_border, down_border, gps, dps, fit, t, n, ratio, se], dtype="object")

	return ret



'''function [DccCount, DccMatrix, DccBpm, TotalofTransition] = ctgDCCDetection(nst,time,toppeak,varargin)
% ctgDCCDetection detects the FHR and catchs the deceleration (DCC) patterns 
% Input 
%       nst             : FHR signal 
%       time            : The minimum time for a valid DCC pattern, d : 15s
%       toppeak         : The toppeak    
%       plotting        : Plotting on/off
% Output
%       DccCount        : The number of DCC pattern
%       DccMatrix       : The DCC matrix
%       DccBpm          : DCC Count / length of the signal
%       TotalTransition : The total count of the transition'''

def get_number_decelerations(signal):

	time = 15
	toppeak = -15

	#print(np.isnan(signal).sum())

	if signal.size == 0 or np.isnan(signal).all() == True:
		print('Empty signal')
		return 0, [0, 0, 0, 0], 0, 0

	# Baseline estimation
	baseline_object = get_baseline(signal);
	fit, fit_baseline = baseline_object[5]
	n1 = baseline_object[7]

	# The difference between baseline and baseline fit
	signal_fit = fit(range(n1))

	if signal_fit.size == 0 or np.isnan(signal_fit).all() == True:
		print('Empty signal fit')
		return 0, [0, 0, 0, 0], 0, 0

	# Catching the transitions
	sign = 1;
	diff = np.sign(signal-signal_fit)
	if diff[0] < 0:
		sign = -1
		

	matrix = []
	k = 0
	start = 0
	for i in range(n1-1):
		if diff[i] != sign and np.isnan(diff[i]) == False:
			# A transition is detected!
			stop = i - 1

			# Processing the transition.
			sign = sign * -1

			# Start and stop indexes of the transition
			ind = range(start,stop+1)

			diff_aux = signal[ind]-signal_fit[ind]

			# The baseline difference between related interval. 
			if diff_aux.size == 0 or np.isnan(diff_aux).all() == True:
				continue

			peak = np.nanmin(diff_aux)

			# Saving of the transition (pattern) to the matrix.
			matrix.append([start, stop, (stop-start)/4, peak])

			# Looking for new transition/pattern
			k = k + 1
			start = i

	matrix = np.array(matrix)
	# The total count of the transition. 
	total_transition = len(matrix)

	# Isolating the short transitions (patterns)
	ele = np.where(matrix[:,2]<time)[0]
	#print(len(ele))
	matrix[ele]=np.nan
	matrix = matrix[~np.isnan(matrix).any(axis=1)]

	# Isolating the transitions (patterns) that could not reach the ...
	# top peak value. 

	ele = np.where(matrix[:,3]>toppeak)[0]
	#print(len(ele))
	matrix[ele]=np.nan
	matrix = matrix[~np.isnan(matrix).any(axis=1)]

	#print(matrix)

	c = matrix.shape[1]
	if c==4:
		dcc_count = len(matrix)
		dcc_matrix = matrix
	else:
		dcc_count = 0
		dcc_matrix = [0, 0, 0, 0]

	dcc_bpm = dcc_count/(len(signal)/4)


	return np.array([dcc_count, dcc_matrix, dcc_bpm, total_transition], dtype="object")


'''function [AccCount, AccMatrix, AccBpm, TotalofTransition] = ctgACCDetection(nst,time,toppeak,varargin)
% ctgDCCDetection detects the FHR and catchs the deceleration (ACC) patterns 
% Input 
%       nst             : FHR signal 
%       time            : The minimum time for a valid ACC pattern, d : 15s
%       toppeak         : The toppeak    
%       plotting        : Plotting on/off
% Output
%       AccCount        : The number of ACC pattern
%       AccMatrix       : The ACC matrix
%       AccBpm          : ACC Count / length of the signal
%       TotalTransition : The total count of the transition'''

def get_number_acelerations(signal):

	time = 15
	toppeak = +15

	# Baseline estimation
	baseline_object = get_baseline(signal);
	fit, fit_baseline = baseline_object[5]
	n1 = baseline_object[7]

	# The difference between baseline and baseline fit
	signal_fit = fit(range(n1))

	# Catching the transitions
	sign = 1;
	diff = np.sign(signal-signal_fit)
	if diff[0] < 0:
		sign = -1

	#print(diff)    

	matrix = []
	k = 0
	start = 0
	for i in range(n1-1):
		if diff[i] != sign and np.isnan(diff[i]) == False:
			# A transition is detected!
			stop = i - 1

			# Processing the transition.
			sign = sign * -1

			# Start and stop indexes of the transition
			ind = range(start,stop+1)

			diff_aux = signal[ind]-signal_fit[ind]

			# The baseline difference between related interval. 
			if diff_aux.size == 0 or np.isnan(diff_aux).all() == True:
				continue

			# The baseline difference between related interval. 
			peak = np.nanmax(diff_aux)

			# Saving of the transition (pattern) to the matrix.
			matrix.append([start, stop, (stop-start)/4, peak])

			# Looking for new transition/pattern
			k = k + 1
			start = i

	matrix = np.array(matrix)
	# The total count of the transition. 
	total_transition = len(matrix)

	# Isolating the short transitions (patterns)
	ele = np.where(matrix[:,2]<time)[0]
	#print(len(ele))
	matrix[ele]=np.nan
	matrix = matrix[~np.isnan(matrix).any(axis=1)]

	# Isolating the transitions (patterns) that could not reach the ...
	# top peak value. 

	ele = np.where(matrix[:,3]<toppeak)[0]
	#print(len(ele))
	matrix[ele]=np.nan
	matrix = matrix[~np.isnan(matrix).any(axis=1)]

	#print(matrix)

	c = matrix.shape[1]
	if c==4:
		acc_count = len(matrix)
		acc_matrix = matrix
	else:
		acc_count = 0
		acc_matrix = [0, 0, 0, 0]

	acc_bpm = acc_count/(len(signal)/4)

	return np.array([acc_count, acc_matrix, acc_bpm, total_transition], dtype="object")


"""
This function calculates the morphological features of the FHR signal"""
def get_morphological_features(signals):

	baselines_np = np.array([get_baseline(row) for row in signals])[:,0]
	dcc_np = np.array([get_number_decelerations(row) for row in signals])[:,0]
	acc_np = np.array([get_number_acelerations(row) for row in signals])[:,0]

	return np.array([baselines_np, dcc_np, acc_np])