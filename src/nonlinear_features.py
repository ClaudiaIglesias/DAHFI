import numpy as np


'''function apen = get_ApEn( dim, r, data, tau )
%ApEn
%   dim : embedded dimension
%   r : tolerance (typically 0.2 * std)
%   data : time-series data
%   tau : delay time for downsampling

%   Changes in version 1
%       Ver 0 had a minor error in the final step of calculating ApEn
%       because it took logarithm after summation of phi's.
%       In Ver 1, I restored the definition according to original paper's
%       definition, to be consistent with most of the work in the
%       literature. Note that this definition won't work for Sample
%       Entropy which doesn't count self-matching case, because the count 
%       can be zero and logarithm can fail.
%
%       A new parameter tau is added in the input argument list, so the users
%       can apply ApEn on downsampled data by skipping by tau. 
%---------------------------------------------------------------------
% coded by Kijoon Lee,  kjlee@ntu.edu.sg
% Ver 0 : Aug 4th, 2011
% Ver 1 : Mar 21st, 2012
%---------------------------------------------------------------------

'''
def get_ApEn(data, dim, r, tau=1):
    
    if tau > 1:
        data = data[::tau]

    N = len(data)
    result = np.zeros(2)

    for j in [1,2]:
        #print(j)
        m = dim+j-1
        phi = np.zeros(N-m+1)
        data_mat = np.zeros((m,N-m+1))

        for i in range(m):
            data_mat[i,:] = data[i:N-m+i+1]

        for i in range(N-m+1):
            #print(data_mat[:,i].shape)
            temp_mat = abs(data_mat - np.matlib.repmat(data_mat[:,i],N-m+1,1).T)
            bool_mat = (temp_mat > r).any(axis=0)
            phi[i] = sum(~bool_mat)/(N-m+1)

        result[j-1] = sum(np.log(phi))/(N-m+1)


    apen = result[0]-result[1]
    
    return apen


'''function saen = ctgFeatureSampEn( dim, r, data, tau )
% SAMPEN Sample Entropy
%   calculates the sample entropy of a given time series data

%   SampEn is conceptually similar to approximate entropy (ApEn), but has
%   following differences:
%       1) SampEn does not count self-matching. The possible trouble of
%       having log(0) is avoided by taking logarithm at the latest step.
%       2) SampEn does not depend on the datasize as much as ApEn does. The
%       comparison is shown in the graph that is uploaded.

%   dim     : embedded dimension
%   r       : tolerance (typically 0.2 * std)
%   data    : time-series data
%   tau     : delay time for downsampling (user can omit this, in which case
%             the default value is 1)
%
%---------------------------------------------------------------------
% coded by Kijoon Lee,  kjlee@ntu.edu.sg
% Mar 21, 2012
%---------------------------------------------------------------------
'''
def get_SampEn(data, dim, r, tau=1):
    
    if tau > 1:
        data = data[::tau]

    N = len(data)
    correl = np.zeros(2)
    data_mat = np.zeros((dim+1,N-dim))
    
    for i in range(dim+1):
        data_mat[i,:] = data[i:N-dim+i-1+1]

    for m in [dim, dim+1]:
        count = np.zeros(N-dim)
        temp_mat = data_mat[0:m,:]

        for i in range(N-m):
            dist = (abs(temp_mat[:,i+1:N-dim+1] - np.matlib.repmat(temp_mat[:,i],N-dim-(i+1),1).T)).max(axis=0)
            D = (dist < r)

            count[i]= sum(D)/(N-dim)

        correl[m-dim+1-1] = sum(count)/(N-dim)

    saen = np.log(correl[0]/correl[1])
        
    return saen


'''function [C, ParsedSequence] = ctgFeatureLZC(dataIn,coding,bPlot)
% FEATURELZC - Calculates Lempel-Ziv Complexity
%
% Synopsis:
% [C ParsedSequence] = featureLZC(dataIn,coding[,bPlot])
%
% Description:
%  Function that calculates Lempel Ziv Complexity
%  Algorithm based on paper by Lempel1976 - "On the complexity of finite
%  sequence"
%  The resulting complexity is normalized to data length. 
%  The missing data are considered to be NaNs
%
% Input:
%  dataIn - [1xn double] data input
%  coding - [double] Indication of the coding:
%                       - 2  - binary
%                       - 3  - ternary
%                       - [] - nocoding
%  bPlot  - [bool][optional] indication to plot the outcome or not
%
% Output:
%  C      - [float] normalized Lempel Ziv complexity
%  ParsedSequence - [nx1 cell]for debbuging purposes only
%
% Examples:
%  [C ParsedSequence] = featureLZC(dataIn,2,0)
%
% See also:
%
% About:
%  Jiri Spilka
%  http://people.ciirc.cvut.cz/~spilkjir
%
% Modifications:
%  JS:2012-11-13 - handles missing signal. Gaps are represented by NaNs
%
'''

def get_LZC(data, coding=2):

    N = len(data)
    data_coded = np.zeros(N)
    if coding == 2:
        for i in range(N-1):
            if np.isnan(data[i+1]) or np.isnan(data[i]):
                data_coded[i+1] = np.nan
            elif data[i+1] > data[i]:
                data_coded[i+1] = 1
            else:
                data_coded[i+1] = 0

    elif coding == 3:
        for i in range(N-1):
            data[i+1] = np.round(data[i+1]/0.01)*0.01
            data[i] = np.round(data[i]/0.01)*0.01
            if data[i+1] > data[i]:
                data_coded[i+1] = 1
            elif data[i+1] == data[i]:
                data_coded[i+1] = 2
            else:
                data_coded[i+1] = 0

    else:
        data_coded = data
        

    N = len(data_coded)
    c = 1
    Q = ''
    S = str(int(data_coded[0]))

    i = 0
    while i <= N-2:
        i = i+1
        SQpi = str(S)+str(Q)
        
        if not np.isnan(data_coded[i]):
            Sstring = str(int(data_coded[i]))
            Q = str(Q)+str(Sstring)
        else:
            S = str(S)+str(Q)
            c = c + 1
            Q = ''


            while i < N and np.isnan(data_coded[i]):
                i=i+1

            i = i-1
            continue

        if ((SQpi.find(Q)==-1) and (i != N)):
            S = str(S)+str(Q)
            c = c + 1
            Q = ''


    c = c + 1


    N = sum(~np.isnan(data_coded))
    b = N/np.log2(N)
    C = c/b

    return C


"""
This function calculates the nonlinear features of a given set of signals."""
def get_nonlinear_features(signals):
    
    apen015_np = np.array([get_ApEn(row,2,0.15,10) for row in signals])
    apen020_np = np.array([get_ApEn(row,2,0.20,10) for row in signals])
    sapen015_np = np.array([get_SampEn(row,2,0.15,10) for row in signals])
    sapen020_np = np.array([get_SampEn(row,2,0.20,10) for row in signals])
    LZC_np = np.array([get_LZC(row,2)for row in signals])
    
    nonlinear = np.array([apen015_np, apen020_np, sapen015_np, sapen020_np, LZC_np])
    
    return nonlinear