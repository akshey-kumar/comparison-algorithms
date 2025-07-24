import numpy as np
from numpy.random import rand, randn, randint
from dPCA import dPCA
import matplotlib.pyplot as plt


'''
The required initialization parameters are:

    X: A multidimensional array containing the trial-averaged data. E.g. X[n,t,s,d] could correspond to the mean 
    response of the n-th neuron at time t in trials with stimulus s and decision d. The observable (e.g. neuron 
    index) needs to come first. 
    labels: Optional; list of characters with which to describe the parameter axes, 
    e.g. 'tsd' to denote time, stimulus and decision axis. All marginalizations (e.g. time-stimulus) are refered to 
    by subsets of those characters (e.g. 'ts'). 
    n_components: Dictionary or integer; if integer use the same number 
    of components in each marginalization, otherwise every (key,value) pair refers to the number of components (
    value) in a marginalization (key).
    '''
# number of neurons, time-points and stimuli
N, T, S = 100, 250, 6

# noise-level and number of trials in each condition
noise, n_samples = 0.2, 10

# build two latent factors
zt = (np.arange(T) / float(T))
zs = (np.arange(S) / float(S))

# build trial-by trial data
trialR = noise * randn(n_samples, N, S, T)
trialR += randn(N)[None, :, None, None] * zt[None, None, None, :]
trialR += randn(N)[None, :, None, None] * zs[None, None, :, None]

# trial-average data
R = np.mean(trialR, 0)

# center data
R -= np.mean(R.reshape((N, -1)), 1)[:, None, None]

dpca = dPCA.dPCA(labels='st', regularizer='auto')
dpca.protect = ['t']

Z = dpca.fit_transform(R,trialR)

time = np.arange(T)

plt.figure(figsize=(16, 7))
plt.subplot(131)
for s in range(S):
    plt.plot(time, Z['t'][0, s])
plt.title('1st time component')

plt.subplot(132)
for s in range(S):
    plt.plot(time, Z['s'][0, s])
plt.title('1st stimulus component')

plt.subplot(133)
for s in range(S):
    plt.plot(time, Z['st'][0, s])

plt.title('1st mixing component')
plt.show()

