import os
import sys
import fnmatch

import numpy as np

datapath = 'target/'
savepath = 'targetdata/'

mceplist = []
for file in os.listdir(datapath):
    if fnmatch.fnmatch(file, '*mc.npy'):
        mceplist.append(file)

INF = 1e+6

FRAME_SIZE = 2000
PHONEME_SIZE = 36
MCEP_SIZE = 40

trainmcep = np.zeros((1, MCEP_SIZE))
trainppg = np.zeros((1, PHONEME_SIZE))

for mcepname in mceplist:
    root, ext = os.path.splitext(mcepname)
    root = root[:-2]
    ppgname = root + 'ppg.npy'
    tmpmcep = np.load(datapath + mcepname)
    tmpppg = np.load(datapath + ppgname)[0]
    trainmcep = np.vstack((trainmcep, tmpmcep))
    trainppg = np.vstack((trainppg, tmpppg))

trainmcep = trainmcep[1:]
trainppg = trainppg[1:]

mceppad = np.zeros((FRAME_SIZE-np.shape(trainmcep)[0]%FRAME_SIZE, MCEP_SIZE))
ppgpad = np.zeros((FRAME_SIZE-np.shape(trainppg)[0]%FRAME_SIZE, PHONEME_SIZE))
trainmcep = np.vstack((trainmcep, mceppad))
trainppg = np.vstack((trainppg, ppgpad))

trainmcep = trainmcep.reshape((-1, FRAME_SIZE, MCEP_SIZE))
trainppg = trainppg.reshape((-1, FRAME_SIZE, PHONEME_SIZE))

np.save(savepath + 'targetmc.npy', trainmcep)
np.save(savepath + 'targetppg.npy', trainppg)
