import os
import sys
import fnmatch

import numpy as np
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

datapath = 'data/'
savepath = 'traindata/'

mfcclist = []
for filename in os.listdir(datapath):
    if fnmatch.fnmatch(filename, '*mf.npy'):
        mfcclist.append(filename)

# mfcclist = ['a01mf.npy'] # debug
# mfcclist = mfcclist[:3] # debug

INF = 1e+6

FRAME_SIZE = 2000
MFCC_SIZE = 40
PHONEME_SIZE = 36

trainmfcc = np.zeros((1, MFCC_SIZE))
trainppg = np.zeros((1, PHONEME_SIZE))

for mfccname in mfcclist:
    root, ext = os.path.splitext(mfccname)
    root = root[:-2]
    ppgname = root + 'ppg.npy'
    tmpmfcc = np.load(datapath + mfccname)
    tmpppg = np.load(datapath + ppgname)
    trainmfcc = np.vstack((trainmfcc, tmpmfcc))
    trainppg = np.vstack((trainppg, tmpppg))

trainmfcc = trainmfcc[1:]
ss = StandardScaler()
ss.fit(trainmfcc)
trainmfcc = ss.transform(trainmfcc)
joblib.dump(ss, 'standard.pkl') 
#trainmfcc = sp.stats.zscore(trainmfcc, axis=0, ddof=1)
#trainmfcc = sp.stats.zscore(trainmfcc, axis=0, ddof=1)+0.5
trainppg = trainppg[1:]

#np.save(datapath + 'flattenmfc.npy', trainmfcc)
#np.save(datapath + 'flattenppg.npy', trainppg)

mfccpad = np.zeros((FRAME_SIZE-np.shape(trainmfcc)[0]%FRAME_SIZE, MFCC_SIZE))
ppgpad = np.zeros((FRAME_SIZE-np.shape(trainppg)[0]%FRAME_SIZE, PHONEME_SIZE))
trainmfcc = np.vstack((trainmfcc, mfccpad))
trainppg = np.vstack((trainppg, ppgpad))

trainmfcc = trainmfcc.reshape((-1, FRAME_SIZE, MFCC_SIZE))
trainppg = trainppg.reshape((-1, FRAME_SIZE, PHONEME_SIZE))

np.save(savepath + 'trainmfc.npy', trainmfcc)
np.save(savepath + 'trainppg.npy', trainppg)


