import os
import sys

sys.path.append('sptk')

import numpy as np

from sptk.sptktools import w2r
from sptk.extract import ext_mcep, ext_mfcc, ext_pitch, ext_f0
from sptk.converter import mcep2vec, pitch2vec, vec2mcep, vec2pitch, synthesize

wavpath = 'main/'
datapath = 'main/'
savepath = 'main/'

if __name__ == '__main__':
    mcep = np.load('main/resultmcep.npy')
    pitch = np.load('main/resultpitch.npy')
    
    s_mname = 'main/result.mcep'
    s_pname = 'main/result.pitch'
    s_rname = savepath + 'result.raw'
    s_wname = savepath + 'result.wav'
    
    vec2mcep(mcep, s_mname)
    vec2pitch(pitch, s_pname)
    synthesize(s_pname, s_mname, s_rname, s_wname)
